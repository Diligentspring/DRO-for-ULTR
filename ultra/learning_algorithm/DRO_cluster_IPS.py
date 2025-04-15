"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the inverse propensity weighting algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils
import pickle

def selu(x):
    # with tf.name_scope('selu') as scope:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x >= 0.0, x, alpha * F.elu(x))

def projector(group_weight_vec):
    z = 1
    sorted_group_weight, _ = torch.sort(group_weight_vec, descending=True)
    # print(group_weight_vec)
    # print(sorted_group_weight)
    p = 0
    for i in range(group_weight_vec.shape[0]):
        if sorted_group_weight[i] > torch.mean(sorted_group_weight[:i + 1] - z):
            p = i
        else:
            break
    # print(p)
    theta = (torch.sum(group_weight_vec[:p + 1]) - z) / (p+1)
    # print(theta)
    w = group_weight_vec - theta
    w = torch.where(w > 0, w, 0)
    # print(w)
    return w  # (1, group_size)


class DRO_cluster_IPS_softmax(BaseAlgorithm):
    """The Inverse Propensity Weighting algorithm for unbiased learning to rank.

    This class implements the training and testing of the Inverse Propensity Weighting algorithm for unbiased learning to rank. See the following paper for more information on the algorithm.

    * Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
    * Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17

    """

    def __init__(self, feature_size, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build DRO_cluster_IPS_softmax.')

        self.hparams = ultra.utils.hparams.HParams(
            propensity_estimator_type='ultra.utils.propensity_estimator.RandomizedPropensityEstimator',
            # the setting file for the predefined click models.
            propensity_estimator_json='./example/PropensityEstimator/randomized_pbm_0.1_1.0_4_1.0.json',
            # learning_rate=0.05,                 # Learning rate.
            learning_rate=exp_settings['ln'],
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_loss',      # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
            lambda_para=0.01,
            c_para=1000
        )

        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.cuda = torch.device('cuda')
        self.train_summary = {}
        self.eval_summary = {}
        self.test_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.feature_size = feature_size
        self.model = self.create_model(self.feature_size)

        # self.group_weight_model = [torch.tensor(0.111641277), torch.tensor(0.114577923), torch.tensor(0.116749604),
        #                            torch.tensor(0.117303605), torch.tensor(0.117215851), torch.tensor(0.113510696),
        #                            torch.tensor(0.1108772), torch.tensor(0.103224903), torch.tensor(0.094898941)]

        with open('train_distribution_qpp.txt', 'rb') as f:
            self.train_distribution = pickle.load(f)
        self.group_weight_model = [torch.tensor(d) for d in self.train_distribution]

        # print(self.group_weight_model[0].is_leaf)
        self.exams = torch.tensor([1, 1, 0.6738, 0.4145, 0.2932, 0.2079, 0.1714, 0.1363, 0.1166, 0.0838, 0.0579])

        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.exams = self.exams.to(device=self.cuda)
            for i in range(len(self.group_weight_model)):
                self.group_weight_model[i] = self.group_weight_model[i].to(device=self.cuda)
                self.group_weight_model[i].requires_grad = True

        # print(self.group_weight_model[0].is_leaf)
        # self.propensity_estimator = ultra.utils.find_class(
        #     self.hparams.propensity_estimator_type)(
        #     self.hparams.propensity_estimator_json)

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        # self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))
        self.PAD_embed = torch.zeros(1, self.feature_size)
        self.PAD_embed = self.PAD_embed.to(dtype = torch.float32)
        self.optimizer_func = torch.optim.Adagrad

        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD
        self.lambda_para = float(exp_settings['lambda_para'])
        self.c_para = float(exp_settings['c_para'])


    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        # Output feed: depends on whether we do a backward step or not.
        # compute propensity weights for the input data.

        self.model.train()
        # self.group_weight_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)


        train_output = self.ranking_model(self.model,
            self.rank_list_size)
        # train_output = torch.nan_to_num(train_output_raw)  # the output of the ranking model may contain nan

        positions = [torch.tensor(input_feed["positions"][i]).to(device=self.cuda) for i in
                     range(len(input_feed["positions"]))]
        propensities = []
        for i in range(len(positions)):
            propensities.append(torch.gather(self.exams, 0, positions[i]))
        self.propensity = torch.stack(propensities, dim=0)
        # print(self.propensity)
        self.propensity_weights = torch.ones_like(self.propensity) / self.propensity
        # train_labels = self.labels / self.propensity

        batch_clusters = input_feed["clusters"]
        # print(batch_clusters)
        # clusters = [torch.tensor(batch_clusters[i]).to(device=self.cuda) + torch.tensor([1]).to(device=self.cuda) for i
        #             in range(len(batch_clusters))]
        clusters = torch.tensor(batch_clusters).to(device=self.cuda) + torch.tensor([1]).to(device=self.cuda)

        # self.cluster = torch.stack(clusters, dim=0)


        p0 = torch.tensor(self.train_distribution).to(device=self.cuda)


        group_weight_values = torch.stack(self.group_weight_model, dim=-1)
        group_weight_values = group_weight_values / p0
        print(group_weight_values)

        group_weight_values = torch.cat((torch.tensor([0.0]).to(device=self.cuda), group_weight_values), dim=-1)
        group_weights = []
        # print(bm25_buckets)


        self.group_weight = torch.gather(group_weight_values, 0, clusters)
        # for i in range(len(clusters)):
        #     # print(bm25_buckets[i])
        #     group_weights.append(torch.gather(group_weight_values, 0, clusters[i]))
        # self.group_weight = torch.stack(group_weights, dim=0).to(device=self.cuda)
        # print(self.group_weight)

        # print(self.group_weight)

        # self.loss = self.softmax_loss_group(train_output, train_labels, self.group_weight.detach())
        self.loss = self.softmax_loss_group(train_output, self.labels, self.propensity_weights, self.group_weight.detach())

        lambda_para = self.lambda_para
        # print(lambda_para)
        # self.group_weight_loss = -self.softmax_loss_group(train_output.detach(), train_labels, self.group_weight) \
        #                          + lambda_para * torch.norm(torch.stack(self.group_weight_model, dim=-1) - p0, p=2)
        self.group_weight_loss = -self.softmax_loss_group(train_output.detach(), self.labels, self.propensity_weights, self.group_weight) \
                                 + lambda_para * torch.norm(torch.stack(self.group_weight_model, dim=-1) - p0, p=2)

        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            for p in params:
                self.loss += self.hparams.l2_loss * self.l2_loss(p)

        opt = self.optimizer_func(self.model.parameters(), self.learning_rate)
        opt.zero_grad(set_to_none=True)
        self.loss.backward()
        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
        opt.step()

        # print(self.group_weight_model[0].is_leaf)

        # opt_group = self.optimizer_func(self.group_weight_model, self.learning_rate)
        # opt_group.zero_grad(set_to_none=True)

        self.group_weight_loss.backward()

        if self.global_step < 1000:
            # self.group_weight_lr = 1 / (1000 * self.lambda_para * (self.global_step+1))
            self.group_weight_lr = 1 / (self.c_para * self.lambda_para * (self.global_step + 1))
        else:
            self.group_weight_lr = 1 / (self.lambda_para * (self.global_step+1))
        # print(self.group_weight_lr)

        for i in range(len(self.group_weight_model)):
            if self.group_weight_model[i].grad != None:
                self.group_weight_model[i].data = self.group_weight_model[i].data - self.group_weight_lr * self.group_weight_model[i].grad
        for i in range(len(self.group_weight_model)):
            if self.group_weight_model[i].grad != None:
                self.group_weight_model[i].grad.zero_()


        # print(self.group_weight_model)

        new_group_weight_model = projector(torch.stack(self.group_weight_model, dim=-1))
        for i in range(len(self.group_weight_model)):
            self.group_weight_model[i] = torch.tensor(new_group_weight_model[i].data, requires_grad=True)

        # print(self.group_weight_model)

        self.global_step += 1
        print(" Loss %f at Global Step %d: " % (self.loss.item(),self.global_step))
        self.train_summary['loss'] = self.loss.item()
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation= False, is_validation=False):
        """Run a step of the model feeding the given inputs for validating process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                self.max_candidate_num)
        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)

            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn, metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False, is_validation)
        if is_validation:
            return None, self.output, self.eval_summary  # loss, outputs, summary.
        else:
            return None, self.output, self.test_summary  # loss, outputs, summary.