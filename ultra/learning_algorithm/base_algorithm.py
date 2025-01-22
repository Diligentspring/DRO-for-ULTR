"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch
import numpy as np
from abc import ABC, abstractmethod

import ultra
import ultra.utils as utils


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
    return loss

def logistic_rank(output):
    list = []
    for i in range(output.shape[-1]):
        dif = output - torch.stack([output[:, i] for _ in range(output.shape[-1])], dim= -1)
        expo = torch.exp(dif) + 1
        log2 = torch.log2(expo)
        local_rank = torch.sum(log2, dim=-1)
        list.append(local_rank)
    rank = torch.stack(list, dim=1)
    return rank

def group_logistic_rank(output, group_weights):
    list = []
    for i in range(output.shape[-1]):
        dif = output - torch.stack([output[:, i] for _ in range(output.shape[-1])], dim= -1)
        expo = torch.exp(dif) + 1
        log2 = torch.log2(expo)
        weighted_log2 = group_weights * log2
        local_rank = torch.sum(weighted_log2, dim=-1)
        list.append(local_rank)
    rank = torch.stack(list, dim=1)
    return rank

class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = torch.optim.adagrad()
        pass


    @abstractmethod
    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a summary containing related information about the step.

        """
        pass

    @abstractmethod
    def validation(self, input_feed):
        """Run a step of the model feeding the given inputs for validating process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a summary containing related information about the step.

        """
        pass

    def remove_padding_for_metric_eval(self, input_id_list, model_output):
        output_scores = torch.unbind(model_output, dim=1)
        if len(output_scores) > len(input_id_list):
            raise AssertionError(
                'Input id list is shorter than output score list when remove padding.')
        # Build mask
        valid_flags = torch.cat((torch.ones(self.letor_features.shape[0]), torch.zeros([1])), dim=0)
        valid_flags = valid_flags.type(torch.bool)
        input_flag_list = []
        for i in range(len(output_scores)):
          index_to_remove = torch.index_select(
                    input=valid_flags, dim=0, index=input_id_list[i])
          input_flag_list.append(index_to_remove)
        # Mask padding documents
        output_scores = list(output_scores)
        for i in range(len(output_scores)):
            if self.is_cuda_avail:
                output_scores[i] = torch.where(
                    input_flag_list[i].to(device=self.cuda),
                    output_scores[i],
                    torch.ones_like(output_scores[i],device=self.cuda) * self.PADDING_SCORE
                )
            else:
                output_scores[i] = torch.where(
                    input_flag_list[i],
                    output_scores[i],
                    torch.ones_like(output_scores[i]) * self.PADDING_SCORE
                )
        return torch.stack(output_scores, dim=1)

    def ranking_model(self,model, list_size):
        """Construct ranking model with the given list size.

        Args:
            model: (BaseRankingModel) The model that is used to compute the ranking score.
            list_size: (int) The top number of documents to consider in the input docids.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        output_scores = self.get_ranking_scores(model = model,
             input_id_list= self.docid_inputs[:list_size])
        return torch.cat(output_scores, 1)

    def get_ranking_scores(self, model, input_id_list, **kwargs):
        """Compute ranking scores with the given inputs.

        Args:
            model: (BaseRankingModel) The model that is used to compute the ranking score.
            input_id_list: (list<torch.Tensor>) A list of tensors containing document ids.
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A tensor with the same shape of input_docids.

        """
            # Build feature padding
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(torch.from_numpy(np.take(letor_features, input_id_list[i],0)))
        return model.build(
            input_feature_list, **kwargs)

    def create_model(self, feature_size):
        """ Initialize the ranking model.

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        if not hasattr(self, "model"):
            model = utils.find_class(
                self.exp_settings['ranking_model'])(
                self.exp_settings['ranking_model_hparams'], feature_size)
        return model

    def create_input_feed(self, input_feed, list_size):
        """Create the input from input_feed to run the model

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.
            list_size: (int) The top number of documents to consider in the input docids.
        """
        self.labels = []
        self.docid_inputs = []
        self.letor_features = input_feed["letor_features"]
        for i in range(list_size):
            self.docid_inputs.append(input_feed[self.docid_inputs_name[i]])
            self.labels.append(input_feed[self.labels_name[i]])
        self.labels = np.transpose(self.labels)
        self.labels = torch.from_numpy(self.labels)
        if self.is_cuda_avail:
            self.labels = self.labels.to(device=self.cuda)
        self.docid_inputs = torch.as_tensor(data=np.array(self.docid_inputs), dtype=torch.int64)

    def create_summary(self, scalar_name, summarize_name, value, is_training, is_validation=True):
        """ Summarize the result of an operation

        Args:
            scalar_name: (str) A string used as the name for the result of the operation to add to Tensorboard.
            summarize_name: (str) A string used as the name for the summarization of the operation.
            value: The value of the result of the operation.
            is_training: (Boolean) if the model is in training mode or eval mode.

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        # self.writer.add_scalar(
        #     scalar_name, value, self.global_step)
        if is_training:
            self.train_summary[summarize_name] = value
        else:
            if is_validation:
                self.eval_summary[summarize_name] = value
            else:
                self.test_summary[summarize_name] = value

    def opt_step(self, opt, params):
        """ Perform an optimization step

        Args:
            opt: Optimization Function to use
            params: Model's parameters

        Returns
            The ranking model that will be used to computer the ranking score.

        """
        # for p in params:
        #     p.grad = None
        opt.zero_grad()
        self.loss.backward()
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradient = torch.nn.utils.clip_grad_norm_(
                params, self.hparams.max_gradient_norm)
        opt.step()

    def pairwise_cross_entropy_loss(
            self, pos_scores, neg_scores, propensity_weights=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(pos_scores)
        label_dis = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
        loss = softmax_cross_entropy_with_logits(
            logits = torch.cat([pos_scores, neg_scores], dim=1), labels = label_dis)* propensity_weights
        return loss

    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        # label_dis = torch.minimum(labels, 1) 这行报错了，似乎也没有用，直接注释掉
        criterion =  torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(output, labels) * propensity_weights
        return torch.mean(torch.sum(loss, dim=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, dim=1)
        sliced_label = torch.unbind(labels, dim=1)
        sliced_propensity = torch.unbind(propensity_weights, dim=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * \
                    sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                # cur_pair_loss = - \
                #     torch.exp(
                #         sliced_output[i]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j])) #这样loss很可能是负数
                cur_pair_loss = torch.log(1 + torch.exp(-cur_label_weight*(sliced_output[i]-sliced_output[j])))

                # cur_pair_loss = torch.exp(sliced_output[j]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j]))

                # if loss is None:
                #     loss = cur_label_weight * cur_pair_loss
                # loss += cur_label_weight * cur_pair_loss * cur_propensity
                if loss is None:
                    loss = cur_pair_loss
                else:
                    loss += cur_pair_loss
        batch_size = labels.size()[0]
        # return torch.sum(loss) / batch_size.type(torch.float32) #有bug
        return torch.sum(loss) / batch_size

    def softmax_loss(self, output, labels, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        # output_rem_nan = torch.nan_to_num(output)
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis)* torch.sum(weighted_labels, 1)
        return torch.sum(loss) / torch.sum(weighted_labels)

    def l2_loss(self, input):
        return torch.sum(input ** 2)/2

    def attrank_loss(self, output, labels, weights=None):
        """
        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if weights is None:
            weights = torch.ones_like(labels)
        label_dis = torch.softmax(labels, dim=-1)
        label_dis = torch.nan_to_num(label_dis)
        # output_rem_nan = torch.nan_to_num(output)
        output_dis = torch.softmax(output, dim=-1)
        loss = (-(torch.log(output_dis) * label_dis + torch.log(1 - output_dis) * (1 - label_dis))*weights).sum(dim=-1).mean()

        # loss = torch.sum(- label_dis * output, -1)

        # loss = softmax_cross_entropy_with_logits(
        #     logits = output, labels = label_dis)* torch.sum(weighted_labels, 1)

        # return torch.sum(loss) / torch.sum(weighted_labels)
        return loss

    def softmax_loss_nosum(self, output, labels, weights=None):
        if weights is None:
            weights = torch.ones_like(labels)
        # print(labels)
        label_dis = labels / torch.sum(labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        # print(label_dis)
        loss = torch.sum(- weights * label_dis * F.log_softmax(output, -1), -1).mean()
        return loss

    def softmax_loss_nodiv(self, output, labels, weights=None):
        if weights is None:
            weights = torch.ones_like(labels)
        # print(labels)
        # label_dis = labels / torch.sum(labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(labels)
        # print(label_dis)
        loss = torch.sum(- weights * label_dis * F.log_softmax(output, -1), -1).mean()
        return loss


    def logistic_loss(self, output, labels, weights=None):
        if weights is None:
            weights = torch.ones_like(labels)
        weighted_labels = labels * weights
        label_dis = torch.nan_to_num(weighted_labels)
        rank = logistic_rank(output)
        loss = torch.sum(label_dis * rank, dim=-1).mean()
        return loss

    def listMLE(self, output, labels, weights=None):
        if weights is None:
            weights = torch.ones_like(labels)
        weights_table = weights.reshape(-1)
        batch_size = labels.shape[0]

        # output = F.normalize(output, p=2, dim=1)
        # loss = -1.0 * torch.sum(output, dim=1)
        # print(loss.shape)
        output = torch.sigmoid(output)
        output = torch.nan_to_num(output)
        exp_output = torch.exp(output)
        exp_output_table = exp_output.reshape(-1)

        # print(output)
        # print(exp_output)

        # print(exp_output.shape)
        # print(exp_output_table.shape)
        sum_exp_output = torch.sum(exp_output, dim=1)
        # loss = torch.add(loss, torch.log(sum_exp_output))


        sorted_label, target_indexs = torch.sort(labels, dim=1, descending=True)
        # print(labels)
        # print(target_indexs)
        target_indexs = target_indexs.permute(1, 0).detach()
        loss = torch.zeros_like(sum_exp_output)

        # compute MLE
        for i in range(self.rank_list_size - 1):
            idx = target_indexs[i] + torch.tensor([self.rank_list_size * i for i in range(batch_size)]).to(device=self.cuda)
            y_i = torch.gather(exp_output_table, 0, idx)

            weight_i = torch.gather(weights_table, 0, idx)
            proportion = y_i/sum_exp_output
            proportion = torch.nan_to_num(proportion)
            
            # print(y_i)
            # print(sum_exp_output)
            #
            # proportion = torch.where(proportion < 0.001, 1.0, proportion)
            # proportion = torch.where(proportion > 1.0, 1.0, proportion)

            # print(proportion)
            loss = loss - torch.nan_to_num(torch.log(proportion)) * weight_i
            # print(loss)
            sum_exp_output = sum_exp_output - y_i
            # print(sum_exp_output)
        return torch.mean(loss)

    # def softmax_loss_group(self, output, labels, propensity_weights=None, group_weights=None):
    #     if propensity_weights is None:
    #         propensity_weights = torch.ones_like(labels)
    #     if group_weights is None:
    #         group_weights = torch.ones_like(labels)
    #
    #     output = torch.nan_to_num(output)
    #
    #     weighted_labels = labels * propensity_weights
    #     group_weighted_labels = weighted_labels * group_weights
    #     label_dis = group_weighted_labels / torch.sum(group_weighted_labels, 1, keepdim=True)
    #     label_dis = torch.nan_to_num(label_dis)
    #     print("label_dis")
    #     print(label_dis)
    #
    #     log_softmax = torch.log(group_weights * torch.exp(output) / torch.sum(group_weights * torch.exp(output), 1, keepdim=True))
    #     log_softmax = torch.nan_to_num(log_softmax)
    #
    #     print("log_softmax")
    #     print(log_softmax)
    #     loss = torch.sum(- label_dis * log_softmax, -1).mean()
    #
    #     # loss = torch.sum(- weights * label_dis * F.log_softmax(output, -1), -1).mean()
    #     return loss

    def softmax_loss_group(self, output, labels, propensity_weights=None, group_weights=None):
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        if group_weights is None:
            group_weights = torch.ones_like(labels[:,0])

        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        # output_rem_nan = torch.nan_to_num(output)
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis)* torch.sum(weighted_labels, 1) * group_weights
        return torch.sum(loss) / torch.sum(torch.sum(weighted_labels, 1) * group_weights)

    def logistic_loss_group(self, output, labels, propensity_weights=None, group_weights=None):
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        if group_weights is None:
            group_weights = torch.ones_like(labels)

        output = torch.nan_to_num(output)
        weighted_labels = labels * propensity_weights * group_weights
        label_dis = torch.nan_to_num(weighted_labels)
        rank = group_logistic_rank(output, group_weights)
        loss = torch.sum(label_dis * rank, dim=-1).mean()

        return loss