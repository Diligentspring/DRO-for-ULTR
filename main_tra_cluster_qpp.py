"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.

    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
from datasets import load_dataset

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import time
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import ultra
import numpy as np
import random

import faiss
import pickle

from ultra.learning_algorithm.dla_pbm import DLA_PBM
from ultra.learning_algorithm.prs_rank_modify import PRSrank_modify
from ultra.learning_algorithm.ips_pbm import IPS_PBM
from ultra.learning_algorithm.navie_algorithm import NavieAlgorithm
from ultra.learning_algorithm.navie_algorithm_pair import NavieAlgorithm_pair
from ultra.learning_algorithm.navie_algorithm_softmax import NavieAlgorithm_softmax
from ultra.learning_algorithm.pbm_click_model import PBM_Click_Model

from ultra.learning_algorithm.DRO_cluster_IPS import DRO_cluster_IPS_softmax
from ultra.learning_algorithm.DRO_cluster_DLA import DRO_cluster_DLA_softmax

from ultra.learning_algorithm.DRO_cluster_PBM_click_model import DRO_cluster_pbm_click_model
from ultra.learning_algorithm.DRO_cluster_regression import DRO_cluster_regression

# rank list size should be read from data
parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--dataset_dir", type=str, default="./tests/data/", help="The directory of the baidu-ultr-uva dataset.")
parser.add_argument("--ULTR_model", type=str, default="DRO-IPS",
                    help="The ULTR model to train.")
parser.add_argument("--model_dir", type=str, default="./tests/tmp_model/", help="The directory for model and "
                                                                                "intermediate outputs.")
parser.add_argument("--output_dir", type=str, default="./tests/tmp_output/", help="The directory to output results.")

parser.add_argument("--click_model_dir", type=str, default=None,
                    help="The directory that contains labels produced by the click model")
parser.add_argument("--data_format", type=str, default="ULTRA", help="The format of the data")
# model
parser.add_argument("--setting_file", type=str, default="./example/offline_setting/dla_exp_settings.json",
                    help="A json file that contains all the settings of the algorithm.")

# general training parameters
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size to use during training.")
parser.add_argument("--max_list_cutoff", type=int, default=0,
                    help="The maximum number of top documents to consider in each rank list (0: no limit).")
parser.add_argument("--selection_bias_cutoff", type=int, default=10,
                    help="The maximum number of top documents to be shown to user "
                         "(which creates selection bias) in each rank list (0: no limit).")
parser.add_argument("--max_train_iteration", type=int, default=10000,
                    help="Limit on the iterations of training (0: no limit).")
parser.add_argument("--start_saving_iteration", type=int, default=0,
                    help="The minimum number of iterations before starting to test and save models. "
                         "(0: no limit).")
parser.add_argument("--steps_per_checkpoint", type=int, default=200,  # 50
                    help="How many training steps to do per checkpoint.")

parser.add_argument("--test_while_train", type=bool, default=False,
                    help="Set to True to test models during the training process.")
parser.add_argument("--test_only", type=bool, default=False,
                    help="Set to True for testing models only.")
parser.add_argument("--ln", type=float, default=0.01,
                    help="Learning rate.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed.")
parser.add_argument("--lambda_para", type=float, default=0.1,
                    help="lambda_para_for_DRO")
parser.add_argument("--c_para", type=float, default=1000,
                    help="c_para_for_DRO")

args = parser.parse_args()

def get_traditional_vectors(info):
    keys = ['bm25', 'bm25_title', 'bm25_abstract', 'tf_idf', 'tf', 'idf', 'ql_jelinek_mercer_short',
            'ql_jelinek_mercer_long', 'ql_dirichlet', 'document_length', 'title_length', 'abstract_length']
    keys_list = [info[k] for k in keys]
    traditional_vectors = [list(row) for row in zip(*keys_list)]
    return traditional_vectors

def get_clusters(info, index):
    # keys = ['bm25', 'bm25_title', 'bm25_abstract', 'tf_idf', 'tf', 'idf', 'ql_jelinek_mercer_short',
    #         'ql_jelinek_mercer_long', 'ql_dirichlet', 'document_length', 'title_length', 'abstract_length']
    # vectors = []
    # for i in range(len(info['click'])):
    #     vector = []
    #     for k in keys:
    #         vector.append(info[k][i])
    #     vectors.append(vector)
    vectors = get_traditional_vectors(info)
    # print(vector)
    D, I = index.search(np.array(vectors), 1)
    return [int(x) for x in I]

def get_qpp_cluster(info, index):
    keys = ['bm25', 'bm25_title', 'bm25_abstract', 'tf_idf', 'tf', 'idf', 'ql_jelinek_mercer_short',
            'ql_jelinek_mercer_long', 'ql_dirichlet', 'document_length', 'title_length', 'abstract_length']
    positions = info['position']
    vectors = []
    for j in range(len(positions)):
        if positions[j] <= 10:
            vector = []
            for k in keys:
                vector.append(info[k][j])
            vectors.append(np.array(vector))
    qpp_vector = np.concatenate((np.mean(vectors, axis=0), np.std(vectors, axis=0)))
    D, I = index.search(np.array([qpp_vector]), 1)
    return int(I[0])

def get_batch_click(model, data_set, batch_size, list_size, index):
    batch_size = int(batch_size)
    length = len(data_set)
    index = index
    docid_inputs, letor_features, labels, positions = [], [], [], []
    rank_list_idxs = []
    query_lengths = []
    bm25s = []
    clusters = []

    for _ in range(batch_size):
        i = int(random.random() * length)
        rank_list_idxs.append(i)
        # batch_clusters = get_clusters(data_set[i], index)
        batch_cluster = get_qpp_cluster(data_set[i], index)
        clusters.append(batch_cluster)
        query_lengths.append(data_set[i]['query_length'])

        traditional_vectors = get_traditional_vectors(data_set[i])

        if len(data_set[i]['click']) >= list_size:
            labels.append(data_set[i]['click'][:list_size])
            positions.append(data_set[i]['position'][:list_size])
            bm25s.append(data_set[i]['bm25'][:list_size])
            # clusters.append(batch_clusters[:list_size])
            docid_inputs.append([i + len(letor_features) for i in range(len(labels[-1]))])
            # letor_features.extend(data_set[i]['query_document_embedding'][:list_size])
            letor_features.extend(traditional_vectors[:list_size])
        else:
            labels.append(data_set[i]['click'])
            positions.append(data_set[i]['position'])
            bm25s.append(data_set[i]['bm25'])
            # clusters.append(batch_clusters)
            docid_inputs.append([i + len(letor_features) for i in range(len(labels[-1]))])
            # letor_features.extend(data_set[i]['query_document_embedding'])
            letor_features.extend(traditional_vectors)

            for _ in range(list_size - len(labels[-1])):
                # letor_features.append([0.0 for _ in range(768)])
                letor_features.append([0.0 for _ in range(12)])
                labels[-1].append(0)
                positions[-1].append(0)
                bm25s[-1].append(-1)
                # clusters[-1].append(-1)
                docid_inputs[-1].append(-1)

    local_batch_size = len(docid_inputs)

    letor_features_length = len(letor_features)
    for i in range(local_batch_size):
        for j in range(list_size):
            if docid_inputs[i][j] < 0:
                docid_inputs[i][j] = letor_features_length
                labels[i][j] = 0
            if positions[i][j] > list_size:
                positions[i][j] = 0
                docid_inputs[i][j] = letor_features_length
                labels[i][j] = 0
            if bm25s[i][j] <= 14.46619701:
                labels[i][j] = 0

    batch_docid_inputs = []
    batch_labels = []
    for length_idx in range(list_size):
        # Batch encoder inputs are just re-indexed docid_inputs.
        batch_docid_inputs.append(
            np.array([docid_inputs[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create
        # labels.
        batch_labels.append(
            np.array([labels[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))

        # To do: position

    # Create input feed map
    input_feed = {}
    input_feed[model.letor_features_name] = np.array(letor_features)
    for l in range(list_size):
        input_feed[model.docid_inputs_name[l]] = batch_docid_inputs[l]
        input_feed[model.labels_name[l]] = batch_labels[l]
    # if hasattr(model, 'positions'):
    #     input_feed[model.positions] = positions
    input_feed["positions"] = positions
    input_feed["query_lengths"] = query_lengths
    input_feed["bm25s"] = bm25s
    input_feed["clusters"] = clusters
    # Create info_map to store other information
    info_map = {
        'rank_list_idxs': rank_list_idxs,
        'input_list': docid_inputs,
        'click_list': labels,
        'letor_features': letor_features,
        'positions': positions,
        'clusters': clusters
    }
    return input_feed, info_map


def get_batch_annotation(index, model, data_set, batch_size, list_size):
    batch_size = int(batch_size)
    docid_inputs, letor_features, labels = [], [], []

    num_remain_data = len(data_set) - index
    for offset in range(min(batch_size, num_remain_data)):
        i = index + offset

        traditional_vectors = get_traditional_vectors(data_set[i])

        labels.append(data_set[i]['label'])
        docid_inputs.append([i + len(letor_features) for i in range(len(labels[-1]))])
        # letor_features.extend(data_set[i]['query_document_embedding'])
        letor_features.extend(traditional_vectors)

        if len(labels[-1]) < list_size:
            for _ in range(list_size - len(labels[-1])):
                # letor_features.append([0.0 for _ in range(768)])
                letor_features.append([0.0 for _ in range(12)])
                labels[-1].append(0)
                docid_inputs[-1].append(-1)

    local_batch_size = len(docid_inputs)

    letor_features_length = len(letor_features)
    for i in range(local_batch_size):
        for j in range(list_size):
            if docid_inputs[i][j] < 0:
                docid_inputs[i][j] = letor_features_length

    batch_docid_inputs = []
    batch_labels = []
    for length_idx in range(list_size):
        # Batch encoder inputs are just re-indexed docid_inputs.
        batch_docid_inputs.append(
            np.array([docid_inputs[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create
        # labels.
        batch_labels.append(
            np.array([labels[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))

        # To do: position

    # Create input feed map
    input_feed = {}
    input_feed[model.letor_features_name] = np.array(letor_features)
    for l in range(list_size):
        input_feed[model.docid_inputs_name[l]] = batch_docid_inputs[l]
        input_feed[model.labels_name[l]] = batch_labels[l]
    # Create info_map to store other information
    info_map = {
        'input_list': docid_inputs,
        'click_list': labels,
    }
    return input_feed, info_map

def get_batch_valid_click(index, model, data_set, batch_size, list_size):
    batch_size = int(batch_size)
    docid_inputs, letor_features, labels, positions, bm25s = [], [], [], [], []

    num_remain_data = len(data_set) - index
    for offset in range(min(batch_size, num_remain_data)):
        i = index + offset

        traditional_vectors = get_traditional_vectors(data_set[i])

        if len(data_set[i]['click']) >= list_size:
            labels.append(data_set[i]['click'][: list_size])
            docid_inputs.append([i + len(letor_features) for i in range(len(labels[-1]))])
            # letor_features.extend(data_set[i]['query_document_embedding'][: list_size])
            letor_features.extend(traditional_vectors[: list_size])
            positions.append(data_set[i]['position'][: list_size])
            bm25s.append(data_set[i]['bm25'][: list_size])
        else:
            labels.append(data_set[i]['click'])
            docid_inputs.append([i + len(letor_features) for i in range(len(labels[-1]))])
            # letor_features.extend(data_set[i]['query_document_embedding'])
            letor_features.extend(traditional_vectors)
            positions.append(data_set[i]['position'])
            bm25s.append(data_set[i]['bm25'])

            if len(labels[-1]) < list_size:
                for _ in range(list_size - len(labels[-1])):
                    # letor_features.append([0.0 for _ in range(768)])
                    letor_features.append([0.0 for _ in range(12)])
                    labels[-1].append(0)
                    docid_inputs[-1].append(-1)
                    positions[-1].append(0)
                    bm25s[-1].append(-1)

    local_batch_size = len(docid_inputs)

    letor_features_length = len(letor_features)

    batch_docid_inputs = []
    batch_labels = []
    for length_idx in range(list_size):
        # Batch encoder inputs are just re-indexed docid_inputs.
        batch_docid_inputs.append(
            np.array([docid_inputs[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create
        # labels.
        batch_labels.append(
            np.array([labels[batch_idx][length_idx]
                      for batch_idx in range(local_batch_size)], dtype=np.float32))

    for i in range(local_batch_size):
        for j in range(list_size):
            if docid_inputs[i][j] < 0:
                docid_inputs[i][j] = letor_features_length
                labels[i][j] = 0
            if positions[i][j] > list_size:
                positions[i][j] = 0
                docid_inputs[i][j] = letor_features_length
                labels[i][j] = 0
            if bm25s[i][j] <= 14.46619701:
                labels[i][j] = 0

    # Create input feed map
    input_feed = {}
    input_feed[model.letor_features_name] = np.array(letor_features)
    for l in range(list_size):
        input_feed[model.docid_inputs_name[l]] = batch_docid_inputs[l]
        input_feed[model.labels_name[l]] = batch_labels[l]
    input_feed["positions"] = positions
    input_feed["bm25s"] = bm25s
    # Create info_map to store other information
    info_map = {
        'input_list': docid_inputs,
        'click_list': labels,
    }
    return input_feed, info_map


def create_model(exp_settings, data_set):
    """Create model and initialize or load parameters in session.

        Args:
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
    """

    model = ultra.utils.find_class(exp_settings['learning_algorithm'])(data_set, exp_settings)
    try:
        checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
        ckpt = torch.load(checkpoint_path)
        print("Reading model parameters from %s" % checkpoint_path)
        model.model.load_state_dict(ckpt)
        model.model.eval()
    except FileNotFoundError:
        print("Created model with fresh parameters.")
    return model


def train(exp_settings):
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    setup_seed(args.seed)

    # Prepare data.

    train_dataset = load_dataset(path=args.model_dir + "baidu-ultr_uva-mlm-ctr/baidu-ultr_uva-mlm-ctr.py",
                                 name="clicks",
                                 split="train",  # ["train", "test"]
                                 cache_dir=args.model_dir + "baidu-ultr_uva-mlm-ctr/parts",
                                 )
    # train_dataset.set_format("torch")

    annotation_dataset = load_dataset(path=args.model_dir + "baidu-ultr_uva-mlm-ctr/baidu-ultr_uva-mlm-ctr.py",
                                      name="annotations",
                                      split="test",  # ["train", "test"]
                                      cache_dir=args.model_dir + "baidu-ultr_uva-mlm-ctr/parts",
                                      )

    # print("Train Rank list size %d" % train_set.rank_list_size)
    # print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = 113
    feature_size = 12

    if args.test_while_train:
        test_dataset = annotation_dataset

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']

    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # # Pad data
    # train_set.pad(exp_settings['max_candidate_num'])
    # valid_set.pad(exp_settings['max_candidate_num'])

    # Create model based on the input layer.

    exp_settings['ln'] = args.ln
    exp_settings['model_dir'] = args.model_dir
    exp_settings['batch_size'] = args.batch_size
    exp_settings['lambda_para'] = args.lambda_para
    exp_settings['c_para'] = args.c_para

    print("Creating model...")

    ULTR_model = args.ULTR_model
    if ULTR_model == 'DRO-IPS':
        model = DRO_cluster_IPS_softmax(feature_size, exp_settings)
    elif ULTR_model == 'DRO-DLA':
        model = DRO_cluster_DLA_softmax(feature_size, exp_settings)

    # Create tensorboard summarizations.
    train_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/train_log')
    valid_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/valid_log')

    test_writer = None
    if args.test_while_train:
        test_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/test_log')
        test_output_file = open(args.model_dir + '/test_output.txt', 'w')

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_perf = None
    best_step = None

    best_perf_test = None
    best_step_test = None


    best_loss = None
    loss_best_step = None
    print("max_train_iter: ", args.max_train_iteration)

    with open('centroids_qpp.txt', 'rb') as f:
        centroids = pickle.load(f)

    # print(centroids)

    index = faiss.IndexFlatL2(24)
    index.add(centroids)

    while True:
        # Get a batch and make a step.
        start_time = time.time()

        # input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True, data_format=args.data_format)

        input_feed, info_map = get_batch_click(model, train_dataset, exp_settings['batch_size'],
                                               exp_settings['selection_bias_cutoff'], index)

        # print(input_feed)
        # print(info_map)
        # break

        step_loss, _, summary = model.train(input_feed)
        # break
        step_time += (time.time() - start_time) / args.steps_per_checkpoint
        loss += step_loss / args.steps_per_checkpoint
        current_step += 1
        # print("Training at step %s" % model.global_step, summary)
        train_writer.add_scalars("Train_loss", summary, model.global_step)

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            print("global step %d learning rate %.4f step-time %.2f loss "
                  "%.4f" % (model.global_step, model.learning_rate,
                            step_time, loss))
            previous_losses.append(loss)

            if hasattr(model, 'propensity_model'):
                propensity_file = open(args.model_dir + '/propensity.txt', 'a')
                propensity_file.write('current_step:' + str(current_step) + '\n')
                propensity = torch.squeeze(model.propensity_model(), dim=0).cpu().detach().numpy()
                for i in range(len(propensity)):
                    propensity_file.write(str(propensity[i]) + ' ')
                propensity_file.write('\n')
                propensity_file.close()

            if hasattr(model, 'propensity_model_logit'):
                propensity_file = open(args.model_dir + '/propensity.txt', 'a')
                propensity_file.write('current_step:' + str(current_step) + '\n')
                # propensity = torch.squeeze(model.propensity_model(), dim=0).cpu().detach().numpy()
                propensity = torch.squeeze(torch.softmax(model.propensity_model_logit(), dim=-1),
                                           dim=0).cpu().detach().numpy()
                for i in range(len(propensity)):
                    propensity_file.write(str(propensity[i]) + ' ')
                propensity_file.write('\n')
                propensity_file.close()

            if hasattr(model, 'name'):
                if model.name == "Regression-EM":
                    propensity_file = open(args.model_dir + '/propensity.txt', 'a')
                    propensity_file.write('current_step:' + str(current_step) + '\n')
                    propensity = torch.squeeze(model.propensity, dim=0).cpu().detach().numpy()
                    for i in range(len(propensity)):
                        propensity_file.write(str(propensity[i]) + ' ')
                    propensity_file.write('\n')
                    propensity_file.close()

            #     # Validate model
            def validate_model(data_set, is_validation= False): #with annotation
                it = 0
                count_batch = 0.0
                summary_list = []
                batch_size_list = []
                while it < len(data_set):
                    # input_feed, info_map = data_input_feed.get_next_batch(
                    #     it, data_set, check_validation=False, data_format=args.data_format)

                    input_feed, info_map = get_batch_annotation(it, model, data_set, exp_settings['batch_size'],
                                                                exp_settings['max_candidate_num'])

                    # print(exp_settings['max_candidate_num'])

                    # _, _, summary = model.validation(input_feed, is_validation=is_validation)
                    _, _, summary = model.validation(input_feed)

                    # summary_list.append(summary)
                    # deep copy the summary dict
                    summary_list.append(copy.deepcopy(summary))
                    batch_size_list.append(len(info_map['input_list']))
                    it += batch_size_list[-1]
                    count_batch += 1.0
                return ultra.utils.merge_Summary(summary_list, batch_size_list)
                # return summary_list


            if args.test_while_train:
                test_summary = validate_model(test_dataset, is_validation=False)
                test_writer.add_scalars('Test Summary while training', test_summary, model.global_step)
                test_output_file.write(str(model.global_step))

                for key, value in test_summary.items():
                    if key != 'cross_entropy_loss':
                        # print(key, value)
                        test_output_file.write(' ' + str(key) + ': ' + str(value))
                        print('test value: ' + str(key) + ': ' + str(value))
                        if key == 'dcg_5':
                            if best_perf_test == None or best_perf_test < value:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % str(
                                                                   exp_settings['learning_algorithm']) + str(
                                                                   model.global_step))
                                torch.save(model.model.state_dict(), checkpoint_path)
                                best_perf_test = value
                                best_step_test = model.global_step
                                print('Save model, test %s:%.4f,step %d' % (key, best_perf_test, best_step_test))
                            else:
                                print('best test %s:%.4f,step %d' % (key, best_perf_test, best_step_test))
                test_output_file.write('\n')

            # Save checkpoint if there is no objective metric
            if best_perf == None and current_step > args.start_saving_iteration:
                checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                torch.save(model.model.state_dict(), checkpoint_path)

            # Save checkpoint when it is the 7000th step
            if current_step == 7000:
                checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % str(exp_settings['learning_algorithm']) \
                                               + str(model.global_step))
                torch.save(model.model.state_dict(), checkpoint_path)
                # save propensity_model, if exists
                if hasattr(model, 'propensity_model'):
                    checkpoint_path_propensity = os.path.join(args.model_dir, "%s.ckpt_propensity" % \
                                                              str(exp_settings['learning_algorithm']) + str(
                        model.global_step))
                    torch.save(model.propensity_model.state_dict(), checkpoint_path_propensity)
                else:
                    pass

            if loss == float('inf'):
                break

            step_time, loss = 0.0, 0.0
            sys.stdout.flush()

            if args.max_train_iteration > 0 and current_step > args.max_train_iteration:
                print("current_step: ", current_step)
                break
    train_writer.close()
    valid_writer.close()
    if args.test_while_train:
        test_writer.close()
        test_output_file.close()

def test(exp_settings):
    # Load test data.
    print("Reading data in %s" % args.data_dir)

    annotation_dataset = load_dataset(path=args.model_dir + "baidu-ultr_uva-mlm-ctr/baidu-ultr_uva-mlm-ctr.py",
                                      name="annotations",
                                      split="test",  # ["train", "test"]
                                      cache_dir=args.model_dir + "baidu-ultr_uva-mlm-ctr/parts",
                                      )
    test_dataset = annotation_dataset
    exp_settings['max_candidate_num'] = 113
    feature_size = 12

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']
    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    exp_settings['ln'] = args.ln
    exp_settings['batch_size'] = args.batch_size
    # Create model and load parameters.

    model = NavieAlgorithm_softmax(feature_size, exp_settings)

    checkpoint_path = args.model_dir
    ckpt = torch.load(checkpoint_path)
    print("Reading model parameters from %s" % checkpoint_path)
    model.model.load_state_dict(ckpt)
    model.model.eval()

    # Create input feed
    # test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
    #                                                                           exp_settings['test_input_hparams'])

    # test_writer = SummaryWriter(log_dir=args.model_dir + '/test_log')

    rerank_scores = []
    summary_list = []
    # Start testing.

    it = 0
    count_batch = 0.0
    batch_size_list = []
    while it < len(test_dataset):
        input_feed, info_map = get_batch_annotation(it, model, test_dataset, exp_settings['batch_size'], exp_settings['selection_bias_cutoff'])
        # print(input_feed)
        _, output_logits, summary = model.validation(input_feed)
        # summary_list.append(summary)
        # deep copy the summary dict
        summary_list.append(copy.deepcopy(summary))
        batch_size_list.append(len(info_map['input_list']))
        for x in range(batch_size_list[-1]):
            rerank_scores.append(output_logits[x])
        it += batch_size_list[-1]
        count_batch += 1.0
        # print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)

    print("\n[Done]")
    test_summary = ultra.utils.merge_Summary(summary_list, batch_size_list)
    print("  eval: %s" % (
        ' '.join(['%s:%.4f' % (key, value) for key, value in test_summary.items()])
    ))

def main(_):
    exp_settings = json.load(open(args.setting_file))
    if args.test_only:
        test(exp_settings)
    else:
        train(exp_settings)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
