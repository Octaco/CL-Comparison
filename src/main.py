# This is a sample Python script.
import argparse
import glob
import logging
import os
import random

import pandas as pd
import numpy as np
import torch
from torch import nn

from datasets import load_dataset
from utils.CustomDataset import CustomDataset
from info_nce import InfoNCE, info_nce

from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

logger = logging.getLogger(__name__)

LOSS_FORMULATIONS = ['triplet', 'INFO_NCE', 'Soft_nearest_neighbour']
LEARNING_ARCHITECTURES = ['SimCLR', 'SimSiam', 'MoCo']


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_data(args):
    code_search_dataset = load_dataset('code_search_net', args.language)

    train_data = code_search_dataset['train']
    test_data = code_search_dataset['test']

    # columns in the dataset: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string',
    # 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens',
    # 'split_name', 'func_code_url']"

    function_code = train_data['func_code_string']
    function_documentation = train_data['func_documentation_string']

    function_code_test = test_data['func_code_string']
    function_documentation_test = test_data['func_documentation_string']

    train_df1 = pd.DataFrame()
    train_df1['text'] = function_documentation

    test_df1 = pd.DataFrame()
    test_df1['text'] = function_documentation

    train_df2 = pd.DataFrame()
    train_df2['text'] = function_code

    test_df2 = pd.DataFrame()
    test_df2['text'] = function_code

    return train_df1, train_df2, test_df1, test_df2


def main(randomt=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_formulation", default="INFO_NCE", type=str, required=False,
                        help="Loss formulation selected in the list: " + ", ".join(LOSS_FORMULATIONS))

    parser.add_argument("--learning_architecture", default=None, type=str, required=False,
                        help="Learning architecture selected in the list: " + ", ".join(LEARNING_ARCHITECTURES))

    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    args = parser.parse_args()

    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.language = 'ruby'
    args.n_labels = 20
    args.num_epochs = 10

    args.MAX_LEN = 512
    args.TRAIN_BATCH_SIZE = 1
    args.TEST_BATCH_SIZE = 1
    args.VALID_BATCH_SIZE = 1
    args.LEARNING_RATE = 1e-5
    args.train_size = 0.8



    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Load Data
    train_df1, train_df2, test_df1, test_df2 = load_data(args)

    # Create Custom Dataset
    train_size = args.train_size
    train_dataset = train_df1.sample(frac=train_size, random_state=200)
    valid_dataset = train_df1.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_df1.reset_index(drop=True)

    train_dataset2 = train_df2.sample(frac=train_size, random_state=200)
    valid_dataset2 = train_df2.drop(train_dataset.index).reset_index(drop=True)
    train_dataset2 = train_dataset.reset_index(drop=True)
    test_dataset2 = test_df2.reset_index(drop=True)

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(valid_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, args)
    validation_set = CustomDataset(valid_dataset, args)
    testing_set = CustomDataset(test_dataset, args)
    # print("len:" , testing_set.__len__())
    # print(testing_set)

    training_set2 = CustomDataset(train_dataset2, args)
    validation_set2 = CustomDataset(valid_dataset2, args)
    testing_set2 = CustomDataset(test_dataset2, args)
    # print("len2:" , testing_set2.__len__())

    # print(testing_set.__getitem__(0))

    train_params = {'batch_size': args.TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': args.TEST_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    ##########################################################
    # model
    ##########################################################

    model = RobertaModel.from_pretrained(args.model_name)
    model.to(args.device)

    if args.loss_formulation == 'triplet':
        loss_formulation = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    elif args.loss_formulation == 'INFO_NCE':
        loss_formulation = InfoNCE(negative_mode='unpaired')
    elif args.loss_formulation == 'Soft_nearest_neighbour':
        loss_formulation = torch.nn.SoftMarginLoss()
    else:
        raise ValueError("Loss formulation selected is not valid. Please select one of the following: " +
                         ", ".join(LOSS_FORMULATIONS))

    # training
    train(args, loss_formulation, model, train_params, training_set, training_set2)


    # prediction

    distances = validation(model, test_params, validation_set, validation_set2)


def validation(model, test_params, validation_set, validation_set2):
    dataloader_eval = DataLoader(validation_set, **test_params)
    dataloader_eval2 = DataLoader(validation_set2, **test_params)
    dataloader_iterator = iter(dataloader_eval)
    data2 = next(dataloader_iterator)
    all_distances = []
    for index, data1 in enumerate(dataloader_eval2):

        try:
            id_1 = data1["ids"]
            id_2 = data2["ids"]
            mask_1 = data1["mask"]
            mask_2 = data2["mask"]

            query = model(id_1, mask_1)[1]  # using pooled values
            positive_key = model(id_2, mask_2)[1]  # using pooled values

            # negative keys
            sample_indices = list(range(len(dataloader_eval2)))
            sample_indices.remove(index)
            sample_idx = random.choices(sample_indices, k=99)

            subset = torch.utils.data.Subset(validation_set2, sample_idx)
            dataloader_subset = DataLoader(subset, **test_params)

            negative_keys = []
            for index2, data in enumerate(dataloader_subset):
                id_3 = data["ids"]
                mask_3 = data["mask"]
                negative_key = model(id_3, mask_3)[1]
                negative_keys.append(torch.tensor(negative_key.clone().detach().numpy()))
                print(index2)

            negative_keys_reshaped = torch.cat(negative_keys, 0)

            # calc Euclidean distance for positive key at first position
            distances = [np.linalg.norm((query - positive_key).detach().numpy())]

            for i in range(len(negative_keys_reshaped)):
                # calc Euclidean distance for negative keys
                distance = np.linalg.norm((query - negative_keys_reshaped[i]).detach().numpy())

                distances.append(distance)

            print(distances)
            all_distances.append(distances)
            data2 = next(dataloader_iterator)

        except StopIteration:
            dataloader_iterator = iter(dataloader_eval)
            data2 = next(dataloader_iterator)
    return all_distances


def train(args, loss_formulation, model, train_params, training_set, training_set2):

    dataloader_train = DataLoader(training_set, **train_params)
    dataloader_train2 = DataLoader(training_set2, **train_params)

    for epoch in range(args.num_epochs):
        dataloader_iterator = iter(dataloader_train)
        data2 = next(dataloader_iterator)

        for index, data1 in enumerate(dataloader_train2):
            try:

                id_1 = data1["ids"]
                id_2 = data2["ids"]

                mask_1 = data1["mask"]
                mask_2 = data2["mask"]

                query = model(id_1, mask_1)[1]  # using pooled values
                positive_key = model(id_2, mask_2)[1]  # using pooled values

                # negative keys

                # subsetindices = [random.randint(0, 15) for i in range(15)]

                sample_indices = list(range(len(dataloader_train2)))
                sample_indices.remove(index)
                sample_idx = random.choices(sample_indices, k=15)
                # sample_indices.append(index)

                subset = torch.utils.data.Subset(training_set2, sample_idx)
                dataloader_subset = DataLoader(subset, **train_params)

                negative_keys = []
                for index2, data in enumerate(dataloader_subset):
                    id_3 = data["ids"]
                    mask_3 = data["mask"]
                    negative_key = model(id_3, mask_3)[1]
                    negative_keys.append(negative_key.clone().detach())

                negative_keys_reshaped = torch.cat(negative_keys, 0)
                # what is the query,
                loss = loss_formulation(query, positive_key, negative_keys_reshaped)

                print(loss)
                loss.backward()

                # for i in range(len(output_1[0])):
                #     loss = loss_function(output_1[0][i], output_2[0][i])
                #     loss.backward()
                #     print(loss)

                print("_________________________")
                print(query[0].shape)
                print(positive_key[0].shape)
                data2 = next(dataloader_iterator)

            except StopIteration:
                dataloader_iterator = iter(dataloader_train)
                data2 = next(dataloader_iterator)


            # querry, seperatly positve and negative and compare to querry.
            # -> 100 numbers each similarity between querry and positive / negative
            # training on training data, predition of cosine similarity  on test data.
            # MRR on similarity numers.


def calculate_mrr_from_distances(distances_lists):
    ranks = []

    for distances in distances_lists:
        # Find the index of the minimum distance (assuming lower distance is better)
        min_distance_index = min(range(len(distances)), key=distances.__getitem__)

        # If the minimum distance is at index 0, consider it as the rank 1
        rank = min_distance_index + 1 if min_distance_index == 0 else len(distances) + 1

        ranks.append(rank)

    mrr = sum(1 / rank if rank != 0 else 0 for rank in ranks) / len(ranks)
    return mrr



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
