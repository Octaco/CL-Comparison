import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset

logger = logging.getLogger(__name__)

LOSS_FORMULATIONS = ['triplet', 'INFO_NCE', 'Soft_nearest_neighbour']
LEARNING_ARCHITECTURES = ['SimCLR', 'SimSiam', 'MoCo']

mrr_path = './data/MRR.txt'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss_formulation", default="INFO_NCE", type=str, required=False,
                        help="Loss formulation selected in the list: " + ", ".join(LOSS_FORMULATIONS))

    parser.add_argument("--learning_architecture", default=None, type=str, required=False,
                        help="Learning architecture selected in the list: " + ", ".join(LEARNING_ARCHITECTURES))

    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-2,
                        help="For distributed training: local_rank")

    parser.add_argument("--log_path", default='../logging', type=str, required=False,
                        help="Path to log files")

    parser.add_argument("--output_dir", default='../data', type=str, required=False,
                        help="Path to output files")

    args = parser.parse_args()

    args.dataset = 'codebert-base'
    args.model_name = 'microsoft/codebert-base'
    args.language = 'ruby'
    args.n_labels = 20
    args.num_epochs = 5

    args.MAX_LEN = 512
    args.TRAIN_BATCH_SIZE = 1
    args.TEST_BATCH_SIZE = 1
    args.VALID_BATCH_SIZE = 1
    args.LEARNING_RATE = 1e-5
    args.train_size = 0.8

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_data(args):
    code_search_dataset = load_dataset('code_search_net', args.language)

    # train_data
    train_data = code_search_dataset['train']

    function_code = train_data['func_code_string']
    function_documentation = train_data['func_documentation_string']

    train_df =pd.DataFrame()
    train_df['doc'] = function_documentation
    train_df['code'] = function_code

    # test_data
    test_data = code_search_dataset['test']

    function_code_test = test_data['func_code_string']
    function_documentation_test = test_data['func_documentation_string']


def main():
    args = parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(filename=args.log_path + '/log_' + args.language + '.txt',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.DEBUG)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, language: %s, loss_formulation: %s",
                   args.local_rank, device, args.n_gpu, args.language, args.loss_formulation)

    # Setup random seed
    set_seed(args)

    load_data(args)


if __name__ == "__main__":
    main()
