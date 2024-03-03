import argparse
import os
import sys
import unittest

import numpy as np

from run import write_mrr_to_file

path = '../data/MRR.txt'
sys.path.append(os.path.abspath(os.path.join('..', 'data/MRR.txt')))

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.language = 'ruby'
args.loss_formulation = "INFO_NCE"

parser = argparse.ArgumentParser()

parser.add_argument("--loss_function", default="INFO_NCE", type=str, required=False)

parser.add_argument("--architecture", default=None, type=str, required=False)

parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")

parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--log_path", default='./logging', type=str, required=False,
                    help="Path to log files")

parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")

parser.add_argument("--train_batch_size", default=16, type=int, required=False, help="Training batch size")

parser.add_argument("--eval_batch_size", default=16, type=int, required=False, help="Evaluation batch size")

parser.add_argument("--learning_rate", default=1e-5, type=float, required=False, help="Learning rate")

parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="Number of training epochs")

parser.add_argument("--train_size", default=0.8, type=float, required=False, help="percentage of train dataset used"
                                                                                  "for training")

parser.add_argument("--data_path", default='../data/', type=str, required=False, help="Path to mrr file")

parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

parser.add_argument("--num_of_accumulation_steps", default=16, type=int, required=False,
                    help="Number of accumulation steps")

parser.add_argument("--num_of_negative_samples", default=15, type=int, required=False, help="Number of negative "
                                                                                            "samples")
parser.add_argument("--num_of_distractors", default=99, type=int, required=False, help="Number of distractors")
parser.add_argument("--GPU", required=False, help="specify the GPU which should be used")

args = parser.parse_args()
args.dataset = 'codebert-base'
args.model_name = 'microsoft/codebert-base'
args.MAX_LEN = 512


class MyTestCase(unittest.TestCase):
    def test_something(self):

        mrr = np.random.rand(1)[0]

        write_mrr_to_file(args, mrr, test=True)

        with open(path, 'r') as f:
            mrr_read = f.read()

        print(mrr_read)
        print(str(mrr) + '\n')
        self.assertTrue(mrr_read.endswith(str(mrr) + '\n'))  # add assertion here


if __name__ == '__main__':
    unittest.main()
