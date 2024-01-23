import argparse
import os
import sys
import unittest

import numpy as np

from src.main import write_mrr_to_file

path = '../data/MRR.txt'
sys.path.append(os.path.abspath(os.path.join('..', 'data/MRR.txt')))

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.language = 'ruby, python'


class MyTestCase(unittest.TestCase):
    def test_something(self):

        mrr = np.random.rand(1)[0]

        write_mrr_to_file(args, mrr, True)

        with open(path, 'r') as f:
            mrr_read = f.read()

        print(mrr_read)
        print(str(mrr) + '\n')
        self.assertTrue(mrr_read.endswith(str(mrr) + '\n'))  # add assertion here


if __name__ == '__main__':
    unittest.main()
