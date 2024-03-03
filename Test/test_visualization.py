import argparse

import torch
from run import visualize_losses
import unittest


class TestVisualization(unittest.TestCase):

    def test_visualize_losses(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.data_path = "../data"

        random_test_losses = torch.rand(10)
        random_val_losses = torch.rand(10)
        visualize_losses(random_test_losses, random_val_losses, args)
        print('Test passed')
        assert True
