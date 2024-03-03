import argparse

import torch
import numpy as np
from run import visualize_losses, visualize_embeddings, visualize_multiple_embeddings
import unittest


class TestVisualization(unittest.TestCase):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_path = "../data"

    def test_visualize_losses(self):

        random_test_losses = torch.rand(10)
        random_val_losses = torch.rand(10)
        visualize_losses(random_test_losses, random_val_losses, self.args)
        print('Test passed')
        assert True

    def test_visualize_embeddings(self):
        query_embedding = np.array([0.1, 0.2, 0.3])  # Example query embedding
        positive_embedding = np.array([0.2, 0.3, 0.4])  # Example positive key embedding
        negative_embeddings = np.array([[0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [2.1, 3.2, 3.2]])  # Example list of negative key embeddings

        visualize_embeddings(self.args, 1,  query_embedding, positive_embedding, negative_embeddings)
