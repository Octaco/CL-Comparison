import unittest
import os

from src.main import calculate_mrr_from_distances


class TestMRRCalculation(unittest.TestCase):

    def test_mrr_calculation(self):
        list1 = [[i / 10.0 for i in range(11) if i > 0]]

        # Create another list by reversing the first list
        list2 = [list1[0][::-1]]

        # Combine both lists into a single list
        distances = [list1[0], list2[0]]

        #mrr = calculate_mrr_from_distances(distances)
        mrr2 = calculate_mrr_from_distances(list1)
        mrr3 = calculate_mrr_from_distances(list2)

        assert mrr2 == 1.0
        assert mrr3 == 0.1


