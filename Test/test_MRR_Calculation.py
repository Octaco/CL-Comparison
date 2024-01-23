import unittest
import os

from src.main import calculate_mrr_from_distances


class TestMRRCalculation(unittest.TestCase):

    def test_mrr_calculation(self):
        a1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        a2 = [0.5, 0.4, 0.1, 0.2, 0.3, 0.6]
        a3 = [0.3, 0.2, 0.6, 0.5, 0.4, 0.1]
        a4 = [0.3, 0.1, 0.3, 0.4, 0.5, 0.6]

        b = [a1, a2, a3, a4]


        list1 = [[i / 10.0 for i in range(11) if i > 0]]

        # Create another list by reversing the first list
        list2 = [list1[0][::-1]]

        # Combine both lists into a single list
        distances = [list1[0], list2[0]]

        mrr = calculate_mrr_from_distances(distances)
        mrr2 = calculate_mrr_from_distances(list1)
        mrr3 = calculate_mrr_from_distances(list2)
        mrra = calculate_mrr_from_distances(b)

        # print(f"mmra : {mrra}")
        assert mrra == 0.5

        # print(mrr)
        assert mrr2 == 1.0
        assert mrr3 == 0.1


