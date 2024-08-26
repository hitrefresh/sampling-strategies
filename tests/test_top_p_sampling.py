import unittest
import torch
from math import log
from src.top_p_sampling import top_p_sampling


class TestTopP(unittest.TestCase):
    def test_top_p_test1(self):
        # General
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([2.0, 4.0, 2.0]), 0.5
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1.0])))
        self.assertTrue(filtered_indices == torch.tensor([1]))

    def test_top_p_test2(self):
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([log(1.), log(2.), log(3.)]), 0.49
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1.0])))
        self.assertTrue(filtered_indices == torch.tensor([2]))

    def test_top_p_test3(self):
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([log(1.), log(2.), log(3.)]), 0.51
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([3/5, 2/5])))
        self.assertTrue(filtered_indices.equal(torch.tensor([2, 1])))

    def test_top_p_test4(self):
        # pm on the edges, 0, 0.0001, 0.9999, 1.0
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([1., 1., 1., 1.]), 1.0
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1/4, 1/4, 1/4, 1/4])))
        self.assertTrue(filtered_indices.equal(torch.tensor([0, 1, 2, 3])))
        

    def test_top_p_test5(self):
        # pm on the edges, 0, 0.0001, 0.9999, 1.0
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([1., 1., 1., 1.]), 0.9999
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1/4, 1/4, 1/4, 1/4])))
        self.assertTrue(filtered_indices.equal(torch.tensor([0, 1, 2, 3])))

    def test_top_p_test6(self):
        # pm on the edges, 0, 0.0001, 0.9999, 1.0
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([1., 2., 3., 4.]), 0.0
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1.])))
        self.assertTrue(filtered_indices.equal(torch.tensor([3])))

    def test_top_p_test7(self):
        # pm on the edges, 0, 0.0001, 0.9999, 1.0
        _, filtered_prob, filtered_indices = top_p_sampling(
            torch.tensor([1., 2., 3., 4.]), 0.0001
        )
        self.assertTrue(torch.allclose(filtered_prob, torch.tensor([1.])))
        self.assertTrue(filtered_indices.equal(torch.tensor([3])))

if __name__ == "__main__":
    unittest.main()
