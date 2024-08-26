import unittest
import torch
from math import log
from src.top_k_sampling import top_k_sampling


class TestTopK(unittest.TestCase):
    def test_top_p_test(self):
        # General
        logits = torch.tensor([1.2, 0.8, 0.5, 0.3])
        sampled_token = top_k_sampling(logits, 1)

        self.assertTrue(sampled_token == torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
