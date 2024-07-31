import unittest

import torch
from torch import tensor

from torch_operation_counter import OperationsCounterMode


class TestCholeskyFunctions(unittest.TestCase):
    def setUp(self):
        self.matrix = tensor([[6., 2.], [2., 4.]])

    def test_cholesky_decomposition(self):
        with OperationsCounterMode() as counter:
            torch.linalg.cholesky(self.matrix)
            self.assertTrue(counter.total_operations > 0)

    def test_cholesky_inverse(self):
        L = torch.linalg.cholesky(self.matrix)
        with OperationsCounterMode() as counter:
            torch.cholesky_inverse(L)
            self.assertTrue(counter.total_operations > 0)


if __name__ == '__main__':
    unittest.main()
