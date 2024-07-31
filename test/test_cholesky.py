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
            expected_operations = 1/3 * 2**3
            self.assertEqual(expected_operations, counter.total_operations)

    def test_cholesky_inverse(self):
        L = torch.linalg.cholesky(self.matrix)
        with OperationsCounterMode() as counter:
            torch.cholesky_inverse(L)
            expected_operations = 2/3 * 2**3
            self.assertEqual(expected_operations, counter.total_operations)


if __name__ == '__main__':
    unittest.main()
