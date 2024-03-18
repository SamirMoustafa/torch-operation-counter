import unittest

from torch import Tensor
from torch.nn import Module

from torch_operation_counter import OperationsCounterMode


class MatrixMultiplicationModule(Module):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def forward(self):
        return self.A @ self.B


class TestMatrixMultiplication(unittest.TestCase):
    def test_correctness(self):
        A = Tensor([[1, 2],
                    [3, 4]])
        B = Tensor([[1, 2],
                    [3, 4]])
        module = MatrixMultiplicationModule(A, B)
        output = module()
        expected_output = A @ B
        self.assertTrue((output == expected_output).all())

    def test_num_ops(self):
        A = Tensor([[1, 2],
                    [3, 4]])
        B = Tensor([[1, 2],
                    [3, 4]])
        module = MatrixMultiplicationModule(A, B)
        # A \in R^{2x2}, B \in R^{2x2}, C = A @ B \in R^{2x2}
        # C_{ij} = \sum_{k=1}^{2} A_{ik} * B_{kj}
        # In each element of C, there are 2 multiplications.
        # There are 4 elements in C
        # Therefore, the total number of operations is 4 * 2 = 8
        with OperationsCounterMode() as ops_counter:
            module()
        self.assertEqual(ops_counter.total_operations, 8)


if __name__ == "__main__":
    unittest.main()