import unittest

from torch.nn import Module
from torch.testing import assert_close
from torch import cuda, device, ones, stack, randint, rand, sparse_coo_tensor, zeros

from torch_ops_counter import OperationsCounterMode


class SparseMatrixModule(Module):
    def __init__(self, row, col, value, size):
        super().__init__()
        self.index = stack([row, col], dim=0)
        self.value = value
        self.size = size

    def forward(self, x):
        y = zeros(self.size, x.size(1), device=x.device)
        for i in range(self.index.size(1)):
            y[self.index[0, i]] += self.value[i] * x[self.index[1, i]]
        return y


class TestSparseMatrixMult(unittest.TestCase):
    def setUp(self):
        self.device = device("cuda" if cuda.is_available() else "cpu")

        size = 1000
        row = randint(3, size=(size // 10,), device=self.device)
        col = randint(3, size=(size // 10,), device=self.device)
        value = rand(size // 10, device=self.device)

        self.size = size
        self.sparse_matrix = sparse_coo_tensor(indices=stack([row, col], dim=0),
                                               values=value,
                                               size=(size, size),
                                               device=self.device,
                                               )
        self.sparse_matrix_module = SparseMatrixModule(row, col, value, size).to(self.device)

    def test_correctness(self):
        x = ones(self.size, 10, device=self.device)
        expected_output = self.sparse_matrix.matmul(x)
        output = self.sparse_matrix_module(x)
        assert_close(output, expected_output)

    def test_num_ops(self):
        x = ones(self.size, 1, device=self.device)
        with OperationsCounterMode() as ops_counter:
            self.sparse_matrix_module(x)
        # # Logging the operations count
        # counter_dict = ops_counter.operations_count
        # for module, operation in counter_dict.items():
        #     for op_name, op_count in operation.items():
        #         print(f"{module}::{op_name}: {op_count} OPs")
        # The number of operations should be 2 * nnz
        expected_ops = 2 * self.sparse_matrix._nnz()
        self.assertEqual(ops_counter.total_operations, expected_ops)


if __name__ == "__main__":
    unittest.main()
