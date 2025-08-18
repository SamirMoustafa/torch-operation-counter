import unittest
import torch
import torch.nn as nn
from torch_operation_counter import OperationsCounterMode


class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.m = 32
        self.n = 64
        self.p = 16
        
    def test_matrix_multiplication_operations(self):
        """Test matrix multiplication operation counting"""
        A = torch.randn(self.batch_size, self.m, self.n, device=self.device)
        B = torch.randn(self.batch_size, self.n, self.p, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Matrix multiplication: batch_size * m * n * p
            C = torch.bmm(A, B)
            
        expected_ops = self.batch_size * self.m * self.n * self.p
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_linear_layer_operations(self):
        """Test linear layer operation counting"""
        x = torch.randn(self.batch_size, self.m, device=self.device)
        weight = torch.randn(self.n, self.m, device=self.device)
        bias = torch.randn(self.n, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Linear: x @ weight.T + bias
            # Matrix multiplication: batch_size * m * n
            # Bias addition: batch_size * n
            y = torch.nn.functional.linear(x, weight, bias)
            
        expected_ops = self.batch_size * self.m * self.n + self.batch_size * self.n
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_einsum_operations(self):
        """Test einsum operation counting"""
        A = torch.randn(self.batch_size, self.m, self.n, device=self.device)
        B = torch.randn(self.batch_size, self.n, self.p, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Einsum: batch matrix multiplication
            C = torch.einsum('bik,bkj->bij', A, B)
            
        # Conservative estimate for einsum
        expected_ops = self.batch_size * self.m * self.n * self.p
        self.assertGreaterEqual(ops_counter.total_operations, expected_ops)
        
    def test_vector_operations(self):
        """Test vector operation counting"""
        a = torch.randn(self.m, device=self.device)
        b = torch.randn(self.m, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Dot product: 2 * m operations (multiplication + addition)
            dot_product = torch.dot(a, b)
            # Cross product: 5 * m operations for 3D vectors
            if self.m == 3:
                cross_product = torch.cross(a, b)
            
        if self.m == 3:
            expected_ops = 2 * self.m + 5 * self.m
        else:
            expected_ops = 2 * self.m
        self.assertEqual(ops_counter.total_operations, expected_ops)


if __name__ == "__main__":
    unittest.main()
