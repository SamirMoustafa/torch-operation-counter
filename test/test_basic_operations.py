import unittest
import torch
import torch.nn as nn
from torch_operation_counter import OperationsCounterMode


class TestBasicOperations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        
    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations counting"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Addition
            y1 = x + 1.0
            # Subtraction
            y2 = x - 1.0
            # Multiplication
            y3 = x * 2.0
            # Division
            y4 = x / 2.0
            # Power
            y5 = x ** 2
            
        # Each operation should count prod(shape) operations
        expected_ops = 5 * x.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_reduction_operations(self):
        """Test reduction operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Sum: n-1 additions for n elements
            y1 = torch.sum(x)
            # Mean: sum + division
            y2 = torch.mean(x)
            # Max: n-1 comparisons for n elements
            y3 = torch.max(x)
            # Min: n-1 comparisons for n elements
            y4 = torch.min(x)
            
        # Sum: x.numel() - 1 additions
        # Mean: (x.numel() - 1) + 1 operations
        # Max/Min: x.numel() - 1 comparisons each
        expected_ops = (x.numel() - 1) + (x.numel() - 1 + 1) + 2 * (x.numel() - 1)
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_tensor_creation_operations(self):
        """Test tensor creation operation counting"""
        shape = (self.batch_size, self.seq_len, self.hidden_size)
        
        with OperationsCounterMode() as ops_counter:
            # Zeros: write operations for each element
            y1 = torch.zeros(shape, device=self.device)
            # Ones: write operations for each element
            y2 = torch.ones(shape, device=self.device)
            # Randn: random generation + write for each element
            y3 = torch.randn(shape, device=self.device)
            # Full: write operations for each element
            y4 = torch.full(shape, 0.5, device=self.device)
            
        expected_ops = y1.numel() + y2.numel() + 2 * y3.numel() + y4.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
if __name__ == "__main__":
    unittest.main()
