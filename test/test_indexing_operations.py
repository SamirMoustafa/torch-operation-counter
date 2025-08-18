import unittest

import torch
import torch.nn as nn
from torch_operation_counter import OperationsCounterMode


class TestIndexingOperations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        
    def test_advanced_indexing_operations(self):
        """Test advanced indexing operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        indices = torch.randint(0, self.seq_len, (5,), device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Advanced indexing: read operations for selected elements
            y = x[:, indices, :]
            
        # Advanced indexing: read operations for selected elements
        expected_ops = indices.numel() * self.batch_size * self.hidden_size
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_masked_operations(self):
        """Test masked operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        mask = torch.randint(0, 2, (self.batch_size, self.seq_len, self.hidden_size), device=self.device).bool()
        
        with OperationsCounterMode() as ops_counter:
            # Masked select: read operations for true elements
            y1 = torch.masked_select(x, mask)
            # Masked fill: write operations for true elements
            y2 = torch.masked_fill(x, mask, 0.0)
            
        # Masked select: read operations for true elements
        # Masked fill: write operations for true elements
        true_elements = mask.sum().item()
        expected_ops = true_elements + true_elements
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_scatter_operations(self):
        """Test scatter operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, device=self.device)
        indices = torch.randint(0, self.seq_len, (self.batch_size, 5), device=self.device)
        src = torch.randn(self.batch_size, 5, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Scatter: write operations for each index
            y = torch.scatter(x, dim=1, index=indices, src=src)
            
        # Scatter: write operations for each index
        expected_ops = indices.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_gather_operations(self):
        """Test gather operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, device=self.device)
        indices = torch.randint(0, self.seq_len, (self.batch_size, 5), device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Gather: read operations for each index
            y = torch.gather(x, dim=1, index=indices)
            
        # Gather: read operations for each index
        expected_ops = indices.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_take_put_operations(self):
        """Test take and put operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, device=self.device)
        indices = torch.randint(0, x.numel(), (10,), device=self.device)
        values = torch.randn(10, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Take: read operations for each index
            y1 = torch.take(x, indices)
            # Put: write operations for each index
            y2 = torch.put(x, indices, values)
            
        # Take: read operations for each index
        # Put: write operations for each index
        expected_ops = indices.numel() + indices.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_where_operations(self):
        """Test where operation counting"""
        condition = torch.randint(0, 2, (self.batch_size, self.seq_len), device=self.device).bool()
        x = torch.randn(self.batch_size, self.seq_len, device=self.device)
        y = torch.randn(self.batch_size, self.seq_len, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Where: conditional selection operations
            result = torch.where(condition, x, y)
            
        # Where: conditional selection operations for each element
        expected_ops = condition.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_roll_flip_operations(self):
        """Test roll and flip operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Roll: shift operations
            y1 = torch.roll(x, shifts=2, dims=1)
            # Flip: reverse operations
            y2 = torch.flip(x, dims=[1])
            
        # Roll: copy operations for each element
        # Flip: copy operations for each element
        expected_ops = x.numel() + x.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_diag_tril_triu_operations(self):
        """Test diagonal and triangular operation counting"""
        x = torch.randn(self.seq_len, self.seq_len, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Diag: extract diagonal elements
            y1 = torch.diag(x)
            # Tril: extract lower triangular
            y2 = torch.tril(x)
            # Triu: extract upper triangular
            y3 = torch.triu(x)
            
        # Diag: read operations for diagonal elements
        # Tril/Triu: read operations for triangular elements
        expected_ops = y1.numel() + y2.numel() + y3.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)


if __name__ == "__main__":
    unittest.main()
