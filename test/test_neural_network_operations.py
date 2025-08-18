import unittest
import torch
import torch.nn as nn
from torch_operation_counter import OperationsCounterMode


class TestNeuralNetworkOperations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        self.num_channels = 16
        self.height = 32
        self.width = 32
        
    def test_convolution_operations(self):
        """Test convolution operation counting"""
        x = torch.randn(self.batch_size, self.num_channels, self.height, self.width, device=self.device)
        conv = nn.Conv2d(self.num_channels, self.num_channels * 2, kernel_size=3, padding=1)
        conv = conv.to(self.device)
        
        with OperationsCounterMode() as ops_counter:
            y = conv(x)
            
        # Convolution operations are handled by utils.conv_ops_count
        # This test verifies the operation is counted
        expected_ops = 18 * self.batch_size * self.num_channels ** 2 * self.height * self.width
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_pooling_operations(self):
        """Test pooling operation counting"""
        x = torch.randn(self.batch_size, self.num_channels, self.height, self.width, device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            # Max pooling: comparisons per window
            y1 = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            # Average pooling: sum + divide per window
            y2 = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            
        # Both operations should be counted
        expected_ops = 7/8 * self.batch_size * self.num_channels * self.height * self.width
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_layer_norm_operations(self):
        """Test layer normalization operation counting"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device)
        ln = nn.LayerNorm(self.hidden_size)
        ln = ln.to(self.device)
        
        with OperationsCounterMode() as ops_counter:
            y = ln(x)
            
        # Layer norm: 6 * numel operations (mean, var, normalize, scale, shift)
        expected_ops = 6 * x.numel()
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
    def test_embedding_operations(self):
        """Test embedding operation counting"""
        embedding = nn.Embedding(1000, self.hidden_size)
        embedding = embedding.to(self.device)
        indices = torch.randint(0, 1000, (self.batch_size, self.seq_len), device=self.device)
        
        with OperationsCounterMode() as ops_counter:
            y = embedding(indices)
            
        # Embedding: indices.numel() * hidden_size operations
        expected_ops = indices.numel() * self.hidden_size
        self.assertEqual(ops_counter.total_operations, expected_ops)
        
if __name__ == "__main__":
    unittest.main()
