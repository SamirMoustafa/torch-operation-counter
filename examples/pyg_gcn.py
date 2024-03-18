# This script requires the torch_geometric library to be installed.
from torch import device, long, randint, randn
from torch.nn import Module, ModuleList, Linear

from torch_geometric.nn import GCNConv

from torch_ops_counter import OperationsCounterMode


class GCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else hidden_channels
            out_channel = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(GCNConv(in_channel, out_channel, normalize=False))
        self.fc = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = device("cpu")

    # Reddit dataset settings
    n = 232_965
    e = 114_615_892 + n
    num_feature = 602
    out_feature = 41

    # # Cora dataset settings
    # n = 2_708
    # e = 5_278 + n
    # num_feature = 1_433
    # out_feature = 7

    # # Pubmed dataset settings
    # n = 19_717
    # e = 44_324 + n
    # num_feature = 500
    # out_feature = 3

    # model hyper-parameters
    hidden_channels = 128

    model = GCN(num_feature, hidden_channels, out_feature, 2).to(device)
    x = randn(n, num_feature).to(device)
    edge_index = randint(n, size=(2, e), dtype=long).to(device)
    edge_weight = randn(e).to(device)

    random_input = (x, edge_index, edge_weight)

    with OperationsCounterMode() as ops_counter:
        model(*random_input)

    counter_dict = ops_counter.operations_count
    for module, operation in counter_dict.items():
        for op_name, op_count in operation.items():
            print(f"{module}::{op_name}: {op_count} OPs")

    print()
    # Print the total number of operations by two layers of GCN with 32 hidden channels.
    # The total number of operations should be similar to 19 GigaOPs as reported in the introduction section of the paper
    # Tailor, Shyam A. et al. “Degree-Quant: Quantization-Aware Training for Graph Neural Networks.”, ICLR 2021
    print(f"{model.__class__.__name__}::Total operations: GigiaOP(s) {ops_counter.total_operations / 1e9}")
    print(f"{model.__class__.__name__}::Total parameters: KiloParam(s) {sum([p.numel() for p in [*model.parameters()][:-1]]) / 1e3}")
