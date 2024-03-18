import torch
import timm
from torch import randn

from torch_ops_counter import OperationsCounterMode


if __name__ == "__main__":
    device = torch.device("cpu")

    model = timm.create_model("vgg19_bn")
    batch_size = 1
    x = randn(batch_size, 3, 224, 224)

    ops_counter = OperationsCounterMode()
    with ops_counter:
        model(x)

    counter_dict = ops_counter.operations_count
    for module, operation in counter_dict.items():
        for op_name, op_count in operation.items():
            print(f"{module}::{op_name}: {op_count} OPs")

    print()
    # Printed number of operations should be close to the number of operations in Figure 1 of the paper
    # “Benchmark Analysis of Representative Deep Neural Network Architectures.”,2018 IEEE:1-8
    print(f"Total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
    print(f"Total parameters: {sum([p.numel() for p in [*model.parameters()][:-1]]) / 1e6} MegaParam(s)")
