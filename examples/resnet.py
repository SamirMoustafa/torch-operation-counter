import torch
from torch import randn
from torchvision.models import resnet101

from torch_operation_counter import OperationsCounterMode


if __name__ == "__main__":
    device = torch.device("cpu")

    model = resnet101()

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
    # Printed number of operations should be close to the number of operations in Table 4 of the paper
    # “Multi-Scale Vision Longformer:A New Vision Transformer for High-Resolution Image Encoding.”,2021 ICCV:2978-2988
    # In the paper they refer to the number of operations as FLOPs, but the number of operations is the same.
    print(f"Total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
    print(f"Total parameters: {sum([p.numel() for p in [*model.parameters()][:-1]]) / 1e6} MegaParam(s)")
