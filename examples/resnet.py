import time

import torch
from torch import randn

from torchvision.models import resnet50 as resnet50
from torchvision.models.quantization import resnet50 as resnet50_quantized


from torch_operation_counter import OperationsCounterMode


def time_profiling(model, x, warmup=10, repeat=25):
    # Warm-up
    [model(x) for _ in range(warmup)]
    # Timing
    start = time.time()
    [model(x) for _ in range(repeat)]
    end = time.time()
    return (end - start) / repeat


if __name__ == "__main__":
    device = torch.device("cpu")
    batch_size = 1
    x = randn(batch_size, 3, 224, 224).to(device)
    ops_counter = OperationsCounterMode()

    model = resnet50(pretrained=True).to(device)
    full_precision_duration = time_profiling(model, x)
    with ops_counter:
        model(x)

    # Printed number of operations should be close to the number of operations in Table 4 of the paper
    # “Multi-Scale Vision Longformer:A New Vision Transformer for High-Resolution Image Encoding.”,2021 ICCV:2978-2988
    # In the paper they refer to the number of operations as FLOPs, but the number of operations is the same.
    print(f"Full precision total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
    print(f"Full precision duration: {full_precision_duration} seconds")

    model = resnet50_quantized(pretrained=True).to(device)
    quantized_duration = time_profiling(model.quant, x)
    with ops_counter:
        model.quant(x)
    print(f"Quantized total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
    print(f"Quantized duration: {quantized_duration} seconds")
