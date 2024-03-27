<div align="center">

<h1> Torch Operation Counter</h1>
<img src="docs/logo/torch-operation-counter-logo-512.png"/>

![python-3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-1.13.1](https://img.shields.io/badge/torch-1.13.1%2B-orange)
![release-version](https://img.shields.io/badge/release-0.1-green)
![license](https://img.shields.io/badge/license-GPL%202-red)
</div>

The `torch-operation-counter` library offers a comprehensive toolkit for analyzing and counting the arithmetic operations (e.g., additions, multiplications) in PyTorch models. 
This tool is invaluable for optimizing model architectures, especially in resource-constrained environments or when aiming to reduce inference times.

## Installation
```bash
pip install torch-operation-counter
```

## Key Components
*torch_operation_counter**: The main module that provides the `OperationsCounterMode` context manager for counting operations in PyTorch models.
 * `counters.py`: Implements various counters for different types of arithmetic operations.
 * `ignore_modules.py`: Contains definitions for modules to be ignored during operation counting.
 * `operation_counter.py`: The core module that provides the functionality to wrap PyTorch models and count operations.
 * `utils.py`: Utility functions to support operation counting and analysis.

## Examples
The `examples` directory contains several scripts demonstrating how to use torch-operation-counter with different models:
 * `resnet18.py`: Demonstrates how to count operations in a ResNet-18 model.
 * `vgg16.py`: Demonstrates how to count operations in a VGG-16 model.
 * `pyg_gcn.py`: Demonstrates how to count operations in a PyG GCN model.

## Usage
To use `torch-operation-counter`, you wrap your PyTorch model with the OperationCounter class and perform a forward pass. Here's a simplified example using a `ResNet18` model:
```bash
import torch
from torchvision.models import resnet18

from torch_operation_counter import OperationsCounterMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18().to(device)
x = torch.randn(1, 3, 224, 224).to(device)

with OperationsCounterMode(model) as ops_counter:
    model(x)
    
print(f"Total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
# >> Total operations ~ 1.821047879 GigiaOP(s)
```

# Benchmarks

The following table shows the number of operations, parameters.

Table 1 for benchmarking image classification models on the ImageNet dataset with single input size `224 x 224`.

| Model              | GigaOP(s) | MegaParam(s) | Top-1 Accuracy | Top-5 Accuracy |
|--------------------|-----------|--------------|----------------|----------------|
| shufflenet_v2_x0_5 | 0.04      | 1.37         | 50.64          | 73.59          |
| mobilenet_v3_small | 0.06      | 2.54         | 66.54          | 86.88          |
| mnasnet0_5         | 0.11      | 2.22         | 66.92          | 86.93          |
| shufflenet_v2_x1_0 | 0.15      | 2.28         | 63.63          | 84.31          |
| mobilenet_v3_large | 0.22      | 5.48         | 73.26          | 91.09          |
| mobilenet_v2       | 0.30      | 3.50         | 69.14          | 88.79          |
| mnasnet1_0         | 0.32      | 4.38         | 73.21          | 91.22          |
| squeezenet1_1      | 0.36      | 1.24         | 57.32          | 79.99          |
| mnasnet1_3         | 0.54      | 6.28         | 76.44          | 93.43          |
| alexnet            | 0.72      | 61.10        | 55.55          | 78.46          |
| squeezenet1_0      | 0.83      | 1.25         | 57.11          | 79.70          |
| googlenet          | 1.52      | 6.62         | 66.17          | 87.20          |
| resnet18           | 1.82      | 11.69        | 66.61          | 87.19          |
| densenet121        | 2.98      | 7.98         | 70.87          | 90.11          |
| densenet169        | 3.54      | 14.15        | 72.38          | 91.00          |
| resnet34           | 3.67      | 21.80        | 69.96          | 89.29          |
| resnet50           | 4.12      | 25.56        | 72.41          | 90.87          |
| resnext50_32x4d    | 4.26      | 25.03        | 75.01          | 92.33          |
| densenet201        | 4.54      | 20.01        | 72.68          | 91.31          |
| vgg11              | 7.63      | 132.86       | 68.75          | 88.68          |
| vgg11_bn           | 7.63      | 132.87       | 67.32          | 87.90          |
| resnet101          | 7.84      | 44.55        | 73.94          | 91.85          |
| densenet161        | 8.13      | 28.68        | 74.21          | 92.09          |
| vgg13              | 11.34     | 133.05       | 69.63          | 89.06          |
| vgg13_bn           | 11.34     | 133.05       | 68.67          | 88.98          |
| wide_resnet50_2    | 11.43     | 68.88        | 74.95          | 92.27          |
| resnet152          | 11.57     | 60.19        | 75.25          | 92.57          |
| vgg16              | 15.50     | 138.36       | 71.49          | 90.45          |
| vgg16_bn           | 15.50     | 138.37       | 71.33          | 90.39          |
| resnext101_32x8d   | 16.48     | 88.79        | 76.58          | 93.22          |
| vgg19              | 19.67     | 143.67       | 72.17          | 90.74          |
| vgg19_bn           | 19.67     | 143.68       | 71.84          | 90.68          |
| wide_resnet101_2   | 22.80     | 126.89       | 74.68          | 92.23          |

Table 2 for benchmarking NLP models on the `512` input sequence lengths.

| Model                                 | GigaOP(s) | MegaParam(s) |
|---------------------------------------|-----------|--------------|
| squeezebert/squeezebert-mnli-headless | 18.7      | 51.1         |
| squeezebert/squeezebert-uncased       | 18.7      | 51.1         |
| distilbert-base-uncased               | 24.26     | 66.96        |
| bert-base-uncased                     | 48.59     | 109.48       |
| roberta-base                          | 48.59     | 124.65       |
| xlm-roberta-base                      | 48.59     | 278.05       |
| microsoft/layoutlm-base-uncased       | 48.6      | 112.63       |
| google/bigbird-roberta-base           | 48.73     | 128.06       |
| gpt2                                  | 48.73     | 124.44       |
| albert-base-v2                        | 48.78     | 11.69        |
| facebook/bart-base                    | 58.34     | 140.01       |
| microsoft/deberta-base                | 72.94     | 139.19       |
| t5-base                               | 116.78    | 223.5        |
| microsoft/layoutlm-large-uncased      | 168.24    | 339.34       |
| facebook/mbart-large-cc25             | 200.68    | 611.9        |

# Contributing to torch-operation-counter

We welcome contributions from the community and are pleased to have you join us in improving `torch-operation-counter`! Here's how you can contribute:

## Reporting Issues

Found a bug or have a feature request? Please feel free to open a new issue. 
When creating a bug report, please include as much detail as possible, such as:
- A clear and descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or error messages

## Making Contributions

Before making contributions, please first discuss the change you wish to make via an issue, email, or any other method with the owners of this repository. This discussion helps avoid duplicating efforts or conflicting changes.

### Pull Request Process

1. Update the README.md or any relevant documentation with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
2. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent.
3. You may merge the Pull Request once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Setting Up Your Environment

To set up your development environment for contributing to `torch-operation-counter`, please follow these steps:
- Fork the repository on GitHub
- Clone your forked repository to your local machine
- Create a new branch for your feature or bug fix
- Make your changes locally and test them thoroughly
- Push your changes to your fork on GitHub and submit a pull request to the main repository

## License
This project is licensed under the terms of the GPL 2.0 license.

