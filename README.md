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

