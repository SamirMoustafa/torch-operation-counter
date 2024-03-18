<div align="center">

<h1> Torch Operation Counter

![python-3.10](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch-1.13.1](https://img.shields.io/badge/torch-1.13.1%2B-orange)
![release-version](https://img.shields.io/badge/release-0.1-green)
![license](https://img.shields.io/badge/license-GPL%202-red)
_________________________
</div>


# Installation
```bash
# TODO: Add installation command after uploading to PyPI
```

# Get Started
```bash
import torch
from torchvision.models import resnet18

from torch_ops_counter import OperationsCounterMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18().to(device)
x = torch.randn(1, 3, 224, 224).to(device)

with OperationsCounterMode(model) as ops_counter:
    model(x)
    
print(f"Total operations: {ops_counter.total_operations / 1e9} GigiaOP(s)")
# >> Total operations: 1.821047879 GigiaOP(s)
```
