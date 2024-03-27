from tqdm import tqdm
from numpy import mean, std
from prettytable import PrettyTable

from torch import cuda, device, randn, hub
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms as T
from torchmetrics import Accuracy

from torch_operation_counter import OperationsCounterMode


def evaluate_accuracy_over_loader(model, loader, device):
    model.eval()
    accuracy_top_1 = Accuracy(num_classes=len(loader.dataset.classes), task="multiclass", top_k=1).to(device)
    accuracy_top_5 = Accuracy(num_classes=len(loader.dataset.classes), task="multiclass", top_k=5).to(device)
    accuracies_top_1 = []
    accuracies_top_5 = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        accuracies_top_1 += [accuracy_top_1(y_pred, y).detach().cpu().numpy() * 100, ]
        accuracies_top_5 += [accuracy_top_5(y_pred, y).detach().cpu().numpy() * 100, ]
    top_1_accuracy = f"{mean(accuracies_top_1):.2f} ± {std(accuracies_top_1):.2f}"
    top_5_accuracy = f"{mean(accuracies_top_5):.2f} ± {std(accuracies_top_5):.2f}"
    return top_1_accuracy, top_5_accuracy


if __name__ == "__main__":
    device = device("cuda" if cuda.is_available() else "cpu")
    batch_size = 1
    x = randn(batch_size, 3, 224, 224).to(device)

    transform = T.Compose([T.Resize(224),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ])
    val_data = ImageNet(root="/home/samirm97cs/datasets/imagenet", split="val", transform=transform)
    loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=16)

    models_names = ["alexnet",
                    "densenet121",
                    "densenet161",
                    "densenet169",
                    "densenet201",
                    "fcn_resnet101",
                    "fcn_resnet50",
                    "googlenet",
                    "inception_v3",
                    "lraspp_mobilenet_v3_large",
                    "mnasnet0_5",
                    "mnasnet1_0",
                    "mnasnet1_3",
                    "mobilenet_v2",
                    "mobilenet_v3_large",
                    "mobilenet_v3_small",
                    "resnet101",
                    "resnet152",
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "resnext101_32x8d",
                    "resnext50_32x4d",
                    "shufflenet_v2_x0_5",
                    "shufflenet_v2_x1_0",
                    "squeezenet1_0",
                    "squeezenet1_1",
                    "vgg11",
                    "vgg11_bn",
                    "vgg13",
                    "vgg13_bn",
                    "vgg16",
                    "vgg16_bn",
                    "vgg19",
                    "vgg19_bn",
                    "wide_resnet101_2",
                    "wide_resnet50_2",
                    ]

    number_of_operations_per_model = {}
    number_of_parameters_per_model = {}
    top_1_accuracy_per_model = {}
    top_5_accuracy_per_model = {}

    pbar = tqdm(models_names)
    for model_name in pbar:
        pbar.set_description(f"Evaluating {model_name}")
        try:
            model = hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
        except RuntimeError:
            print(f"Model {model_name} not available in the hub.")
            continue
        except ValueError:
            print(f"Model {model_name} has no pretrained weights.")
            continue
        except Exception as e:
            print(f"Model {model_name} failed to load with error: {e}")
            continue

        try:
            model = model.to(device)
            ops_counter = OperationsCounterMode(model)
            with ops_counter:
                model(x)

            number_of_operations_per_model[model_name] = ops_counter.total_operations
            number_of_parameters_per_model[model_name] = sum(p.numel() for p in model.parameters())
            top_1_accuracy, top_5_accuracy = evaluate_accuracy_over_loader(model, loader, device)
            top_1_accuracy_per_model[model_name] = top_1_accuracy
            top_5_accuracy_per_model[model_name] = top_5_accuracy
        except Exception as e:
            print(f"Model {model_name} failed to evaluate with error: {e}")
            if model_name in number_of_operations_per_model:
                del number_of_operations_per_model[model_name]
            if model_name in number_of_parameters_per_model:
                del number_of_parameters_per_model[model_name]
            if model_name in top_1_accuracy_per_model:
                del top_1_accuracy_per_model[model_name]
            if model_name in top_5_accuracy_per_model:
                del top_5_accuracy_per_model[model_name]
            continue

    table = PrettyTable()
    table.field_names = ["Model", "Operations GigaOP(s)", "Parameters MegaParam(s)", "Top-1 Accuracy", "Top-5 Accuracy"]
    for model_name in models_names:
        if model_name in number_of_operations_per_model:
            table.add_row([model_name,
                           number_of_operations_per_model[model_name] / 1e9,
                           number_of_parameters_per_model[model_name] / 1e6,
                           top_1_accuracy_per_model[model_name],
                           top_5_accuracy_per_model[model_name],
                           ])

    table.sortby = "Operations GigaOP(s)"
    print(table)
