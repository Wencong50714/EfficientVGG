import torch
from torch import nn
from torchprofile import profile_macs
from vgg import VGG
import os
from data import prepare_data
from train import evaluate
import time


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def get_model_and_dataloader(model_path="models/model.pth"):
    if not os.path.exists(model_path):
        print("Warning: Model file not found at", model_path)
        print("Please train the model first!")
        exit(1)

    model = VGG.load(path=model_path).cuda()

    _, dataloader = prepare_data()

    return model, dataloader


@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model = model.to("cpu")
    model.eval()
    model = model.to("cuda")
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency


def evaluate_and_print_metrics(model, dataloader, model_name, count_nonzero_only=False):
    accuracy = evaluate(model, dataloader["test"])
    model_size = get_model_size(model, count_nonzero_only=count_nonzero_only)

    batch_size = 10
    input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()

    macs = get_model_macs(model, input_tensor)
    params = get_num_parameters(model)
    latency = measure_latency(model, input_tensor)

    print(f"{model_name} ====================")
    print(f"accuracy={accuracy:.2f}%")
    print(f"Latency (CPU) ={latency * 1000:.2f} ms")
    print(f"size={model_size / MiB:.2f} MiB")
    print(f"MACs={macs / 1e6:.2f} M")
    print(f"Params={params / 1e6:.2f} M")
    print("====================\n")


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
