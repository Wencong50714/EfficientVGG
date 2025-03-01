import torch
from vgg import *
from matplotlib import pyplot as plt
from data import prepare_data
from utility import *
from train import evaluate
from prune import FineGrainedPruner, ChannelPruner
import copy


def visualize_output(dataset):
    plt.figure(figsize=(20, 10))
    for index in range(40):
        image, label = dataset["test"][index]

        # Model inference
        model.eval()
        with torch.inference_mode():
            # No need to call cuda() on the model since it's already moved
            pred = model(image.unsqueeze(dim=0).cuda())
            pred = pred.argmax(dim=1)

        # Convert from CHW to HWC for visualization
        image = image.permute(1, 2, 0)

        # Convert from class indices to class names
        pred = dataset["test"].classes[pred]
        label = dataset["test"].classes[label]

        # Visualize the image
        plt.subplot(4, 10, index + 1)
        plt.imshow(image)
        plt.title(f"pred: {pred}" + "\n" + f"label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def evaluate_and_print_metrics(model, dataloader, model_name, count_nonzero_only=False):
    accuracy = evaluate(model, dataloader["test"])
    model_size = get_model_size(model, count_nonzero_only=count_nonzero_only)
    macs = get_model_macs(model, torch.randn(1, 3, 32, 32).cuda())
    params = get_num_parameters(model)
    latency = measure_latency(model, torch.randn(1, 3, 32, 32).cuda())

    print(f"{model_name} ====================")
    print(f"accuracy={accuracy:.2f}%")
    print(f"Latency (CPU) ={latency * 1000:.2f} ms")
    print(f"size={model_size / MiB:.2f} MiB")
    print(f"MACs={macs / 1e6:.2f} M")
    print(f"Params={params / 1e6:.2f} M")
    print("====================\n")


if __name__ == "__main__":
    # ====== Base Model ======
    model, dataloader = get_model_and_dataloader()

    # Fine Grained Prune
    fd_model = copy.deepcopy(model)
    fg_pruner = FineGrainedPruner()
    fg_pruner.prune(fd_model, dataloader)
    fg_pruner.finetune(fd_model, dataloader)

    # Channel Prune
    channel_model = copy.deepcopy(model)
    channel_pruner = ChannelPruner()
    channel_pruner.prune(channel_model, 0.3)
    channel_pruner.finetune(channel_model, dataloader)

    evaluate_and_print_metrics(model, dataloader, "Raw model")
    evaluate_and_print_metrics(
        fd_model, dataloader, "Fine Grained Prune", count_nonzero_only=True
    )
    evaluate_and_print_metrics(
        channel_model, dataloader, "Channel Prune", count_nonzero_only=True
    )
