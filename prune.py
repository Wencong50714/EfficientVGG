from typing import Union, List
import numpy as np
import torch
import torch.nn as nn  # Missing import for nn
from matplotlib import pyplot as plt
from utility import *
from train import evaluate, train
import math
from tqdm.auto import tqdm
import copy


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements * sparsity)
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    # Step 3: calculate the pruning threshold
    threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance > threshold
    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


def test_fine_grained_prune(
    test_tensor=torch.tensor(
        [
            [-0.46, -0.40, 0.39, 0.19, 0.37],
            [0.00, 0.40, 0.17, -0.15, 0.16],
            [-0.20, -0.23, 0.36, 0.25, 0.03],
            [0.24, 0.41, 0.07, 0.13, -0.15],
            [0.48, -0.09, -0.36, 0.12, 0.45],
        ]
    ),
    test_mask=torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, True, False, False, False],
            [True, False, False, False, True],
        ]
    ),
    target_sparsity=0.75,
    target_nonzeros=None,
):
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap="tab20c")
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(
                    j,
                    i,
                    f"{tensor[i, j].item():.2f}",
                    ha="center",
                    va="center",
                    color="k",
                )

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, "dense tensor")

    sparsity_before_pruning = get_sparsity(test_tensor)
    mask = fine_grained_prune(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparsity(test_tensor)
    sparsity_of_mask = get_sparsity(mask)

    plot_matrix(test_tensor, ax_right, "sparse tensor")
    fig.tight_layout()
    plt.savefig("./images/test_fine_grained_prune.png")

    print("* Test fine_grained_prune()")
    print(f"    target sparsity: {target_sparsity:.2f}")
    print(f"        sparsity before pruning: {sparsity_before_pruning:.2f}")
    print(f"        sparsity after pruning: {sparsity_after_pruning:.2f}")
    print(f"        sparsity of pruning mask: {sparsity_of_mask:.2f}")

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print("* Test passed.")
        else:
            print("* Test failed.")
    else:
        if mask.count_nonzero() == target_nonzeros:
            print("* Test passed.")
        else:
            print("* Test failed.")


@torch.no_grad()
def sensitivity_scan(
    model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True
):
    """
    Performs a layer-wise sensitivity analysis by applying different sparsity levels to each layer
    and measuring the impact on model accuracy.

    This function temporarily prunes each convolutional/fully-connected layer at various sparsity levels
    and evaluates the model's performance to determine how sensitive each layer is to pruning.
    """
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [
        (name, param) for (name, param) in model.named_parameters() if param.dim() > 1
    ]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(
            sparsities,
            desc=f"scanning {i_layer}/{len(named_conv_weights)} weight - {name}",
        ):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader["test"])
            if verbose:
                print(f"\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%", end="")
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(
                f"\r    sparsity=[{','.join(['{:.2f}'.format(x) for x in sparsities])}]: accuracy=[{', '.join(['{:.2f}%'.format(x) for x in accuracy])}]",
                end="",
            )
        accuracies.append(accuracy)
    return sparsities, accuracies


def plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel("sparsity")
            ax.set_ylabel("top-1 accuracy")
            ax.legend(
                [
                    "accuracy after pruning",
                    f"{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy",
                ]
            )
            ax.grid(axis="x")
            plot_index += 1
    fig.suptitle("Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig("./images/plot_sensitivity_scan")


def generate_sparsity_dict_from_sensitivity(
    model,
    dataloader,
    target_accuracy_drop=0.02,
    scan_step=0.1,
    scan_start=0.4,
    scan_end=0.9,
    visualize=False,
):
    """
    Generate a sparsity dictionary based on sensitivity analysis

    Args:
        model: The neural network model to analyze
        dataloader: DataLoader for evaluation during sensitivity analysis
        target_accuracy_drop: Maximum acceptable drop in accuracy (as a fraction)
        scan_step: Step size for sparsity scanning
        scan_start: Starting sparsity value for scanning
        scan_end: Ending sparsity value for scanning
        visualize: Whether to visualize sensitivity analysis results

    Returns:
        Dictionary mapping layer names to optimal sparsity values
    """
    # Get base accuracy
    base_accuracy = evaluate(model, dataloader["test"])

    # Perform sensitivity scan
    sparsities, layer_accuracies = sensitivity_scan(
        model, dataloader, scan_step=scan_step, scan_start=scan_start, scan_end=scan_end
    )

    if visualize:
        plot_sensitivity_scan(model, sparsities, layer_accuracies, base_accuracy)

    # Determine per-layer sparsity based on sensitivity
    sparsity_dict = {}
    layer_idx = 0

    for name, param in model.named_parameters():
        if param.dim() > 1:  # Conv or FC layer
            # Find highest sparsity that maintains accuracy within threshold
            acceptable_accuracy = base_accuracy - (base_accuracy * target_accuracy_drop)
            max_sparsity = scan_start  # Default to minimum if none found

            # Search through sparsities from highest to lowest
            for i, sparsity in reversed(list(enumerate(sparsities))):
                if layer_accuracies[layer_idx][i] >= acceptable_accuracy:
                    max_sparsity = sparsity
                    break

            sparsity_dict[name] = max_sparsity
            print(f"Layer {name}: assigned sparsity {max_sparsity:.2f}")
            layer_idx += 1

    return sparsity_dict


class FineGrainedPruner:
    def __init__(self):
        self.masks = {}
        self.best_accuracy = 0.0
        self.average_sparsity = 0.0

    @torch.no_grad()
    def _apply(self, model):
        """
        Apply stored pruning masks to model parameters

        Args:
            model: The neural network model to apply masks to
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @torch.no_grad()
    def prune(self, model, dataloader):
        """
        Generate pruning masks based on sensitivity analysis

        Args:
            model: The model to analyze and prune
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary of pruning masks for each layer
        """
        print("================= Sensitivity Scan =================")
        sparsity_dict = generate_sparsity_dict_from_sensitivity(model, dataloader)
        self.average_sparsity = (
            sum(sparsity_dict.values()) / len(sparsity_dict) if sparsity_dict else 0.0
        )
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])

        self.masks = masks
        self._apply(model)

    def finetune(self, model, dataloader, num_epochs=5):
        """
        Finetune the pruned model to recover accuracy

        Args:
            model: The pruned model to finetune
            dataloader: Dictionary with 'train' and 'test' DataLoaders
            num_epochs: Number of finetuning epochs

        Returns:
            Finetuned model and best accuracy achieved
        """
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train(model, dataloader["train"], criterion, optimizer, scheduler)
            self._apply(model)  # warning call back here !!!

            accuracy = evaluate(model, dataloader["test"])
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            print(
                f"Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {self.best_accuracy:.2f}%"
            )

        return model, self.best_accuracy


class UniformFineGrainedPruner:
    def __init__(self, sparsity):
        self.sparsity = sparsity
        self.best_accuracy = 0.0
        self.mask = {}

    @torch.no_grad()
    def _apply(self, model):
        """
        Apply stored pruning masks to model parameters

        Args:
            model: The neural network model to apply masks to
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @torch.no_grad()
    def prune(self, model, dataloader):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, self.sparsity)

        self.masks = masks
        self._apply(model)

    def finetune(self, model, dataloader, num_epochs=5):
        """
        Finetune the pruned model to recover accuracy

        Args:
            model: The pruned model to finetune
            dataloader: Dictionary with 'train' and 'test' DataLoaders
            num_epochs: Number of finetuning epochs

        Returns:
            Finetuned model and best accuracy achieved
        """
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train(model, dataloader["train"], criterion, optimizer, scheduler)
            self._apply(model)  # warning call back here !!!

            accuracy = evaluate(model, dataloader["test"])
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            print(
                f"Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {self.best_accuracy:.2f}%"
            )

        return model, self.best_accuracy


class ChannelPruner:
    def __init__(self):
        """
        Initialize the channel pruner
        """
        self.best_accuracy = 0.0

    def _get_num_channels_to_keep(self, channels: int, prune_ratio: float) -> int:
        """
        Calculate the number of channels to preserve after pruning
        """
        return int(round((1 - prune_ratio) * channels))

    def _get_input_channel_importance(self, weight):
        """
        Compute importance score for each input channel

        Args:
            weight: Weight tensor of a convolutional layer

        Returns:
            Tensor of importance scores for each input channel
        """
        importances = []
        # compute the importance for each input channel
        for i_c in range(weight.shape[1]):
            channel_weight = weight.detach()[:, i_c]
            importance = torch.norm(channel_weight, p=2)
            importances.append(importance.view(1))
        return torch.cat(importances)

    @torch.no_grad()
    def prune(self, model, prune_ratio: Union[List, float]) -> nn.Module:
        """
        Perform channel pruning on the model

        Args:
            prune_ratio: Ratio of channels to prune (float for uniform pruning,
                        list for layer-specific pruning)
            sort_before_pruning: Whether to sort channels by importance before pruning

        Returns:
            Pruned model
        """
        # sanity check of provided prune_ratio
        assert isinstance(prune_ratio, (float, list))
        n_conv = len([m for m in model.backbone if isinstance(m, nn.Conv2d)])
        if isinstance(prune_ratio, list):
            assert len(prune_ratio) == n_conv - 1
        else:  # convert float to list
            prune_ratio = [prune_ratio] * (n_conv - 1)

        # we only apply pruning to the backbone features
        all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
        all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
        # apply pruning
        assert len(all_convs) == len(all_bns)

        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            prev_bn = all_bns[i_ratio]
            next_conv = all_convs[i_ratio + 1]
            original_channels = prev_conv.out_channels
            n_keep = self._get_num_channels_to_keep(original_channels, p_ratio)

            # prune the output of the previous conv and bn
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

            # prune the input of the next conv
            next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])

    @torch.no_grad()
    def _apply_channel_sorting(self, model):
        """
        Sort channels by importance before pruning

        Args:
            model: The neural network model

        Returns:
            Model with sorted channels
        """
        # fetch all the conv and bn layers from the backbone
        all_convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
        all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]

        # iterate through conv layers
        for i_conv in range(len(all_convs) - 1):
            prev_conv = all_convs[i_conv]
            prev_bn = all_bns[i_conv]
            next_conv = all_convs[i_conv + 1]

            # compute importance according to input channels
            importance = self._get_input_channel_importance(next_conv.weight)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True)

            # apply to previous conv and its following bn
            prev_conv.weight.copy_(
                torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
            )
            for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )

            # apply to the next conv input
            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )

        return model

    def finetune(self, model, dataloader, num_epochs=5):
        """
        Finetune the pruned model to recover accuracy

        Args:
            model: The pruned model to finetune
            dataloader: Dictionary with 'train' and 'test' DataLoaders
            num_epochs: Number of finetuning epochs

        Returns:
            Finetuned model and best accuracy achieved
        """
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train(model, dataloader["train"], criterion, optimizer, scheduler)
            accuracy = evaluate(model, dataloader["test"])
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy
            print(
                f"Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {self.best_accuracy:.2f}%"
            )

        return model, self.best_accuracy


if __name__ == "__main__":
    # ====== Base Model ======
    model, dataloader = get_model_and_dataloader()

    # Fine Grained Prune
    fg_path = "models/fg_model.pth"
    if os.path.exists(fg_path):
        fg_model = VGG.load(path=fg_path).cuda()
    else:
        fg_model = copy.deepcopy(model)
        fg_pruner = FineGrainedPruner()
        fg_pruner.prune(fg_model, dataloader)
        fg_pruner.finetune(fg_model, dataloader)
        fg_model.save(path=fg_path)

    # Uniform Fine Grained Prune
    ufg_path = "models/ufg_model.pth"
    if os.path.exists(ufg_path):
        ufg_model = VGG.load(path=ufg_path).cuda()
    else:
        average_sparsity = fg_pruner.average_sparsity
        ufg_model = copy.deepcopy(model)
        ufg_pruner = UniformFineGrainedPruner(average_sparsity)
        ufg_pruner.prune(ufg_model, dataloader)
        ufg_pruner.finetune(ufg_model, dataloader)
        ufg_model.save(path=ufg_path)

    channel_path = "models/channel_mode.pth"
    if os.path.exists(channel_path):
        channel_model = VGG.load(path=channel_path).cuda()
    else:
        # Channel Prune
        channel_model = copy.deepcopy(model)
        channel_pruner = ChannelPruner()
        channel_pruner.prune(channel_model, 0.3)
        channel_pruner.finetune(channel_model, dataloader)

    evaluate_and_print_metrics(model, dataloader, "Raw model")
    evaluate_and_print_metrics(
        fg_model, dataloader, "Fine Grained Prune", count_nonzero_only=True
    )
    evaluate_and_print_metrics(
        ufg_model, dataloader, "Uniform Fine Grained Prune", count_nonzero_only=True
    )
    evaluate_and_print_metrics(
        channel_model, dataloader, "Channel Prune", count_nonzero_only=True
    )
