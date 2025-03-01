from collections import OrderedDict, defaultdict
import os
import torch
from torch import nn
from torchprofile import profile_macs


class VGG(nn.Module):
    # VGG19 architecture:
    # - 2 conv layers of 64 filters, followed by maxpool
    # - 2 conv layers of 128 filters, followed by maxpool
    # - 4 conv layers of 256 filters, followed by maxpool
    # - 4 conv layers of 512 filters, followed by maxpool
    # - 4 conv layers of 512 filters, followed by maxpool
    # Total: 16 convolutional layers + 3 fully connected layers = 19 layers
    ARCH = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))

        # Replace single linear layer with three fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 1, 1] (for 32x32 input)
        x = self.backbone(x)

        # avgpool: [N, 512, 1, 1] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x

    def save(self, path):
        """Save the model to the given path."""
        # Ensure directory exists
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path="models/model.pth"):
        """Load a model from the given path."""
        model = cls()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model


def print_model_info(self, verbose=False) -> None:
    if verbose:
        print(self.backbone)
        print(self.classifier)

    num_params = 0
    for param in self.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print("#Params:", num_params)

    # Use the device that the model is on
    device = next(self.parameters()).device
    num_macs = profile_macs(self, torch.zeros(1, 3, 32, 32).to(device))
    print("#MACs:", num_macs)
