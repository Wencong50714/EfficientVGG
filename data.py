import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from matplotlib import pyplot as plt


def get_datasets(root="data/cifar10"):
    """
    Create CIFAR-10 datasets for training and testing.

    Args:
        root: Path where the dataset will be stored

    Returns:
        Dictionary containing train and test datasets
    """
    transforms = {
        "train": Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
            ]
        ),
        "test": ToTensor(),
    }

    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )

    return dataset


def get_dataloaders(dataset, batch_size=512, num_workers=0):
    """
    Create DataLoaders for training and testing.

    Args:
        dataset: Dictionary containing train and test datasets
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading

    Returns:
        Dictionary containing train and test DataLoaders
    """
    dataflow = {}
    for split in ["train", "test"]:
        dataflow[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataflow


def prepare_data(batch_size=512, num_workers=0, root="data/cifar10"):
    """
    Prepare datasets and dataloaders in one step.

    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        root: Path where the dataset will be stored

    Returns:
        Tuple of (datasets, dataloaders) dictionaries
    """
    datasets = get_datasets(root)
    dataloaders = get_dataloaders(datasets, batch_size, num_workers)

    return datasets, dataloaders


def visualize_dataset_samples(dataset, num_per_class=4):
    """
    Visualize sample images from each class in the dataset.

    Args:
        dataset: CIFAR-10 dataset
        num_per_class: Number of samples to show per class
    """
    samples = [[] for _ in range(10)]
    for image, label in dataset:
        if len(samples[label]) < num_per_class:
            samples[label].append(image)

        # Break if we have collected enough samples
        if all(len(class_samples) >= num_per_class for class_samples in samples):
            break

    total_samples = 10 * num_per_class
    rows = num_per_class
    cols = 10

    plt.figure(figsize=(20, 9))
    for index in range(total_samples):
        label = index % 10
        image = samples[label][index // 10]

        # Convert from CHW to HWC for visualization
        image = image.permute(1, 2, 0)

        # Convert from class index to class name
        class_name = dataset.classes[label]

        # Visualize the image
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
