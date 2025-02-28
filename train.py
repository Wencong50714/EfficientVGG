import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
import argparse
import os
import json
from datetime import datetime

from vgg import VGG
from data import prepare_data

def set_seed(seed=0):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(
    model: nn.Module,
    dataflow,
    criterion: nn.Module,
    optimizer,
    scheduler: LambdaLR,
) -> None:
    """Train the model for one epoch."""
    model.train()

    for inputs, targets in tqdm(dataflow, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataflow
) -> float:
    """Evaluate the model and return accuracy."""
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()

if __name__ == "__main__":
    # default arguments
    parser = argparse.ArgumentParser(description="Train a VGG model on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.4, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--save", type=str, default="models/model.pth", help="Path to save the model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Prepare data
    datasets, dataflow = prepare_data(batch_size=args.batch_size, num_workers=args.workers)
    
    # Create model
    model = VGG().cuda()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    
    # Learning rate scheduler
    num_epochs = args.epochs
    steps_per_epoch = len(dataflow["train"])
    
    # Define the piecewise linear scheduler
    lr_lambda = lambda step: np.interp(
        [step / steps_per_epoch],
        [0, num_epochs * 0.3, num_epochs],
        [0, 1, 0]
    )[0]
    
    # Create and visualize learning rate schedule
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    for epoch_num in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        train(model, dataflow["train"], criterion, optimizer, scheduler)
        metric = evaluate(model, dataflow["test"])
        print(f"epoch {epoch_num}: {metric}")
    
    # Save the trained model and metadata
    # Create a directory structure for saving
    if args.save.endswith('.pth'):
        # Standard file path provided
        model.save(args.save)
    else:
        # Use as directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = args.save if os.path.isdir(args.save) else f"models/{args.save}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pth")
        model.save(model_path)
        
        # Save training metadata
        metadata = {
            "accuracy": metric,
            "epochs": num_epochs,
            "batch_size": args.batch_size,
            "initial_lr": args.lr,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Training metadata saved to {save_dir}/metadata.json")

