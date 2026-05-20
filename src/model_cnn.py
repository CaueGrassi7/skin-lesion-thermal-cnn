"""
CNN model definitions and training utilities.

Includes a custom lightweight CNN and a transfer-learning wrapper around
pre-trained torchvision models (e.g. ResNet, EfficientNet).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ThermalCNN(nn.Module):
    """Lightweight CNN designed for thermal skin lesion classification."""

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        """
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        raise NotImplementedError


def build_transfer_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a transfer-learning model from a torchvision backbone.

    Args:
        backbone: Name of the torchvision model (e.g. 'resnet18', 'efficientnet_b0').
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: If True, freeze all layers except the final classifier.

    Returns:
        A PyTorch model with a replaced classification head.
    """
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device (CPU or CUDA).

    Returns:
        Average training loss over the epoch.
    """
    raise NotImplementedError


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Evaluate the model on a DataLoader.

    Args:
        model: Trained model.
        loader: DataLoader for evaluation (val or test).
        device: Target device.

    Returns:
        Tuple of (accuracy, true_labels, predicted_labels).
    """
    raise NotImplementedError


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Save model weights and training state to a .pth file.

    Args:
        model: Model to save.
        path: Destination file path.
        epoch: Current epoch number (stored in checkpoint metadata).
        optimizer: Optional optimizer state to include.
    """
    raise NotImplementedError
