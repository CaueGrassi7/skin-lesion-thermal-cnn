"""
CNN model definitions and training utilities.

Includes a custom lightweight CNN and a transfer-learning wrapper around
pre-trained torchvision models (e.g. ResNet, EfficientNet).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


class ThermalCNN(nn.Module):
    """Lightweight CNN designed for thermal skin lesion classification.

    Architecture: three conv blocks (conv→BN→ReLU→MaxPool) followed by
    a two-layer MLP classifier with dropout.
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        """
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, in_channels, 224, 224) → (B, 32, 112, 112)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            # Block 2: → (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 3: → (B, 128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings (before final linear).

        Used for classical ML feature extraction (SVM / Random Forest).

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Embedding tensor of shape (B, 256).
        """
        x = self.features(x)
        # Run classifier up to (but not including) the last Linear
        for layer in list(self.classifier.children())[:-1]:
            x = layer(x)
        return x


def _adapt_first_conv(old_conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    """Return a copy of `old_conv` accepting `in_channels` input channels.

    The pre-trained RGB filters are averaged across the channel dimension to a
    single-channel filter, then that filter is replicated across all
    `in_channels` inputs (so a multi-frame temporal stack starts from the same
    pre-trained edge/texture detectors on every channel). This preserves the
    ImageNet features while matching the dataset's single-channel frames — or,
    for temporal input, a stack of them (see `TemporalThermalDataset`).
    """
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    if pretrained:
        mean_filter = old_conv.weight.data.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
        new_conv.weight.data = mean_filter.repeat(1, in_channels, 1, 1)
    return new_conv


def build_transfer_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    in_channels: int = 1,
) -> nn.Module:
    """Build a transfer-learning model from a torchvision backbone.

    The first conv layer is adapted to accept `in_channels` input channels by
    averaging the pre-trained RGB weights across the channel dimension and
    replicating the result (see `_adapt_first_conv`).

    Args:
        backbone: Name of the torchvision model ('resnet18', 'efficientnet_b0').
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: If True, freeze all layers except the final classifier.
        in_channels: Number of input channels (1 for a single grayscale frame,
            >1 for a temporal multi-frame stack).

    Returns:
        A PyTorch model with a replaced classification head.
    """
    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
        # Adapt first conv: (64, 3, 7, 7) → (64, in_channels, 7, 7)
        model.conv1 = _adapt_first_conv(model.conv1, in_channels, pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        model.features[0][0] = _adapt_first_conv(model.features[0][0], in_channels, pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}. Use 'resnet18' or 'efficientnet_b0'."
        )

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Unfreeze only the final classifier
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return the average loss and accuracy.

    Accuracy is computed from the same forward pass used for the loss,
    avoiding a redundant pass over the training set.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device (CPU, CUDA, or MPS).

    Returns:
        Tuple of (average training loss, training accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


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
    model.eval()
    correct = 0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    accuracy = correct / total
    return accuracy, all_true, all_pred


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
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
