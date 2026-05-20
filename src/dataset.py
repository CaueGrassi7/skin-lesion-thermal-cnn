"""
Dataset loading, splitting, and augmentation for thermal skin lesion images.

Provides a PyTorch Dataset subclass and utility functions for building
train/validation/test DataLoaders with optional data augmentation.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class ThermalLesionDataset(Dataset):
    """PyTorch Dataset for thermal skin lesion images."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: Path to the processed data directory.
            split: One of 'train', 'val', or 'test'.
            transform: Optional torchvision transforms applied to each image.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (image_tensor, label) for the given index."""
        raise NotImplementedError


def get_transforms(split: str) -> Callable:
    """Return torchvision transforms appropriate for the given split.

    Training split includes augmentations (flip, rotation, color jitter).
    Val/test split returns only resize and normalize.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        A composed torchvision transform.
    """
    raise NotImplementedError


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return train, val, and test DataLoaders.

    Args:
        data_root: Path to the processed data directory.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    raise NotImplementedError
