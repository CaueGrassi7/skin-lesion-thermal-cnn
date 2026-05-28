"""
Dataset loading, splitting, and augmentation for thermal skin lesion images.

Provides a PyTorch Dataset subclass and utility functions for building
train/validation/test DataLoaders with patient-level splits to prevent
data leakage across frames of the same patient.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["Healthy", "Sick"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


def _collect_samples(data_root: Path) -> list[dict]:
    """Walk data_root/Class/Patient/ and return one record per frame."""
    samples = []
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[class_dir.name]
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            match = re.search(r"Paciente_?(\d+)", patient_dir.name)
            patient_id = match.group(1) if match else patient_dir.name
            for frame_path in sorted(patient_dir.glob("*.png")):
                samples.append(
                    {
                        "path": frame_path,
                        "label": label,
                        "patient_id": patient_id,
                    }
                )
    return samples


def _patient_split(
    samples: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split samples into train/val/test grouping by patient to avoid leakage.

    Patients are split per class so each split preserves the original
    class ratio as closely as possible.
    """
    # Build per-class list of unique patient IDs (sorted for reproducibility)
    class_patients: dict[int, list[str]] = defaultdict(list)
    seen: set[tuple] = set()
    for s in samples:
        key = (s["label"], s["patient_id"])
        if key not in seen:
            seen.add(key)
            class_patients[s["label"]].append(s["patient_id"])

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    for label, patient_ids in class_patients.items():
        rng = np.random.default_rng(seed + label)
        ids = list(patient_ids)
        rng.shuffle(ids)

        n = len(ids)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))

        test_ids.update(ids[:n_test])
        val_ids.update(ids[n_test : n_test + n_val])
        train_ids.update(ids[n_test + n_val :])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in samples:
        pid = s["patient_id"]
        if pid in test_ids:
            splits["test"].append(s)
        elif pid in val_ids:
            splits["val"].append(s)
        else:
            splits["train"].append(s)

    return splits


class ThermalLesionDataset(Dataset):
    """PyTorch Dataset for thermal skin lesion images."""

    def __init__(
        self,
        samples: list[dict],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


def compute_mean_std(samples: list[dict]) -> Tuple[float, float]:
    """Compute per-channel mean and std over a list of samples.

    Iterates all frames — call only on the training split and reuse the
    result for val/test normalization.
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    count = 0

    for s in samples:
        img = np.array(Image.open(s["path"]).convert("L"), dtype=np.float32) / 255.0
        pixel_sum += img.sum()
        pixel_sq_sum += (img**2).sum()
        count += img.size

    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean**2)
    return float(mean), float(std)


def get_transforms(
    split: str,
    mean: float = 0.5,
    std: float = 0.5,
) -> Callable:
    """Return torchvision transforms for the given split.

    Args:
        split: One of 'train', 'val', or 'test'.
        mean: Dataset mean for normalization (single channel).
        std: Dataset std for normalization (single channel).
    """
    normalize = transforms.Normalize(mean=[mean], std=[std])

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return train, val, and test DataLoaders.

    If mean/std are not provided they are computed from the training split.

    Args:
        data_root: Path to the processed data directory.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        val_ratio: Fraction of patients for validation.
        test_ratio: Fraction of patients for test.
        seed: Random seed for reproducible splits.
        mean: Dataset mean for normalization; computed if None.
        std: Dataset std for normalization; computed if None.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_root = Path(data_root)
    samples = _collect_samples(data_root)
    splits = _patient_split(samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    if mean is None or std is None:
        mean, std = compute_mean_std(splits["train"])

    datasets = {
        split: ThermalLesionDataset(split_samples, transform=get_transforms(split, mean, std))
        for split, split_samples in splits.items()
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }

    return loaders["train"], loaders["val"], loaders["test"]
