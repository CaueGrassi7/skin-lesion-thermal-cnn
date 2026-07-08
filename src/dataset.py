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


def _collect_samples(data_root: Path, frame_stride: int = 1) -> list[dict]:
    """Walk data_root/Class/Patient/ and return one record per frame.

    Args:
        data_root: Root directory with Class/Patient/*.png structure.
        frame_stride: Keep 1 in every N frames per patient (in sequence
            order). Frames from the same thermal sequence are highly
            correlated (near-duplicate), so a stride > 1 reduces redundancy
            and helps prevent the model from memorizing patient-specific
            sequences instead of learning generalizable lesion features.
    """
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
            frame_paths = sorted(patient_dir.glob("*.png"))[::frame_stride]
            for frame_path in frame_paths:
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


def _patient_kfold_split(
    samples: list[dict],
    k: int = 5,
    seed: int = 42,
) -> list[dict[str, list[dict]]]:
    """Split samples into k patient-level folds.

    Each patient is assigned to exactly one of k folds. For fold i, that
    fold's patients become the test set, the next fold (i+1 mod k) becomes
    the validation set (for early stopping), and the remaining k-2 folds
    become the training set. Every patient is used as test exactly once and
    as validation exactly once across the k iterations.

    Each patient is assigned as a whole (both of its classes go to the same
    fold) — in this dataset a majority of patients contribute frames to
    *both* classes (a healthy region and a lesion region imaged for the same
    person), so a given patient_id is not reliably tied to a single label,
    and splitting a patient's frames across folds would leak identity-level
    features (thermal signature, skin tone) between train and test.

    Fold assignment is stratified by class frame count (not just patient
    count): a plain `idx % k` split can produce folds whose test-set class
    ratio drifts far from the global ratio purely by chance (observed
    empirically: one fold's test set was 65/35 vs. a global ratio of 55/45),
    which adds a confound on top of genuine model variance when comparing
    folds. Patients are sorted by their class-0-minus-class-1 frame-count
    imbalance and dealt out in a "snake" order (0,1,...,k-1,k-1,...,1,0,0,...) —
    this keeps each fold's patient count exactly balanced (unlike a greedy
    min-cost assignment, which can pathologically pile patients into a few
    folds on repeated ties) while alternating direction spreads
    heavily-skewed patients evenly across folds.

    Args:
        samples: Full sample list (as returned by `_collect_samples`).
        k: Number of folds.
        seed: Random seed for reproducible fold assignment.

    Returns:
        List of k dicts, each with 'train'/'val'/'test' sample lists.
    """
    patient_class_counts: dict[str, list[int]] = {}
    for s in samples:
        counts = patient_class_counts.setdefault(s["patient_id"], [0, 0])
        counts[s["label"]] += 1

    # Randomize tie order deterministically, then stable-sort by class
    # imbalance (descending) so patients with the same imbalance keep a
    # randomized relative order.
    rng = np.random.default_rng(seed)
    patient_ids = list(patient_class_counts)
    rng.shuffle(patient_ids)
    patient_ids.sort(
        key=lambda pid: patient_class_counts[pid][0] - patient_class_counts[pid][1],
        reverse=True,
    )

    snake_order = list(range(k)) + list(range(k - 1, -1, -1))
    patient_fold = {
        pid: snake_order[i % len(snake_order)] for i, pid in enumerate(patient_ids)
    }

    folds = []
    for test_fold in range(k):
        val_fold = (test_fold + 1) % k
        fold_splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for s in samples:
            f = patient_fold[s["patient_id"]]
            if f == test_fold:
                fold_splits["test"].append(s)
            elif f == val_fold:
                fold_splits["val"].append(s)
            else:
                fold_splits["train"].append(s)
        folds.append(fold_splits)

    return folds


def _build_loaders_from_splits(
    splits: dict[str, list[dict]],
    batch_size: int = 32,
    num_workers: int = 0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """Build train/val/test DataLoaders from an already-computed split dict.

    Mean/std are computed from the split's training samples if not provided.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, mean, std).
    """
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
    return loaders["train"], loaders["val"], loaders["test"], mean, std


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


def compute_class_weights(samples: list[dict], num_classes: int = len(CLASSES)) -> torch.Tensor:
    """Compute inverse-frequency class weights for a (possibly imbalanced) split.

    Per-fold class ratios in the patient-level k-fold split can drift far
    from the global ratio (see `_patient_kfold_split` docstring) — an
    unweighted CrossEntropyLoss lets the model collapse to predicting the
    majority class in a skewed fold. Weight_c = N / (num_classes * count_c),
    so a class with half the average frequency gets ~2x the loss weight.

    Args:
        samples: Sample list (as returned by `_collect_samples`) for the
            split the loss will be computed on — typically the training split.
        num_classes: Total number of classes.

    Returns:
        Float tensor of shape (num_classes,), one weight per class index.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        counts[s["label"]] += 1
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


TEMPORAL_FEATURE_NAMES = ["slope", "delta_first_last", "std_across_frames", "range"]
TEMPORAL_FEATURE_DIM = len(TEMPORAL_FEATURE_NAMES)


def compute_temporal_features(samples: list[dict]) -> dict[str, np.ndarray]:
    """Compute per-sequence thermal-dynamics features from frame intensity over time.

    The dataset is *dynamic* thermal imaging (DTI) — diagnostic signal lives
    partly in how a lesion's temperature evolves across the sequence (warming/
    cooling rate), not just in a single frame's static appearance. The CNN/
    classical pipeline otherwise treats every frame as i.i.d. and never sees
    that dynamic. This computes, per (patient_id, label) sequence, simple
    scalar summaries of the mean-intensity trajectory across its frames (in
    the same order used by `_collect_samples`, i.e. sorted filename order
    after striding):

    - slope: least-squares slope of mean frame intensity vs. frame index
      (warming/cooling rate)
    - delta_first_last: last frame's mean intensity minus the first frame's
    - std_across_frames: std of per-frame mean intensity (temporal variability)
    - range: max minus min of per-frame mean intensity

    Args:
        samples: Sample list (as returned by `_collect_samples`), covering
            one or more full patient sequences (do not pass a frame subset
            that cuts a sequence short — the trend would be meaningless).

    Returns:
        Dict mapping "{patient_id}_{label}" (matching the group-id convention
        used by `aggregate_by_group`) to a float array of shape
        (TEMPORAL_FEATURE_DIM,), in the order of `TEMPORAL_FEATURE_NAMES`.
    """
    sequences: dict[str, list] = defaultdict(list)
    for s in samples:
        key = f"{s['patient_id']}_{s['label']}"
        sequences[key].append(s["path"])

    features: dict[str, np.ndarray] = {}
    for key, paths in sequences.items():
        means = np.array(
            [np.asarray(Image.open(p).convert("L"), dtype=np.float32).mean() / 255.0 for p in paths]
        )
        if len(means) < 2:
            features[key] = np.zeros(TEMPORAL_FEATURE_DIM, dtype=np.float32)
            continue
        frame_idx = np.arange(len(means))
        slope = float(np.polyfit(frame_idx, means, 1)[0])
        delta_first_last = float(means[-1] - means[0])
        std_across_frames = float(means.std())
        value_range = float(means.max() - means.min())
        features[key] = np.array(
            [slope, delta_first_last, std_across_frames, value_range], dtype=np.float32
        )
    return features


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
                # Geometric-only augmentation: no brightness/contrast jitter,
                # since absolute pixel intensity carries thermal/diagnostic meaning.
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
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
    frame_stride: int = 1,
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
        frame_stride: Keep 1 in every N frames per patient; see `_collect_samples`.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_root = Path(data_root)
    samples = _collect_samples(data_root, frame_stride=frame_stride)
    splits = _patient_split(samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    train_loader, val_loader, test_loader, _, _ = _build_loaders_from_splits(
        splits, batch_size=batch_size, num_workers=num_workers, mean=mean, std=std
    )
    return train_loader, val_loader, test_loader
