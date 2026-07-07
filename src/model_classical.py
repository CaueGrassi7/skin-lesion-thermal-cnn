"""
Classical machine learning classifiers for thermal skin lesion images.

Feature strategy: CNN-extracted embeddings from ThermalCNN's penultimate layer
(dim=256), which are methodologically stronger than hand-crafted HOG/LBP features
and align with the reference literature (Magalhães et al. 2021).
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.svm import SVC
from torch.utils.data import DataLoader


def extract_cnn_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings from a trained CNN.

    Supports ThermalCNN (via its extract_embeddings() method) and
    torchvision-style backbones such as ResNet18 (via a temporary swap of the
    final `fc` layer for nn.Identity, restored afterwards).

    Args:
        model: Trained model — either a ThermalCNN or a torchvision backbone
            with a top-level `fc` layer (e.g. ResNet18).
        loader: DataLoader to extract features from (train, val, or test).
        device: Target device (CPU, CUDA, or MPS).

    Returns:
        Tuple of (embeddings of shape (N, embed_dim), labels of shape (N,)).
    """
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    original_fc = None
    use_native_embeddings = hasattr(model, "extract_embeddings")
    if not use_native_embeddings:
        if not hasattr(model, "fc"):
            raise AttributeError(
                "Model has neither extract_embeddings() nor a top-level fc layer."
            )
        original_fc = model.fc
        model.fc = nn.Identity()

    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                if use_native_embeddings:
                    embeddings = model.extract_embeddings(images)
                else:
                    embeddings = model(images)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
    finally:
        if original_fc is not None:
            model.fc = original_fc

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def build_svm(kernel: str = "rbf", C: float = 1.0, **kwargs: Any) -> ClassifierMixin:
    """Build a scikit-learn SVM classifier with probability estimates enabled.

    Args:
        kernel: SVM kernel type ('linear', 'rbf', 'poly').
        C: Regularisation parameter.
        **kwargs: Additional keyword arguments forwarded to SVC.

    Returns:
        An unfitted sklearn SVC instance.
    """
    return SVC(kernel=kernel, C=C, probability=True, random_state=42, **kwargs)


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    **kwargs: Any,
) -> ClassifierMixin:
    """Build a scikit-learn Random Forest classifier.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree; None for unlimited.
        **kwargs: Additional keyword arguments forwarded to RandomForestClassifier.

    Returns:
        An unfitted sklearn RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )


def _effective_group_cv(groups: np.ndarray, cv: int) -> int:
    """Clamp `cv` to the number of unique groups available.

    `GroupKFold(n_splits=cv)` raises if `cv` exceeds the number of unique
    groups — this happens on small folds/subsets (e.g. the QUICK_TEST subset
    has only a couple of training patients). Returns the clamped value, or 1
    if there are too few groups to search at all.
    """
    return max(1, min(cv, len(np.unique(groups))))


def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_iter: int = 8,
    cv: int = 3,
    random_state: int = 42,
) -> ClassifierMixin:
    """Random-search SVM hyperparameters with a patient-grouped inner CV.

    `GroupKFold` on `groups` (patient ids) keeps frames from the same patient
    out of the inner validation fold — otherwise near-duplicate frames from
    one patient would leak between the inner train/val split, making the
    search overfit to patient identity rather than genuine class signal.
    `class_weight="balanced"` compensates for per-fold class-ratio drift
    (see `compute_class_weights` in `dataset.py`).

    Args:
        X_train: Training feature matrix of shape (N, feat_dim).
        y_train: Training labels of shape (N,).
        groups: Patient (or patient+class) id per row, shape (N,), used to
            group frames for `GroupKFold` so the inner search stays
            patient-level clean.
        n_iter: Number of random parameter combinations to try.
        cv: Number of inner group folds (clamped to the number of unique
            groups if smaller).
        random_state: Seed for the random search and the resulting estimator.

    Returns:
        The best-scoring fitted SVC (probability=True, refit on all of X_train).
        Falls back to an untuned default (still class-balanced) SVC if too
        few groups are available to run a grouped inner CV.
    """
    if len(np.unique(y_train)) < 2:
        # Degenerate split (e.g. a tiny QUICK_TEST subset) — SVC can't fit a
        # single class. Fall back to a prior-frequency dummy predictor.
        return DummyClassifier(strategy="prior").fit(X_train, y_train)

    effective_cv = _effective_group_cv(groups, cv)
    if effective_cv < 2:
        return build_svm(kernel="rbf", C=1.0, class_weight="balanced").fit(X_train, y_train)

    param_dist = {
        "C": [0.1, 1.0, 10.0, 50.0],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "kernel": ["rbf"],
        "class_weight": ["balanced"],
    }
    search = RandomizedSearchCV(
        SVC(probability=True, random_state=random_state),
        param_distributions=param_dist,
        n_iter=min(n_iter, 16),
        cv=GroupKFold(n_splits=effective_cv),
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_iter: int = 8,
    cv: int = 3,
    random_state: int = 42,
) -> ClassifierMixin:
    """Random-search Random Forest hyperparameters with a patient-grouped inner CV.

    See `tune_svm` for why `GroupKFold` and `class_weight="balanced"` matter here.

    Args:
        X_train: Training feature matrix of shape (N, feat_dim).
        y_train: Training labels of shape (N,).
        groups: Patient (or patient+class) id per row, shape (N,).
        n_iter: Number of random parameter combinations to try.
        cv: Number of inner group folds (clamped to the number of unique
            groups if smaller).
        random_state: Seed for the random search and the resulting estimator.

    Returns:
        The best-scoring fitted RandomForestClassifier (refit on all of X_train).
        Falls back to an untuned default (still class-balanced) forest if too
        few groups are available to run a grouped inner CV.
    """
    if len(np.unique(y_train)) < 2:
        # Degenerate split (e.g. a tiny QUICK_TEST subset) — fall back to a
        # prior-frequency dummy predictor.
        return DummyClassifier(strategy="prior").fit(X_train, y_train)

    effective_cv = _effective_group_cv(groups, cv)
    if effective_cv < 2:
        return build_random_forest(n_estimators=100, class_weight="balanced").fit(X_train, y_train)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=min(n_iter, 16),
        cv=GroupKFold(n_splits=effective_cv),
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_


def train_classical(
    classifier: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Fit a classical classifier on training features.

    Args:
        classifier: An unfitted sklearn classifier.
        X_train: Training feature matrix of shape (N, embed_dim).
        y_train: Training labels of shape (N,).

    Returns:
        The fitted classifier.
    """
    return classifier.fit(X_train, y_train)


def predict_classical(
    classifier: ClassifierMixin,
    X: np.ndarray,
    n_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probability scores from a fitted classifier.

    If `classifier` was fit on a degenerate single-class split (e.g. a tiny
    QUICK_TEST fold — see the fallback in `tune_svm`/`tune_random_forest`),
    `predict_proba` only returns one column. Zero-pads the missing class
    columns (via `classifier.classes_`) so callers can always assume a
    `(N, n_classes)` probability matrix, e.g. when concatenating predictions
    across folds.

    Args:
        classifier: A fitted sklearn classifier with predict_proba support.
        X: Feature matrix of shape (N, embed_dim).
        n_classes: Total number of classes in the task (not just the ones
            `classifier` happened to see during fitting).

    Returns:
        Tuple of (predicted labels of shape (N,), probability scores of shape (N, C)).
    """
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)

    classes = np.asarray(classifier.classes_).astype(int)
    if len(classes) < n_classes:
        full_prob = np.zeros((y_prob.shape[0], n_classes), dtype=y_prob.dtype)
        full_prob[:, classes] = y_prob
        y_prob = full_prob

    return y_pred, y_prob
