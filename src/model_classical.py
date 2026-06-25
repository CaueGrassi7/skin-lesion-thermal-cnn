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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader


def extract_cnn_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings from a trained CNN.

    Args:
        model: Trained ThermalCNN with an extract_embeddings() method.
        loader: DataLoader to extract features from (train, val, or test).
        device: Target device (CPU, CUDA, or MPS).

    Returns:
        Tuple of (embeddings of shape (N, embed_dim), labels of shape (N,)).
    """
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embeddings = model.extract_embeddings(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

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
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probability scores from a fitted classifier.

    Args:
        classifier: A fitted sklearn classifier with predict_proba support.
        X: Feature matrix of shape (N, embed_dim).

    Returns:
        Tuple of (predicted labels of shape (N,), probability scores of shape (N, C)).
    """
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)
    return y_pred, y_prob
