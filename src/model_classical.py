"""
Classical machine learning classifiers for thermal skin lesion images.

Implements SVM and Random Forest pipelines using hand-crafted features
(HOG, LBP) or CNN-extracted embeddings as input representations.
"""

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """Extract Histogram of Oriented Gradients (HOG) features from images.

    Args:
        images: Array of images with shape (N, H, W) or (N, H, W, C).

    Returns:
        Feature matrix of shape (N, feature_dim).
    """
    raise NotImplementedError


def extract_lbp_features(images: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Extract Local Binary Pattern (LBP) texture features from images.

    Args:
        images: Array of grayscale images with shape (N, H, W).
        radius: Radius of the circular LBP neighbourhood.
        n_points: Number of circularly symmetric neighbour set points.

    Returns:
        Feature matrix of shape (N, n_points + 2).
    """
    raise NotImplementedError


def build_svm(kernel: str = "rbf", C: float = 1.0, **kwargs: Any) -> ClassifierMixin:
    """Build a scikit-learn SVM classifier.

    Args:
        kernel: SVM kernel type ('linear', 'rbf', 'poly').
        C: Regularisation parameter.
        **kwargs: Additional keyword arguments forwarded to SVC.

    Returns:
        An unfitted sklearn SVC instance.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def train_classical(
    classifier: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ClassifierMixin:
    """Fit a classical classifier on training features.

    Args:
        classifier: An unfitted sklearn classifier.
        X_train: Training feature matrix of shape (N, feature_dim).
        y_train: Training labels of shape (N,).

    Returns:
        The fitted classifier.
    """
    raise NotImplementedError


def predict_classical(
    classifier: ClassifierMixin,
    X: np.ndarray,
) -> np.ndarray:
    """Generate predictions from a fitted classical classifier.

    Args:
        classifier: A fitted sklearn classifier.
        X: Feature matrix of shape (N, feature_dim).

    Returns:
        Predicted label array of shape (N,).
    """
    raise NotImplementedError
