"""
Evaluation metrics and result visualisation for skin lesion classifiers.

Computes accuracy, precision, recall, F1-score, and AUC-ROC, and generates
confusion matrices and ROC curves saved to the results directory.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth label array of shape (N,).
        y_pred: Predicted label array of shape (N,).
        y_prob: Predicted probability array of shape (N, C); required for AUC-ROC.

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1, auc_roc (if y_prob given).
    """
    raise NotImplementedError


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot and optionally save a normalised confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings for axis labels.
        save_path: If provided, saves the figure to this path.
    """
    raise NotImplementedError


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities of shape (N, C).
        class_names: List of class name strings.
        save_path: If provided, saves the figure to this path.
    """
    raise NotImplementedError


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot loss and accuracy learning curves over training epochs.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_accs: Per-epoch training accuracies.
        val_accs: Per-epoch validation accuracies.
        save_path: If provided, saves the figure to this path.
    """
    raise NotImplementedError


def save_results(
    metrics: dict[str, float],
    output_path: str | Path,
    model_name: str = "model",
) -> None:
    """Append evaluation metrics to a CSV results file.

    Args:
        metrics: Dictionary of metric name → value.
        output_path: Path to the output CSV file.
        model_name: Identifier string for this model run.
    """
    raise NotImplementedError
