"""
Evaluation metrics and result visualisation for skin lesion classifiers.

Computes accuracy, precision, recall, F1-score, and AUC-ROC, and generates
confusion matrices and ROC curves saved to the results directory.
"""

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


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
        Dictionary with keys: accuracy, precision, recall, specificity, f1,
        and auc_roc (only if y_prob is provided).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Specificity: TN / (TN + FP) — binary only; for multiclass average per class
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificities = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics["auc_roc"] = auc(fpr, tpr)

    return metrics


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
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")

    thresh = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                    ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


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
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, name in enumerate(class_names[:n_classes]):
        binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


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
    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_losses, label="Train")
    ax_loss.plot(epochs, val_losses, label="Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, train_accs, label="Train")
    ax_acc.plot(epochs, val_accs, label="Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def save_results(
    metrics: dict[str, float],
    output_path: str | Path,
    model_name: str = "model",
) -> None:
    """Append evaluation metrics to a CSV results file.

    Creates the file with a header row if it does not exist yet.

    Args:
        metrics: Dictionary of metric name → value.
        output_path: Path to the output CSV file.
        model_name: Identifier string for this model run.
    """
    output_path = Path(output_path)
    fieldnames = ["model"] + list(metrics.keys())
    write_header = not output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"model": model_name, **metrics})
