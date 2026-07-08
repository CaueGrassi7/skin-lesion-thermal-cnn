"""Patient-level k-fold cross-validation pipeline.

Orchestrates the full experiment: builds patient-grouped k-fold splits,
trains ThermalCNN and ResNet18 per fold, extracts embeddings + thermal-dynamics
features for SVM/Random Forest, evaluates every model at both frame and
patient/sequence level, and persists metrics, pooled predictions, training
histories, and checkpoints to `config.results_dir`.

Entry point: `run_cross_validation(config, device)`, called from
`scripts/train_cv.py`. Kept separate from the training/evaluation notebook so
long runs (hours) can execute as a background process with a log file,
independent of a live Jupyter kernel; the notebook loads this module's saved
outputs for analysis and does not retrain anything.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.dataset import (
    ThermalLesionDataset,
    _build_loaders_from_splits,
    _collect_samples,
    _patient_kfold_split,
    compute_class_weights,
    compute_temporal_features,
    get_transforms,
)
from src.evaluate import aggregate_by_group, compute_metrics
from src.model_classical import (
    extract_cnn_embeddings,
    predict_classical,
    tune_random_forest,
    tune_svm,
)
from src.model_cnn import (
    ThermalCNN,
    build_transfer_model,
    evaluate_model,
    save_checkpoint,
    train_one_epoch,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = ["ThermalCNN", "ResNet18", "SVM", "Random Forest"]
FILE_SLUG = {
    "ThermalCNN": "thermal_cnn", "ResNet18": "resnet18", "SVM": "svm", "Random Forest": "rf",
}


@dataclass
class TrainConfig:
    """Hyperparameters and paths for one cross-validation run.

    Defaults match the full-dataset run; `scripts/train_cv.py --quick-test`
    overrides `k_folds`/`num_epochs`/`patience`/`data_root` for a fast
    end-to-end smoke test on `data/processed/_subset_validation`.
    """

    data_root: Path
    results_dir: Path
    frame_stride: int = 5
    k_folds: int = 5
    num_classes: int = 2
    lr: float = 1e-3
    weight_decay: float = 5e-4  # increased from 1e-4 to reduce overfitting
    num_epochs: int = 50
    patience: int = 7
    batch_size: int = 32
    resnet_phase1_epochs: int = 10
    resnet_phase1_patience: int = 4
    resnet_phase2_lr: float = 1e-4
    svm_rf_tune_iter: int = 8
    svm_rf_tune_cv: int = 3
    seed: int = 42

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-safe dict (Path fields as strings)."""
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["results_dir"] = str(self.results_dir)
        return d


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick CUDA > MPS > CPU. `torch.cuda.is_available()` is always False on
    Apple Silicon — MPS is ~7x faster than CPU for ThermalCNN on that hardware,
    so it must be checked explicitly rather than falling through to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _training_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    model_name: str,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    patience: int,
    class_weights: Optional[torch.Tensor] = None,
    val_group_ids: Optional[list] = None,
    checkpoint_path: Optional[Path] = None,
) -> tuple[nn.Module, dict]:
    """Train with early stopping on validation AUC-ROC.

    AUC-ROC is used instead of val_loss as the stopping criterion because
    val_loss can diverge (overconfidence) even while discriminative capacity
    stays stable — AUC is more robust to imbalance and matches the metric of
    interest for this problem.

    `class_weights`, if given, weights CrossEntropyLoss by inverse class
    frequency in the training split (`compute_class_weights`) — without it, a
    fold with skewed class ratios (see `_patient_kfold_split`) can make the
    model collapse to predicting the majority class.

    `val_group_ids`, if given, aggregates validation probabilities by
    patient/sequence (same logic as `aggregate_by_group`) before computing the
    AUC used for early stopping/scheduling — frame-level AUC is noisy with few
    validation patients (~19 per fold), and neighboring frames from the same
    sequence are near-duplicates, so this aggregation reduces noise in the
    stopping criterion.

    Returns the model (restored to its best-val-AUC state) and a dict with
    per-epoch loss/accuracy/AUC history.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_val_auc = -float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_true, _ = evaluate_model(model, val_loader, device)

        model.eval()
        val_loss = 0.0
        val_probs = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                val_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
        val_loss /= len(val_loader.dataset)

        if val_group_ids is not None:
            val_prob_2col = np.stack(
                [1 - np.array(val_probs), np.array(val_probs)], axis=1,
            )
            val_true_agg, _, val_prob_agg = aggregate_by_group(
                val_true, val_prob_2col, val_group_ids,
            )
            val_auc = roc_auc_score(val_true_agg, val_prob_agg[:, 1])
        else:
            val_auc = roc_auc_score(val_true, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        scheduler.step(val_auc)

        improved = val_auc > best_val_auc
        marker = " *" if improved else ""
        logger.info(
            "[%s] epoca %02d/%d | loss treino=%.4f val=%.4f | "
            "acc treino=%.4f val=%.4f | auc val=%.4f%s",
            model_name, epoch, num_epochs, train_loss, val_loss,
            train_acc, val_acc, val_auc, marker,
        )

        if improved:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                save_checkpoint(model, str(checkpoint_path), epoch, optimizer)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "  -> Early stopping na epoca %d (sem melhora ha %d epocas)", epoch, patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def _predict_probs(
    model: nn.Module, loader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, y_prob) for every sample in `loader`."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_true.extend(labels.tolist())
            all_pred.extend(preds)
            all_prob.extend(probs)
    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def _temporal_feature_matrix(
    fold_samples: list[dict], temporal_features: dict[str, np.ndarray],
) -> np.ndarray:
    """Map each sample in `fold_samples` (same order) to its sequence's
    precomputed thermal-dynamics feature vector (patient_id + label)."""
    keys = [f"{s['patient_id']}_{s['label']}" for s in fold_samples]
    return np.stack([temporal_features[k] for k in keys])


def _train_and_evaluate_fold(
    fold_splits: dict[str, list[dict]],
    fold_idx: int,
    config: TrainConfig,
    device: torch.device,
    temporal_features: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict], dict, dict]:
    """Train and evaluate all 4 models on one cross-validation fold.

    Besides frame-level metrics, aggregates test predictions by sequence
    (patient_id + class — a patient can contribute frames to both classes, see
    `_patient_kfold_split`) via mean probability before argmax. Neighboring
    frames within a thermal sequence are near-duplicates, so this aggregation
    should reduce per-frame noise and stabilize the across-fold variance seen
    in frame-level metrics.

    Returns:
        results: list of dicts (one per model, frame level) with fold, model, and metrics.
        results_seq: same, at patient/sequence level (aggregated).
        predictions: dict model -> (y_true, y_pred, y_prob) on the test fold (frame level).
        histories: dict with ThermalCNN's and ResNet18's training history.
    """
    results_dir = config.results_dir
    train_loader, val_loader, test_loader, mean, std = _build_loaders_from_splits(
        fold_splits, batch_size=config.batch_size, num_workers=0,
    )
    # test_loader doesn't shuffle (shuffle=False for splits != "train"), so
    # prediction order matches fold_splits["test"] order. Same for val.
    test_seq_ids = [f"{s['patient_id']}_{s['label']}" for s in fold_splits["test"]]
    val_seq_ids = [f"{s['patient_id']}_{s['label']}" for s in fold_splits["val"]]

    # Fold class weights — corrects per-fold class imbalance (see
    # `_patient_kfold_split`), which without correction could make a model
    # collapse to predicting the majority class in a skewed fold.
    class_weights = compute_class_weights(fold_splits["train"]).to(device)

    results: list[dict] = []
    results_seq: list[dict] = []
    predictions: dict = {}

    def record(model_name, y_true, y_pred, y_prob):
        results.append({
            "fold": fold_idx, "model": model_name, **compute_metrics(y_true, y_pred, y_prob),
        })
        seq_true, seq_pred, seq_prob = aggregate_by_group(y_true, y_prob, test_seq_ids)
        results_seq.append({
            "fold": fold_idx, "model": model_name,
            **compute_metrics(seq_true, seq_pred, seq_prob),
        })
        predictions[model_name] = (y_true, y_pred, y_prob)

    # --- ThermalCNN ---
    cnn_model = ThermalCNN(num_classes=config.num_classes, in_channels=1)
    cnn_model, cnn_history = _training_loop(
        cnn_model, train_loader, val_loader, device,
        model_name=f"ThermalCNN[fold {fold_idx}]",
        lr=config.lr, weight_decay=config.weight_decay,
        num_epochs=config.num_epochs, patience=config.patience,
        class_weights=class_weights, val_group_ids=val_seq_ids,
        checkpoint_path=results_dir / f"checkpoint_thermal_cnn_fold{fold_idx}.pth",
    )
    cnn_true, cnn_pred, cnn_prob = _predict_probs(cnn_model, test_loader, device)
    record("ThermalCNN", cnn_true, cnn_pred, cnn_prob)

    # --- ResNet18 (phase 1: frozen backbone, phase 2: full fine-tuning) ---
    resnet_model = build_transfer_model(
        "resnet18", num_classes=config.num_classes, pretrained=True, freeze_backbone=True,
    )
    resnet_model, _ = _training_loop(
        resnet_model, train_loader, val_loader, device,
        model_name=f"ResNet18-fase1[fold {fold_idx}]",
        lr=config.lr, weight_decay=config.weight_decay,
        num_epochs=config.resnet_phase1_epochs, patience=config.resnet_phase1_patience,
        class_weights=class_weights, val_group_ids=val_seq_ids,
    )
    for param in resnet_model.parameters():
        param.requires_grad = True
    resnet_model, resnet_history = _training_loop(
        resnet_model, train_loader, val_loader, device,
        model_name=f"ResNet18-fase2[fold {fold_idx}]",
        lr=config.resnet_phase2_lr, weight_decay=config.weight_decay,
        num_epochs=config.num_epochs, patience=config.patience,
        class_weights=class_weights, val_group_ids=val_seq_ids,
        checkpoint_path=results_dir / f"checkpoint_resnet18_fold{fold_idx}.pth",
    )
    res_true, res_pred, res_prob = _predict_probs(resnet_model, test_loader, device)
    record("ResNet18", res_true, res_pred, res_prob)

    # --- SVM and Random Forest on ResNet18 embeddings + thermal dynamics ---
    # Uses ResNet18 (the highest-AUC model) as the feature extractor rather
    # than ThermalCNN — its more discriminative embeddings should propagate to
    # SVM/RF as well.
    #
    # Deterministic loader (no shuffle, no geometric augmentation) for
    # extracting train embeddings: reusing train_loader here reshuffled order
    # on every extraction and applied random augmentation, injecting
    # unnecessary noise into the features fed to SVM/RF — the goal is a stable
    # representation of the frame, not training diversity.
    train_eval_loader = DataLoader(
        ThermalLesionDataset(fold_splits["train"], transform=get_transforms("val", mean, std)),
        batch_size=config.batch_size, shuffle=False,
    )
    X_train_emb, y_train_emb = extract_cnn_embeddings(resnet_model, train_eval_loader, device)
    X_test_emb, y_test_emb = extract_cnn_embeddings(resnet_model, test_loader, device)

    # Thermal-dynamics features (warming/cooling rate etc., see
    # `compute_temporal_features`) concatenated to the embedding — the dataset
    # is *dynamic* thermal imaging (DTI), but up to this point the pipeline
    # treats every frame independently and never sees how temperature evolves
    # across the sequence.
    X_train_temporal = _temporal_feature_matrix(fold_splits["train"], temporal_features)
    X_test_temporal = _temporal_feature_matrix(fold_splits["test"], temporal_features)

    scaler = StandardScaler().fit(np.hstack([X_train_emb, X_train_temporal]))
    X_train_full = scaler.transform(np.hstack([X_train_emb, X_train_temporal]))
    X_test_full = scaler.transform(np.hstack([X_test_emb, X_test_temporal]))

    # groups=patient_id (not patient_id+label) for the inner GroupKFold of the
    # hyperparameter search: a patient contributes frames to both classes, so
    # grouping by patient_id alone guarantees no frame from the same patient
    # leaks between the inner search fold and its inner validation fold.
    train_patient_ids = np.array([s["patient_id"] for s in fold_splits["train"]])

    svm = tune_svm(
        X_train_full, y_train_emb, groups=train_patient_ids,
        n_iter=config.svm_rf_tune_iter, cv=config.svm_rf_tune_cv,
    )
    svm_pred, svm_prob = predict_classical(svm, X_test_full, n_classes=config.num_classes)
    record("SVM", y_test_emb, svm_pred, svm_prob)

    rf = tune_random_forest(
        X_train_full, y_train_emb, groups=train_patient_ids,
        n_iter=config.svm_rf_tune_iter, cv=config.svm_rf_tune_cv,
    )
    rf_pred, rf_prob = predict_classical(rf, X_test_full, n_classes=config.num_classes)
    record("Random Forest", y_test_emb, rf_pred, rf_prob)

    logger.info("--- Fold %d concluido ---", fold_idx)
    for r, r_seq in zip(results, results_seq):
        auc = r.get("auc_roc", float("nan"))
        auc_seq = r_seq.get("auc_roc", float("nan"))
        logger.info(
            "  %-15s frame: acc=%.4f f1=%.4f auc=%.4f  |  paciente: acc=%.4f f1=%.4f auc=%.4f",
            r["model"], r["accuracy"], r["f1"], auc,
            r_seq["accuracy"], r_seq["f1"], auc_seq,
        )

    return (
        results, results_seq, predictions,
        {"ThermalCNN": cnn_history, "ResNet18": resnet_history},
    )


def _save_outputs(
    config: TrainConfig,
    all_results: list[dict],
    all_results_seq: list[dict],
    all_predictions: dict,
    fold0_histories: dict,
) -> None:
    """Persist everything the analysis notebook needs, without requiring it
    to retrain anything: per-fold metrics (frame + patient level), pooled
    test predictions per model, fold-0 training histories, and the config
    used for this run."""
    results_dir = config.results_dir

    folds_df = pd.DataFrame(all_results)
    folds_df.to_csv(results_dir / "metrics_cv_folds.csv", index=False)

    folds_seq_df = pd.DataFrame(all_results_seq)
    folds_seq_df.to_csv(results_dir / "metrics_cv_folds_patient.csv", index=False)

    for name in MODEL_NAMES:
        y_true = np.array(all_predictions[name]["y_true"])
        y_pred = np.array(all_predictions[name]["y_pred"])
        y_prob = np.concatenate(all_predictions[name]["y_prob"], axis=0)
        np.savez(
            results_dir / f"predictions_{FILE_SLUG[name]}.npz",
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        )

    for model_name, history in fold0_histories.items():
        history_path = results_dir / f"history_{FILE_SLUG[model_name]}_fold0.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    with open(results_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_json_dict(), f, indent=2)

    logger.info("Metricas, predicoes e historico salvos em %s", results_dir)


def run_cross_validation(config: TrainConfig, device: torch.device) -> None:
    """Run the full patient-level k-fold cross-validation experiment and save
    all outputs under `config.results_dir`."""
    config.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Coletando amostras de %s (frame_stride=%d)...", config.data_root, config.frame_stride,
    )
    samples = _collect_samples(config.data_root, frame_stride=config.frame_stride)
    folds = _patient_kfold_split(samples, k=config.k_folds, seed=config.seed)

    # Thermal-dynamics features (one per patient+class sequence) computed once
    # over the full dataset and reused across all folds — a fixed property of
    # each patient's own frames, independent of which fold uses it as
    # train/val/test.
    temporal_features = compute_temporal_features(samples)

    logger.info(
        "Total de amostras: %s | sequencias com features de dinamica termica: %s",
        f"{len(samples):,}", f"{len(temporal_features):,}",
    )
    for i, f in enumerate(folds):
        n_train_p = len(set(s["patient_id"] for s in f["train"]))
        n_val_p = len(set(s["patient_id"] for s in f["val"]))
        n_test_p = len(set(s["patient_id"] for s in f["test"]))
        logger.info(
            "Fold %d: train=%s (%dp) val=%s (%dp) test=%s (%dp)",
            i, f"{len(f['train']):,}", n_train_p,
            f"{len(f['val']):,}", n_val_p, f"{len(f['test']):,}", n_test_p,
        )

    all_results: list[dict] = []
    all_results_seq: list[dict] = []
    all_predictions = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in MODEL_NAMES}
    fold0_histories: dict = {}

    for i, fold in enumerate(folds):
        logger.info("%s\nFOLD %d/%d\n%s", "=" * 70, i + 1, config.k_folds, "=" * 70)
        results, results_seq, predictions, histories = _train_and_evaluate_fold(
            fold, i, config, device, temporal_features,
        )
        all_results.extend(results)
        all_results_seq.extend(results_seq)
        for name, (y_true, y_pred, y_prob) in predictions.items():
            all_predictions[name]["y_true"].extend(np.asarray(y_true).tolist())
            all_predictions[name]["y_pred"].extend(np.asarray(y_pred).tolist())
            all_predictions[name]["y_prob"].append(np.asarray(y_prob))
        if i == 0:
            fold0_histories = histories

    _save_outputs(config, all_results, all_results_seq, all_predictions, fold0_histories)
    logger.info("Validacao cruzada concluida.")
