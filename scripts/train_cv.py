#!/usr/bin/env python
"""CLI entry point for patient-level k-fold cross-validation training.

Runs the full experiment (ThermalCNN, ResNet18, SVM, Random Forest) and
writes metrics, pooled predictions, training histories, and checkpoints to
`results/runs/<run-name>/` (one self-contained folder per run, plus a RUN.md
documenting what changed for that run — see `--notes`). The training/evaluation
notebook only reads a run folder's outputs — it never retrains anything, so
this script is the single source of truth for a run's numbers.

Usage:
    # Full run (~hours) — always pass --notes so RUN.md documents what this
    # run is testing relative to the previous one.
    python scripts/train_cv.py --notes "Added per-fold class weighting and
    patient-grouped SVM/RF tuning."

    # Fast smoke test on the small subset, before committing to a full run
    python scripts/train_cv.py --quick-test

Long full runs should be launched detached so they survive a closed terminal:
    nohup python scripts/train_cv.py --notes "..." > /dev/null 2>&1 &
Progress is written to <run-dir>/train_cv.log either way.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import TrainConfig, get_device, run_cross_validation, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run on data/processed/_subset_validation with few folds/epochs, "
             "to smoke-test the pipeline end-to-end before a full run.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override the dataset root.")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this run's folder under results/runs/. Default: a timestamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Free-text description of what changed for this run (vs. the previous one), written to RUN.md. "
             "Strongly recommended for every non-quick-test run.",
    )
    parser.add_argument(
        "--notes-file", type=Path, default=None,
        help="Path to a file whose contents are used as --notes (overrides --notes if both are given).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Override the output directory entirely, bypassing results/runs/<run-name>/. "
             "Mainly for smoke tests / scratch runs that shouldn't be tracked as a numbered run.",
    )
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--temporal-mode", type=str, default="off", choices=["off", "context", "diff"],
        help="Temporal (dynamic-thermal) CNN input. 'off' = classic single frame; "
             "'context' = stack of frames sampled across the sequence; "
             "'diff' = anchor + (later frame - anchor) reheating maps. Default: off "
             "('context'/'diff' empirically hurt patient-level AUC — see CLAUDE.md).",
    )
    parser.add_argument(
        "--temporal-channels", type=int, default=5,
        help="Number of input channels when --temporal-mode != off (anchor + N-1 sequence frames).",
    )
    parser.add_argument(
        "--label-source", type=str, default="folder", choices=["folder", "biopsy"],
        help="Ground-truth source: 'folder' = on-disk Healthy/Sick naming; "
             "'biopsy' = biopsy-confirmed labels linked from --labels-xlsx (clean "
             "benign×cancer task, drops ambiguous/inconsistent/pre-cancer). Default: folder.",
    )
    parser.add_argument(
        "--labels-xlsx", type=Path, default=None,
        help="Path to Labels_Skin_Cancer.xlsx (biopsy labels). "
             "Defaults to data/raw/Labels_Skin_Cancer.xlsx when --label-source biopsy.",
    )
    parser.add_argument(
        "--classical-level", type=str, default="sequence", choices=["sequence", "frame"],
        help="Granularity of the SVM/Random Forest models. 'sequence' (default) "
             "pools each sequence's frame embeddings into one row (matches the "
             "unit of analysis); 'frame' classifies frames independently then "
             "pools at prediction time (legacy). Neural models are unaffected.",
    )
    parser.add_argument(
        "--thermal-roi", type=str, default="full", choices=["full", "center", "hot"],
        help="Region the thermal-dynamics trajectory is measured over: 'full' = "
             "whole frame (original); 'center' = central half; 'hot' = Otsu "
             "hottest region (lesion proxy). Localising concentrates the "
             "reheating signal on the lesion. Default: full.",
    )
    parser.add_argument(
        "--thermal-features", type=str, default="basic", choices=["basic", "rich"],
        help="Thermal-feature set fed to SVM/RF: 'basic' = 4 trajectory summaries "
             "(original); 'rich' = adds reheating-curve shape + spatial-"
             "heterogeneity descriptors. Default: basic.",
    )
    parser.add_argument(
        "--seq-pool", type=str, default="mean", choices=["mean", "meanstdmax"],
        help="How a sequence's per-frame embeddings are pooled for the "
             "sequence-level classical models: 'mean' (original) or 'meanstdmax' "
             "(mean⊕std⊕max — a sequence-level distribution summary). Default: mean.",
    )
    parser.add_argument(
        "--no-ensemble", action="store_true",
        help="Disable the ResNet18+SVM+RF probability-averaging ensemble (on by default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _git_info() -> tuple[str, "bool | None"]:
    """Return (short commit hash, has_uncommitted_changes) for reproducibility.

    Best-effort — falls back to ("unknown", None) outside a git repo or if git
    isn't on PATH, since that shouldn't block a training run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", None


def _write_run_md(run_dir: Path, run_name: str, notes: "str | None") -> None:
    commit, dirty = _git_info()
    dirty_str = "sim" if dirty else ("não" if dirty is False else "desconhecido")
    lines = [
        f"# Run: {run_name}",
        "",
        f"- **Timestamp**: {datetime.now().isoformat(timespec='seconds')}",
        f"- **Git commit**: `{commit}` (mudanças não commitadas: {dirty_str})",
        "",
        "## O que mudou nesta run",
        "",
        notes or "_Nenhuma nota fornecida — passe `--notes` ou `--notes-file` ao rodar `scripts/train_cv.py`._",
        "",
        "## Configuração",
        "",
        "Ver `train_config.json` nesta mesma pasta para os hiperparâmetros exatos.",
        "",
    ]
    (run_dir / "RUN.md").write_text("\n".join(lines))


def _append_run_outcome(run_dir: Path, status: str, duration_str: str) -> None:
    with open(run_dir / "RUN.md", "a") as f:
        f.write(f"\n## Resultado\n\n- **Status**: {status}\n- **Duração**: {duration_str}\n")


def main() -> None:
    args = parse_args()

    notes = args.notes_file.read_text().strip() if args.notes_file is not None else args.notes
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = args.data_root if args.data_root is not None else (
        PROJECT_ROOT / "data/processed/_subset_validation" if args.quick_test
        else PROJECT_ROOT / "data/processed/TR_heatSkin"
    )
    results_dir = args.results_dir if args.results_dir is not None else (PROJECT_ROOT / "results" / "runs" / run_name)

    labels_xlsx = args.labels_xlsx
    if args.label_source == "biopsy" and labels_xlsx is None:
        labels_xlsx = PROJECT_ROOT / "data/raw/Labels_Skin_Cancer.xlsx"

    config = TrainConfig(
        data_root=data_root,
        results_dir=results_dir,
        frame_stride=args.frame_stride,
        k_folds=args.k_folds or (3 if args.quick_test else 5),
        num_epochs=3 if args.quick_test else 50,
        patience=2 if args.quick_test else 7,
        temporal_mode=args.temporal_mode,
        temporal_channels=args.temporal_channels,
        label_source=args.label_source,
        labels_xlsx=labels_xlsx,
        classical_level=args.classical_level,
        thermal_roi=args.thermal_roi,
        thermal_features=args.thermal_features,
        seq_pool=args.seq_pool,
        ensemble=not args.no_ensemble,
        seed=args.seed,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    _write_run_md(config.results_dir, run_name, notes)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(config.results_dir / "train_cv.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    set_seed(config.seed)
    device = get_device()
    logging.info(f"Run: {run_name}")
    logging.info(f"Dispositivo: {device}")
    logging.info(f"Config: {config.to_json_dict()}")

    start = datetime.now()
    try:
        run_cross_validation(config, device)
        status = "concluído com sucesso"
    except Exception:
        status = "falhou — ver train_cv.log"
        raise
    finally:
        _append_run_outcome(config.results_dir, status, str(datetime.now() - start))


if __name__ == "__main__":
    main()
