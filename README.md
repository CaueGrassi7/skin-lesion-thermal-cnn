# Skin Lesion Classification in Dynamic Thermal Infrared Images

**Undergraduate Thesis (TCC) — Bachelor of Computer Science, Federal University of Uberlândia (UFU), 2026**

> 📄 **Full thesis (defended):** [repositorio.ufu.br/handle/123456789/49570](https://repositorio.ufu.br/handle/123456789/49570)

## Overview

This work investigates whether **Convolutional Neural Networks (CNNs)** and **classical machine learning models (SVM, Random Forest)** can distinguish **benign from malignant/suspect skin lesions** using **Dynamic Thermal Infrared Imaging (DTI)** — a non-invasive imaging modality that captures a lesion's thermal reheating trajectory after a controlled cooling stimulus, rather than a single static photograph.

The central hypothesis is that the diagnostic signal is not in a lesion's static appearance (a single frame yields near-chance discrimination, AUC ≈ 0.45–0.49) but in its **thermal dynamics over time** — how quickly and unevenly a lesion re-warms relative to surrounding tissue.

The study was conducted on **TR_heatSkin**, a private clinical dataset (108,548 processed frames, ~168 usable thermal sequences from ~93 patients), with ground-truth labels derived from **biopsy-confirmed histopathology** rather than visual/folder heuristics.

## Key Contributions

- **End-to-end reproducible pipeline** for DTI preprocessing, patient-level cross-validation, model training, and rigorous statistical evaluation.
- **Biopsy-linked labeling system** that reconciles raw clinical folders with histopathology reports, explicitly excluding ambiguous cases rather than guessing a diagnosis.
- **Patient-grouped k-fold cross-validation** with per-fold class reweighting, preventing data leakage across train/validation/test splits.
- **Sequence-level classical ML** (SVM / Random Forest on pooled CNN embeddings + thermal-dynamics features) evaluated against transfer-learning (ResNet18) and from-scratch (custom `ThermalCNN`) architectures.
- **Rigorous evaluation methodology**: pooled AUC with stratified bootstrap 95% confidence intervals — appropriate for a modest sample size, rather than reporting an optimistic mean of per-fold scores.
- **Systematic ablations** on temporal input encoding, ROI-restricted thermal features, richer feature sets, embedding pooling strategies, and model ensembling — each documented with results, including negative ones.

## Key Findings

- **Best result:** pooled AUC ≈ **0.66** (95% CI up to ~0.74), achieved consistently by Random Forest, SVM, and fine-tuned ResNet18 — statistically indistinguishable from one another.
- **Sample size, not architecture, is the binding constraint.** Every tested enrichment (channel-stacked temporal input, ROI-restricted thermal features, richer feature sets, alternative embedding pooling, probability-averaging ensembles) landed within the baseline's confidence interval — none produced a statistically meaningful improvement.
- **Naive temporal channel-stacking hurts, not helps.** Feeding the CNN multiple frames as stacked input channels caused it to treat them as an unordered bag rather than a sequence, adding overfitting surface without capturing genuine dynamics. Disabling it improved patient-level AUC across every strong model (ResNet18 +0.04, Random Forest +0.04, SVM +0.07).
- **A CNN trained from scratch performs at chance level** (AUC ≈ 0.48–0.54) — with only ~134 training sequences, transfer learning (pre-trained ResNet18) is essential; this negative result is reported transparently rather than omitted.
- **Biopsy-confirmed labels cost little accuracy** relative to the original folder-naming heuristic (≈0.01–0.02 AUC), confirming the dataset's original heuristic labels were largely reliable, while providing a methodologically sounder ground truth.

## Methodology at a Glance

```
Raw DTI frames (108,548, 224×224 grayscale)
    → Preprocessing (denoising, normalization — no histogram equalization,
      since thermal gradient magnitude is diagnostically meaningful)
    → Biopsy-confirmed label linkage (ambiguous cases excluded, never guessed)
    → Patient-level stratified 5-fold cross-validation (seed=42)
    → Per-fold class reweighting
    → CNN embeddings (custom ThermalCNN / ResNet18, 1-channel adapted)
        ├─ Sequence-level SVM / Random Forest (patient-grouped tuning)
        └─ End-to-end CNN fine-tuning
    → Pooled AUC + stratified bootstrap 95% confidence intervals
```

## Repository Structure

```
skin-lesion-thermal-cnn/
├── src/
│   ├── preprocessing.py     # Load → denoise → normalize → resize pipeline
│   ├── labels.py            # Biopsy-confirmed label linkage
│   ├── dataset.py           # PyTorch Dataset, patient-level k-fold split, thermal-dynamics features
│   ├── model_cnn.py         # ThermalCNN + transfer-learning models
│   ├── model_classical.py   # CNN-embedding extraction + SVM / Random Forest
│   ├── evaluate.py          # Metrics, ROC, pooled AUC + bootstrap CI
│   └── pipeline.py          # Orchestrates the full cross-validation experiment
├── scripts/
│   └── train_cv.py          # CLI entry point for the full experiment
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_evaluation.ipynb  # Loads and analyzes results from a completed run
├── results/                 # Metrics, figures, and per-run experiment artifacts
├── requirements.txt
└── LICENSE
```

## Tech Stack

Python 3.11 · PyTorch (CUDA / Apple Silicon MPS) · torchvision (ResNet18 transfer learning) · scikit-learn (SVM, Random Forest) · pandas / NumPy · Jupyter

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Experiment

```bash
# Full patient-level k-fold cross-validation
python scripts/train_cv.py --notes "Description of what this run tests."

# Fast smoke test on a small validation subset
python scripts/train_cv.py --quick-test
```

Each run is written to its own `results/runs/<run-name>/` folder — hyperparameters, per-epoch logs, pooled predictions, and checkpoints — so results remain independently inspectable and comparable across experiments. `notebooks/03_evaluation.ipynb` loads a completed run and produces the final figures and tables (confusion matrices, ROC curves, pooled AUC with confidence intervals).

## Note on Data

The `TR_heatSkin` dataset is private and provided by the thesis advisor; it is not included in this repository. The codebase is published to document the methodology and enable reproducibility of the analysis pipeline on equivalent data.

---

**Author:** Cauê Grassi — B.Sc. Computer Science, Federal University of Uberlândia (UFU)
**Advisor:** [see full thesis for advisor and committee details]
