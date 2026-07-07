# Run: legacy_run2_single_split_improved

**Legado** — run manual anterior à arquitetura atual (`scripts/train_cv.py` +
`results/runs/<run-name>/` + `RUN.md` gerado automaticamente). Movida para cá
apenas para manter o histórico de execuções num só lugar; **não é compatível**
com `notebooks/03_evaluation.ipynb`, que espera `train_config.json`,
`metrics_cv_folds*.csv`, `predictions_*.npz` e `history_*_fold0.json` — esta
pasta não tem nenhum desses (ela usa um único split treino/val/teste, não
k-fold, e `metrics.csv` já no formato final).

Para inspecionar: `metrics.csv` e os `.png` (matrizes de confusão, ROC, curvas
de treino) nesta pasta podem ser abertos diretamente, sem o notebook.
