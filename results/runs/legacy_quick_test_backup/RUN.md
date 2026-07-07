# Run: legacy_quick_test_backup

**Legado** — backup manual de um smoke test anterior à arquitetura atual
(`scripts/train_cv.py --quick-test` + `results/runs/<run-name>/` + `RUN.md`
gerado automaticamente). Movida para cá apenas para manter o histórico de
execuções num só lugar; **não é compatível** com `notebooks/03_evaluation.ipynb`,
que espera `train_config.json`, `metrics_cv_folds*.csv`, `predictions_*.npz` e
`history_*_fold0.json` — esta pasta só tem `metrics_quicktest.csv` e os
checkpoints.
