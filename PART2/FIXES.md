# FIXES (2026-11-08)

## Root causes of missing metrics in `outputs/results/model_comparison.csv`
1. **MSE column was never written.** `src/evaluation/evaluate.py::build_comparison` computed MSE (`regression_test_metrics.csv` had it) but
   dropped it when building the comparison table. **Fixed:** `test_MSE` is now carried through; baseline values are unchanged (MSE = RMSE²).
2. **Framework regression rows (xgboost, keras, fastai, tabnet, flash) are empty because those models have not run.** The recorded errors in
   `outputs/results/run_status.csv` are `ModuleNotFoundError` (xgboost, tensorflow, fastai, torch, flash not installed here; nothing was installed).
   This is an environment limit, not a metrics bug. No values were filled in.

## Code fixes
- fastai: `ps` was passed directly to `tabular_learner` (TypeError); now `config=tabular_config(ps=...)`.
- XGBoost: prediction pinned to `iteration_range=(0, best_iteration+1)` so save/load cannot change outputs.
- Report generation creates `docs/` if missing.
- Added a completeness guard in `build_comparison`: an evaluated regression model must have RMSE, MSE, MAE, R², Directional Accuracy, and any
  model without metrics must carry a recorded error (no silent skips). Violations raise `AssertionError`.
- Added `scripts/check_environment.py` (reports importable frameworks, installs nothing).
- Tests: new `test_comparison_has_all_five_regression_metrics`; Flash tests cannot pass via the fallback. Result: **21 passed, 12 skipped, 0 failed**
  (the 12 skips are framework tests whose libraries are absent).

## Requirements
`requirements.txt` (plus `requirements-base.txt`, `requirements-flash.txt`): numpy<2, pandas<2.3, scikit-learn<1.6, xgboost 2.x, tensorflow 2.15-2.17,
torch 2.1-2.5, fastai 2.7.x, pytorch-tabnet 4.1, pytorch-lightning 2.x; Flash in its own env. **Versions are unverified (not installed here).**

## Verified
`python run_evaluation.py` regenerates all CSVs/figures; the 4 baseline regression rows carry all five metrics; classification metrics, baselines and
efficacy ranks are unchanged. Dataset, splits and Person 1 files untouched.

## Still failing / not executed (honest status)
`xgboost_reg/clf`, `keras_reg/clf`, `fastai_reg/clf`, `tabnet_reg/clf`, `flash_reg/clf`: not trained, no metrics. After installing the requirements run
`python scripts/check_environment.py`, `pytest -q`, `python run_models.py`, `python run_evaluation.py`. Further version/API errors may appear on first run;
they will be recorded in `run_status.csv`.
