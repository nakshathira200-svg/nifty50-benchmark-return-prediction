# PROGRESS — Manish S (PES2UG24AM085) — modelling   (last updated 2026-11-08, completion pass)

## STATUS: NOT COMPLETE against the stated completion conditions — the 10 framework models have NOT been trained.
Reason (verified): the sandbox used to build this had **no network** (`pip`: "Name or service not known"), so xgboost, tensorflow/keras,
torch, fastai, pytorch-tabnet, pytorch-lightning, lightning-flash and pytest could not be installed. Everything that could run here did.

## Completed and VERIFIED by execution
- [x] Inspected Part 1 (data, EDA, features, splits, tests, brief PDF). Part 1 files unchanged.
- [x] Data prep with leakage guards, train-only scaler (`src/data/prepare.py`); test file not read during training.
- [x] Metrics (regression + classification incl. confusion matrix, CI) and tests of known values.
- [x] 7 baselines trained, saved, reloaded and evaluated -> **real results** in `outputs/results/` and `docs/RESULTS.md`.
- [x] Harness verified end-to-end for all 10 model slots using sklearn stand-ins (`--standin --tag smoke_standin`): train -> save -> load ->
      evaluate -> tables/figures. Stand-in output lives only in `*_smoke_standin` folders and is watermarked; it is NOT a result.
- [x] Comparison tables, 10+ figure types, feature-importance (native + permutation), auto-generated RESULTS.md.
- [x] `python run_models.py` and `python run_evaluation.py` run from a clean state; framework failures are logged, not hidden.
- [x] Tests: 20 passed, 12 skipped, 0 failed (`python scripts/run_tests_without_pytest.py`; use `pytest -q` where pytest is installed).
- [x] Docs: README (Part 2 section), METHODOLOGY_PART2, RESULTS (generated), FINAL_REPORT_TEMPLATE, VIVA_QA, DECISIONS (appended).

## Completion-pass changes (static review; frameworks still not installable here)
- FIXED fastai: `ps` was passed directly to `tabular_learner` (would raise TypeError in Learner); now `config=tabular_config(ps=...)`.
- FIXED XGBoost: prediction now uses `iteration_range=(0, best_iteration+1)` explicitly, so save/load cannot change predictions.
- FIXED docs generation: `docs/` is created if missing before writing RESULTS.md.
- Tests: Flash real test now requires the `flash` package and `allow_fallback=False` (a fallback can no longer pass as Flash);
  added Lightning-fallback tests and a regression test for the fastai fix. 12 framework tests SKIP because libraries are absent.
- `requirements*.txt` rewritten with compatible version ranges (unverified offline; numpy<2 for TabNet/TF).
- Baseline results re-generated and unchanged in value; no framework model has been trained or evaluated.

## Written but NOT executed (needs a machine with the frameworks)
- [ ] XGBoost reg/clf   (`src/models/xgboost_model.py`)
- [ ] Keras reg/clf     (`src/models/keras_model.py`)
- [ ] fastai reg/clf    (`src/models/fastai_model.py`)
- [ ] TabNet reg/clf    (`src/models/tabnet_model.py`)
- [ ] Lightning Flash reg/clf (`src/models/flash_model.py`; falls back to PyTorch-Lightning MLP, labelled, if Flash fails)
These were desk-checked against the framework APIs and are syntax-checked, but any first run may expose API/version issues.

## Errors recorded (real, from `outputs/results/run_status.csv`)
xgboost: ModuleNotFoundError xgboost | keras: ModuleNotFoundError tensorflow | fastai: ModuleNotFoundError fastai |
tabnet: ModuleNotFoundError torch | flash: ModuleNotFoundError flash AND torch (fallback also unavailable). Tracebacks in `logs/`.

## EXACT NEXT STEPS (on your machine)
1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `pytest -q`  -> the ten `test_*_real` tests must PASS (not skip). Fix any failure before step 3 (tracebacks point to the module).
3. `python run_models.py`   (if one framework fails, the rest still run; see the SUMMARY line and `outputs/results/run_status.csv`)
4. Lightning Flash: if step 3 shows `pytorch_lightning_fallback` or a failure, try a separate env per `requirements-flash.txt`, then
   `python run_models.py --models flash_reg flash_clf`.
5. `python run_evaluation.py`  -> regenerates all tables, figures and `docs/RESULTS.md`; check the `implementation` column.
6. Re-run with other seeds for the neural models (`python run_models.py --models keras_reg ... --seed 7`) before trusting small gaps.
7. Write the report in your own words from `docs/FINAL_REPORT_TEMPLATE.md`; the brief also asks for a PPT (not created here) and a GitHub repo.


## FIVE FRAMEWORK PAIRS STILL TO BE TRAINED AND EVALUATED (no results exist for any of them)
| # | Framework | Regression slot | Classification slot | Status |
|---|---|---|---|---|
| 1 | XGBoost | `xgboost_reg` | `xgboost_clf` | NOT trained (xgboost not installed) |
| 2 | TensorFlow / Keras | `keras_reg` | `keras_clf` | NOT trained (tensorflow not installed) |
| 3 | PyTorch + fastai | `fastai_reg` | `fastai_clf` | NOT trained (fastai not installed) |
| 4 | PyTorch TabNet | `tabnet_reg` | `tabnet_clf` | NOT trained (torch / pytorch-tabnet not installed) |
| 5 | PyTorch Lightning Flash | `flash_reg` | `flash_clf` | NOT trained (flash and torch not installed) |

All metrics in `outputs/results/`, `deliverables/model_results/` and `docs/RESULTS.md` come from the 7 baselines only.
Nothing was estimated or filled in for the pairs above. `*_smoke_standin` folders are pipeline tests with sklearn stand-ins, not results.


See `FIXES.md` for the latest fixes (MSE column added to model_comparison.csv; completeness guard; environment checker).
