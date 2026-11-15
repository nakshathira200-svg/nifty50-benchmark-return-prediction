# Part 2 — Modelling, Evaluation and Comparison (Manish S — PES2UG24AM085)

Five frameworks × {regression, classification}: **XGBoost, TensorFlow/Keras, PyTorch + fastai, PyTorch + TabNet, PyTorch + Lightning Flash**,
plus 7 baselines. Part 1's datasets are reused unchanged. **Read `PROGRESS.md` first: it states exactly what has and has not been executed.**

> **Status:** 7 baselines trained + evaluated (real). The 10 framework models (XGBoost, Keras, fastai, TabNet, Flash x reg/clf) are implemented but **NOT yet trained**: the frameworks were not installable where this was built. Install `requirements.txt`, run `pytest -q` (12 framework tests must pass, not skip), then `python run_models.py && python run_evaluation.py`.

## Team contribution — Manish S (PES2UG24AM085)\n\n## Quick start
```bash
python -m venv .venv && source .venv/bin/activate       # Python 3.10 or 3.11 recommended
pip install -r requirements.txt                          # XGBoost, TensorFlow, PyTorch, fastai, TabNet, Lightning
# Lightning Flash often conflicts with the above; use a separate environment: see requirements-flash.txt
pytest -q                                                # the *_real tests skip any framework that is not installed
python run_models.py                                     # trains baselines + all 10 models; saves to models/ (train+val data only)
python run_evaluation.py                                 # loads saved models, evaluates on test, writes tables/figures/docs/RESULTS.md
```
Run one model on its own: `python run_models.py --models xgboost_reg keras_clf` or
`python -m src.models.tabnet_model --task regression`. Baselines only: `python run_models.py --models baselines`.
Failures never abort the run; the real error is recorded in `outputs/results/run_status.csv` and `logs/`.

## Layout added by Part 2
```text
src/config.py                 paths, seeds, targets, feature policy
src/data/prepare.py           loads Part 1 splits, leakage guards, train-only scaler
src/models/                   base.py, baselines.py, xgboost_model.py, keras_model.py, fastai_model.py,
                              tabnet_model.py, flash_model.py (+_lightning_utils.py), registry.py, standins.py
src/training.py               training orchestration, status logging, per-model CLI
src/evaluation/               metrics.py, evaluate.py, plots.py, report.py
run_models.py / run_evaluation.py
models/                       saved models + preprocessing/ (scaler, feature list)
outputs/results/              metrics + comparison CSVs, run_status.csv, feature importances
outputs/model_graphs/         comparison, ROC, confusion-matrix, importance figures
outputs/*_smoke_standin/      HARNESS TEST ONLY (sklearn stand-ins; never real results)
deliverables/model_results/   copy of the final tables;  deliverables/graphs/part2/  copy of the final figures
docs/METHODOLOGY_PART2.md  docs/RESULTS.md (auto-generated)  docs/FINAL_REPORT_TEMPLATE.md  docs/VIVA_QA.md
scripts/                      feature_set_ablation.py, run_tests_without_pytest.py
```

## Key design decisions
* Default features = 17 stationary predictors (raw price levels dropped; see `docs/METHODOLOGY_PART2.md` §3). `--feature-set all` restores 21.
* Test data is never read by `run_models.py`. Validation drives early stopping and tuning. Scaler is fit on train only.
* If Lightning Flash cannot run, the code falls back to a plain PyTorch-Lightning MLP and labels it `pytorch_lightning_fallback`.
* `--standin` swaps in sklearn stand-ins purely to smoke-test the pipeline; it refuses to write to the `main` results.

## Academic-integrity note
The brief says "NO AI Generated Content" (online help for coding is allowed). `docs/FINAL_REPORT_TEMPLATE.md` is a skeleton to fill in
your own words; `docs/VIVA_QA.md` is a study aid. Check with your instructor before reusing any generated prose.
