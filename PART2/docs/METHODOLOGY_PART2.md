# Part 2 Methodology — Modelling and Evaluation

## 1. Inputs inherited from Part 1 (reused unchanged)
`deliverables/datasets/{train,validation,test}.csv` (3,251 / 695 / 698 rows, chronological 70/15/15, boundary rows purged so that
no next-day target crosses into the following partition). Part 1 files were not modified.

## 2. Targets
| Task | Column | Definition |
|---|---|---|
| Regression | `target_next_day_return` | next observed session close / today's close − 1 |
| Classification | `target_next_day_up` | 1 if that return > 0, else 0 (train share of up days: 52.8%) |

## 3. Features
* Excluded from every model: both targets, `target_date`, `price_date`, `adj_close`, `volume`, `dividends`, `stock_splits`.
* **Default set = 17 stationary features** (returns, SMA/EMA ratios, MACD ratio, volatilities, intraday range, RSI, calendar).
* The 4 raw price levels (`open/high/low/close`) from Part 1's 21 predictors are dropped by default: the training `close` maximum
  is 15,315, yet 93% of validation rows are above it. Trees cannot extrapolate beyond the training range and neural nets would
  receive inputs far outside what they saw. `--feature-set all` reproduces the 21-feature setting.
* **This choice is principled, not proven empirically.** `scripts/feature_set_ablation.py` (train → validation only) gives
  small, mixed differences (see `docs/RESULTS.md`).
* `day_of_week` is treated as a numeric feature in all frameworks so all five see identical inputs.

## 4. Leakage controls (all enforced in code and tested in `tests/test_modelling.py`)
1. `assert_no_leakage` runs on every data load: no forbidden/target columns as features, splits chronologically ordered with no
   overlap, no target date inside the next partition, no feature with |corr| ≥ 0.5 to a target on train.
2. `StandardScaler` is fitted on **train only** and merely applied to validation/test; it is saved with the models.
3. `run_models.py` never reads `test.csv` (`build_bundle(include_test=False)`).
4. Validation data is used for early stopping and hyper-parameter selection; test is used once, in `run_evaluation.py`.
5. Regression targets are standardised with **train** mean/std for neural nets (returns are ~1e-2) and un-standardised on prediction.
6. Permutation importance is computed on validation, not test.

## 5. Models (5 frameworks × {regression, classification})
| Framework | Architecture / settings | Tuning & stopping on validation |
|---|---|---|
| XGBoost | `XGBRegressor/XGBClassifier`, hist trees, subsample 0.8, colsample 0.8, λ=5, min_child_weight 5 | grid max_depth {2,3,4} × lr {0.02,0.05}; early stopping 50 rounds; best by val RMSE / logloss |
| Keras | Dense 64-32 (ReLU), dropout 0.2, L2 1e-4, Adam 1e-3, batch 64; MSE or sigmoid+BCE | EarlyStopping(val_loss, patience 20, restore best) |
| fastai | `tabular_learner`, layers [64,32], dropout 0.2, wd 1e-2, lr 1e-3 | EarlyStoppingCallback + SaveModelCallback on `valid_loss`; explicit chronological split |
| TabNet | `pytorch-tabnet`, n_d=n_a=8, 3 steps, γ=1.3, λ_sparse 1e-3, Adam 2e-2 | `patience=20` on val RMSE / logloss |
| Lightning Flash | `TabularRegressor/TabularClassifier`, backbone `category_embedding` | custom best-weights early stopping on val loss; **falls back to a plain PyTorch-Lightning MLP if Flash fails; labelled `pytorch_lightning_fallback`** |

Baselines: regression — train-mean, zero, persistence (yesterday's return), Ridge (alpha on val); classification — majority class
(always up), persistence (yesterday's direction), Logistic regression (C on val). Seed 42 everywhere (`src/config.py`).

## 6. Metrics
Regression: MAE, MSE, RMSE, R², directional accuracy (predicted sign vs. realised sign, "up" = strictly > 0). Classification:
accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a 95% binomial half-width on accuracy.

## 7. Comparison criteria
* **Efficacy / prediction performance** — measured on test (rank by RMSE or ROC-AUC; baselines included).
* **Training time** — measured wall-clock seconds (CPU), logged by `run_models.py`.
* **Simplicity** — objective proxy: non-blank, non-comment lines in `src/models/<framework>_model.py`; plus an analyst rating
  1–5 (setup effort, number of hyper-parameters, abstraction hiding complexity).
* **Transparency** — analyst rating 1–5 (can you inspect the learned model? native feature importance? interpretable parts?).
  These ratings are judgement, not measurement; justify them in your own words in the report.

## 8. Limitations — read before quoting results
* **Framework code has been desk-checked and unit-tested only where the framework was installable.** The sandbox used to write
  this code had no network, so XGBoost, Keras, fastai, TabNet and Flash were never executed there. Run `pytest -q`: the 10
  `*_real` tests prove each framework trains + saves + loads on your machine; skipped tests mean the framework is missing.
* One chronological split, one seed per model: neural-net results vary run to run. Re-run with several `--seed` values before
  drawing conclusions from small gaps. Walk-forward validation is not implemented.
* With 698 test days, accuracy is known only to about ±3.7 pp; many differences between models will be noise.
* Next-day index returns are close to unpredictable from price-derived features; near-baseline results are expected.
* The stand-in run (`--standin --tag smoke_standin`) only verifies the harness and is never a result.


## Completion-pass addendum (2026-11-08)
- **Leakage controls (verified by tests):** chronological Part 1 split (train 2007-11 to 2021-03, validation to 2024-01, test 2024-01 to 2026-11); boundary rows whose
  next-day target falls in the next partition are purged; scaler fit on train only; test file not read by `run_models.py`; targets and identifiers are blocked as inputs.
- **Metrics:** regression - MAE, MSE, RMSE, R², directional accuracy; classification - accuracy (+95% binomial CI), precision, recall, F1, ROC-AUC, confusion matrix.
  Baselines (train-mean, zero, persistence, ridge, majority-class, persistence, logistic) anchor every comparison.
- **Known selection effect:** XGBoost depth/learning rate and all early stopping use the validation set, so validation scores are slightly optimistic; test is untouched.
- **Bugs fixed in this pass:** fastai `ps` argument (now via `tabular_config`), XGBoost best-iteration prediction after save/load.
- **Not executed:** XGBoost, Keras, fastai, TabNet, Lightning Flash (libraries unavailable, no network). They are code-complete and statically reviewed only;
  no metric for them exists. Their rows in all tables read `failed` with the real ModuleNotFoundError.
