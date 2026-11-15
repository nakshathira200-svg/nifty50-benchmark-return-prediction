# Project Decisions Log

Record decisions so the data pipeline and model comparisons use the same definitions.

| Date | Decision / question | Agreed choice | Reason / owner |
|---|---|---|---|
| | Index series or 50 constituent stocks? | Pending | Project brief title/objective says NIFTY 50; sample data shows HDFC Bank stock rows. |
| | Prediction horizon | Pending | |
| | Regression target formula | Pending | |
| | Classification threshold / labels | Pending | |
| | Historical date range | Pending | |
| | Train / validation / test boundaries | Pending | |

## Part 2 decisions (appended; Part 1 table above is unchanged)

| Decision | Choice | Reason |
|---|---|---|
| Model feature set | 17 stationary features (raw open/high/low/close excluded); `--feature-set all` restores Part 1's 21 | 93% of validation `close` values exceed the training maximum; ablation on validation was inconclusive, so the reason is non-stationarity |
| Hyper-parameter tuning | Small XGBoost grid on validation; fixed architectures + validation early stopping for NNs | keeps runs fast and test untouched |
| Target scaling | Standardise regression target with train mean/std for NNs | returns are ~1e-2 |
| Lightning Flash failure policy | Fall back to plain PyTorch-Lightning MLP, label `pytorch_lightning_fallback` | Flash is in maintenance mode with strict version pins |
| Classification threshold | 0.5 on predicted probability | matches the label definition; ROC-AUC reported as threshold-free check |
| Dependency pins | numpy<2, torch<2.6, fastai 2.7.x, TF<2.18, TabNet 4.1 | mutual compatibility; unverified offline |
