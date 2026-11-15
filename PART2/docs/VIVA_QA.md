# Viva Preparation — Questions and Study Notes

> **Study aid only.** The brief says "NO AI Generated Content": do not paste these answers into the report or slides, and be able
> to explain every point in your own words. Where an answer depends on *your* run, it says so. Numbers marked (†) are from
> this repo's real outputs; check `docs/RESULTS.md` for the current values.

## A. Project and data
**1. What are you predicting?** The NIFTY 50 index (`^NSEI`) next-session close-to-close return (regression) and whether it is
positive (classification, 1 = up, 0 = zero/negative).

**2. Why chronological splitting instead of a random split?** Random splits let the model train on days after the ones it is
tested on and mix neighbouring, autocorrelated days across sets, which inflates scores. Train < validation < test in time mimics
real deployment. Sizes†: 3,251 / 695 / 698 rows.

**3. What is "purging" at the split boundary?** The target for the last row of a partition is the *next* day's return; if that
day is in the next partition, information crosses the boundary. Part 1 drops those rows (two in total).

**4. How did you prevent data leakage?** Targets/IDs excluded from features; every feature uses only information at or before
the prediction day; scaler fitted on train only; validation used for tuning/early stopping; test read once at evaluation;
automated checks in `assert_no_leakage` and `tests/test_modelling.py`.

**5. Why did you drop `open/high/low/close` as features?** They are price *levels*, which trend. 93%† of validation rows have a
close above the training maximum, so these inputs are out of range for the test period: trees cannot extrapolate and networks
are unreliable there. Our own ablation on validation was inconclusive, so the justification is the argument, not a measured
gain. Option `--feature-set all` restores them.

**6. Why was volume excluded?** Part 1 found reported index volume inconsistent (often zero).

## B. Frameworks
**7. How does XGBoost work, and why is it comparatively transparent?** Sequentially adds shallow trees, each fitting the
residual errors (gradient boosting) with regularisation. Gain-based feature importance and inspectable trees; needs no scaling.

**8. What does early stopping do and what data does it use?** It stops adding trees/epochs when validation loss stops improving
and keeps the best model. It uses the chronological validation set, never test.

**9. What is TabNet and where does its interpretability come from?** An attention-based network for tabular data: at each of
several decision steps it selects a sparse subset of features (sparsemax masks). Aggregated masks give feature importance.

**10. What does fastai add over plain PyTorch?** A high-level `Learner`, tabular data handling, training loop and callbacks
(early stopping, best-model saving), so a model is a few lines; the cost is that much behaviour is hidden.

**11. What is Lightning Flash and why might it fail to install?** A task-level library on PyTorch Lightning. It has been in
maintenance mode and has historically pinned older Python/PyTorch/Lightning versions, so it may conflict with the other
frameworks. This repo falls back to a plain PyTorch-Lightning MLP and labels it `pytorch_lightning_fallback`, never as Flash.
**Be ready to say honestly which implementation your `run_status.csv` shows.**

**12. Why is the regression target standardised for neural networks?** Daily returns are ~0.01 in size; an MSE loss on such
tiny numbers lets networks collapse to predicting the mean. Standardising with *train* mean/std fixes scale; predictions are
converted back.

**13. Why scale features, and why only on train?** Gradient-based models train poorly with differently scaled inputs. Fitting
the scaler on validation/test would leak their statistics into training.

## C. Metrics and results
**14. Define MAE, MSE, RMSE, R².** Mean absolute / squared error; RMSE is √MSE in return units; R² = 1 − SSE/SST, where 0
means no better than predicting the mean and negative means worse.

**15. Why can R² be negative, and is that a bug?** No: on out-of-sample data a model can do worse than the mean. Persistence
(predict yesterday's return) scores far below zero† because it doubles the variance of the error.

**16. What is directional accuracy and why 50% matters?** Share of days where the predicted sign equals the realised sign.
50% is chance; with 698 test days the 95% noise band is about ±3.7 pp, so 52–53% is not clearly better than chance.

**17. Why does F1 look good for "always predict up"?** Recall is 1.0 and about half the days are up, so F1 ≈ 0.68† despite zero
skill (ROC-AUC = 0.5). Use ROC-AUC and compare with the majority baseline.

**18. What do precision, recall, ROC-AUC mean here?** Precision: of days predicted up, share that were up. Recall: of up days,
share predicted up. AUC: probability a random up day is scored higher than a random down day (0.5 = chance).

**19. Did the complex models beat the baselines?** *Answer from your own `docs/RESULTS.md`.* Report differences relative to the
noise band. A "no, not meaningfully" is a valid and expected result for next-day index returns.

**20. Why is next-day return so hard to predict?** Efficient-market arguments: public price history is quickly priced in, and
daily returns have a very low signal-to-noise ratio.

## D. Comparison and critique
**21. How did you compare simplicity, transparency, efficacy?** Simplicity: lines of code in each model module + a rating;
transparency: can we inspect/interpret the model (native importance, trees, attention masks) — a rating; efficacy: measured
test metrics and rank. State that ratings are judgements and defend them.

**22. How do you know the feature importance is meaningful?** Native importance depends on the model; permutation importance
(on validation) is model-agnostic. With near-zero skill, importances are unstable and should not be over-interpreted.

**23. Limitations of your study?** Single split and (by default) single seed; small test set; no walk-forward validation; limited
hyper-parameter search; no transaction costs; daily data only.

**24. What would you do next?** Multiple seeds and walk-forward evaluation; confidence intervals via bootstrap; additional
features (macro, other indices, volatility indices); economic evaluation of a trading rule with costs.

**25. If a framework failed to run, what do you report?** The exact error from `outputs/results/run_status.csv`, what you
tried, and the closest valid substitute, clearly labelled. Never present a substitute or an estimate as the original framework.
