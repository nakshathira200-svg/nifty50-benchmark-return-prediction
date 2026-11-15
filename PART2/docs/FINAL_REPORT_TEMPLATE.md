# Final Report Template — NIFTY 50 Benchmark Return Prediction

> The project brief says **"NO AI Generated Content"**. This file is a **skeleton**: headings, prompts, and pointers to where each
> number or figure lives. Write the prose yourself, in your own words. Confirm with your instructor what is allowed.

Team / names / roll numbers: ______   Date: ______   GitHub repository: ______

## 1. Introduction
- What is being predicted and why does it matter? (NIFTY 50 next-day return; regression and up/down classification)
- Why compare five frameworks? State your hypothesis before looking at results.
- Which instrument did the team agree on (index `^NSEI` vs constituent stocks)? Reference `docs/DECISIONS.md`.

## 2. Methodology
### 2.1 Data (Person 1) — source, date range, rows, cleaning (`docs/PART1_METHODOLOGY.md`, `deliverables/PART1_RUN_SUMMARY.json`)
### 2.2 Features and targets — table from `docs/DATA_DICTIONARY.md`; explain why 17 features are used (`docs/METHODOLOGY_PART2.md` §3)
### 2.3 Split and leakage prevention — dates from `deliverables/datasets/split_summary.csv`; purge; scaler on train only
### 2.4 Models — one short paragraph per framework (your own understanding): what it is, key hyper-parameters, how validation is used
### 2.5 Baselines and metrics — why baselines matter; define each metric
### 2.6 Comparison criteria — define simplicity, transparency, efficacy in your own words; state your ratings and the evidence

## 3. Exploratory Data Analysis
Insert (from `outputs/eda/`): closing price, return distribution, rolling volatility, drawdown, correlation, volume.
For each figure: one observation + one implication for modelling.

## 4. Results
| Item | Source file |
|---|---|
| Regression table (MAE, MSE, RMSE, R², directional accuracy) | `outputs/results/regression_test_metrics.csv` |
| Classification table (Acc, Prec, Rec, F1, AUC, confusion matrix) | `outputs/results/classification_test_metrics.csv` |
| Full comparison (time, params, simplicity, transparency, rank) | `outputs/results/model_comparison.csv` |
| Figures | `outputs/model_graphs/*.png` |
| Feature importance | `outputs/model_graphs/feature_importance_<model>.png` |
- Report each framework's result **and** the baselines. Say which differences exceed the ±3.7 pp noise band.
- Discuss training time vs performance. Discuss which features matter and whether this is plausible financially.
- Check `implementation` in the tables: only report a model as "Lightning Flash" if that column says `lightning_flash`.

## 5. Discussion of the three criteria
Simplicity / Transparency / Efficacy: one paragraph each, backed by a number or an observation from the code.

## 6. Conclusion / Summary
- Answer the hypothesis from the Introduction honestly, including a "no model beat the baseline" outcome if that is what you found.
- Limitations (single split, single seed, daily data only, no transaction costs) and what you would do next.

## References / Appendix
Library versions (`pip freeze`), how to reproduce (`README.md`), seeds, hardware, any model that failed and why (`outputs/results/run_status.csv`).
