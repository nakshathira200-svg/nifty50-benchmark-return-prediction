# NIFTY 50 Benchmark Return Prediction — Part 1

This folder contains the completed **Data & Preprocessing** work led by **Nakshathira B (PES2UG24AM096)**.

## Team contribution — Nakshathira B (PES2UG24AM096)\n\n## Your scope

1. Collect the historical NIFTY 50 dataset.
2. Inspect and clean missing and duplicate values.
3. Perform exploratory data analysis (EDA) and create visualizations.
4. Engineer features such as returns, moving averages, and volatility.
5. Define regression and classification targets.
6. Split the data chronologically into training, validation, and test sets.

## Folder map

```text
mlbd-part 1/
├── references/             # Original project brief and source references
├── data/
│   ├── raw/                 # Untouched downloaded source data
│   ├── interim/             # Intermediate cleaned/transformed data
│   └── processed/           # Final model-ready datasets
├── notebooks/               # EDA and experimentation notebooks
├── src/data/                # Reusable preprocessing scripts
├── outputs/eda/              # EDA charts and summary tables
├── deliverables/
│   ├── datasets/             # Final train/validation/test files
│   ├── graphs/               # Figures to share with the team
│   └── preprocessing/        # Code and preprocessing notes to share
├── docs/                     # Data dictionary, decisions, methodology notes
└── logs/                     # Download/preprocessing run notes
```

## Confirmed project setup

- **Instrument:** Yahoo Finance NIFTY 50 index ticker `^NSEI` (one index time series, not 50 constituent-stock series).
- **Frequency:** Daily trading observations.
- **Forecast horizon:** Next observed trading session.
- **Regression target:** Next-session close-to-close return.
- **Classification target:** `1` when the next-session return is positive; otherwise `0`.
- **Splitting:** Chronological train/validation/test partitions, with boundary purging to prevent next-session targets crossing partitions.

## Data handling rules

- Keep downloaded source files unchanged in `data/raw/`.
- Record the data source, retrieval date, date range, ticker(s), and any download limitations.
- Sort observations by trading date before creating lagged features or targets.
- Calculate predictive features using information available at or before the prediction time.
- Shift future targets backward only to label the current observation; remove rows whose future target is unavailable.
- Fit any learned preprocessing (for example, imputation/scaling, if used) on training data only, then apply it to validation and test data.
- Keep training, validation, and test periods in chronological order and prevent overlap.

## Suggested file naming

- `data/raw/nifty50_<ticker>_<start>_<end>_daily.csv`
- `data/interim/nifty50_cleaned.csv`
- `data/processed/nifty50_features_targets.csv`
- `data/processed/train.csv`, `validation.csv`, `test.csv`
- `outputs/eda/01_price_history.png`, `02_returns.png`, etc.

## Academic project note

The supplied project brief says **“NO AI Generated Content”** and separately allows online help for coding. Use this workspace to organize your own analysis and code. Check your instructor's rules before using generated prose, figures, or other generated material in submitted work. Keep explanations and report writing in your own words.
