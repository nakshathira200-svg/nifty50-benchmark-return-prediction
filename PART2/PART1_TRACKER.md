# Part 1 — Deliverables Tracker

Use this as a working checklist. Mark an item complete only after checking the output.

## 1. Data collection
- [ ] Confirm index-only versus constituent-stock interpretation with the team.
- [ ] Confirm ticker(s), date range, and daily frequency.
- [ ] Download historical data from the agreed source.
- [ ] Save untouched download(s) in `data/raw/`.
- [ ] Record source, retrieval date, ticker(s), date range, and columns.

## 2. Data quality and cleaning
- [ ] Inspect shape, data types, date range, and duplicate rows.
- [ ] Check duplicate dates and missing values by column.
- [ ] Check impossible/inconsistent OHLC values and non-positive prices.
- [ ] Decide how to handle missing values and document the reason.
- [ ] Save cleaned intermediate data in `data/interim/`.

## 3. EDA
- [ ] Summary statistics and data-quality summary.
- [ ] Price/close history plot.
- [ ] Daily return distribution and time-series plot.
- [ ] Volume analysis, if volume is meaningful and available.
- [ ] Correlation analysis for candidate numeric features.
- [ ] Rolling volatility / moving-average plots.
- [ ] Save figures in `outputs/eda/` and selected figures in `deliverables/graphs/`.

## 4. Feature engineering
- [ ] Define return formula and prediction horizon.
- [ ] Add lagged returns (document lag windows).
- [ ] Add moving averages / price-to-moving-average features.
- [ ] Add rolling volatility features.
- [ ] Check every feature is available at prediction time (no look-ahead leakage).
- [ ] Document engineered columns in `docs/DATA_DICTIONARY.md`.

## 5. Targets
- [ ] Define regression target precisely.
- [ ] Define classification target and threshold precisely.
- [ ] Confirm target alignment with the team.
- [ ] Check class balance and target distribution.
- [ ] Remove rows with unavailable future targets.

## 6. Chronological split and handoff
- [ ] Agree train/validation/test date boundaries with the team.
- [ ] Split chronologically (no random shuffling).
- [ ] Verify dates do not overlap and are in ascending order.
- [ ] Check target distribution in each split.
- [ ] Save final files in `data/processed/` and copies in `deliverables/datasets/`.
- [ ] Share preprocessing code, EDA figures, data dictionary, and split files with Persons 2 and 3.
- [ ] Explain feature columns, target definitions, split boundaries, and any caveats to the modelling team.
