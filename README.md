# NIFTY 50 Benchmark Return Prediction

A three-member machine-learning project for next-observed-session NIFTY 50 index return forecasting and market-direction classification.

## Team

| Member | Name | SRN |
|---|---|---|
| 1 | Nakshathira B | PES2UG24AM096 |
| 2 | Manish S | PES2UG24AM085 |
| 3 | Shanmuga Vel | PES2UG24AM083 |

## Project structure

- `PART1/` — data collection, cleaning, EDA, feature engineering and chronological data splits (Nakshathira B).
- `PART2/` — baseline modelling, evaluation framework and baseline results (Manish S).
- `PART3/` — advanced models, ensembles and explainability (Shanmuga Vel).

The three folders are kept separate because they contain files with overlapping names and distinct execution stages. Do not merge their contents into one flat directory.

## Prediction setup

- Instrument: Yahoo Finance `^NSEI`, a single NIFTY 50 index time series.
- Horizon: next observed trading session, not next calendar day.
- Regression: next-session close-to-close return.
- Classification: positive return (`1`) versus zero or negative return (`0`).
- Evaluation: chronological train/validation/test partitions; the held-out test set has 698 observations.

## Important interpretation of results

Part 2 reports seven genuine baseline evaluations. Its ten planned framework model slots were not successfully trained in that execution run; smoke-test stand-ins are not experimental results. Part 3 is a later execution stage with trained Decision Tree, XGBoost and TabNet models, weighted ensembles, regression stacking and explainability outputs. Keep these stages distinct when presenting the work.

The reported predictive gains are small and classification performance is close to chance. The results do not establish a profitable trading strategy.

## How to navigate

Read each subfolder's `README.md` first. Each part has its own dependencies, commands, methodology and deliverables. Use the corresponding handoff documentation to reproduce that stage.

## Academic integrity

The supplied project brief states “NO AI Generated Content” while allowing online help for coding. Check the instructor's rules before submitting AI-assisted wording or other generated material. Ensure the team reviews and can explain all submitted work.
