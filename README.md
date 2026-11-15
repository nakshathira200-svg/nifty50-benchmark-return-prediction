# NIFTY 50 Benchmark Return Prediction

A Machine Learning project focused on predicting next-session NIFTY 50 index returns and market direction using historical market data, technical indicators, baseline models, advanced machine learning algorithms, ensemble methods, and explainability techniques.

## Team Members

| Sl. No. | Name | SRN |
|---|---|---|
| 1 | Nakshathira B | PES2UG24AM096 |
| 2 | Manish S | PES2UG24AM085 |
| 3 | Shanmuga Vel | PES2UG24AM083 |

## Project Overview

The project investigates whether historical NIFTY 50 market data and engineered technical indicators can provide useful information for predicting future index movements.

The project addresses two prediction tasks:

- **Regression:** Predict the next observed trading session's close-to-close return.
- **Classification:** Predict whether the next observed trading session's return will be positive or non-positive.

The workflow includes data collection, preprocessing, exploratory data analysis, feature engineering, chronological dataset splitting, baseline evaluation, advanced modelling, ensemble learning, and model explainability.

## Project Objectives

- Collect historical NIFTY 50 index data.
- Clean and preprocess the collected dataset.
- Perform exploratory data analysis (EDA).
- Engineer technical indicators and relevant market features.
- Generate regression and classification targets.
- Create chronological training, validation, and testing datasets.
- Implement and evaluate baseline machine learning models.
- Develop and compare advanced machine learning models.
- Explore ensemble techniques to improve prediction performance.
- Apply explainability techniques to understand model behaviour.

## Dataset Information

| Attribute | Description |
|---|---|
| Data Source | Yahoo Finance |
| Instrument | NIFTY 50 Index |
| Ticker | `^NSEI` |
| Frequency | Daily |
| Downloaded Rows | 4,696 |
| Cleaned Rows | 4,696 |
| Model-Ready Rows | 4,646 |
| Prediction Horizon | Next observed trading session |

The project uses the NIFTY 50 index as a single time series rather than modelling its 50 constituent stocks individually.

## Methodology

The project is divided into three major stages.

### Part 1 – Data Collection and Preprocessing

**Contributor: Nakshathira B**

- Historical NIFTY 50 data collection using Yahoo Finance.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature engineering using returns, moving averages, momentum indicators, and volatility.
- Creation of regression and classification targets.
- Chronological train, validation, and test splitting.

### Part 2 – Baseline Modelling and Evaluation

**Contributor: Manish S**

- Implementation of regression and classification baselines.
- Development of a reusable modelling and evaluation framework.
- Performance evaluation using regression and classification metrics.
- Comparison of baseline models.
- Feature importance analysis and visualisation.
- Documentation of modelling methodology and execution status.

### Part 3 – Advanced Modelling and Explainability

**Contributor: Shanmuga Vel**

- Implementation and evaluation of Decision Tree, XGBoost, and TabNet models.
- Regression and classification experiments.
- Weighted ensemble modelling.
- Regression stacking experiments.
- Model comparison and performance analysis.
- Explainability using SHAP and TabNet attention mechanisms.

## Model Evaluation

The project uses the following evaluation metrics.

**Regression Metrics**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Directional Accuracy

**Classification Metrics**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Key Results

The reported Part 3 test results include:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Train Mean Baseline | 0.008627 | 0.006194 | -0.000274 |
| Decision Tree | 0.008641 | 0.006213 | -0.003564 |
| XGBoost | 0.008680 | 0.006215 | -0.012732 |
| TabNet | 0.008579 | 0.006221 | +0.010824 |
| Weighted Ensemble | 0.008605 | 0.006211 | +0.004696 |

TabNet achieved the lowest RMSE among the individual advanced regression models. The weighted ensemble achieved a lower MAE than its two component models.

The results indicate limited predictive signal in next-session NIFTY 50 returns. They do not establish a profitable trading strategy.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** TensorFlow/Keras, PyTorch
- **Advanced Modelling:** PyTorch TabNet
- **Explainability:** SHAP
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

## Repository Structure

```text
NIFTY50-Benchmark-Return-Prediction/
│
├── PART1/
│   ├── data/
│   ├── src/
│   ├── notebooks/
│   ├── tests/
│   ├── docs/
│   ├── deliverables/
│   └── README.md
│
├── PART2/
│   ├── data/
│   ├── src/
│   ├── scripts/
│   ├── outputs/
│   ├── docs/
│   ├── tests/
│   └── README.md
│
├── PART3/
│   ├── data/
│   ├── src/
│   ├── outputs/
│   ├── deliverables/
│   ├── docs/
│   └── README.md
│
├── NIFTY50_Project_Report_Final_Corrected.docx
├── NIFTY50_Benchmark_Return_Prediction_Presentation_Corrected.pptx
├── .gitignore
└── README.md
```

Each part contains its respective source code, documentation, datasets, experiments, and deliverables.

## Project Outcomes

The project provides an end-to-end machine learning workflow for NIFTY 50 next-session return prediction, covering data preparation, baseline evaluation, advanced modelling, ensemble experiments, and explainability.

The comparative evaluation highlights the challenges of predicting short-term financial market movements and the importance of evaluating advanced models against simple baselines.

## References

- Yahoo Finance – Historical NIFTY 50 Market Data: https://finance.yahoo.com/
- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- PyTorch Documentation: https://pytorch.org/
- PyTorch TabNet: https://github.com/dreamquark-ai/tabnet
- SHAP Documentation: https://shap.readthedocs.io/
