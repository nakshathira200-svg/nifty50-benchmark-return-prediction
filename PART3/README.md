# NIFTY 50 Benchmark Return Prediction — Part 3
## Deep Learning (TabNet), Ensembling & Explainability (SHAP)

This workspace contains the complete implementation, trained models, evaluation benchmarks, and deliverable artifacts for **Shanmuga Vel (PES2UG24AM083)**.

---

## Team contribution — Shanmuga Vel (PES2UG24AM083)\n\n## 1. Scope & Responsibilities

| Responsibility | Status | Implementation Details & Deliverables |
|---|---|---|
| **1. Implement PyTorch TabNet** | Complete | Canonical deep tabular architecture (`src/models/tabnet_model.py`) with 3 decision steps, Sparsemax attention, and GLU feature transformations. |
| **2. Train Regression & Classification Models** | Complete | Trained on Train split (3,251 days) with early stopping on Validation split (695 days). Models saved to `models/` and `deliverables/trained_models/`. |
| **3. Compare TabNet against XGBoost** | Complete | Head-to-head benchmarking against tuned XGBoost and Decision Tree baselines on test split (698 days). Reports RMSE, MAE, R², Directional Accuracy, ROC-AUC, F1, CI95. |
| **4. Implement Ensemble Learning** | Complete | Multi-paradigm convex weighted blending ($w^*=0.62$ TabNet + $0.38$ XGBoost) and Stacking (Ridge meta-learner) with error decorrelation analysis ($\rho = 0.584$). |
| **5. Perform SHAP Analysis** | Complete | Dual explainability via TreeSHAP on XGBoost and KernelSHAP on TabNet. Computed on test observations. |
| **6. Generate Feature Importance & Visualizations** | Complete | 11 publication-grade figures covering attention masks, beeswarm plots, ROC curves, confusion matrices, and dependence plots. |

---

## 2. Directory Layout

```text
Part3/
├── data/
│   └── processed/                  # Chronological splits (train.csv, validation.csv, test.csv)
├── deliverables/                   # Official Handoff Deliverables
│   ├── trained_models/             # Serialized TabNet, XGBoost, and Ensemble models
│   ├── model_comparison/          # Benchmark tables (CSV) and comparison charts (PNG)
│   ├── ensemble_results/           # Ensemble weights, metrics, and tuning curve
│   └── shap_visualizations/        # SHAP beeswarm, bar, dependence, and TabNet mask plots
├── docs/
│   ├── PART3_METHODOLOGY.md        # Mathematical formulation of TabNet, Ensembles, and SHAP
│   ├── PART3_RESULTS.md            # Experimental findings, metric analysis, and tables
│   └── PART3_VIVA_QA.md            # 10 comprehensive viva questions & answers
├── models/                         # Working model checkpoints & preprocessing scalers
├── notebooks/
│   └── Part3_TabNet_Ensemble_SHAP.ipynb  # Interactive walkthrough notebook
├── outputs/
│   ├── figures/                    # 11 high-resolution evaluation and explainability plots
│   ├── predictions/                # Out-of-sample prediction CSVs
│   └── results/                    # Validation and test metric CSVs
├── src/
│   ├── config.py                   # Central paths, hyperparameters, and seeds
│   ├── data/
│   │   └── loader.py               # Data ingestion with strict anti-leakage guards
│   ├── evaluation/
│   │   └── metrics.py              # Regression and classification metric calculators
│   ├── explainability/
│   │   ├── shap_explainer.py       # TreeSHAP and KernelSHAP implementations
│   │   └── tabnet_masks.py         # TabNet step-wise attention mask extractors
│   ├── models/
│   │   ├── base.py                 # Abstract base class
│   │   ├── tabnet_model.py         # PyTorch TabNet regressor and classifier
│   │   ├── xgboost_model.py        # Tuned XGBoost regressor and classifier
│   │   ├── decision_tree_model.py  # Decision Tree baseline
│   │   └── ensemble_model.py       # Weighted and Stacking ensembles
│   └── visualization/
│       └── plots.py                # Publication-quality plotting routines
├── tests/
│   ├── conftest.py                 # Pytest configuration
│   └── test_part3.py               # Unit and integration test suite
├── run_part3.py                    # End-to-end one-click execution pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation
```

---

## 3. Quick Start & Execution

### Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Pipeline End-to-End
To retrain models, re-evaluate metrics, recompute SHAP values, and generate all figures:
```powershell
python run_part3.py
```

### Run Test Suite
```powershell
pytest -q tests/test_part3.py
```
*(All 5 tests pass cleanly).*

---

## 4. Key Performance Highlights

### Regression Benchmark (Out-of-Sample Test Split: 698 Trading Days)
- **TabNet Regressor**: Lowest RMSE (**0.008579**) and the **only individual model with positive $R^2$ (+0.010824)**.
- **XGBoost Regressor**: RMSE = 0.008680, $R^2$ = -0.012732.
- **Ensemble Blend ($0.62$ TabNet + $0.38$ XGBoost)**: MAE of **0.006211**, lower than its two component models (TabNet: 0.006221; XGBoost: 0.006215).

### Classification Benchmark (Out-of-Sample Test Split: 698 Trading Days)
- **TabNet Classifier**: Highest Accuracy (**51.29%**), highest ROC-AUC (**0.5211**), and highest positive Precision (**52.05%**).
- **XGBoost Classifier**: Accuracy = 50.00%, ROC-AUC = 0.4962.

---

## 5. Explainability Highlights
1. **TabNet Attentive Masks**:
   - Step 0 targets macro cyclicality (`month_cos`) and immediate momentum (`ema_12_ratio`).
   - Step 1 targets intermediate trend anchors (`ema_26_ratio`).
   - Step 2 targets intra-week effects (`day_of_week`) and rolling volatility (`volatility_20`).
2. **SHAP Analysis**:
   - `volatility_20` acts as a regime dampener.
   - `rsi_14` demonstrates sharp mean-reverting pressure when $>70$.
