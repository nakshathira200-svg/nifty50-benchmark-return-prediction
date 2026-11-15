"""Part 3 Orchestration: Deep Learning (TabNet), Ensembling, and Explainability (SHAP).
Executes end-to-end training, comparative evaluation, ensemble learning, SHAP explanations,
and generates publication-quality figures and deliverable artifacts.
"""
from __future__ import annotations
import os
import time
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import confusion_matrix

from src.config import (
    ROOT,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    DELIVERABLES_DIR,
    SEED,
    set_global_seed,
)
from src.data.loader import build_bundle, save_preprocessing
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel
from src.models.xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel
from src.models.decision_tree_model import DecisionTreeRegressorModel, DecisionTreeClassifierModel
from src.models.ensemble_model import (
    WeightedEnsembleRegressor,
    WeightedEnsembleClassifier,
    StackingEnsembleRegressor,
)
from src.explainability.shap_explainer import (
    explain_xgboost,
    explain_tabnet,
    shap_to_dataframe,
)
from src.explainability.tabnet_masks import (
    extract_attention_masks,
    summarize_attention_by_step,
)
from src.visualization.plots import (
    plot_regression_comparison,
    plot_classification_comparison,
    plot_actual_vs_predicted,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_tabnet_attention_masks,
    plot_shap_summary,
    plot_shap_feature_importance_comparison,
    plot_shap_dependence,
    plot_ensemble_tuning_curve,
    plot_error_correlation,
)


def run_pipeline():
    print("=" * 80)
    print("  PERSON 3: DEEP LEARNING (TABNET), ENSEMBLING & EXPLAINABILITY (SHAP)")
    print("=" * 80)

    set_global_seed(SEED)

    # 1. Load Data Bundle
    print("\n[Step 1/6] Loading data splits and verifying anti-leakage invariants...")
    bundle = build_bundle("stationary", include_test=True)
    save_preprocessing(bundle, MODELS_DIR, "stationary")
    print(f"  Train: {bundle.X_train.shape[0]} rows | Validation: {bundle.X_val.shape[0]} rows | Test: {bundle.X_test.shape[0]} rows")
    print(f"  Engineered features ({len(bundle.features)}): {', '.join(bundle.features)}")

    # 2. Train Models
    print("\n[Step 2/6] Training individual and baseline models...")
    models_dict = {}
    train_times = {}

    # TabNet Regressor
    print("  [Train] PyTorch TabNet Regressor...")
    t0 = time.perf_counter()
    tab_reg = TabNetRegressorModel(seed=SEED)
    tab_reg.fit(bundle.X_train, bundle.y_train_reg, bundle.X_val, bundle.y_val_reg, bundle.features)
    train_times["TabNet_Regressor"] = time.perf_counter() - t0
    tab_reg.save(MODELS_DIR / "tabnet_reg")
    models_dict["TabNet_Regressor"] = tab_reg
    print(f"          Done in {train_times['TabNet_Regressor']:.2f}s | Epochs: {tab_reg.epochs_trained_}")

    # TabNet Classifier
    print("  [Train] PyTorch TabNet Classifier...")
    t0 = time.perf_counter()
    tab_clf = TabNetClassifierModel(seed=SEED)
    tab_clf.fit(bundle.X_train, bundle.y_train_clf, bundle.X_val, bundle.y_val_clf, bundle.features)
    train_times["TabNet_Classifier"] = time.perf_counter() - t0
    tab_clf.save(MODELS_DIR / "tabnet_clf")
    models_dict["TabNet_Classifier"] = tab_clf
    print(f"          Done in {train_times['TabNet_Classifier']:.2f}s | Epochs: {tab_clf.epochs_trained_}")

    # XGBoost Regressor
    print("  [Train] XGBoost Regressor (with hyperparameter tuning)...")
    t0 = time.perf_counter()
    xgb_reg = XGBoostRegressorModel(seed=SEED)
    xgb_reg.fit(bundle.X_train, bundle.y_train_reg, bundle.X_val, bundle.y_val_reg, bundle.features)
    train_times["XGBoost_Regressor"] = time.perf_counter() - t0
    xgb_reg.save(MODELS_DIR / "xgboost_reg")
    models_dict["XGBoost_Regressor"] = xgb_reg
    print(f"          Done in {train_times['XGBoost_Regressor']:.2f}s | Best params: {xgb_reg.best_params_}")

    # XGBoost Classifier
    print("  [Train] XGBoost Classifier (with hyperparameter tuning)...")
    t0 = time.perf_counter()
    xgb_clf = XGBoostClassifierModel(seed=SEED)
    xgb_clf.fit(bundle.X_train, bundle.y_train_clf, bundle.X_val, bundle.y_val_clf, bundle.features)
    train_times["XGBoost_Classifier"] = time.perf_counter() - t0
    xgb_clf.save(MODELS_DIR / "xgboost_clf")
    models_dict["XGBoost_Classifier"] = xgb_clf
    print(f"          Done in {train_times['XGBoost_Classifier']:.2f}s | Best params: {xgb_clf.best_params_}")

    # Decision Tree Baselines
    print("  [Train] Decision Tree Baselines (tuned on validation)...")
    dt_reg = DecisionTreeRegressorModel(seed=SEED)
    dt_reg.fit(bundle.X_train, bundle.y_train_reg, bundle.X_val, bundle.y_val_reg, bundle.features)
    dt_reg.save(MODELS_DIR / "decision_tree_reg")
    models_dict["DecisionTree_Regressor"] = dt_reg

    dt_clf = DecisionTreeClassifierModel(seed=SEED)
    dt_clf.fit(bundle.X_train, bundle.y_train_clf, bundle.X_val, bundle.y_val_clf, bundle.features)
    dt_clf.save(MODELS_DIR / "decision_tree_clf")
    models_dict["DecisionTree_Classifier"] = dt_clf

    # Ridge Regression Baseline
    ridge = Ridge(alpha=100.0, random_state=SEED).fit(bundle.X_train, bundle.y_train_reg)
    # Logistic Regression Baseline
    logistic = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED).fit(bundle.X_train, bundle.y_train_clf)

    # 3. Implement Ensemble Learning
    print("\n[Step 3/6] Implementing Ensemble Learning (TabNet + XGBoost)...")
    # Weighted Average Regressor
    ens_reg = WeightedEnsembleRegressor(seed=SEED)
    ens_reg.fit_components(tab_reg, xgb_reg, bundle.X_val, bundle.y_val_reg, bundle.features)
    ens_reg.save(MODELS_DIR / "ensemble_weighted_reg")
    models_dict["Ensemble_Weighted_Regressor"] = ens_reg
    print(f"  Optimal Regression Blend: {ens_reg.w_tabnet_:.2f} * TabNet + {1 - ens_reg.w_tabnet_:.2f} * XGBoost (Val RMSE: {ens_reg.best_params_['val_rmse']:.5f})")

    # Stacking Regressor
    stack_reg = StackingEnsembleRegressor(seed=SEED)
    stack_reg.fit_components(tab_reg, xgb_reg, bundle.X_val, bundle.y_val_reg, bundle.features)
    stack_reg.save(MODELS_DIR / "ensemble_stacking_reg")
    models_dict["Ensemble_Stacking_Regressor"] = stack_reg
    print(f"  Stacking Regressor (Ridge meta-model): coef={stack_reg.best_params_['meta_coef']}")

    # Weighted Average Classifier
    ens_clf = WeightedEnsembleClassifier(seed=SEED)
    ens_clf.fit_components(tab_clf, xgb_clf, bundle.X_val, bundle.y_val_clf, bundle.features)
    ens_clf.save(MODELS_DIR / "ensemble_weighted_clf")
    models_dict["Ensemble_Weighted_Classifier"] = ens_clf
    print(f"  Optimal Classification Blend: {ens_clf.w_tabnet_:.2f} * TabNet + {1 - ens_clf.w_tabnet_:.2f} * XGBoost (Val ROC-AUC: {ens_clf.best_params_['val_roc_auc']:.4f})")

    # 4. Comprehensive Evaluation on Validation and Test Splits
    print("\n[Step 4/6] Evaluating models across Validation and Test sets...")
    reg_test_records = []
    reg_val_records = []

    # Helper dictionary for regression predictions
    reg_eval_models = {
        "Baseline_TrainMean": {
            "predict_fn": lambda X: np.full(len(X), bundle.y_train_reg.mean()),
            "framework": "baseline", "simplicity": 5, "transparency": 5, "params": 1
        },
        "Baseline_Persistence": {
            "predict_fn": lambda X: (bundle.y_train_reg.mean() if len(X) == 0 else
                                     X[:, bundle.features.index("daily_return")] * bundle.scaler.scale_[bundle.features.index("daily_return")] +
                                     bundle.scaler.mean_[bundle.features.index("daily_return")]),
            "framework": "baseline", "simplicity": 5, "transparency": 5, "params": 1
        },
        "Baseline_Ridge": {
            "predict_fn": ridge.predict,
            "framework": "linear", "simplicity": 5, "transparency": 5, "params": len(bundle.features) + 1
        },
        "DecisionTree_Regressor": {
            "predict_fn": dt_reg.predict,
            "framework": "decision_tree", "simplicity": 4, "transparency": 4, "params": dt_reg.n_params_
        },
        "XGBoost_Regressor": {
            "predict_fn": xgb_reg.predict,
            "framework": "xgboost", "simplicity": 4, "transparency": 4, "params": xgb_reg.n_params_
        },
        "TabNet_Regressor": {
            "predict_fn": tab_reg.predict,
            "framework": "tabnet", "simplicity": 3, "transparency": 4, "params": tab_reg.n_params_
        },
        "Ensemble_Weighted_Regressor": {
            "predict_fn": ens_reg.predict,
            "framework": "ensemble", "simplicity": 3, "transparency": 4, "params": ens_reg.n_params_
        },
        "Ensemble_Stacking_Regressor": {
            "predict_fn": stack_reg.predict,
            "framework": "ensemble", "simplicity": 3, "transparency": 3, "params": stack_reg.n_params_
        },
    }

    test_reg_preds = {}
    for name, spec in reg_eval_models.items():
        pv = spec["predict_fn"](bundle.X_val)
        pt = spec["predict_fn"](bundle.X_test)
        test_reg_preds[name] = pt

        mv = regression_metrics(bundle.y_val_reg, pv)
        mt = regression_metrics(bundle.y_test_reg, pt)

        reg_val_records.append({"model": name, **mv})
        reg_test_records.append({
            "model": name,
            "framework": spec["framework"],
            "simplicity_rating": spec["simplicity"],
            "transparency_rating": spec["transparency"],
            "n_parameters": spec["params"],
            "train_time_sec": round(train_times.get(name, 0.05), 3),
            **mt,
        })
        # Save prediction file
        pd.DataFrame({"price_date": bundle.test_index, "y_true": bundle.y_test_reg, "y_pred": pt}).to_csv(
            PREDICTIONS_DIR / f"{name}_test_predictions.csv", index=False
        )

    df_reg_val = pd.DataFrame(reg_val_records)
    df_reg_test = pd.DataFrame(reg_test_records)
    df_reg_test.to_csv(RESULTS_DIR / "regression_test_metrics.csv", index=False)
    df_reg_val.to_csv(RESULTS_DIR / "regression_validation_metrics.csv", index=False)

    # Classification Models Evaluation
    clf_test_records = []
    clf_val_records = []

    clf_eval_models = {
        "Baseline_MajorityClass": {
            "predict_fn": lambda X: np.full(len(X), float(bundle.y_train_clf.mean() >= 0.5)),
            "framework": "baseline", "simplicity": 5, "transparency": 5, "params": 1
        },
        "Baseline_LogisticRegression": {
            "predict_fn": lambda X: logistic.predict_proba(X)[:, 1],
            "framework": "linear", "simplicity": 5, "transparency": 5, "params": len(bundle.features) + 1
        },
        "DecisionTree_Classifier": {
            "predict_fn": dt_clf.predict,
            "framework": "decision_tree", "simplicity": 4, "transparency": 4, "params": dt_clf.n_params_
        },
        "XGBoost_Classifier": {
            "predict_fn": xgb_clf.predict,
            "framework": "xgboost", "simplicity": 4, "transparency": 4, "params": xgb_clf.n_params_
        },
        "TabNet_Classifier": {
            "predict_fn": tab_clf.predict,
            "framework": "tabnet", "simplicity": 3, "transparency": 4, "params": tab_clf.n_params_
        },
        "Ensemble_Weighted_Classifier": {
            "predict_fn": ens_clf.predict,
            "framework": "ensemble", "simplicity": 3, "transparency": 4, "params": ens_clf.n_params_
        },
    }

    test_clf_probs = {}
    test_cms = {}
    for name, spec in clf_eval_models.items():
        pv = spec["predict_fn"](bundle.X_val)
        pt = spec["predict_fn"](bundle.X_test)
        test_clf_probs[name] = pt

        mv = classification_metrics(bundle.y_val_clf, pv)
        mt = classification_metrics(bundle.y_test_clf, pt)
        test_cms[name] = np.array([[mt["TN"], mt["FP"]], [mt["FN"], mt["TP"]]])

        clf_val_records.append({"model": name, **mv})
        clf_test_records.append({
            "model": name,
            "framework": spec["framework"],
            "simplicity_rating": spec["simplicity"],
            "transparency_rating": spec["transparency"],
            "n_parameters": spec["params"],
            "train_time_sec": round(train_times.get(name, 0.05), 3),
            **mt,
        })
        pd.DataFrame({"price_date": bundle.test_index, "y_true": bundle.y_test_clf, "y_prob": pt, "y_pred": (pt >= 0.5).astype(int)}).to_csv(
            PREDICTIONS_DIR / f"{name}_test_predictions.csv", index=False
        )

    df_clf_val = pd.DataFrame(clf_val_records)
    df_clf_test = pd.DataFrame(clf_test_records)
    df_clf_test.to_csv(RESULTS_DIR / "classification_test_metrics.csv", index=False)
    df_clf_val.to_csv(RESULTS_DIR / "classification_validation_metrics.csv", index=False)

    # Consolidated Comparison Table
    df_comparison = pd.concat([
        df_reg_test.assign(task="regression"),
        df_clf_test.assign(task="classification"),
    ], ignore_index=True)
    df_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # 5. Explainability & SHAP Analysis
    print("\n[Step 5/6] Performing Explainability Analysis (TabNet Masks & SHAP)...")

    # A. TabNet Attention Masks
    print("  Extracting TabNet Step-Wise Attention Masks...")
    agg_attention_df, step_dfs = extract_attention_masks(tab_reg, bundle.X_test, bundle.features)
    agg_attention_df.to_csv(RESULTS_DIR / "tabnet_aggregate_attention.csv", index=False)
    for step_idx, df_step in step_dfs.items():
        df_step.to_csv(RESULTS_DIR / f"tabnet_attention_mask_step_{step_idx}.csv", index=False)

    step_summary_df = summarize_attention_by_step(step_dfs, bundle.features)
    step_summary_df.to_csv(RESULTS_DIR / "tabnet_step_attention_summary.csv", index=False)
    print("  Top 5 features attended by TabNet:")
    for _, row in step_summary_df.head(5).iterrows():
        print(f"    - {row['feature']:<18}: total_attention={row['total_attention']:.3f} (Step 0: {row['step_0_mean_attention']:.3f}, Step 1: {row['step_1_mean_attention']:.3f}, Step 2: {row['step_2_mean_attention']:.3f})")

    # B. TreeSHAP for XGBoost
    print("  Computing TreeSHAP for XGBoost...")
    xgb_shap_vals, xgb_expected, _ = explain_xgboost(xgb_reg, bundle.X_test, bundle.features)
    df_xgb_shap = shap_to_dataframe(xgb_shap_vals, bundle.features)
    df_xgb_shap.to_csv(RESULTS_DIR / "xgboost_shap_importance.csv", index=False)

    # C. KernelSHAP for TabNet
    print("  Computing KernelSHAP for TabNet (evaluating representative test sample)...")
    eval_sample_size = 60
    tab_shap_vals, tab_expected, _ = explain_tabnet(
        tab_reg,
        bundle.X_train,
        bundle.X_test[:eval_sample_size],
        bundle.features,
        n_kmeans=8,
        max_evals=150,
    )
    df_tab_shap = shap_to_dataframe(tab_shap_vals, bundle.features)
    df_tab_shap.to_csv(RESULTS_DIR / "tabnet_shap_importance.csv", index=False)

    # 6. Generate Figures
    print("\n[Step 6/6] Generating Publication-Quality Figures...")

    # Plot 1: Regression comparison
    plot_regression_comparison(df_reg_test, FIGURES_DIR / "01_regression_comparison.png")
    # Plot 2: Classification comparison
    plot_classification_comparison(df_clf_test, FIGURES_DIR / "02_classification_comparison.png")
    # Plot 3: Actual vs Predicted
    selected_preds = {
        "TabNet": test_reg_preds["TabNet_Regressor"],
        "XGBoost": test_reg_preds["XGBoost_Regressor"],
        "Ensemble": test_reg_preds["Ensemble_Weighted_Regressor"],
    }
    plot_actual_vs_predicted(bundle.test_index, bundle.y_test_reg, selected_preds, FIGURES_DIR / "03_actual_vs_predicted_regression.png")
    # Plot 4: ROC Curves
    selected_probs = {
        "TabNet": test_clf_probs["TabNet_Classifier"],
        "XGBoost": test_clf_probs["XGBoost_Classifier"],
        "DecisionTree": test_clf_probs["DecisionTree_Classifier"],
        "LogisticRegression": test_clf_probs["Baseline_LogisticRegression"],
        "Ensemble Blend": test_clf_probs["Ensemble_Weighted_Classifier"],
    }
    plot_roc_curves(bundle.y_test_clf, selected_probs, FIGURES_DIR / "04_roc_curves.png")
    # Plot 5: Confusion Matrices
    selected_cms = {
        "TabNet": test_cms["TabNet_Classifier"],
        "XGBoost": test_cms["XGBoost_Classifier"],
        "Decision Tree": test_cms["DecisionTree_Classifier"],
        "Ensemble Blend": test_cms["Ensemble_Weighted_Classifier"],
    }
    plot_confusion_matrices(selected_cms, FIGURES_DIR / "05_confusion_matrices.png")
    # Plot 6: TabNet Attention Masks
    plot_tabnet_attention_masks(step_summary_df, FIGURES_DIR / "06_tabnet_attention_masks_heatmap.png")
    # Plot 7: SHAP Beeswarm Summary Plot (TreeSHAP on full test set)
    plot_shap_summary(xgb_shap_vals, bundle.X_test, bundle.features, FIGURES_DIR / "07_shap_summary_beeswarm.png",
                      title="TreeSHAP Beeswarm Summary (Feature Value Impact on Return Prediction)")
    # Plot 8: SHAP Feature Importance Comparison
    plot_shap_feature_importance_comparison(df_tab_shap, df_xgb_shap, FIGURES_DIR / "08_shap_feature_importance_comparison.png")
    # Plot 9: SHAP Dependence Plots
    top_3_feats = df_xgb_shap["feature"].head(3).tolist()
    plot_shap_dependence(xgb_shap_vals, bundle.X_test, bundle.features, top_3_feats, FIGURES_DIR / "09_shap_dependence_plots.png")
    # Plot 10: Ensemble Blending Optimization Curve
    plot_ensemble_tuning_curve(ens_reg.best_params_["loss_curve"], "RMSE", ens_reg.w_tabnet_, FIGURES_DIR / "10_ensemble_weight_tuning.png")
    # Plot 11: Error Correlation Analysis
    e_tab = bundle.y_test_reg - test_reg_preds["TabNet_Regressor"]
    e_xgb = bundle.y_test_reg - test_reg_preds["XGBoost_Regressor"]
    plot_error_correlation(e_tab, e_xgb, FIGURES_DIR / "11_prediction_error_correlation.png")

    print(f"  All figures saved to {FIGURES_DIR}")

    # Copy Deliverables to Deliverable directories
    print("\nSynchronizing Deliverables directory...")
    # Trained models
    for mdir in ["tabnet_reg", "tabnet_clf", "xgboost_reg", "xgboost_clf", "ensemble_weighted_reg", "ensemble_weighted_clf"]:
        src_m = MODELS_DIR / mdir
        dst_m = DELIVERABLES_DIR / "trained_models" / mdir
        if src_m.exists():
            if dst_m.exists():
                shutil.rmtree(dst_m)
            shutil.copytree(src_m, dst_m)

    # Model comparison
    for f in ["model_comparison.csv", "regression_test_metrics.csv", "classification_test_metrics.csv"]:
        shutil.copy2(RESULTS_DIR / f, DELIVERABLES_DIR / "model_comparison" / f)
    for g in ["01_regression_comparison.png", "02_classification_comparison.png", "03_actual_vs_predicted_regression.png", "04_roc_curves.png", "05_confusion_matrices.png"]:
        shutil.copy2(FIGURES_DIR / g, DELIVERABLES_DIR / "model_comparison" / g)

    # Ensemble results
    ensemble_df = pd.DataFrame([
        {"Model": "XGBoost Regressor", "RMSE": df_reg_test.loc[df_reg_test.model == "XGBoost_Regressor", "RMSE"].values[0],
         "MAE": df_reg_test.loc[df_reg_test.model == "XGBoost_Regressor", "MAE"].values[0],
         "R2": df_reg_test.loc[df_reg_test.model == "XGBoost_Regressor", "R2"].values[0]},
        {"Model": "TabNet Regressor", "RMSE": df_reg_test.loc[df_reg_test.model == "TabNet_Regressor", "RMSE"].values[0],
         "MAE": df_reg_test.loc[df_reg_test.model == "TabNet_Regressor", "MAE"].values[0],
         "R2": df_reg_test.loc[df_reg_test.model == "TabNet_Regressor", "R2"].values[0]},
        {"Model": "Ensemble Weighted", "RMSE": df_reg_test.loc[df_reg_test.model == "Ensemble_Weighted_Regressor", "RMSE"].values[0],
         "MAE": df_reg_test.loc[df_reg_test.model == "Ensemble_Weighted_Regressor", "MAE"].values[0],
         "R2": df_reg_test.loc[df_reg_test.model == "Ensemble_Weighted_Regressor", "R2"].values[0]},
        {"Model": "Ensemble Stacking", "RMSE": df_reg_test.loc[df_reg_test.model == "Ensemble_Stacking_Regressor", "RMSE"].values[0],
         "MAE": df_reg_test.loc[df_reg_test.model == "Ensemble_Stacking_Regressor", "MAE"].values[0],
         "R2": df_reg_test.loc[df_reg_test.model == "Ensemble_Stacking_Regressor", "R2"].values[0]},
    ])
    ensemble_df.to_csv(DELIVERABLES_DIR / "ensemble_results" / "ensemble_regression_comparison.csv", index=False)
    for g in ["10_ensemble_weight_tuning.png", "11_prediction_error_correlation.png"]:
        shutil.copy2(FIGURES_DIR / g, DELIVERABLES_DIR / "ensemble_results" / g)

    # SHAP visualizations
    for f in ["xgboost_shap_importance.csv", "tabnet_shap_importance.csv", "tabnet_step_attention_summary.csv"]:
        shutil.copy2(RESULTS_DIR / f, DELIVERABLES_DIR / "shap_visualizations" / f)
    for g in ["06_tabnet_attention_masks_heatmap.png", "07_shap_summary_beeswarm.png", "08_shap_feature_importance_comparison.png", "09_shap_dependence_plots.png"]:
        shutil.copy2(FIGURES_DIR / g, DELIVERABLES_DIR / "shap_visualizations" / g)

    print("\n" + "=" * 80)
    print("  PERSON 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n=== Regression Test Results ===")
    print(df_reg_test[["model", "RMSE", "MAE", "R2", "Directional_Accuracy"]].to_string(index=False))
    print("\n=== Classification Test Results ===")
    print(df_clf_test[["model", "Accuracy", "ROC_AUC", "F1", "Precision", "Recall"]].to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
