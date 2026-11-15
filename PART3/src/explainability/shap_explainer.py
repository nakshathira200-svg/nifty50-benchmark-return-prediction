"""SHAP (SHapley Additive exPlanations) analysis for XGBoost and TabNet models.
Implements TreeSHAP for tree-based models and KernelSHAP with background summarization for TabNet.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import shap

from src.models.xgboost_model import XGBoostRegressorModel, XGBoostClassifierModel
from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel


def explain_xgboost(
    model: XGBoostRegressorModel | XGBoostClassifierModel,
    X: np.ndarray,
    feature_names: list,
) -> Tuple[np.ndarray, float, shap.Explainer]:
    """Computes exact TreeSHAP values for XGBoost."""
    explainer = shap.TreeExplainer(model.model_)
    shap_values = explainer.shap_values(X)
    expected_value = float(
        explainer.expected_value
        if np.isscalar(explainer.expected_value)
        else explainer.expected_value[0]
    )
    return shap_values, expected_value, explainer


def explain_tabnet(
    model: TabNetRegressorModel | TabNetClassifierModel,
    X_train_bg: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list,
    n_kmeans: int = 10,
    max_evals: int = 200,
) -> Tuple[np.ndarray, float, shap.KernelExplainer]:
    """Computes KernelSHAP values for TabNet using k-means summary of the training distribution."""
    # Summarize background dataset using k-means to optimize runtime while preserving marginals
    bg_summary = shap.kmeans(X_train_bg, n_kmeans)

    predict_fn = model.predict
    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    shap_values = explainer.shap_values(X_eval, nsamples=max_evals)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0] if len(shap_values) == 1 else shap_values[1])
    else:
        shap_values = np.array(shap_values)

    expected_value = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0])
    return shap_values, expected_value, explainer


def shap_to_dataframe(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Formats SHAP values into a DataFrame and calculates mean absolute SHAP importance."""
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs = df_shap.abs().mean().sort_values(ascending=False)
    summary_df = pd.DataFrame({
        "feature": mean_abs.index,
        "mean_abs_shap": mean_abs.values,
        "relative_importance": mean_abs.values / mean_abs.sum() if mean_abs.sum() > 0 else mean_abs.values,
    })
    return summary_df
