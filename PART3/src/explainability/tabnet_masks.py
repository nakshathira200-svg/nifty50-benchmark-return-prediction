"""TabNet Attention Mask analysis and interpretation.
TabNet's Attentive Transformer computes a sparse feature selection mask M[i] at each decision step i.
This module extracts and summarizes step-wise and aggregated attention masks.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.models.tabnet_model import TabNetRegressorModel, TabNetClassifierModel


def extract_attention_masks(
    model: TabNetRegressorModel | TabNetClassifierModel,
    X: np.ndarray,
    feature_names: list,
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Extracts aggregate attention matrix and individual step masks M[step] for all samples.
    Returns:
        agg_df: DataFrame of aggregated attention weights (n_samples, n_features)
        step_dfs: Dict mapping step index (0, 1, 2) to DataFrame of mask values (n_samples, n_features)
    """
    explain_matrix, masks = model.explain(X)

    agg_df = pd.DataFrame(explain_matrix, columns=feature_names)
    step_dfs = {}
    for step_idx, mask_arr in masks.items():
        step_dfs[step_idx] = pd.DataFrame(mask_arr, columns=feature_names)

    return agg_df, step_dfs


def summarize_attention_by_step(
    step_dfs: Dict[int, pd.DataFrame],
    feature_names: list,
) -> pd.DataFrame:
    """Computes mean attention allocated to each feature at each decision step."""
    summary_data = {"feature": feature_names}
    for step_idx, df in step_dfs.items():
        summary_data[f"step_{step_idx}_mean_attention"] = [float(df[col].mean()) for col in feature_names]

    res_df = pd.DataFrame(summary_data)
    # Add overall mean across steps
    step_cols = [c for c in res_df.columns if c.startswith("step_")]
    res_df["total_attention"] = res_df[step_cols].sum(axis=1)
    res_df = res_df.sort_values("total_attention", ascending=False).reset_index(drop=True)
    return res_df
