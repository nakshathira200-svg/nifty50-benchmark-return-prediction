"""Explainability module providing SHAP analysis and TabNet Attention Mask extraction."""
from src.explainability.shap_explainer import (
    explain_xgboost,
    explain_tabnet,
    shap_to_dataframe,
)
from src.explainability.tabnet_masks import (
    extract_attention_masks,
    summarize_attention_by_step,
)

__all__ = [
    "explain_xgboost",
    "explain_tabnet",
    "shap_to_dataframe",
    "extract_attention_masks",
    "summarize_attention_by_step",
]
