"""Visualization library for comparative charts, confusion matrices, ROC curves, and SHAP plots."""
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

__all__ = [
    "plot_regression_comparison",
    "plot_classification_comparison",
    "plot_actual_vs_predicted",
    "plot_roc_curves",
    "plot_confusion_matrices",
    "plot_tabnet_attention_masks",
    "plot_shap_summary",
    "plot_shap_feature_importance_comparison",
    "plot_shap_dependence",
    "plot_ensemble_tuning_curve",
    "plot_error_correlation",
]
