"""Publication-quality visualization library for Part 3.
Generates comprehensive figures for model evaluation, comparisons, ensemble dynamics,
TabNet attention masks, and SHAP explainability.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_regression_comparison(df_reg: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing regression models across RMSE, MAE, R², and Directional Accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NIFTY 50 Regression Performance: TabNet vs XGBoost vs Baselines (Test Split)", fontweight="bold")

    palette = sns.color_palette("muted", len(df_reg))

    # 1. RMSE
    ax = axes[0, 0]
    sns.barplot(data=df_reg, x="model", y="RMSE", ax=ax, palette=palette)
    ax.set_title("Root Mean Squared Error (Lower is Better)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.5f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    # 2. MAE
    ax = axes[0, 1]
    sns.barplot(data=df_reg, x="model", y="MAE", ax=ax, palette=palette)
    ax.set_title("Mean Absolute Error (Lower is Better)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.5f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    # 3. R2 Score
    ax = axes[1, 0]
    sns.barplot(data=df_reg, x="model", y="R2", ax=ax, palette=palette)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("R² Score (Higher is Better; >0 beats mean)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom' if p.get_height() >= 0 else 'top', fontsize=8,
                    xytext=(0, 2 if p.get_height() >= 0 else -10), textcoords='offset points')

    # 4. Directional Accuracy
    ax = axes[1, 1]
    sns.barplot(data=df_reg, x="model", y="Directional_Accuracy", ax=ax, palette=palette)
    ax.axhline(0.5, color="red", linestyle=":", label="Random Walk (50%)")
    ax.set_title("Directional Accuracy (Up/Down sign match)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_classification_comparison(df_clf: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing classification models across Accuracy, ROC-AUC, F1, and Precision."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NIFTY 50 Classification Performance: TabNet vs XGBoost vs Baselines (Test Split)", fontweight="bold")

    palette = sns.color_palette("deep", len(df_clf))

    # 1. Accuracy
    ax = axes[0, 0]
    sns.barplot(data=df_clf, x="model", y="Accuracy", ax=ax, palette=palette)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Accuracy (with 95% Confidence Bounds)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p, ci in zip(ax.patches, df_clf["Accuracy_CI95_halfwidth"]):
        val = p.get_height()
        ax.errorbar(p.get_x() + p.get_width() / 2., val, yerr=ci, color="black", capsize=3)
        ax.annotate(f"{val*100:.1f}%", (p.get_x() + p.get_width() / 2., val + ci),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    # 2. ROC-AUC
    ax = axes[0, 1]
    sns.barplot(data=df_clf, x="model", y="ROC_AUC", ax=ax, palette=palette)
    ax.axhline(0.5, color="red", linestyle=":", label="Chance (0.50)")
    ax.set_title("ROC-AUC Score (Discriminative Power)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    # 3. F1 Score
    ax = axes[1, 0]
    sns.barplot(data=df_clf, x="model", y="F1", ax=ax, palette=palette)
    ax.set_title("F1 Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    # 4. Recall vs Precision
    ax = axes[1, 1]
    melted = df_clf.melt(id_vars=["model"], value_vars=["Precision", "Recall"], var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Precision vs Recall Trade-off")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Time-series and scatter comparison of actual returns vs predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.2, 1]})

    # 1. Time-series zoom (last 100 test days for clarity)
    ax = axes[0]
    zoom = slice(-100, None)
    ax.plot(dates[zoom], y_true[zoom], label="Actual Return", color="black", linewidth=1.5, alpha=0.8)
    styles = [("--", "blue"), ("-.", "green"), (":", "red")]
    for (name, pred), (ls, col) in zip(preds_dict.items(), styles):
        ax.plot(dates[zoom], pred[zoom], label=f"{name}", linestyle=ls, color=col, linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Realized vs Predicted Returns (Last 100 Test Trading Days)")
    ax.set_ylabel("Next-Day Return")
    ax.legend(loc="upper right")

    # 2. Scatter plot: Actual vs Predicted
    ax = axes[1]
    colors = ["blue", "green", "red"]
    for (name, pred), col in zip(preds_dict.items(), colors):
        ax.scatter(y_true, pred, alpha=0.3, s=16, label=name, color=col)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.plot([-0.05, 0.05], [-0.05, 0.05], color="black", linestyle=":", label="Identity (x=y)")
    ax.set_xlim(-0.04, 0.04)
    ax.set_ylim(-0.02, 0.02)
    ax.set_title("Scatter: Predicted Return vs Realized Return (Test Split)")
    ax.set_xlabel("Realized Next-Day Return")
    ax.set_ylabel("Predicted Return")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Receiver Operating Characteristic curves for all classification models."""
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.500)")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (name, prob), col in zip(probs_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, prob)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, linewidth=1.8, label=f"{name} (AUC = {score:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curves: Discriminative Capability on Out-of-Sample Test Set", fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Side-by-side confusion matrix heatmaps."""
    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Down/Flat (0)", "Up (1)"],
            yticklabels=["Down/Flat (0)", "Up (1)"],
        )
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        ax.set_title(f"{name}\nAccuracy: {acc*100:.1f}%")
        ax.set_xlabel("Predicted Direction")
        ax.set_ylabel("Actual Direction")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_tabnet_attention_masks(
    step_summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Heatmap and stacked bar chart showing TabNet's step-wise attention allocation across features."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.2]})

    step_cols = [c for c in step_summary_df.columns if c.startswith("step_")]
    plot_df = step_summary_df.set_index("feature")[step_cols]

    # 1. Heatmap
    ax = axes[0]
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("TabNet Step-Wise Attention Heatmap M[step]")
    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Feature")

    # 2. Stacked Bar Chart
    ax = axes[1]
    plot_df.plot(kind="barh", stacked=True, colormap="viridis", ax=ax)
    ax.set_title("Cumulative Attention Allocated Per Feature Across Decision Steps")
    ax.set_xlabel("Cumulative Attention Weight")
    ax.set_ylabel("Feature")
    ax.legend(title="Step", loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    out_path: Path,
    title: str = "SHAP Summary Beeswarm Plot",
) -> None:
    """SHAP beeswarm summary plot showing impact and directionality of features."""
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_feature_importance_comparison(
    tab_shap_df: pd.DataFrame,
    xgb_shap_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Compares mean |SHAP| feature importance between TabNet and XGBoost."""
    merged = pd.merge(
        tab_shap_df[["feature", "relative_importance"]].rename(columns={"relative_importance": "TabNet"}),
        xgb_shap_df[["feature", "relative_importance"]].rename(columns={"relative_importance": "XGBoost"}),
        on="feature",
    ).sort_values("TabNet", ascending=False)

    melted = merged.melt(id_vars=["feature"], var_name="Framework", value_name="Relative_Importance")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=melted, x="Relative_Importance", y="feature", hue="Framework", ax=ax, palette=["#1f77b4", "#ff7f0e"])
    ax.set_title("SHAP Global Feature Importance Comparison: TabNet vs XGBoost", fontweight="bold")
    ax.set_xlabel("Normalized Mean |SHAP Value| (Contribution to Prediction)")
    ax.set_ylabel("Engineered Feature")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    top_features: list,
    out_path: Path,
) -> None:
    """Multi-panel SHAP dependence plots for top features showing non-linear response curves."""
    import shap
    n_feats = len(top_features)
    fig, axes = plt.subplots(1, n_feats, figsize=(5.5 * n_feats, 4.5))
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_features):
        feat_idx = feature_names.index(feat)
        plt.sca(ax)
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            ax=ax,
        )
        ax.set_title(f"SHAP Dependence: {feat}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_tuning_curve(
    loss_curve: list,
    metric_name: str,
    optimal_w: float,
    out_path: Path,
) -> None:
    """Plots validation loss / evaluation metric as ensemble weight varies between TabNet and XGBoost."""
    df_curve = pd.DataFrame(loss_curve)
    w_col = [c for c in df_curve.columns if "weight" in c][0]
    val_col = [c for c in df_curve.columns if c != w_col][0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_curve[w_col], df_curve[val_col], color="#2b5c8f", linewidth=2.2, label=f"Validation {metric_name}")
    ax.axvline(optimal_w, color="red", linestyle="--", label=f"Optimal Weight (w* = {optimal_w:.2f})")
    ax.scatter([optimal_w], [df_curve.loc[df_curve[w_col] == optimal_w, val_col].values[0]],
               color="red", s=70, zorder=5)

    ax.set_xlabel("Ensemble Weight assigned to TabNet (w)")
    ax.set_ylabel(f"Validation {metric_name}")
    ax.set_title(f"Ensemble Blending Optimization: Validation {metric_name} vs Weight w", fontweight="bold")
    ax.annotate(f"Pure XGBoost\n(w = 0.0)", (0.02, df_curve[val_col].iloc[0]), fontsize=9, alpha=0.8)
    ax.annotate(f"Pure TabNet\n(w = 1.0)", (0.85, df_curve[val_col].iloc[-1]), fontsize=9, alpha=0.8)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_error_correlation(
    e_tabnet: np.ndarray,
    e_xgboost: np.ndarray,
    out_path: Path,
) -> None:
    """Residual error scatter and correlation between TabNet and XGBoost,
    providing empirical evidence for variance reduction via ensembling."""
    r = float(np.corrcoef(e_tabnet, e_xgboost)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(e_xgboost, e_tabnet, alpha=0.35, color="#4c72b0", s=18)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Trendline
    m, b = np.polyfit(e_xgboost, e_tabnet, 1)
    x_vals = np.linspace(e_xgboost.min(), e_xgboost.max(), 100)
    ax.plot(x_vals, m * x_vals + b, color="red", linestyle=":", label=f"Fit (r = {r:.3f})")

    ax.set_title("Residual Error Correlation: TabNet vs XGBoost (Test Split)", fontweight="bold")
    ax.set_xlabel("XGBoost Residual Error (y_true - y_pred)")
    ax.set_ylabel("TabNet Residual Error (y_true - y_pred)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
