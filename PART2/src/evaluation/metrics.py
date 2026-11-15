"""Regression and classification metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)


def directional_accuracy(y_true, y_pred) -> float:
    """Share of days where predicted sign matches realised sign. 'Up' means strictly > 0, consistent with the
    classification label (zero return counts as not-up)."""
    return float(np.mean((np.asarray(y_pred) > 0) == (np.asarray(y_true) > 0)))


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred), "Directional_Accuracy": directional_accuracy(y_true, y_pred)}


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 and len(np.unique(y_true)) > 1 else 0.5
    n = len(y_true); acc = accuracy_score(y_true, y_hat)
    return {"Accuracy": acc, "Precision": precision_score(y_true, y_hat, zero_division=0),
            "Recall": recall_score(y_true, y_hat, zero_division=0), "F1": f1_score(y_true, y_hat, zero_division=0),
            "ROC_AUC": auc, "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
            "Accuracy_CI95_halfwidth": float(1.96 * np.sqrt(acc * (1 - acc) / n))}
