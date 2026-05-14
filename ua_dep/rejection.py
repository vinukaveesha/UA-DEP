from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def coverage_accuracy_curve(
    uncertainty: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size: int = 200,
) -> pd.DataFrame:
    if len(uncertainty) == 0:
        raise ValueError("Empty uncertainty array.")

    min_u = float(np.min(uncertainty))
    max_u = float(np.max(uncertainty))
    if max_u == min_u:
        taus = np.array([min_u], dtype=np.float32)
    else:
        taus = np.linspace(min_u, max_u, num=grid_size)

    rows = []
    n = len(y_true)
    for tau in taus:
        accepted = uncertainty <= tau
        coverage = float(np.mean(accepted))
        if np.any(accepted):
            acc = float(np.mean(y_true[accepted] == y_pred[accepted]))
            n_acc = int(np.sum(accepted))
        else:
            acc = np.nan
            n_acc = 0
        rows.append(
            {
                "tau": float(tau),
                "coverage": coverage,
                "accepted_accuracy": acc,
                "accepted_count": n_acc,
                "total_count": int(n),
                "objective": 0.0 if np.isnan(acc) else float(acc * coverage),
            }
        )

    return pd.DataFrame(rows)


def choose_tau(
    curve_df: pd.DataFrame,
    min_coverage: float = 0.80,
) -> float:
    eligible = curve_df[curve_df["coverage"] >= min_coverage].copy()
    if eligible.empty:
        eligible = curve_df.copy()

    eligible = eligible.dropna(subset=["accepted_accuracy"])
    if eligible.empty:
        return float(curve_df["tau"].max())

    # Select the threshold that maximizes accepted-sample accuracy while
    # respecting minimum coverage, then prefer higher coverage and lower tau.
    best_acc = eligible["accepted_accuracy"].max()
    best = eligible[eligible["accepted_accuracy"] == best_acc]
    best = best.sort_values(
        by=["coverage", "tau"],
        ascending=[False, True],
    )
    return float(best.iloc[0]["tau"])


def roc_error_detection_curve(
    uncertainty: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    if len(uncertainty) == 0:
        raise ValueError("Empty uncertainty array.")

    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error_labels = (y_true != y_pred).astype(np.int64)

    # Assumption: in the absence of validation corruptions, ROC thresholding is
    # framed as validation error detection where higher uncertainty indicates a
    # higher probability of misclassification.
    if np.unique(error_labels).size < 2:
        return pd.DataFrame(
            columns=[
                "tau",
                "fpr",
                "tpr",
                "youden_j",
                "coverage",
                "accepted_accuracy",
                "accepted_count",
                "total_count",
            ]
        )

    fpr, tpr, thresholds = roc_curve(error_labels, uncertainty)

    rows = []
    n = len(y_true)
    for tau, fpr_i, tpr_i in zip(thresholds, fpr, tpr):
        if not np.isfinite(tau):
            continue
        accepted = uncertainty <= tau
        if np.any(accepted):
            acc = float(np.mean(y_true[accepted] == y_pred[accepted]))
            n_acc = int(np.sum(accepted))
        else:
            acc = np.nan
            n_acc = 0
        rows.append(
            {
                "tau": float(tau),
                "fpr": float(fpr_i),
                "tpr": float(tpr_i),
                "youden_j": float(tpr_i - fpr_i),
                "coverage": float(np.mean(accepted)),
                "accepted_accuracy": acc,
                "accepted_count": n_acc,
                "total_count": int(n),
            }
        )

    return pd.DataFrame(rows)


def choose_tau_roc(
    roc_df: pd.DataFrame,
    min_coverage: float = 0.80,
) -> float:
    if roc_df.empty:
        raise ValueError("ROC threshold selection requires non-empty ROC curve data.")

    eligible = roc_df[roc_df["coverage"] >= min_coverage].copy()
    if eligible.empty:
        eligible = roc_df.copy()

    best_j = eligible["youden_j"].max()
    best = eligible[eligible["youden_j"] == best_j]
    best = best.sort_values(
        by=["tpr", "fpr", "coverage", "tau"],
        ascending=[False, True, False, True],
    )
    return float(best.iloc[0]["tau"])


def apply_rejection_gate(uncertainty: np.ndarray, tau: float) -> np.ndarray:
    return uncertainty <= tau
