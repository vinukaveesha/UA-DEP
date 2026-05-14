from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def multiclass_nll(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, dict[str, np.ndarray]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True)
    bin_ids = np.clip(bin_ids, 1, n_bins)

    ece = 0.0
    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        count = int(np.sum(mask))
        bin_counts[b - 1] = count
        if count == 0:
            continue
        acc = float(np.mean(correctness[mask]))
        conf = float(np.mean(confidences[mask]))
        bin_acc[b - 1] = acc
        bin_conf[b - 1] = conf
        ece += (count / len(y_true)) * abs(acc - conf)

    rel_data = {
        "bins_left": bins[:-1],
        "bins_right": bins[1:],
        "bin_accuracy": bin_acc,
        "bin_confidence": bin_conf,
        "bin_counts": bin_counts,
    }
    return float(ece), rel_data


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    ece_bins: int = 15,
) -> dict:
    ece, rel_data = expected_calibration_error(y_true, probs, n_bins=ece_bins)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "nll": multiclass_nll(y_true, probs),
        "ece": ece,
        "brier_score": multiclass_brier_score(y_true, probs, num_classes=num_classes),
        "reliability": {
            "bins_left": rel_data["bins_left"].tolist(),
            "bins_right": rel_data["bins_right"].tolist(),
            "bin_accuracy": rel_data["bin_accuracy"].tolist(),
            "bin_confidence": rel_data["bin_confidence"].tolist(),
            "bin_counts": rel_data["bin_counts"].tolist(),
        },
    }


def corrupted_detection_auroc(clean_scores: np.ndarray, corrupted_scores: np.ndarray) -> float:
    y = np.concatenate(
        [np.zeros_like(clean_scores, dtype=np.int64), np.ones_like(corrupted_scores, dtype=np.int64)]
    )
    s = np.concatenate([clean_scores, corrupted_scores])
    return float(roc_auc_score(y, s))
