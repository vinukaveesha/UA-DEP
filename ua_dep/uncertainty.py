from __future__ import annotations

import numpy as np


def _sanitize_member_probs(member_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(member_probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)

    row_sums = probs.sum(axis=2, keepdims=True)
    bad = row_sums <= 0.0
    row_sums = np.where(bad, 1.0, row_sums)
    probs = probs / row_sums
    if np.any(bad):
        probs[bad.squeeze(-1)] = 1.0 / probs.shape[2]
    return probs


def _sanitize_avg_probs(avg_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(avg_probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    bad = row_sums <= 0.0
    row_sums = np.where(bad, 1.0, row_sums)
    probs = probs / row_sums
    if np.any(bad):
        probs[bad.squeeze(-1)] = 1.0 / probs.shape[1]
    return probs


def average_probabilities(member_probs: np.ndarray) -> np.ndarray:
    probs = _sanitize_member_probs(member_probs)
    return np.mean(probs, axis=0)


def predictive_entropy(avg_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = _sanitize_avg_probs(avg_probs)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def variation_ratio(member_probs: np.ndarray) -> np.ndarray:
    member_probs = _sanitize_member_probs(member_probs)
    member_preds = np.argmax(member_probs, axis=2)  # [M, N]
    m, n = member_preds.shape
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        values, counts = np.unique(member_preds[:, i], return_counts=True)
        out[i] = 1.0 - (counts.max() / float(m))
    return out


def ensemble_probability_variance(member_probs: np.ndarray) -> np.ndarray:
    # Mean variance across classes.
    member_probs = _sanitize_member_probs(member_probs)
    return np.var(member_probs, axis=0).mean(axis=1)


def one_minus_confidence(avg_probs: np.ndarray) -> np.ndarray:
    avg_probs = _sanitize_avg_probs(avg_probs)
    return 1.0 - np.max(avg_probs, axis=1)


def compute_uncertainty_scores(member_probs: np.ndarray) -> dict[str, np.ndarray]:
    avg_probs = average_probabilities(member_probs)
    return {
        "predictive_entropy": predictive_entropy(avg_probs),
        "variation_ratio": variation_ratio(member_probs),
        "ensemble_probability_variance": ensemble_probability_variance(member_probs),
        "one_minus_confidence": one_minus_confidence(avg_probs),
    }
