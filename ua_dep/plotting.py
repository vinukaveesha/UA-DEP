from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _finite_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0], dtype=np.float64)
    return arr


def _stable_hist_bins(a: np.ndarray, b: np.ndarray, max_bins: int = 30) -> np.ndarray:
    merged = np.concatenate([a, b]).astype(np.float64, copy=False)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return np.linspace(0.0, 1.0, 11, dtype=np.float64)

    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(0.0, 1.0, 11, dtype=np.float64)

    # NumPy 2.x can fail for degenerate/tiny spans with many bins; use explicit safe edges.
    span = hi - lo
    if span <= 0.0 or span <= np.finfo(np.float64).eps * max(1.0, abs(lo), abs(hi)):
        center = (lo + hi) / 2.0
        delta = max(abs(center) * 1e-6, 1e-6)
        lo = center - delta
        hi = center + delta

    bins = min(max_bins, max(10, int(np.sqrt(merged.size))))
    edges = np.linspace(lo, hi, bins + 1, dtype=np.float64)
    if np.any(edges[:-1] >= edges[1:]):
        center = (lo + hi) / 2.0
        delta = max(abs(center) * 1e-3, 1e-3)
        edges = np.linspace(center - delta, center + delta, 11, dtype=np.float64)
    return edges


def plot_reliability_diagram(reliability: dict, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    bin_acc = np.array(reliability["bin_accuracy"], dtype=np.float32)
    bin_conf = np.array(reliability["bin_confidence"], dtype=np.float32)
    bins_left = np.array(reliability["bins_left"], dtype=np.float32)
    bins_right = np.array(reliability["bins_right"], dtype=np.float32)
    width = bins_right - bins_left

    plt.figure(figsize=(6, 5))
    plt.bar(bins_left, bin_acc, width=width, align="edge", alpha=0.7, edgecolor="black", label="Accuracy")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(bin_conf, bin_acc, "o-", color="red", label="Bin mean")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confidence_histogram(probs: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    conf = np.max(probs, axis=1)
    conf = np.asarray(conf, dtype=np.float64)
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        conf = np.array([0.0], dtype=np.float64)
    conf = np.clip(conf, 0.0, 1.0)
    plt.figure(figsize=(6, 4))
    plt.hist(conf, bins=np.linspace(0.0, 1.0, 21), color="#3a7", edgecolor="black", alpha=0.8)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coverage_accuracy_curve(curve_df: pd.DataFrame, tau: float, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(curve_df["coverage"], curve_df["accepted_accuracy"], color="#247", lw=2)
    match = curve_df.iloc[(curve_df["tau"] - tau).abs().argsort()[:1]]
    if not match.empty:
        cov = float(match.iloc[0]["coverage"])
        acc = float(match.iloc[0]["accepted_accuracy"])
        plt.scatter([cov], [acc], color="red", label=f"tau={tau:.4f}")
        plt.legend()
    plt.xlabel("Coverage")
    plt.ylabel("Accepted Accuracy")
    plt.title("Coverage-Accuracy Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_uncertainty_distributions(
    clean_scores: np.ndarray,
    corrupted_scores: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    clean = _finite_1d(clean_scores)
    corrupt = _finite_1d(corrupted_scores)
    bins = _stable_hist_bins(clean, corrupt, max_bins=30)

    plt.figure(figsize=(6, 4))
    plt.hist(clean, bins=bins, density=True, alpha=0.6, label="Clean", color="#2a6")
    plt.hist(corrupt, bins=bins, density=True, alpha=0.6, label="Corrupted", color="#c44")
    plt.xlabel("Uncertainty score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
