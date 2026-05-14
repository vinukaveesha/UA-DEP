from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import corrupted_detection_auroc
from .plotting import plot_uncertainty_distributions
from .uncertainty import average_probabilities, compute_uncertainty_scores
from .utils import ensure_dir, sanitize_id


BASES = np.array(list("ACGT"))


@dataclass
class SimulationConfig:
    substitution_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    truncation_rates: tuple[float, ...] = (0.10, 0.25, 0.50)
    chimeric_rates: tuple[float, ...] = (0.50,)
    random_seed: int = 123


def _sanitize_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float64)
    fill = float(np.median(arr[finite]))
    return np.where(finite, arr, fill)


def _substitute_sequence(seq: str, rate: float, rng: np.random.Generator) -> str:
    n = len(seq)
    if n == 0:
        return seq
    k = max(1, int(round(rate * n)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    arr = np.array(list(seq), dtype="U1")
    for pos in idx:
        curr = arr[pos]
        if curr not in {"A", "C", "G", "T"}:
            arr[pos] = rng.choice(BASES)
            continue
        choices = BASES[BASES != curr]
        arr[pos] = rng.choice(choices)
    return "".join(arr.tolist())


def _truncate_sequence(seq: str, rate: float, rng: np.random.Generator) -> str:
    n = len(seq)
    if n < 2:
        return seq
    keep = max(1, int(round((1.0 - rate) * n)))
    keep = min(keep, n)
    if keep == n:
        return seq
    start = int(rng.integers(0, n - keep + 1))
    return seq[start : start + keep]


def _make_chimeric(
    seq: str,
    seq_label: int,
    all_sequences: list[str],
    all_labels: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> str:
    diff_pool = np.where(all_labels != seq_label)[0]
    if len(diff_pool) == 0:
        return seq

    donor_idx = int(rng.choice(diff_pool))
    donor = all_sequences[donor_idx]
    if not donor:
        return seq

    ratio = float(np.clip(rate, 0.1, 0.9))
    split_a = max(1, min(len(seq) - 1, int(round(len(seq) * ratio))))
    split_b = max(1, min(len(donor) - 1, int(round(len(donor) * (1.0 - ratio)))))
    return seq[:split_a] + donor[split_b:]


def _corrupt_sequences(
    sequences: list[str],
    labels: np.ndarray,
    corruption_type: str,
    rate: float,
    rng: np.random.Generator,
) -> list[str]:
    out = []
    for seq, y in zip(sequences, labels):
        if corruption_type == "substitution":
            out.append(_substitute_sequence(seq, rate=rate, rng=rng))
        elif corruption_type == "truncation":
            out.append(_truncate_sequence(seq, rate=rate, rng=rng))
        elif corruption_type == "chimeric":
            out.append(
                _make_chimeric(
                    seq=seq,
                    seq_label=int(y),
                    all_sequences=sequences,
                    all_labels=labels,
                    rate=rate,
                    rng=rng,
                )
            )
        else:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")
    return out


def run_cyber_exposure_simulation(
    test_sequences: list[str],
    y_test: np.ndarray,
    sample_ids: list[str],
    clean_uncertainty_scores: dict[str, np.ndarray],
    tau: float,
    uncertainty_metric: str,
    predict_member_probs_fn: Callable[[list[str]], np.ndarray],
    output_dir: Path,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SimulationConfig()
    ensure_dir(output_dir)

    rng = np.random.default_rng(cfg.random_seed)

    scenarios: list[tuple[str, float]] = []
    scenarios += [("substitution", r) for r in cfg.substitution_rates]
    scenarios += [("truncation", r) for r in cfg.truncation_rates]
    scenarios += [("chimeric", r) for r in cfg.chimeric_rates]

    clean_metric_scores = _sanitize_scores(clean_uncertainty_scores[uncertainty_metric])

    rows = []
    scenario_predictions_dir = ensure_dir(output_dir / "predictions")
    scenario_plots_dir = ensure_dir(output_dir / "plots")

    for ctype, rate in tqdm(scenarios, desc="Cyber simulations"):
        corrupted_sequences = _corrupt_sequences(
            sequences=test_sequences,
            labels=y_test,
            corruption_type=ctype,
            rate=rate,
            rng=rng,
        )

        member_probs = predict_member_probs_fn(corrupted_sequences)  # [M, N, C]
        avg_probs = average_probabilities(member_probs)
        y_pred = np.argmax(avg_probs, axis=1)

        scores = compute_uncertainty_scores(member_probs)
        metric_scores = _sanitize_scores(scores[uncertainty_metric])

        auroc = corrupted_detection_auroc(clean_metric_scores, metric_scores)

        accepted = metric_scores <= tau
        rejection_rate = float(np.mean(~accepted))
        accepted_acc = float(np.mean(y_pred[accepted] == y_test[accepted])) if np.any(accepted) else float("nan")

        tag = sanitize_id(f"{ctype}_{rate:.2f}")

        pred_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "true_label": y_test,
                "pred_label": y_pred,
                "confidence": np.max(avg_probs, axis=1),
                "accepted": accepted,
                "uncertainty": metric_scores,
                "predictive_entropy": scores["predictive_entropy"],
                "variation_ratio": scores["variation_ratio"],
                "ensemble_probability_variance": scores["ensemble_probability_variance"],
                "one_minus_confidence": scores["one_minus_confidence"],
                "corruption_type": ctype,
                "corruption_rate": rate,
            }
        )
        pred_df.to_csv(scenario_predictions_dir / f"{tag}_predictions.csv", index=False)

        plot_uncertainty_distributions(
            clean_scores=clean_metric_scores,
            corrupted_scores=metric_scores,
            title=f"{ctype} @ {rate:.0%}",
            out_path=scenario_plots_dir / f"{tag}_uncertainty_distribution.png",
        )

        rows.append(
            {
                "corruption_type": ctype,
                "corruption_rate": rate,
                "auroc_uncertainty_detection": auroc,
                "rejection_rate": rejection_rate,
                "accepted_accuracy": accepted_acc,
                "mean_uncertainty_clean": float(np.mean(clean_metric_scores)),
                "mean_uncertainty_corrupted": float(np.mean(metric_scores)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["corruption_type", "corruption_rate"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "cyber_simulation_summary.csv", index=False)
    return summary_df
