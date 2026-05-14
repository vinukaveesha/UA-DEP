#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as:
#   python3 scripts/run_ua_dep.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ua_dep.config import DataConfig, PipelinePaths, RejectionConfig, TrainConfig
from ua_dep.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UA-DEP: Uncertainty-Aware Deep Ensemble Pipeline")

    parser.add_argument("--raw-dir", type=Path, default=Path("sars_cov_2_10class_dataset/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--min-length", type=int, default=25000)
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--uncertainty-metric",
        type=str,
        default="ensemble_probability_variance",
        choices=[
            "predictive_entropy",
            "variation_ratio",
            "ensemble_probability_variance",
            "one_minus_confidence",
        ],
    )
    parser.add_argument(
        "--threshold-selection-method",
        type=str,
        default="coverage_accuracy",
        choices=["coverage_accuracy", "roc"],
    )
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--tau-grid-size", type=int, default=200)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        min_length=args.min_length,
        max_n_fraction=args.max_n_fraction,
        k=args.k,
        random_state=args.split_seed,
    )

    seeds = [args.base_seed + i for i in range(args.ensemble_size)]
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        seeds=seeds,
    )

    reject_cfg = RejectionConfig(
        metric=args.uncertainty_metric,
        min_coverage=args.min_coverage,
        tau_grid_size=args.tau_grid_size,
        threshold_selection_method=args.threshold_selection_method,
    )

    pipeline_cfg = PipelineConfig(
        data=data_cfg,
        train=train_cfg,
        reject=reject_cfg,
        paths=PipelinePaths(output_root=args.output_dir),
        device=args.device,
    )

    result = run_pipeline(pipeline_cfg)
    summary = result["summary"]

    print("UA-DEP pipeline complete")
    print(f"Samples: {summary['num_samples']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Test macro F1: {summary['test_macro_f1']:.4f}")
    print(f"Selected tau: {summary['tau']:.6f}")


if __name__ == "__main__":
    main()
