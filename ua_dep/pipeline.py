from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio import SeqIO

from .calibration import apply_temperature, fit_temperature_scaling, softmax_numpy
from .config import DataConfig, PipelinePaths, RejectionConfig, TrainConfig
from .cyber_simulation import SimulationConfig, run_cyber_exposure_simulation
from .data_processing import build_processed_dataset
from .features import extract_kmer_features, kmer_frequency_vector
from .metrics import classification_metrics
from .plotting import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_coverage_accuracy_curve,
    plot_reliability_diagram,
)
from .rejection import (
    apply_rejection_gate,
    choose_tau,
    choose_tau_roc,
    coverage_accuracy_curve,
    roc_error_detection_curve,
)
from .train import predict_logits, train_single_model
from .uncertainty import average_probabilities, compute_uncertainty_scores
from .utils import ensure_dir, ensure_output_dirs, set_global_seed, write_json


@dataclass
class PipelineConfig:
    data: DataConfig
    train: TrainConfig
    reject: RejectionConfig
    paths: PipelinePaths
    device: str = "cpu"


def _load_sequences_in_label_order(fasta_path: Path, labels_df: pd.DataFrame) -> list[str]:
    seq_map = {record.id: str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")}
    sequences = []
    for sid in labels_df["sample_id"]:
        seq = seq_map.get(sid)
        if seq is None:
            raise KeyError(f"Missing sequence for sample_id={sid}")
        sequences.append(seq)
    return sequences


def _subset(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[idx], y[idx]


def _fit_feature_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_feature_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _split_stats(labels_df: pd.DataFrame, split_map: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, idx in split_map.items():
        split_df = labels_df.iloc[idx]
        out[split_name] = {
            "count": int(len(split_df)),
            "class_counts": split_df["class_name"].value_counts().sort_index().to_dict(),
        }
    return out


def _bootstrap_indices(num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(num_samples, size=num_samples, replace=True).astype(np.int64)


def _predict_member_probs(
    models: list,
    temperatures: list[float],
    X: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    member_probs = []
    for model, temp in zip(models, temperatures):
        logits = predict_logits(model, X=X, batch_size=batch_size, num_workers=num_workers, device=device)
        logits = apply_temperature(logits, temp)
        probs = softmax_numpy(logits)
        member_probs.append(probs)
    return np.stack(member_probs, axis=0)


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    set_global_seed(cfg.data.random_state)

    ensure_dir(cfg.data.processed_dir)
    output_dirs = ensure_output_dirs(cfg.paths.output_root)

    # 1-4: combine FASTA + labels and clean/filter sequences.
    labels_df, combined_fasta, _ = build_processed_dataset(
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        min_length=cfg.data.min_length,
        max_n_fraction=cfg.data.max_n_fraction,
    )

    # 5: k-mer feature extraction.
    X, y, _, _ = extract_kmer_features(
        fasta_path=combined_fasta,
        labels_df=labels_df,
        k=cfg.data.k,
        processed_dir=cfg.data.processed_dir,
    )

    # 6: stratified splits.
    from .splits import create_stratified_splits

    split_map = create_stratified_splits(
        labels_df=labels_df,
        random_state=cfg.data.random_state,
        processed_dir=cfg.data.processed_dir,
    )

    split_stats = _split_stats(labels_df, split_map)
    write_json(Path(output_dirs["metrics"]) / "split_summary.json", split_stats)

    class_df = labels_df[["class_index", "class_name"]].drop_duplicates().sort_values("class_index")
    class_names = class_df["class_name"].tolist()
    num_classes = len(class_names)

    sequences_all = _load_sequences_in_label_order(combined_fasta, labels_df)

    train_idx = split_map["train"]
    cal_idx = split_map["calibration"]
    val_idx = split_map["validation"]
    test_idx = split_map["test"]

    X_train, y_train = _subset(X, y, train_idx)
    X_cal, y_cal = _subset(X, y, cal_idx)
    X_val, y_val = _subset(X, y, val_idx)
    X_test, y_test = _subset(X, y, test_idx)

    scaler_mean, scaler_std = _fit_feature_scaler(X_train)
    X_train = _apply_feature_scaler(X_train, scaler_mean, scaler_std)
    X_cal = _apply_feature_scaler(X_cal, scaler_mean, scaler_std)
    X_val = _apply_feature_scaler(X_val, scaler_mean, scaler_std)
    X_test = _apply_feature_scaler(X_test, scaler_mean, scaler_std)

    np.savez_compressed(
        cfg.data.processed_dir / f"feature_scaler_k{cfg.data.k}.npz",
        mean=scaler_mean,
        std=scaler_std,
    )

    test_sample_ids = labels_df.iloc[test_idx]["sample_id"].tolist()
    test_sequences = [sequences_all[i] for i in test_idx]

    # 7-9: train single CNN and deep ensemble.
    seeds = cfg.train.seeds[: cfg.train.ensemble_size]
    if len(seeds) != cfg.train.ensemble_size:
        raise ValueError("Not enough seeds provided for requested ensemble_size.")

    models = []
    temperatures = []

    bootstrap_records = []
    bootstrap_indices_payload: dict[str, np.ndarray] = {}

    single_seed = seeds[0]
    single_bootstrap_idx = _bootstrap_indices(num_samples=len(X_train), seed=single_seed)
    single_unique = int(np.unique(single_bootstrap_idx).size)
    bootstrap_records.append(
        {
            "member_index": 0,
            "seed": int(single_seed),
            "replace": True,
            "sample_size": int(len(single_bootstrap_idx)),
            "unique_samples": single_unique,
            "duplicate_samples": int(len(single_bootstrap_idx) - single_unique),
            "out_of_bag_samples": int(len(X_train) - single_unique),
        }
    )
    bootstrap_indices_payload["member_00"] = single_bootstrap_idx

    single_model_path = Path(output_dirs["models"]) / f"single_cnn_seed{single_seed}.pt"
    single_hist_path = Path(output_dirs["metrics"]) / "single_cnn_history.json"
    single_model, _ = train_single_model(
        X_train=X_train[single_bootstrap_idx],
        y_train=y_train[single_bootstrap_idx],
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        config=cfg.train,
        seed=single_seed,
        model_out_path=single_model_path,
        history_out_path=single_hist_path,
        device=cfg.device,
    )

    ensemble_val_probs = []
    ensemble_test_probs = []

    for i, seed in enumerate(seeds):
        if i == 0:
            model = single_model
            model_path = single_model_path
        else:
            member_bootstrap_idx = _bootstrap_indices(num_samples=len(X_train), seed=seed)
            member_unique = int(np.unique(member_bootstrap_idx).size)
            bootstrap_records.append(
                {
                    "member_index": i,
                    "seed": int(seed),
                    "replace": True,
                    "sample_size": int(len(member_bootstrap_idx)),
                    "unique_samples": member_unique,
                    "duplicate_samples": int(len(member_bootstrap_idx) - member_unique),
                    "out_of_bag_samples": int(len(X_train) - member_unique),
                }
            )
            bootstrap_indices_payload[f"member_{i:02d}"] = member_bootstrap_idx

            model_path = Path(output_dirs["models"]) / f"ensemble_member_{i:02d}_seed_{seed}.pt"
            hist_path = Path(output_dirs["metrics"]) / f"ensemble_member_{i:02d}_history.json"
            model, _ = train_single_model(
                X_train=X_train[member_bootstrap_idx],
                y_train=y_train[member_bootstrap_idx],
                X_val=X_val,
                y_val=y_val,
                num_classes=num_classes,
                config=cfg.train,
                seed=seed,
                model_out_path=model_path,
                history_out_path=hist_path,
                device=cfg.device,
            )

        # 10: temperature scaling per member.
        logits_cal = predict_logits(
            model,
            X=X_cal,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            device=cfg.device,
        )
        temp = fit_temperature_scaling(logits_cal, y_cal, device=cfg.device)
        temperatures.append(temp)

        logits_val = predict_logits(
            model,
            X=X_val,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            device=cfg.device,
        )
        logits_test = predict_logits(
            model,
            X=X_test,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            device=cfg.device,
        )

        val_probs = softmax_numpy(apply_temperature(logits_val, temp))
        test_probs = softmax_numpy(apply_temperature(logits_test, temp))
        ensemble_val_probs.append(val_probs)
        ensemble_test_probs.append(test_probs)

        models.append(model)

        write_json(
            Path(output_dirs["metrics"]) / f"ensemble_member_{i:02d}_calibration.json",
            {
                "member_index": i,
                "seed": seed,
                "temperature": temp,
                "model_path": str(model_path),
                "bootstrap_sampling": True,
            },
        )

    write_json(
        Path(output_dirs["metrics"]) / "ensemble_bootstrap_sampling.json",
        {
            "sampling": "with_replacement",
            "train_size": int(len(X_train)),
            "members": bootstrap_records,
        },
    )
    np.savez_compressed(
        Path(output_dirs["metrics"]) / "ensemble_bootstrap_indices.npz",
        **bootstrap_indices_payload,
    )

    write_json(
        Path(output_dirs["metrics"]) / "ensemble_temperatures.json",
        {
            "seeds": seeds,
            "temperatures": temperatures,
        },
    )

    ensemble_val_probs = np.stack(ensemble_val_probs, axis=0)
    ensemble_test_probs = np.stack(ensemble_test_probs, axis=0)

    # 11-13: ensemble prediction, uncertainty, rejection threshold.
    avg_val_probs = average_probabilities(ensemble_val_probs)
    y_pred_val = np.argmax(avg_val_probs, axis=1)
    val_uncertainties = compute_uncertainty_scores(ensemble_val_probs)

    rejection_metric = cfg.reject.metric
    if rejection_metric not in val_uncertainties:
        raise ValueError(f"Unknown rejection metric: {rejection_metric}")

    curve_df = coverage_accuracy_curve(
        uncertainty=val_uncertainties[rejection_metric],
        y_true=y_val,
        y_pred=y_pred_val,
        grid_size=cfg.reject.tau_grid_size,
    )
    selection_method = cfg.reject.threshold_selection_method
    tau_selection_note: str | None = None

    if selection_method == "coverage_accuracy":
        tau = choose_tau(curve_df, min_coverage=cfg.reject.min_coverage)
    elif selection_method == "roc":
        roc_df = roc_error_detection_curve(
            uncertainty=val_uncertainties[rejection_metric],
            y_true=y_val,
            y_pred=y_pred_val,
        )
        roc_df.to_csv(Path(output_dirs["metrics"]) / "roc_error_detection_curve_validation.csv", index=False)
        if roc_df.empty:
            tau_selection_note = (
                "roc_threshold_selection_fallback_to_coverage_accuracy_"
                "because_validation_error_labels_have_one_class"
            )
            tau = choose_tau(curve_df, min_coverage=cfg.reject.min_coverage)
        else:
            tau = choose_tau_roc(roc_df, min_coverage=cfg.reject.min_coverage)
    else:
        raise ValueError(f"Unknown threshold_selection_method: {selection_method}")

    curve_df.to_csv(Path(output_dirs["metrics"]) / "coverage_accuracy_curve_validation.csv", index=False)

    rejection_gate_payload = {
        "metric": rejection_metric,
        "tau": tau,
        "min_coverage": cfg.reject.min_coverage,
        "threshold_selection_method": selection_method,
    }
    if tau_selection_note is not None:
        rejection_gate_payload["threshold_selection_note"] = tau_selection_note

    write_json(
        Path(output_dirs["metrics"]) / "rejection_gate.json",
        rejection_gate_payload,
    )

    # 14: clean test metrics + plots.
    avg_test_probs = average_probabilities(ensemble_test_probs)
    y_pred_test = np.argmax(avg_test_probs, axis=1)
    confidence_test = np.max(avg_test_probs, axis=1)
    test_uncertainties = compute_uncertainty_scores(ensemble_test_probs)

    accepted_test = apply_rejection_gate(test_uncertainties[rejection_metric], tau=tau)

    clean_metrics = classification_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        probs=avg_test_probs,
        num_classes=num_classes,
    )

    clean_metrics["rejection"] = {
        "metric": rejection_metric,
        "threshold_selection_method": selection_method,
        "tau": float(tau),
        "coverage": float(np.mean(accepted_test)),
        "rejection_rate": float(np.mean(~accepted_test)),
        "accepted_accuracy": float(np.mean(y_pred_test[accepted_test] == y_test[accepted_test]))
        if np.any(accepted_test)
        else float("nan"),
    }
    if tau_selection_note is not None:
        clean_metrics["rejection"]["threshold_selection_note"] = tau_selection_note

    write_json(Path(output_dirs["metrics"]) / "test_clean_metrics.json", clean_metrics)

    cm = np.array(clean_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names=class_names, out_path=Path(output_dirs["plots"]) / "confusion_matrix.png")
    plot_reliability_diagram(
        clean_metrics["reliability"],
        out_path=Path(output_dirs["plots"]) / "reliability_diagram.png",
    )
    plot_confidence_histogram(avg_test_probs, out_path=Path(output_dirs["plots"]) / "confidence_histogram.png")
    plot_coverage_accuracy_curve(
        curve_df,
        tau=tau,
        out_path=Path(output_dirs["plots"]) / "coverage_accuracy_curve_validation.png",
    )

    test_curve_df = coverage_accuracy_curve(
        uncertainty=test_uncertainties[rejection_metric],
        y_true=y_test,
        y_pred=y_pred_test,
        grid_size=cfg.reject.tau_grid_size,
    )
    test_curve_df.to_csv(Path(output_dirs["metrics"]) / "coverage_accuracy_curve_test.csv", index=False)
    plot_coverage_accuracy_curve(
        test_curve_df,
        tau=tau,
        out_path=Path(output_dirs["plots"]) / "coverage_accuracy_curve_test.png",
    )

    pred_df = pd.DataFrame(
        {
            "sample_id": test_sample_ids,
            "true_label": y_test,
            "pred_label": y_pred_test,
            "confidence": confidence_test,
            "accepted": accepted_test,
            "predictive_entropy": test_uncertainties["predictive_entropy"],
            "variation_ratio": test_uncertainties["variation_ratio"],
            "ensemble_probability_variance": test_uncertainties["ensemble_probability_variance"],
            "one_minus_confidence": test_uncertainties["one_minus_confidence"],
        }
    )
    pred_df.to_csv(Path(output_dirs["predictions"]) / "test_predictions.csv", index=False)

    np.savez_compressed(
        Path(output_dirs["predictions"]) / "test_probabilities.npz",
        avg_probs=avg_test_probs,
        member_probs=ensemble_test_probs,
        y_true=y_test,
        y_pred=y_pred_test,
        sample_ids=np.array(test_sample_ids),
    )

    # 15-16: cyber-exposure simulation.
    def predict_member_probs_from_sequences(seqs: list[str]) -> np.ndarray:
        X_new = np.stack([kmer_frequency_vector(s, k=cfg.data.k) for s in seqs], axis=0)
        X_new = _apply_feature_scaler(X_new, scaler_mean, scaler_std)
        return _predict_member_probs(
            models=models,
            temperatures=temperatures,
            X=X_new,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            device=cfg.device,
        )

    cyber_summary = run_cyber_exposure_simulation(
        test_sequences=test_sequences,
        y_test=y_test,
        sample_ids=test_sample_ids,
        clean_uncertainty_scores=test_uncertainties,
        tau=tau,
        uncertainty_metric=rejection_metric,
        predict_member_probs_fn=predict_member_probs_from_sequences,
        output_dir=Path(output_dirs["cyber_simulation"]),
        config=SimulationConfig(),
    )

    # 17: save outputs across outputs/* already done progressively.
    write_json(
        Path(output_dirs["metrics"]) / "pipeline_config.json",
        {
            "data": {
                **asdict(cfg.data),
                "raw_dir": str(cfg.data.raw_dir),
                "processed_dir": str(cfg.data.processed_dir),
            },
            "train": asdict(cfg.train),
            "reject": asdict(cfg.reject),
            "paths": {"output_root": str(cfg.paths.output_root)},
            "device": cfg.device,
        },
    )

    summary = {
        "num_samples": int(len(labels_df)),
        "num_classes": num_classes,
        "rejection_metric": rejection_metric,
        "threshold_selection_method": selection_method,
        "tau": float(tau),
        "test_accuracy": clean_metrics["accuracy"],
        "test_macro_f1": clean_metrics["macro_f1"],
        "cyber_scenarios": int(len(cyber_summary)),
    }
    write_json(Path(output_dirs["metrics"]) / "run_summary.json", summary)

    return {
        "summary": summary,
        "test_metrics": clean_metrics,
        "cyber_summary_path": str(Path(output_dirs["cyber_simulation"]) / "cyber_simulation_summary.csv"),
    }
