from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("sars_cov_2_10class_dataset/raw")
    processed_dir: Path = Path("processed")
    min_length: int = 25000
    max_n_fraction: float = 0.05
    k: int = 3
    random_state: int = 42


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    ensemble_size: int = 10
    hidden_dim: int = 128
    dropout: float = 0.2
    label_smoothing: float = 0.0
    num_workers: int = 0
    seeds: list[int] = field(default_factory=lambda: list(range(10)))


@dataclass
class RejectionConfig:
    metric: str = "ensemble_probability_variance"
    min_coverage: float = 0.80
    tau_grid_size: int = 200
    threshold_selection_method: str = "coverage_accuracy"


@dataclass
class PipelinePaths:
    output_root: Path = Path("outputs")

    @property
    def model_dir(self) -> Path:
        return self.output_root / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.output_root / "predictions"

    @property
    def plots_dir(self) -> Path:
        return self.output_root / "plots"

    @property
    def cyber_dir(self) -> Path:
        return self.output_root / "cyber_simulation"
