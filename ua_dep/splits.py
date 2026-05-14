from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ensure_dir


def create_stratified_splits(
    labels_df: pd.DataFrame,
    random_state: int,
    processed_dir: Path,
) -> dict[str, np.ndarray]:
    ensure_dir(processed_dir)

    y = labels_df["class_index"].to_numpy()
    idx = np.arange(len(labels_df))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    y_temp = y[temp_idx]
    cal_idx, temp2_idx = train_test_split(
        temp_idx,
        test_size=2 / 3,
        random_state=random_state,
        stratify=y_temp,
    )

    y_temp2 = y[temp2_idx]
    val_idx, test_idx = train_test_split(
        temp2_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp2,
    )

    split_map = {
        "train": np.sort(train_idx),
        "calibration": np.sort(cal_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }

    split_col = np.full(len(labels_df), "", dtype=object)
    for split_name, split_idx in split_map.items():
        split_col[split_idx] = split_name

    if np.any(split_col == ""):
        raise RuntimeError("Some samples were not assigned to a split.")

    split_df = labels_df.copy()
    split_df["split"] = split_col
    split_df.to_csv(processed_dir / "splits.csv", index=False)

    np.savez_compressed(
        processed_dir / "split_indices.npz",
        train=split_map["train"],
        calibration=split_map["calibration"],
        validation=split_map["validation"],
        test=split_map["test"],
    )

    return split_map
