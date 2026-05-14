from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


VALID_BASES = set("ACGTN")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "models": root / "models",
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "plots": root / "plots",
        "cyber_simulation": root / "cyber_simulation",
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    return "".join(ch for ch in seq if ch in VALID_BASES)


def n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    return seq.count("N") / len(seq)


def is_valid_sequence(seq: str, min_length: int, max_n_frac: float) -> bool:
    return len(seq) >= min_length and n_fraction(seq) <= max_n_frac


def sanitize_id(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed)
