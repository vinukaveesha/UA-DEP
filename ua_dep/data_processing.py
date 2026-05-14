from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from .utils import clean_sequence, ensure_dir, is_valid_sequence, n_fraction, sanitize_id, write_json


@dataclass
class RecordSample:
    sample_id: str
    original_id: str
    class_name: str
    class_index: int
    source_file: str
    length: int
    n_fraction: float
    sequence: str


def _find_fasta_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset path does not exist: {raw_dir}. "
            "Pass the directory that contains class folders."
        )

    fasta_files = sorted(raw_dir.glob("*/ncbi_dataset/data/genomic.fna"))
    if fasta_files:
        return fasta_files
    # Fallback for alternative nested layouts.
    fasta_files = sorted(raw_dir.rglob("genomic.fna"))
    if fasta_files:
        return fasta_files

    # Provide useful hints from nearby locations to make path issues obvious.
    nearby = sorted(raw_dir.parent.rglob("genomic.fna")) if raw_dir.parent.exists() else []
    hint = ""
    if nearby:
        preview = ", ".join(str(p) for p in nearby[:5])
        if len(nearby) > 5:
            preview += ", ..."
        hint = f" Nearby genomic.fna files found under parent: {preview}"

    raise FileNotFoundError(
        f"No genomic.fna files found under: {raw_dir}. "
        "Expected layout: <raw_dir>/<class>/ncbi_dataset/data/genomic.fna."
        + hint
    )


def _extract_class_name(raw_dir: Path, fasta_path: Path) -> str:
    rel = fasta_path.relative_to(raw_dir)
    class_dir = rel.parts[0]
    # Convert e.g., alpha_1000 -> alpha
    if class_dir.endswith("_1000"):
        return class_dir[:-5]
    return class_dir


def _iter_samples(
    raw_dir: Path,
    min_length: int,
    max_n_fraction: float,
) -> Iterator[RecordSample]:
    fasta_files = _find_fasta_files(raw_dir)

    class_names = sorted({_extract_class_name(raw_dir, fp) for fp in fasta_files})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for fasta_path in tqdm(fasta_files, desc="Class FASTA files"):
        class_name = _extract_class_name(raw_dir, fasta_path)
        class_index = class_to_idx[class_name]
        for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
            cleaned = clean_sequence(str(record.seq))
            if not is_valid_sequence(cleaned, min_length=min_length, max_n_frac=max_n_fraction):
                continue

            raw_id = record.id or f"seq_{i}"
            sample_id = sanitize_id(f"{class_name}__{raw_id}")
            yield RecordSample(
                sample_id=sample_id,
                original_id=raw_id,
                class_name=class_name,
                class_index=class_index,
                source_file=str(fasta_path),
                length=len(cleaned),
                n_fraction=n_fraction(cleaned),
                sequence=cleaned,
            )


def build_processed_dataset(
    raw_dir: Path,
    processed_dir: Path,
    min_length: int,
    max_n_fraction: float,
) -> tuple[pd.DataFrame, Path, Path]:
    ensure_dir(processed_dir)

    combined_fasta = processed_dir / "combined.fasta"
    labels_csv = processed_dir / "labels.csv"
    stats_json = processed_dir / "build_stats.json"
    class_map_json = processed_dir / "class_mapping.json"

    rows: list[dict] = []
    seen = set()
    with combined_fasta.open("w", encoding="utf-8") as fasta_out:
        for sample in tqdm(
            _iter_samples(raw_dir, min_length=min_length, max_n_fraction=max_n_fraction),
            desc="Build processed dataset",
        ):
            sample_id = sample.sample_id
            suffix = 1
            while sample_id in seen:
                sample_id = f"{sample.sample_id}_{suffix}"
                suffix += 1
            seen.add(sample_id)

            rows.append(
                {
                    "sample_id": sample_id,
                    "original_id": sample.original_id,
                    "class_name": sample.class_name,
                    "class_index": sample.class_index,
                    "source_file": sample.source_file,
                    "length": sample.length,
                    "n_fraction": sample.n_fraction,
                }
            )

            fasta_out.write(f">{sample_id}\n")
            seq = sample.sequence
            for i in range(0, len(seq), 80):
                fasta_out.write(seq[i : i + 80] + "\n")

    if not rows:
        raise RuntimeError(
            "No valid sequences found after filtering. "
            "Check min_length/max_n_fraction thresholds or dataset path."
        )

    labels_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    labels_df.to_csv(labels_csv, index=False)

    class_mapping = (
        labels_df[["class_name", "class_index"]]
        .drop_duplicates()
        .sort_values("class_index")
        .set_index("class_name")["class_index"]
        .to_dict()
    )

    class_counts = labels_df["class_name"].value_counts().sort_index().to_dict()
    stats = {
        "num_sequences": int(len(labels_df)),
        "num_classes": int(labels_df["class_name"].nunique()),
        "min_length": int(labels_df["length"].min()),
        "max_length": int(labels_df["length"].max()),
        "mean_length": float(labels_df["length"].mean()),
        "mean_n_fraction": float(labels_df["n_fraction"].mean()),
        "class_counts": class_counts,
        "filters": {
            "min_length": min_length,
            "max_n_fraction": max_n_fraction,
        },
    }

    write_json(stats_json, stats)
    write_json(class_map_json, class_mapping)

    return labels_df, combined_fasta, labels_csv
