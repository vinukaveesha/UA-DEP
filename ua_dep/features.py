from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from .utils import ensure_dir


ASCII_MAP = np.full(256, -1, dtype=np.int16)
ASCII_MAP[ord("A")] = 0
ASCII_MAP[ord("C")] = 1
ASCII_MAP[ord("G")] = 2
ASCII_MAP[ord("T")] = 3


def build_kmer_vocab(k: int) -> list[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]


def kmer_frequency_vector(seq: str, k: int) -> np.ndarray:
    num_kmers = 4**k
    out = np.zeros(num_kmers, dtype=np.float32)
    n = len(seq)
    if n < k:
        return out

    seq_int = ASCII_MAP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    m = n - k + 1
    windows = np.stack([seq_int[i : i + m] for i in range(k)], axis=0)  # [k, m]
    valid = np.all(windows >= 0, axis=0)
    if np.any(valid):
        idx = np.zeros(m, dtype=np.int64)
        for i in range(k):
            idx = (idx * 4) + windows[i]
        idx = idx[valid]
        out = np.bincount(idx, minlength=num_kmers).astype(np.float32)
        out /= float(valid.sum())
    return out


def extract_kmer_features(
    fasta_path: Path,
    labels_df: pd.DataFrame,
    k: int,
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], Path]:
    ensure_dir(processed_dir)

    feature_path = processed_dir / f"features_k{k}.npz"
    vocab_path = processed_dir / f"kmer_vocab_k{k}.txt"

    seq_map = {record.id: str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")}

    X = np.zeros((len(labels_df), 4**k), dtype=np.float32)
    y = labels_df["class_index"].to_numpy(dtype=np.int64)

    for i, sample_id in enumerate(tqdm(labels_df["sample_id"], desc=f"k-mer(k={k})")):
        seq = seq_map.get(sample_id)
        if seq is None:
            raise KeyError(f"Sample ID {sample_id} from labels not found in FASTA.")
        X[i] = kmer_frequency_vector(seq=seq, k=k)

    vocab = build_kmer_vocab(k)

    np.savez_compressed(
        feature_path,
        X=X,
        y=y,
        sample_ids=labels_df["sample_id"].to_numpy(),
    )

    with vocab_path.open("w", encoding="utf-8") as f:
        for token in vocab:
            f.write(f"{token}\n")

    return X, y, vocab, feature_path
