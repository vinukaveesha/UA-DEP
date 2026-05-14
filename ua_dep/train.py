from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from .config import TrainConfig
from .dataset import KmerDataset
from .model import CNN1DClassifier
from .utils import ensure_dir, set_global_seed, write_json


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = KmerDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def _evaluate_loss_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    preds_all = []
    y_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu().numpy())
            y_all.append(yb.cpu().numpy())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(preds_all)
    acc = float((y_true == y_pred).mean())
    return float(np.mean(losses)) if losses else 0.0, acc


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: TrainConfig,
    seed: int,
    model_out_path: Path,
    history_out_path: Path,
    device: str = "cpu",
) -> tuple[CNN1DClassifier, dict]:
    set_global_seed(seed)
    device_t = torch.device(device)

    model = CNN1DClassifier(
        num_classes=num_classes,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device_t)

    train_loader = _make_loader(
        X=X_train,
        y=y_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = _make_loader(
        X=X_val,
        y=y_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.label_smoothing))

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    history = {
        "seed": seed,
        "config": asdict(config),
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    pbar = trange(config.epochs, desc=f"Train seed={seed}", leave=False)
    for _ in pbar:
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss, val_acc = _evaluate_loss_accuracy(model, val_loader, device_t)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.3f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= config.patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    ensure_dir(model_out_path.parent)
    torch.save(
        {
            "seed": seed,
            "num_classes": num_classes,
            "state_dict": best_state,
            "config": asdict(config),
            "best_val_loss": best_val_loss,
        },
        model_out_path,
    )
    write_json(history_out_path, history)

    return model, history


def load_model(model_path: Path, device: str = "cpu") -> tuple[CNN1DClassifier, dict]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = CNN1DClassifier(
        num_classes=int(ckpt["num_classes"]),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def predict_logits(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: str = "cpu",
    num_workers: int = 0,
) -> np.ndarray:
    device_t = torch.device(device)
    y_dummy = np.zeros(len(X), dtype=np.int64)
    loader = _make_loader(X, y_dummy, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out_logits = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device_t)
            logits = model(xb)
            out_logits.append(logits.cpu().numpy())

    return np.concatenate(out_logits, axis=0)
