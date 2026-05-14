from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature_scaling(
    logits: np.ndarray,
    y_true: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.05,
    device: str = "cpu",
) -> float:
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_true, dtype=torch.long, device=device)

    log_temp = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits_t / temperature, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
    return temperature


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    return logits / t


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    z = logits - np.max(logits, axis=1, keepdims=True)
    z = np.clip(z, -80.0, 80.0)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    probs = exp_z / denom
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    bad = row_sum <= 0.0
    row_sum = np.where(bad, 1.0, row_sum)
    probs = probs / row_sum
    if np.any(bad):
        probs[bad.squeeze(-1)] = 1.0 / probs.shape[1]
    return probs
