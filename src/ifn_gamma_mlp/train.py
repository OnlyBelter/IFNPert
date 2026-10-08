"""Training loops for population pretraining and CINEMA-OT fine-tuning."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ifn_gamma_mlp.model import IFNGMLP


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS device.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(
    pairs: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    inputs = torch.as_tensor(pairs["inputs"], dtype=torch.float32)
    targets = torch.as_tensor(pairs["targets"], dtype=torch.float32)
    weights = torch.as_tensor(
        pairs.get("weights", np.ones(len(inputs), dtype=np.float32)),
        dtype=torch.float32,
    )
    if len(inputs) != len(targets) or len(inputs) != len(weights):
        raise ValueError("Input, target, and sample-weight counts must match.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, targets, weights),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    use_weights: bool,
) -> torch.Tensor:
    profile_loss = torch.mean((prediction - target) ** 2, dim=1)
    if not use_weights:
        return profile_loss.mean()
    weights = torch.clamp(weights, min=1e-6)
    return torch.sum(profile_loss * weights) / torch.sum(weights)


def _evaluate_loss(
    model: IFNGMLP,
    loader: DataLoader,
    device: torch.device,
    use_weights: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_profiles = 0
    with torch.no_grad():
        for inputs, targets, weights in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            loss = _weighted_mse(
                model.predict_profile(inputs),
                targets,
                weights,
                use_weights,
            )
            batch_size = len(inputs)
            total_loss += float(loss) * batch_size
            total_profiles += batch_size
    if not total_profiles:
        raise ValueError("Validation partition contains no pairs.")
    return total_loss / total_profiles


def fit_stage(
    model: IFNGMLP,
    train_pairs: dict[str, Any],
    validation_pairs: dict[str, Any],
    settings: dict[str, Any],
    learning_rate: float,
    device: torch.device,
    seed: int,
    use_match_weights: bool,
) -> tuple[IFNGMLP, pd.DataFrame]:
    """Fit one stage and restore the checkpoint with the lowest validation loss."""
    batch_size = int(settings.get("batch_size", 64))
    train_loader = _make_loader(
        train_pairs,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        validation_pairs,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    max_epochs = int(settings.get("max_epochs", 100))
    patience = int(settings.get("patience", 10))
    min_delta = float(settings.get("min_delta", 1e-5))
    if max_epochs < 1 or patience < 1:
        raise ValueError("Training epochs and early-stopping patience must be positive.")

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    records: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_profiles = 0
        for inputs, targets, weights in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_mse(
                model.predict_profile(inputs),
                targets,
                weights,
                use_match_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size_now = len(inputs)
            total_loss += float(loss.detach()) * batch_size_now
            total_profiles += batch_size_now

        train_loss = total_loss / max(1, total_profiles)
        validation_loss = _evaluate_loss(
            model,
            validation_loader,
            device,
            use_match_weights,
        )
        records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame.from_records(records)


def save_checkpoint(
    model: IFNGMLP,
    path: Path,
    n_genes: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    stage: str,
) -> None:
    """Save model weights together with the architecture needed to reload them."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_genes": n_genes,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "stage": stage,
        },
        path,
    )
