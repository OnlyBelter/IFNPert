"""Evaluate profile predictions against simple baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ifn_gamma_mlp.model import IFNGMLP


def _profile_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(observed_centered**2, axis=1)
        * np.sum(predicted_centered**2, axis=1)
    )
    correlations = np.full(len(observed), np.nan, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def _predict_profiles(
    model: IFNGMLP,
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model.predict_profile(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_partition(
    model: IFNGMLP,
    test_pairs: dict[str, Any],
    train_pairs: dict[str, Any],
    stage: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Score the MLP and baselines per cell type on an untouched partition."""
    inputs = np.asarray(test_pairs["inputs"], dtype=np.float32)
    targets = np.asarray(test_pairs["targets"], dtype=np.float32)
    test_metadata = test_pairs["metadata"].reset_index(drop=True)
    train_targets = np.asarray(train_pairs["targets"], dtype=np.float32)
    train_metadata = train_pairs["metadata"].reset_index(drop=True)
    if len(inputs) != len(targets) or len(inputs) != len(test_metadata):
        raise ValueError(f"{stage} test profiles and metadata are not aligned.")

    predictions = {
        "mlp": _predict_profiles(model, inputs, device, batch_size),
        "identity": inputs.copy(),
        "cell_type_mean": np.empty_like(targets),
    }
    for cell_type in sorted(test_metadata["cell_type"].astype(str).unique()):
        test_mask = test_metadata["cell_type"].astype(str).to_numpy() == cell_type
        train_mask = train_metadata["cell_type"].astype(str).to_numpy() == cell_type
        if not train_mask.any():
            raise ValueError(f"No training profiles are available for {cell_type!r}.")
        mean_profile = train_targets[train_mask].mean(axis=0)
        predictions["cell_type_mean"][test_mask] = mean_profile

    metric_rows: list[dict[str, float | int | str]] = []
    for cell_type in sorted(test_metadata["cell_type"].astype(str).unique()):
        mask = test_metadata["cell_type"].astype(str).to_numpy() == cell_type
        observed = targets[mask]
        for method, predicted in predictions.items():
            predicted_type = predicted[mask]
            profile_rmse = np.sqrt(np.mean((observed - predicted_type) ** 2, axis=1))
            profile_r = _profile_pearson(observed, predicted_type)
            metric_rows.append(
                {
                    "stage": stage,
                    "cell_type": cell_type,
                    "method": method,
                    "n_profiles": int(mask.sum()),
                    "mean_profile_rmse": float(profile_rmse.mean()),
                    "median_profile_rmse": float(np.median(profile_rmse)),
                    "mean_profile_pearson_r": float(np.nanmean(profile_r)),
                }
            )
    return {
        "metrics": pd.DataFrame.from_records(metric_rows),
        "predictions": predictions,
        "inputs": inputs,
        "targets": targets,
        "metadata": test_metadata,
    }
