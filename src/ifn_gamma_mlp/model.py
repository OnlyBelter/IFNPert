"""Configurable dense model for predicting IFNγ-treated GEPs."""

from __future__ import annotations

import math

import torch
from torch import nn


class IFNGMLP(nn.Module):
    """Map one unperturbed 773-gene profile to an IFNγ-treated profile."""

    def __init__(
        self,
        n_genes: int = 773,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        prediction_target: str = "full",
        source_scale: str = "scaled",
        scale_factor: float = 1.0,
        target_sum: float = 2856.0,
    ) -> None:
        super().__init__()
        if not isinstance(prediction_target, str) or prediction_target not in {
            "full",
            "residual",
        }:
            raise ValueError(
                "prediction_target must be either 'full' or 'residual'."
            )
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")
        if source_scale not in {"scaled", "log1p"}:
            raise ValueError("source_scale must be either 'scaled' or 'log1p'.")
        if scale_factor <= 0 or target_sum <= 0:
            raise ValueError("scale_factor and target_sum must be positive.")
        if source_scale == "log1p" and scale_factor < math.log1p(target_sum):
            raise ValueError(
                "scale_factor must be at least log1p(target_sum) for closed profiles."
            )
        self.prediction_target = prediction_target
        self.source_scale = source_scale
        self.scale_factor = float(scale_factor)
        self.target_sum = float(target_sum)
        layers: list[nn.Module] = []
        input_dim = n_genes
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, width),
                    nn.LayerNorm(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = width
        layers.append(nn.Linear(input_dim, n_genes))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
        if self.prediction_target == "residual":
            output_layer = self.network[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, gep: torch.Tensor) -> torch.Tensor:
        """Return a profile or residual in the configured expression space."""
        if self.source_scale == "log1p":
            output = self.network(gep / self.scale_factor)
            if self.prediction_target == "residual":
                return self.scale_factor * torch.tanh(output)
            return self.scale_factor * torch.sigmoid(output)
        return self.network(gep)

    def predict_profile(self, gep: torch.Tensor) -> torch.Tensor:
        """Return a full stimulated profile regardless of prediction mode."""
        prediction = self(gep)
        if self.prediction_target == "residual":
            prediction = gep + prediction
        if self.source_scale != "log1p":
            return prediction

        prediction = torch.clamp(prediction, min=0.0, max=self.scale_factor)
        linear = torch.expm1(prediction)
        row_sum = linear.sum(dim=1, keepdim=True)
        safe_row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        normalized = linear * (self.target_sum / safe_row_sum)

        # Keep an all-zero clipped prediction finite by falling back to its input.
        input_linear = torch.expm1(torch.clamp(gep, min=0.0, max=self.scale_factor))
        input_sum = input_linear.sum(dim=1, keepdim=True)
        safe_input_sum = torch.where(
            input_sum > 0,
            input_sum,
            torch.ones_like(input_sum),
        )
        normalized_input = input_linear * (self.target_sum / safe_input_sum)
        normalized = torch.where(row_sum > 0, normalized, normalized_input)
        return torch.log1p(normalized)
