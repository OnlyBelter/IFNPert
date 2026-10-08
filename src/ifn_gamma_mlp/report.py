"""Write per-cell-type test plots and the run summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_PLOT_COLORS = ("#0072B2", "#E69F00", "#009E73")


def _mean_profile_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    profile_rmse = np.sqrt(np.mean((observed - predicted) ** 2, axis=1))
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(observed_centered**2, axis=1)
        * np.sum(predicted_centered**2, axis=1)
    )
    profile_pearson = np.full(len(observed), np.nan, dtype=np.float64)
    valid = denominator > 0
    profile_pearson[valid] = numerator[valid] / denominator[valid]
    return float(profile_rmse.mean()), float(np.nanmean(profile_pearson))


def _write_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_test_partition(
    result: dict[str, Any],
    stage: str,
    path: Path,
    random_state: int,
    max_points_per_type: int,
    n_sample_plot: int | None = None,
    n_sample_plot_seed: int | None = None,
) -> None:
    metadata = result["metadata"]
    observed = result["targets"]
    predicted = result["predictions"]["mlp"]
    cell_types = sorted(metadata["cell_type"].astype(str).unique())
    rng = np.random.default_rng(
        random_state if n_sample_plot_seed is None else n_sample_plot_seed
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), squeeze=False)
    for ax, cell_type in zip(axes.ravel(), cell_types, strict=False):
        mask = metadata["cell_type"].astype(str).to_numpy() == cell_type
        observed_type = observed[mask]
        predicted_type = predicted[mask]
        if n_sample_plot is not None:
            if n_sample_plot < 1:
                raise ValueError("evaluation.n_sample_plot must be positive.")
            n_profiles = min(n_sample_plot, len(observed_type))
            chosen_profiles = rng.choice(
                len(observed_type),
                size=n_profiles,
                replace=False,
            )
            metric_observed = observed_type[chosen_profiles]
            metric_predicted = predicted_type[chosen_profiles]
            y = metric_observed.ravel()
            y_hat = metric_predicted.ravel()
            for profile_index, (profile_observed, profile_predicted) in enumerate(
                zip(metric_observed, metric_predicted, strict=True)
            ):
                ax.scatter(
                    profile_observed,
                    profile_predicted,
                    s=8,
                    alpha=0.5,
                    color=SAMPLE_PLOT_COLORS[
                        profile_index % len(SAMPLE_PLOT_COLORS)
                    ],
                    label=f"Sample {profile_index + 1}",
                    rasterized=True,
                )
            ax.legend(
                loc="lower right",
                fontsize=8,
                markerscale=1.5,
                frameon=True,
                framealpha=0.85,
            )
        else:
            y = observed_type.ravel()
            y_hat = predicted_type.ravel()
            n_points = min(len(y), max_points_per_type)
            chosen_points = (
                rng.choice(len(y), size=n_points, replace=False)
                if len(y) > n_points
                else np.arange(len(y))
            )
            y = y[chosen_points]
            y_hat = y_hat[chosen_points]
            metric_observed = observed_type
            metric_predicted = predicted_type
            ax.scatter(y, y_hat, s=5, alpha=0.22, rasterized=True)
        lower = min(float(y.min()), float(y_hat.min()))
        upper = max(float(y.max()), float(y_hat.max()))
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        mean_rmse, mean_pearson = _mean_profile_metrics(
            metric_observed,
            metric_predicted,
        )
        ax.text(
            0.04,
            0.96,
            f"RMSE = {mean_rmse:.3f}\n"
            f"Pearson r = {mean_pearson:.3f}\n"
            f"({'shown profiles' if n_sample_plot is not None else 'all profiles'})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
        ax.set(
            xlabel="Observed IFNγ-treated expression",
            ylabel="Predicted IFNγ-treated expression",
            title=(
                f"{cell_type} "
                f"(n={n_profiles} of {int(mask.sum())} test profiles shown)"
                if n_sample_plot is not None
                else f"{cell_type} (n={int(mask.sum())} profiles)"
            ),
        )
    for ax in axes.ravel()[len(cell_types) :]:
        ax.set_visible(False)
    fig.suptitle(stage)
    fig.tight_layout()
    _write_figure(fig, path)


def write_report(
    output_dir: Path,
    stage_one: dict[str, Any],
    stage_two: dict[str, Any],
    stage_one_history: pd.DataFrame,
    stage_two_history: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_one_plot = output_dir / "stage1_population_test.png"
    stage_two_plot = output_dir / "stage2_cinema_test.png"
    training_plot = output_dir / "training_history.png"
    metrics_path = output_dir / "test_metrics.csv"
    summary_path = output_dir / "run_summary.md"

    _plot_test_partition(
        stage_one,
        "Population-pseudo-pair test set",
        stage_one_plot,
        int(config.get("random_state", 0)),
        int(config.get("evaluation", {}).get("max_points_per_cell_type", 10_000)),
        n_sample_plot=int(
            config.get("evaluation", {}).get("n_sample_plot", 3)
        ),
        n_sample_plot_seed=int(
            config.get("evaluation", {}).get("n_sample_plot_seed", 0)
        ),
    )
    _plot_test_partition(
        stage_two,
        "CINEMA-OT fine-tuning test set",
        stage_two_plot,
        int(config.get("random_state", 0)) + 1,
        int(config.get("evaluation", {}).get("max_points_per_cell_type", 10_000)),
        n_sample_plot=int(
            config.get("evaluation", {}).get("n_sample_plot", 3)
        ),
        n_sample_plot_seed=int(
            config.get("evaluation", {}).get("n_sample_plot_seed", 0)
        ),
    )

    history = pd.concat(
        [
            stage_one_history.assign(stage="population_pretraining"),
            stage_two_history.assign(stage="cinema_finetuning"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for stage_name, rows in history.groupby("stage"):
        ax.plot(rows["epoch"], rows["validation_loss"], label=f"{stage_name} validation")
        ax.plot(rows["epoch"], rows["train_loss"], linestyle="--", label=f"{stage_name} train")
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title="Training history")
    ax.legend(fontsize=8)
    _write_figure(fig, training_plot)

    metrics = pd.concat(
        [stage_one["metrics"], stage_two["metrics"]],
        ignore_index=True,
    )
    metrics.to_csv(metrics_path, index=False)
    test_profiles = len(stage_one["metadata"]) + len(stage_two["metadata"])
    expression_config = config.get("expression", {})
    interpretation = (
        "Random pseudo-pairs test population-average prediction and do not "
        "represent observed before/after measurements of the same cell. "
        "CINEMA-OT pairs are inferred counterfactuals. This dataset has no "
        "donor or replicate labels, so the test results are internal "
        "cell-level evaluations rather than independent biological validation."
    )
    if expression_config.get("source_scale", "scaled") == "scaled":
        interpretation += (
            " The supplied .X expression values were scaled before source "
            "splitting, so that preprocessing is transductive with respect to "
            "the test cells."
        )
    else:
        interpretation += (
            " The selected .raw HVGs are reclosed to the configured "
            "linear-space total before log1p modeling, changing their "
            "per-gene values relative to the all-gene .raw matrix."
        )
    lines = [
        "# IFNγ GEP prediction",
        "",
        "- Prediction target: "
        f"{config.get('model', {}).get('prediction_target', 'full')}",
        "- Expression source scale: "
        f"{expression_config.get('source_scale', 'scaled')}",
        "- Model input scale factor: "
        f"{expression_config.get('scale_factor', 1)}",
        "- Linear-space profile target sum: "
        f"{expression_config.get('target_sum', 'not configured')}",
        f"- Selected genes: {len(stage_one['targets'][0])}",
        f"- Population test pseudo-pairs: {len(stage_one['metadata']):,}",
        f"- CINEMA-OT test pairs: {len(stage_two['metadata']):,}",
        f"- Eligible cell types: {metrics['cell_type'].nunique()}",
        f"- Test profile rows across both stages: {test_profiles:,}",
        "",
        "## Test metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Figures",
        "",
        "- `stage1_population_test.png`",
        "- `stage2_cinema_test.png`",
        "- `training_history.png`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary": summary_path,
        "metrics": metrics_path,
        "stage1_plot": stage_one_plot,
        "stage2_plot": stage_two_plot,
        "training_plot": training_plot,
    }
