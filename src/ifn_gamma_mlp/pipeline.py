"""Orchestrate IFNγ data preparation, MLP training, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ifn_gamma_mlp.data import (
    PARTITIONS,
    generate_cinema_pairs,
    generate_population_pairs,
    find_eligible_cell_types,
    load_dataset,
    partition_source_cells,
    prepare_hvg_partitions,
)
from ifn_gamma_mlp.evaluate import evaluate_partition
from ifn_gamma_mlp.model import IFNGMLP
from ifn_gamma_mlp.report import write_report
from ifn_gamma_mlp.train import fit_stage, save_checkpoint, select_device


def _output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["outputs"]["directory"]).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    if path.exists() and any(path.iterdir()) and not config["outputs"].get(
        "overwrite",
        False,
    ):
        raise FileExistsError(
            f"Output directory {path} is not empty. Set outputs.overwrite: true "
            "or choose another outputs.directory."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_pair_sets(
    output: Path,
    stage: str,
    pair_sets: dict[str, dict[str, Any]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for partition in PARTITIONS:
        pair_set = pair_sets[partition]
        arrays[f"{partition}_inputs"] = pair_set["inputs"]
        arrays[f"{partition}_targets"] = pair_set["targets"]
        arrays[f"{partition}_weights"] = pair_set["weights"]
        pair_set["metadata"].to_csv(
            output / f"{stage}_{partition}_pair_metadata.csv",
            index=False,
        )
    np.savez_compressed(output / f"{stage}_pairs.npz", **arrays)


def _save_checkpoint(
    model: IFNGMLP,
    path: Path,
    n_genes: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    stage: str,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_genes": n_genes,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "stage": stage,
            "prediction_target": model.prediction_target,
            "source_scale": model.source_scale,
            "scale_factor": model.scale_factor,
            "target_sum": model.target_sum,
        },
        path,
    )


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("random_state", 0))
    model_config = config["model"]
    expression_config = config.get("expression", {})
    source_scale = expression_config.get("source_scale", "scaled")
    if source_scale not in {"scaled", "log1p"}:
        raise ValueError("expression.source_scale must be 'scaled' or 'log1p'.")
    scale_factor = float(
        expression_config.get("scale_factor", 8.0 if source_scale == "log1p" else 1.0)
    )
    target_sum = float(expression_config.get("target_sum", 2856.0))
    if scale_factor <= 0 or target_sum <= 0:
        raise ValueError("expression.scale_factor and expression.target_sum must be positive.")
    prediction_target = model_config.get("prediction_target", "full")
    if not isinstance(prediction_target, str) or prediction_target not in {
        "full",
        "residual",
    }:
        raise ValueError(
            "model.prediction_target must be either 'full' or 'residual'."
        )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output = _output_dir(config)
    resolved_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    resolved_config.setdefault("model", {})["prediction_target"] = prediction_target
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    adata = load_dataset(config)
    cell_types, counts = find_eligible_cell_types(adata, config)
    counts.to_csv(output / "cell_counts.csv", index=False)
    source_partitions, source_manifest = partition_source_cells(
        adata,
        config,
        cell_types,
    )
    source_manifest.to_csv(output / "source_partitions.csv", index=False)
    partitions, gene_names = prepare_hvg_partitions(source_partitions, config)
    pd.Series(gene_names, name="gene").to_csv(
        output / "gene_names.csv",
        index=False,
    )

    population_pairs = generate_population_pairs(
        partitions,
        config,
        cell_types,
    )
    _save_pair_sets(output, "population", population_pairs)

    cinema_pairs = {
        partition_name: generate_cinema_pairs(
            partitions[partition_name],
            config,
            partition_name,
        )
        for partition_name in PARTITIONS
    }
    _save_pair_sets(output, "cinema_ot", cinema_pairs)

    hidden_dims = tuple(int(width) for width in model_config["hidden_dims"])
    if not hidden_dims or any(width < 1 for width in hidden_dims):
        raise ValueError("model.hidden_dims must contain positive widths.")
    dropout = float(model_config.get("dropout", 0.2))
    device = select_device(config.get("device", "auto"))
    model = IFNGMLP(
        n_genes=len(gene_names),
        hidden_dims=hidden_dims,
        dropout=dropout,
        prediction_target=prediction_target,
        source_scale=source_scale,
        scale_factor=scale_factor,
        target_sum=target_sum,
    )

    model, population_history = fit_stage(
        model=model,
        train_pairs=population_pairs["train"],
        validation_pairs=population_pairs["validation"],
        settings=config["pretraining"],
        learning_rate=float(config["pretraining"]["learning_rate"]),
        device=device,
        seed=seed + 1,
        use_match_weights=False,
    )
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        model,
        checkpoint_dir / "population_pretrained.pt",
        len(gene_names),
        hidden_dims,
        dropout,
        "population_pretraining",
    )
    population_history.to_csv(
        output / "population_training_history.csv",
        index=False,
    )

    fine_tuning = config["fine_tuning"]
    model, cinema_history = fit_stage(
        model=model,
        train_pairs=cinema_pairs["train"],
        validation_pairs=cinema_pairs["validation"],
        settings=fine_tuning,
        learning_rate=float(fine_tuning["learning_rate"]),
        device=device,
        seed=seed + 2,
        use_match_weights=bool(fine_tuning.get("use_match_weights", True)),
    )
    _save_checkpoint(
        model,
        checkpoint_dir / "cinema_finetuned.pt",
        len(gene_names),
        hidden_dims,
        dropout,
        "cinema_finetuning",
    )
    cinema_history.to_csv(output / "cinema_training_history.csv", index=False)

    # Score the stage-one checkpoint before fine-tuning changes its weights.
    population_model = IFNGMLP(
        n_genes=len(gene_names),
        hidden_dims=hidden_dims,
        dropout=dropout,
        prediction_target=prediction_target,
        source_scale=source_scale,
        scale_factor=scale_factor,
        target_sum=target_sum,
    ).to(device)
    pretrained = torch.load(
        checkpoint_dir / "population_pretrained.pt",
        map_location=device,
        weights_only=False,
    )
    population_model.load_state_dict(pretrained["state_dict"])
    evaluation_batch_size = int(config.get("evaluation", {}).get("batch_size", 256))
    population_test = evaluate_partition(
        model=population_model,
        test_pairs=population_pairs["test"],
        train_pairs=population_pairs["train"],
        stage="population_pretraining",
        device=device,
        batch_size=evaluation_batch_size,
    )
    cinema_test = evaluate_partition(
        model=model,
        test_pairs=cinema_pairs["test"],
        train_pairs=cinema_pairs["train"],
        stage="cinema_finetuning",
        device=device,
        batch_size=evaluation_batch_size,
    )

    metrics = pd.concat(
        [population_test["metrics"], cinema_test["metrics"]],
        ignore_index=True,
    )
    metrics.to_csv(output / "test_metrics.csv", index=False)
    np.savez_compressed(
        output / "test_predictions.npz",
        genes=np.asarray(gene_names, dtype=str),
        population_targets=population_test["targets"],
        population_mlp=population_test["predictions"]["mlp"],
        population_identity=population_test["predictions"]["identity"],
        population_cell_type_mean=population_test["predictions"]["cell_type_mean"],
        cinema_targets=cinema_test["targets"],
        cinema_mlp=cinema_test["predictions"]["mlp"],
        cinema_identity=cinema_test["predictions"]["identity"],
        cinema_cell_type_mean=cinema_test["predictions"]["cell_type_mean"],
    )
    population_test["metadata"].to_csv(
        output / "population_test_metadata.csv",
        index=False,
    )
    cinema_test["metadata"].to_csv(
        output / "cinema_test_metadata.csv",
        index=False,
    )
    report = write_report(
        output,
        population_test,
        cinema_test,
        population_history,
        cinema_history,
        config,
    )
    print(
        f"Completed IFNγ MLP workflow with {len(gene_names)} genes on {device}. "
        f"Outputs: {output}"
    )
    return {"output_dir": output, "metrics": metrics, **report}
