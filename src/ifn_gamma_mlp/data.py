"""Load PBMC data, split source cells, and construct training pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PARTITIONS = ("train", "validation", "test")


def dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def load_dataset(config: dict[str, Any]) -> ad.AnnData:
    """Load a local AnnData file or the Dong et al. dataset from pertpy."""
    path_value = config["data"].get("path")
    if path_value:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = Path(config["_project_root"]) / path
        if not path.is_file():
            raise FileNotFoundError(f"AnnData input does not exist: {path}")
        return ad.read_h5ad(path)

    if config["data"].get("dataset", "dong_2023") != "dong_2023":
        raise ValueError("Only dong_2023 or a local .h5ad input is supported.")
    try:
        import pertpy as pt
    except ImportError as exc:
        raise ImportError(
            "The default dataset loader requires pertpy in the active environment."
        ) from exc
    return pt.data.dong_2023()


def find_eligible_cell_types(
    adata: ad.AnnData,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Find cell types with enough control and IFNγ source cells."""
    metadata = config["metadata"]
    conditions = config["conditions"]
    condition_key = metadata["condition"]
    cell_type_key = metadata["cell_type"]
    required = [conditions["control"], conditions["ifng"]]
    for column in (condition_key, cell_type_key):
        if column not in adata.obs:
            raise KeyError(f"Required .obs column {column!r} is missing.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Observation IDs and gene IDs must be unique.")
    if "highly_variable" not in adata.var:
        raise ValueError("Input AnnData must include .var['highly_variable'].")

    counts = (
        adata.obs.groupby([cell_type_key, condition_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    minimum = int(config["data"].get("minimum_cells_per_condition", 20))
    pivot = counts.pivot_table(
        index=cell_type_key,
        columns=condition_key,
        values="n_cells",
        fill_value=0,
        aggfunc="sum",
    )
    eligible = sorted(
        str(cell_type)
        for cell_type in pivot.index
        if all(int(pivot.loc[cell_type].get(label, 0)) >= minimum for label in required)
    )
    if not eligible:
        raise ValueError(
            "No cell type meets the minimum control and IFNγ cell counts."
        )

    counts[cell_type_key] = counts[cell_type_key].astype(str)
    counts[condition_key] = counts[condition_key].astype(str)
    return eligible, counts


def _select_hvg_names(
    training_partition: ad.AnnData,
    config: dict[str, Any],
) -> list[str]:
    """Select features from training cells, or use the existing HVG annotation."""
    expression = config.get("expression", {})
    source_scale = expression.get("source_scale", "scaled")
    n_hvg_value = expression.get("n_hvg")
    if n_hvg_value is not None:
        if source_scale != "log1p":
            raise ValueError(
                "expression.n_hvg can only be used with source_scale='log1p'."
            )
        if isinstance(n_hvg_value, bool) or int(n_hvg_value) != n_hvg_value:
            raise ValueError("expression.n_hvg must be a positive integer.")
        n_hvg = int(n_hvg_value)
        if n_hvg < 1:
            raise ValueError("expression.n_hvg must be a positive integer.")
        if training_partition.raw is None:
            raise ValueError(
                "Training cells need a .raw matrix to select HVGs from scratch."
            )
        if n_hvg > training_partition.raw.n_vars:
            raise ValueError(
                f"expression.n_hvg={n_hvg} exceeds the "
                f"{training_partition.raw.n_vars} genes in .raw."
            )
        import scanpy as sc

        training_raw = training_partition.raw.to_adata()
        sc.pp.highly_variable_genes(
            training_raw,
            flavor=str(expression.get("hvg_flavor", "seurat")),
            n_top_genes=n_hvg,
            inplace=True,
        )
        selected = training_raw.var["highly_variable"].fillna(False).astype(bool)
        gene_names = training_raw.var_names[selected].astype(str).tolist()
        if len(gene_names) != n_hvg:
            raise RuntimeError(
                f"Scanpy selected {len(gene_names)} HVGs; expected {n_hvg}."
            )
        return gene_names

    if "highly_variable" not in training_partition.var:
        raise ValueError(
            "Input AnnData must include .var['highly_variable'] when "
            "expression.n_hvg is not set."
        )
    mask = (
        training_partition.var["highly_variable"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    gene_names = training_partition.var_names[mask].astype(str).tolist()
    if not gene_names:
        raise ValueError("The highly_variable annotation selects no genes.")
    return gene_names


def prepare_hvg_partitions(
    partitions: dict[str, ad.AnnData],
    config: dict[str, Any],
) -> tuple[dict[str, ad.AnnData], list[str]]:
    """Select HVGs from train cells and prepare the same genes in all splits."""
    expression = config.get("expression", {})
    source_scale = expression.get("source_scale", "scaled")
    if source_scale not in {"scaled", "log1p"}:
        raise ValueError("expression.source_scale must be 'scaled' or 'log1p'.")
    gene_names = _select_hvg_names(partitions["train"], config)
    target_sum = float(expression.get("target_sum", 2856))
    scale_factor = float(expression.get("scale_factor", 8))
    if target_sum <= 0 or scale_factor <= 0:
        raise ValueError(
            "expression.target_sum and expression.scale_factor must be positive."
        )

    prepared: dict[str, ad.AnnData] = {}
    for partition_name, partition in partitions.items():
        if source_scale == "scaled":
            var_names = partition.var_names.astype(str)
            var_index = {gene: index for index, gene in enumerate(var_names)}
            missing = [gene for gene in gene_names if gene not in var_index]
            if missing:
                raise ValueError(
                    f"{len(missing)} selected HVGs are missing from .X: "
                    f"{', '.join(missing[:10])}."
                )
            indices = [var_index[gene] for gene in gene_names]
            partition_data = partition[:, indices].copy()
            if not np.isfinite(dense_float32(partition_data.X)).all():
                raise ValueError(
                    "The scaled HVG expression matrix contains non-finite values."
                )
        else:
            if partition.raw is None:
                raise ValueError(
                    "expression.source_scale='log1p' requires a .raw matrix."
                )
            raw_names = partition.raw.var_names.astype(str)
            if not raw_names.is_unique:
                raise ValueError(".raw gene IDs must be unique.")
            raw_index = {gene: index for index, gene in enumerate(raw_names)}
            missing = [gene for gene in gene_names if gene not in raw_index]
            if missing:
                raise ValueError(
                    f"{len(missing)} selected HVGs are missing from .raw: "
                    f"{', '.join(missing[:10])}."
                )
            indices = [raw_index[gene] for gene in gene_names]
            raw_matrix = partition.raw.X[:, indices]
            partition_data = ad.AnnData(
                X=raw_matrix,
                obs=partition.obs.copy(),
                var=partition.raw.var.iloc[indices].copy(),
            )
            log_values = dense_float32(partition_data.X)
            if not np.isfinite(log_values).all() or np.any(log_values < 0):
                raise ValueError(
                    ".raw log1p HVG expression must be finite and nonnegative."
                )
            linear_values = np.expm1(log_values.astype(np.float64))
            row_sums = linear_values.sum(axis=1, keepdims=True)
            if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
                raise ValueError(
                    "Every source profile must have a positive selected-gene sum."
                )
            linear_values *= target_sum / row_sums
            normalized_log = np.log1p(linear_values)
            if not np.isfinite(normalized_log).all():
                raise ValueError(
                    "Normalized log1p HVG expression contains non-finite values."
                )
            if float(normalized_log.max()) > scale_factor + 1e-6:
                raise ValueError(
                    "Normalized log1p expression exceeds expression.scale_factor."
                )
            partition_data.X = normalized_log.astype(np.float32)
        prepared[partition_name] = partition_data
    return prepared, gene_names


def partition_source_cells(
    adata: ad.AnnData,
    config: dict[str, Any],
    eligible_cell_types: list[str],
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """Partition source cell IDs by type and condition before creating pairs."""
    metadata = config["metadata"]
    condition_key = metadata["condition"]
    cell_type_key = metadata["cell_type"]
    split_config = config["splits"]
    fractions = {
        "train": float(split_config["train_fraction"]),
        "validation": float(split_config["validation_fraction"]),
        "test": float(split_config["test_fraction"]),
    }
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Source split fractions must sum to 1.")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("Every source split must have a positive fraction.")

    rng = np.random.default_rng(int(config.get("random_state", 0)))
    assigned: dict[str, list[int]] = {name: [] for name in PARTITIONS}
    rows: list[dict[str, str]] = []
    obs = adata.obs
    for cell_type in eligible_cell_types:
        for condition in (
            config["conditions"]["control"],
            config["conditions"]["ifng"],
        ):
            mask = (
                (obs[cell_type_key].astype(str).to_numpy() == cell_type)
                & (obs[condition_key].astype(str).to_numpy() == condition)
            )
            indices = np.flatnonzero(mask)
            if len(indices) < 3:
                raise ValueError(
                    f"{cell_type} / {condition} cannot support three source splits."
                )
            shuffled = rng.permutation(indices)
            n_test = max(1, int(round(len(indices) * fractions["test"])))
            n_validation = max(
                1,
                int(round(len(indices) * fractions["validation"])),
            )
            n_train = len(indices) - n_test - n_validation
            if n_train < 1:
                raise ValueError(
                    f"{cell_type} / {condition} has no training source cells."
                )
            selections = {
                "train": shuffled[:n_train],
                "validation": shuffled[n_train : n_train + n_validation],
                "test": shuffled[n_train + n_validation :],
            }
            for partition_name, selected in selections.items():
                assigned[partition_name].extend(selected.tolist())
                rows.extend(
                    {
                        "cell_id": str(adata.obs_names[index]),
                        "cell_type": cell_type,
                        "condition": condition,
                        "partition": partition_name,
                    }
                    for index in selected
                )

    partitions = {
        name: adata[np.sort(indices)].copy()
        for name, indices in assigned.items()
    }
    manifest = pd.DataFrame.from_records(rows)
    if manifest["cell_id"].duplicated().any():
        raise RuntimeError("A source cell was assigned to multiple partitions.")
    return partitions, manifest


def generate_population_pairs(
    partitions: dict[str, ad.AnnData],
    config: dict[str, Any],
    eligible_cell_types: list[str],
) -> dict[str, dict[str, Any]]:
    """Draw random control/IFNγ pseudo-pairs within each type and partition."""
    metadata = config["metadata"]
    conditions = config["conditions"]
    condition_key = metadata["condition"]
    cell_type_key = metadata["cell_type"]
    pairs_per_type = int(config["pretraining"]["pairs_per_cell_type"])
    split_counts = {
        name: int(round(pairs_per_type * float(config["splits"][f"{name}_fraction"])))
        for name in ("validation", "test")
    }
    split_counts["train"] = (
        pairs_per_type - split_counts["validation"] - split_counts["test"]
    )
    if any(count < 1 for count in split_counts.values()):
        raise ValueError("Each source partition needs pseudo-pair draws per type.")

    seed = int(config.get("random_state", 0))
    outputs: dict[str, dict[str, Any]] = {}
    for split_index, partition_name in enumerate(PARTITIONS):
        partition = partitions[partition_name]
        matrix = dense_float32(partition.X)
        condition_values = partition.obs[condition_key].astype(str).to_numpy()
        type_values = partition.obs[cell_type_key].astype(str).to_numpy()
        rng = np.random.default_rng(seed + 101 * (split_index + 1))
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        records: list[dict[str, str]] = []

        for cell_type in eligible_cell_types:
            controls = np.flatnonzero(
                (type_values == cell_type)
                & (condition_values == conditions["control"])
            )
            ifng = np.flatnonzero(
                (type_values == cell_type)
                & (condition_values == conditions["ifng"])
            )
            if not len(controls) or not len(ifng):
                raise ValueError(
                    f"{partition_name} source split lacks cells for {cell_type}."
                )
            sampled_controls = rng.choice(
                controls,
                size=split_counts[partition_name],
                replace=True,
            )
            sampled_ifng = rng.choice(
                ifng,
                size=split_counts[partition_name],
                replace=True,
            )
            inputs.extend(matrix[sampled_controls])
            targets.extend(matrix[sampled_ifng])
            records.extend(
                {
                    "pair_id": f"{partition_name}_{cell_type}_{draw:05d}",
                    "cell_type": cell_type,
                    "partition": partition_name,
                    "control_cell_id": str(partition.obs_names[control_index]),
                    "ifng_cell_id": str(partition.obs_names[ifng_index]),
                }
                for draw, (control_index, ifng_index) in enumerate(
                    zip(sampled_controls, sampled_ifng, strict=True)
                )
            )
        outputs[partition_name] = {
            "inputs": np.asarray(inputs, dtype=np.float32),
            "targets": np.asarray(targets, dtype=np.float32),
            "weights": np.ones(len(inputs), dtype=np.float32),
            "metadata": pd.DataFrame.from_records(records),
        }
    return outputs


def generate_cinema_pairs(
    partition: ad.AnnData,
    config: dict[str, Any],
    partition_name: str,
) -> dict[str, Any]:
    """Run CINEMA-OT within a source partition and reconstruct GEP pairs."""
    try:
        import pertpy as pt
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("CINEMA-OT pairing requires pertpy and scanpy.") from exc

    metadata = config["metadata"]
    conditions = config["conditions"]
    condition_key = metadata["condition"]
    cell_type_key = metadata["cell_type"]
    relevant = partition.obs[condition_key].astype(str).isin(
        [conditions["control"], conditions["ifng"]]
    )
    contrast = partition[relevant].copy()
    values = contrast.obs[condition_key].astype(str).to_numpy()
    control_mask = values == conditions["control"]
    ifng_mask = values == conditions["ifng"]
    n_controls = int(control_mask.sum())
    n_ifng = int(ifng_mask.sum())
    if min(n_controls, n_ifng) < 2:
        raise ValueError(
            f"CINEMA-OT {partition_name} partition lacks control or IFNγ cells."
        )

    # Recompute PCA in each partition so held-out profiles do not fit this representation.
    requested_pcs = int(config["cinema_ot"].get("n_pcs", 20))
    n_pcs = min(requested_pcs, contrast.n_obs - 1, contrast.n_vars - 1)
    if n_pcs < 2:
        raise ValueError(f"CINEMA-OT {partition_name} partition has too few PCA features.")
    sc.pp.pca(
        contrast,
        n_comps=n_pcs,
        random_state=int(config.get("random_state", 0)),
    )
    dim = min(
        int(config["cinema_ot"].get("dim", 20)),
        n_pcs,
        n_controls - 1,
        n_ifng - 1,
    )
    if dim < 2:
        raise ValueError(f"CINEMA-OT {partition_name} partition has too few cells for ICA.")

    result = pt.tl.Cinemaot().causaleffect(
        contrast,
        pert_key=condition_key,
        control=conditions["control"],
        return_matching=True,
        use_rep="X_pca",
        dim=dim,
        thres=float(config["cinema_ot"].get("threshold", 0.15)),
        smoothness=float(config["cinema_ot"].get("smoothness", 1e-4)),
        eps=float(config["cinema_ot"].get("eps", 1e-3)),
        solver=config["cinema_ot"].get("solver", "Sinkhorn"),
        preweight_label=cell_type_key,
        random_state=int(config.get("random_state", 0)),
    )

    control_ids = contrast.obs_names[control_mask].astype(str).to_numpy()
    treated_ids = result.obs_names.astype(str).to_numpy()
    matching = np.asarray(result.obsm["ot"], dtype=np.float64)
    if matching.shape != (len(treated_ids), len(control_ids)):
        raise ValueError(
            f"CINEMA-OT {partition_name} matching axes are not aligned."
        )
    observed = dense_float32(contrast[treated_ids].X)
    effects = dense_float32(result.X)
    if observed.shape != effects.shape:
        raise ValueError(
            f"CINEMA-OT {partition_name} effects do not align with treated GEPs."
        )
    # CINEMA-OT stores treated minus the transport-weighted control profile.
    counterfactual = observed - effects

    row_sums = matching.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(
            f"CINEMA-OT {partition_name} has a zero-mass transport row."
        )
    probabilities = matching / row_sums
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, None)),
        axis=1,
    )
    confidence = np.clip(
        1.0 - entropy / np.log(max(2, len(control_ids))),
        0.0,
        1.0,
    ).astype(np.float32)
    if float(confidence.sum()) <= 1e-8:
        confidence[:] = 1.0

    pair_metadata = result.obs.copy().reset_index(drop=True)
    pair_metadata["ifng_cell_id"] = treated_ids
    pair_metadata["cell_type"] = pair_metadata[cell_type_key].astype(str)
    pair_metadata["partition"] = partition_name
    return {
        "inputs": counterfactual.astype(np.float32, copy=False),
        "targets": observed.astype(np.float32, copy=False),
        "weights": confidence,
        "metadata": pair_metadata,
        "control_cell_ids": control_ids,
    }
