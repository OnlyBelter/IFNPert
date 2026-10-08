import anndata as ad
import ifn_gamma_mlp.report as report
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ifn_gamma_mlp.data import (
    generate_population_pairs,
    find_eligible_cell_types,
    partition_source_cells,
    prepare_hvg_partitions,
)
from ifn_gamma_mlp.evaluate import evaluate_partition
from ifn_gamma_mlp.model import IFNGMLP
from ifn_gamma_mlp.train import fit_stage, select_device


def make_config():
    return {
        "data": {"minimum_cells_per_condition": 5},
        "metadata": {"condition": "condition", "cell_type": "cell_type"},
        "conditions": {"control": "control", "ifng": "ifng"},
        "expression": {"gene_list": "highly_variable"},
        "splits": {
            "train_fraction": 0.72,
            "validation_fraction": 0.08,
            "test_fraction": 0.2,
        },
        "pretraining": {"pairs_per_cell_type": 20},
        "random_state": 7,
    }


def test_source_splits_and_random_pairs_are_disjoint():
    rng = np.random.default_rng(7)
    obs = pd.DataFrame(
        {
            "condition": ["control"] * 30 + ["ifng"] * 30,
            "cell_type": ["B cell"] * 60,
        },
        index=[f"cell_{i}" for i in range(60)],
    )
    var = pd.DataFrame(
        {"highly_variable": [True, True, True, False]},
        index=["g1", "g2", "g3", "g4"],
    )
    adata = ad.AnnData(
        X=rng.normal(size=(60, 4)).astype(np.float32),
        obs=obs,
        var=var,
    )
    config = make_config()

    cell_types, _ = find_eligible_cell_types(adata, config)
    source_partitions, manifest = partition_source_cells(adata, config, cell_types)
    partitions, genes = prepare_hvg_partitions(source_partitions, config)
    pairs = generate_population_pairs(partitions, config, cell_types)

    assert genes == ["g1", "g2", "g3"]
    assert all(partition.n_vars == 3 for partition in partitions.values())
    assert {
        split: len(pair_set["inputs"]) for split, pair_set in pairs.items()
    } == {"train": 14, "validation": 2, "test": 4}
    source_ids = {
        split: set(manifest.loc[manifest["partition"] == split, "cell_id"])
        for split in ("train", "validation", "test")
    }
    assert source_ids["train"].isdisjoint(source_ids["validation"])
    assert source_ids["train"].isdisjoint(source_ids["test"])
    assert source_ids["validation"].isdisjoint(source_ids["test"])


def test_log1p_source_maps_same_hvgs_and_closes_each_profile():
    rng = np.random.default_rng(19)
    obs = pd.DataFrame(
        {
            "condition": ["control"] * 30 + ["ifng"] * 30,
            "cell_type": ["B"] * 60,
        },
        index=[f"cell_{i}" for i in range(60)],
    )
    hvg_names = ["g1", "g2", "g3"]
    raw_names = ["g3", "g5", "g1", "g4", "g2"]
    adata = ad.AnnData(
        X=rng.normal(size=(60, 4)).astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(
            {"highly_variable": [True, True, True, False]},
            index=["g1", "g2", "g3", "g4"],
        ),
    )
    raw = ad.AnnData(
        X=rng.uniform(0, 2, size=(60, 5)).astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=raw_names),
    )
    adata.raw = raw
    config = make_config()
    config["expression"].update(
        {"source_scale": "log1p", "scale_factor": 8, "target_sum": 100}
    )

    partitions = {name: adata.copy() for name in ("train", "validation", "test")}
    prepared, genes = prepare_hvg_partitions(partitions, config)
    raw_index = {gene: index for index, gene in enumerate(raw_names)}
    raw_hvg = raw.X[0, [raw_index[gene] for gene in hvg_names]]
    expected = np.log1p(np.expm1(raw_hvg) * 100 / np.expm1(raw_hvg).sum())

    assert genes == hvg_names
    assert all(part.var_names.tolist() == hvg_names for part in prepared.values())
    assert all(part.shape == (60, 3) for part in prepared.values())
    np.testing.assert_allclose(prepared["train"].X[0], expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.expm1(prepared["train"].X).sum(axis=1),
        np.full(60, 100),
        rtol=1e-6,
    )


def test_n_hvg_is_selected_from_training_partition_only(monkeypatch):
    rng = np.random.default_rng(29)
    obs = pd.DataFrame(
        {
            "condition": ["control"] * 30 + ["ifng"] * 30,
            "cell_type": ["B"] * 60,
        },
        index=[f"cell_{i}" for i in range(60)],
    )
    raw = ad.AnnData(
        X=np.log1p(rng.uniform(0, 5, size=(60, 6)).astype(np.float32)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(6)]),
    )
    adata = ad.AnnData(
        X=rng.normal(size=(60, 3)).astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(
            {"highly_variable": [True, True, True]},
            index=["old_1", "old_2", "old_3"],
        ),
    )
    adata.raw = raw
    config = make_config()
    config["expression"].update(
        {
            "source_scale": "log1p",
            "n_hvg": 3,
            "scale_factor": 8,
            "target_sum": 100,
        }
    )
    cell_types, _ = find_eligible_cell_types(adata, config)
    source_partitions, manifest = partition_source_cells(adata, config, cell_types)
    captured = {}

    def fake_hvg(train_raw, flavor, n_top_genes, inplace):
        captured["obs_names"] = set(train_raw.obs_names)
        captured["flavor"] = flavor
        captured["n_top_genes"] = n_top_genes
        train_raw.var["highly_variable"] = [
            index < n_top_genes for index in range(train_raw.n_vars)
        ]

    monkeypatch.setattr("scanpy.pp.highly_variable_genes", fake_hvg)
    prepared, genes = prepare_hvg_partitions(source_partitions, config)

    train_ids = set(
        manifest.loc[manifest["partition"] == "train", "cell_id"]
    )
    heldout_ids = set(
        manifest.loc[manifest["partition"] != "train", "cell_id"]
    )
    assert captured["obs_names"] == train_ids
    assert captured["obs_names"].isdisjoint(heldout_ids)
    assert captured["flavor"] == "seurat"
    assert captured["n_top_genes"] == 3
    assert genes == ["gene_0", "gene_1", "gene_2"]
    assert all(part.var_names.tolist() == genes for part in prepared.values())


def test_configurable_mlp_maps_full_profile_dimension():
    model = IFNGMLP(
        n_genes=773,
        hidden_dims=(512, 512, 256, 128),
        dropout=0.2,
    )
    output = model(torch.zeros(5, 773))
    assert output.shape == (5, 773)
    linear_layers = [
        layer for layer in model.network if isinstance(layer, torch.nn.Linear)
    ]
    assert [layer.out_features for layer in linear_layers] == [
        512,
        512,
        256,
        128,
        773,
    ]
    normalization_layers = [
        layer for layer in model.network if isinstance(layer, torch.nn.LayerNorm)
    ]
    assert [layer.normalized_shape for layer in normalization_layers] == [
        (512,),
        (512,),
        (256,),
        (128,),
    ]


def test_residual_mode_starts_at_identity_and_full_mode_is_direct():
    inputs = torch.randn(4, 8)
    residual_model = IFNGMLP(
        n_genes=8,
        hidden_dims=(12, 6),
        dropout=0.0,
        prediction_target="residual",
    )
    full_model = IFNGMLP(
        n_genes=8,
        hidden_dims=(12, 6),
        dropout=0.0,
        prediction_target="full",
    )

    torch.testing.assert_close(residual_model(inputs), torch.zeros_like(inputs))
    torch.testing.assert_close(residual_model.predict_profile(inputs), inputs)
    torch.testing.assert_close(full_model.predict_profile(inputs), full_model(inputs))
    with np.testing.assert_raises_regex(ValueError, "prediction_target"):
        IFNGMLP(n_genes=8, prediction_target="invalid")


def test_log1p_residual_uses_tanh_and_recloses_bounded_profiles():
    model = IFNGMLP(
        n_genes=3,
        hidden_dims=(8,),
        dropout=0.0,
        prediction_target="residual",
        source_scale="log1p",
        scale_factor=8,
        target_sum=100,
    )
    inputs = torch.log1p(torch.tensor([[10.0, 20.0, 70.0]]))

    torch.testing.assert_close(model(inputs), torch.zeros_like(inputs))
    torch.testing.assert_close(model.predict_profile(inputs), inputs)
    with torch.no_grad():
        model.network[-1].bias.fill_(100)
    residual = model(inputs)
    prediction = model.predict_profile(inputs)

    assert torch.all(residual <= 8)
    assert torch.all(residual >= -8)
    assert torch.all(prediction >= 0)
    assert torch.all(prediction <= 8)
    torch.testing.assert_close(
        torch.expm1(prediction).sum(dim=1),
        torch.tensor([100.0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_log1p_residual_training_keeps_processed_profiles_closed():
    rng = np.random.default_rng(23)
    input_linear = rng.uniform(1, 20, size=(40, 8))
    input_linear *= 100 / input_linear.sum(axis=1, keepdims=True)
    target_linear = input_linear * rng.uniform(0.8, 1.2, size=(40, 8))
    target_linear *= 100 / target_linear.sum(axis=1, keepdims=True)
    inputs = np.log1p(input_linear).astype(np.float32)
    targets = np.log1p(target_linear).astype(np.float32)
    model = IFNGMLP(
        n_genes=8,
        hidden_dims=(16, 8),
        dropout=0.0,
        prediction_target="residual",
        source_scale="log1p",
        scale_factor=8,
        target_sum=100,
    )
    model, history = fit_stage(
        model=model,
        train_pairs={"inputs": inputs[:32], "targets": targets[:32]},
        validation_pairs={"inputs": inputs[32:], "targets": targets[32:]},
        settings={
            "batch_size": 8,
            "max_epochs": 5,
            "patience": 5,
            "weight_decay": 0,
        },
        learning_rate=1e-3,
        device=select_device("cpu"),
        seed=23,
        use_match_weights=False,
    )

    prediction = model.predict_profile(torch.from_numpy(inputs[32:]))
    predicted_totals = torch.expm1(prediction).sum(dim=1)
    assert np.isfinite(history[["train_loss", "validation_loss"]]).all().all()
    assert torch.all(prediction >= 0)
    assert torch.all(prediction <= 8)
    torch.testing.assert_close(
        predicted_totals,
        torch.full_like(predicted_totals, 100),
        rtol=1e-5,
        atol=1e-4,
    )


def test_training_loop_fits_and_returns_finite_history():
    rng = np.random.default_rng(12)
    inputs = rng.normal(size=(80, 12)).astype(np.float32)
    targets = (0.4 * inputs + 0.1).astype(np.float32)
    fit_pairs = {
        "inputs": inputs[:64],
        "targets": targets[:64],
        "weights": np.ones(64, dtype=np.float32),
    }
    validation_pairs = {
        "inputs": inputs[64:],
        "targets": targets[64:],
        "weights": np.ones(16, dtype=np.float32),
    }
    model = IFNGMLP(n_genes=12, hidden_dims=(16, 8), dropout=0.0)

    model, history = fit_stage(
        model,
        fit_pairs,
        validation_pairs,
        {
            "batch_size": 16,
            "max_epochs": 8,
            "patience": 8,
            "weight_decay": 0.0,
        },
        learning_rate=1e-3,
        device=select_device("cpu"),
        seed=12,
        use_match_weights=False,
    )

    assert len(history) > 0
    assert np.isfinite(history[["train_loss", "validation_loss"]]).all().all()
    prediction = model(torch.from_numpy(inputs[:2]))
    assert prediction.shape == (2, 12)


def test_residual_training_learns_deltas_and_evaluation_returns_full_profiles():
    rng = np.random.default_rng(31)
    inputs = rng.normal(size=(80, 12)).astype(np.float32)
    targets = (0.4 * inputs + 0.1).astype(np.float32)
    train_pairs = {
        "inputs": inputs[:64],
        "targets": targets[:64],
        "weights": np.ones(64, dtype=np.float32),
    }
    validation_pairs = {
        "inputs": inputs[64:],
        "targets": targets[64:],
        "weights": np.ones(16, dtype=np.float32),
    }
    model = IFNGMLP(
        n_genes=12,
        hidden_dims=(16, 8),
        dropout=0.0,
        prediction_target="residual",
    )
    initial = model.predict_profile(torch.from_numpy(inputs[64:])).detach().numpy()

    model, history = fit_stage(
        model,
        train_pairs,
        validation_pairs,
        {
            "batch_size": 16,
            "max_epochs": 40,
            "patience": 12,
            "weight_decay": 0.0,
        },
        learning_rate=1e-3,
        device=select_device("cpu"),
        seed=31,
        use_match_weights=False,
    )
    trained = model.predict_profile(torch.from_numpy(inputs[64:])).detach().numpy()
    assert np.mean((initial - targets[64:]) ** 2) > np.mean(
        (trained - targets[64:]) ** 2
    )
    assert np.isfinite(history[["train_loss", "validation_loss"]]).all().all()

    metadata = pd.DataFrame({"cell_type": ["B"] * 16})
    evaluation = evaluate_partition(
        model=model,
        test_pairs={
            "inputs": inputs[64:],
            "targets": targets[64:],
            "metadata": metadata,
        },
        train_pairs={
            "targets": targets[:64],
            "metadata": pd.DataFrame({"cell_type": ["B"] * 64}),
        },
        stage="test",
        device=select_device("cpu"),
        batch_size=8,
    )
    assert evaluation["predictions"]["mlp"].shape == targets[64:].shape
    np.testing.assert_allclose(
        evaluation["predictions"]["mlp"],
        model.predict_profile(torch.from_numpy(inputs[64:])).detach().numpy(),
    )


def test_stage_two_plot_samples_complete_profiles_reproducibly(monkeypatch, tmp_path):
    cell_types = ["B"] * 5 + ["NK"] * 2
    targets = np.arange(28, dtype=np.float32).reshape(7, 4)
    predictions = targets + np.arange(28, dtype=np.float32).reshape(7, 4) * 0.04
    result = {
        "metadata": pd.DataFrame({"cell_type": cell_types}),
        "targets": targets,
        "predictions": {"mlp": predictions},
        "metrics": pd.DataFrame(
            {
                "cell_type": ["B", "NK"],
                "method": ["mlp", "mlp"],
                "mean_profile_rmse": [99.0, 99.0],
                "mean_profile_pearson_r": [-99.0, -99.0],
            }
        ),
    }
    figures = []
    monkeypatch.setattr(
        report,
        "_write_figure",
        lambda fig, path: figures.append(fig),
    )

    for index in range(2):
        report._plot_test_partition(
            result,
            "CINEMA-OT test",
            tmp_path / f"stage2_{index}.png",
            random_state=99,
            max_points_per_type=10_000,
            n_sample_plot=3,
            n_sample_plot_seed=17,
        )

    try:
        for fig in figures:
            assert len(fig.axes[0].collections) == 3
            assert len(fig.axes[1].collections) == 2
            assert [
                len(collection.get_offsets())
                for collection in fig.axes[0].collections
            ] == [4, 4, 4]
            assert "3 of 5 test profiles shown" in fig.axes[0].get_title()
            assert "2 of 2 test profiles shown" in fig.axes[1].get_title()
            assert [
                text.get_text() for text in fig.axes[0].get_legend().get_texts()
            ] == ["Sample 1", "Sample 2", "Sample 3"]
            sample_colors = [
                tuple(collection.get_facecolors()[0])
                for collection in fig.axes[0].collections
            ]
            assert len(set(sample_colors)) == 3
            annotation = fig.axes[0].texts[0].get_text()
            assert "(shown profiles)" in annotation
            assert "full test set" not in annotation
            shown_profiles = [
                np.asarray(collection.get_offsets()) for collection in fig.axes[0].collections
            ]
            shown_y = np.stack([profile[:, 0] for profile in shown_profiles])
            shown_y_hat = np.stack([profile[:, 1] for profile in shown_profiles])
            expected_rmse, expected_pearson = report._mean_profile_metrics(
                shown_y,
                shown_y_hat,
            )
            assert f"RMSE = {expected_rmse:.3f}" in annotation
            assert f"Pearson r = {expected_pearson:.3f}" in annotation
        np.testing.assert_array_equal(
            np.concatenate(
                [
                    collection.get_offsets()
                    for collection in figures[0].axes[0].collections
                ]
            ),
            np.concatenate(
                [
                    collection.get_offsets()
                    for collection in figures[1].axes[0].collections
                ]
            ),
        )
    finally:
        for fig in figures:
            plt.close(fig)
