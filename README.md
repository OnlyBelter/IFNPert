# IFNPert: IFNγ GEP prediction

IFNPert provides a single-cell workflow for predicting an IFNγ-stimulated
gene-expression profile (GEP) from a control profile. A multilayer perceptron
(MLP) is pretrained on random population pseudo-pairs, then fine-tuned on
CINEMA-OT-inferred control counterfactuals. See the
[workflow rationale](predict_IFN_perturbation.md).

## Quick start

The Conda environment uses Python 3.12 and is named `inf_pert`. From the
repository root, create the environment and run the default configuration:

```bash
conda env create -f environment.yml
conda activate inf_pert
python src/run_ifn_prediction.py --config src/config.ifn_gamma_mlp.yaml
```

To update an existing environment, run
`conda env update -n inf_pert -f environment.yml`.

## Data and configuration

The default settings are in
[`src/config.ifn_gamma_mlp.yaml`](src/config.ifn_gamma_mlp.yaml). The file
defines the data source, metadata labels, expression processing, source-cell
splits, training stages, CINEMA-OT parameters, evaluation, and output path.
Pass another YAML file with `--config` to run a different configuration.

The default input is `data/dong_2023.h5ad`. Set `data.path` to `null` to load
`pertpy.data.dong_2023()` instead. The AnnData object must include the
configured condition and cell-type columns and a
`.var['highly_variable']` annotation. The default `log1p` expression mode also
requires `.raw`. Each cell type must have at least
`data.minimum_cells_per_condition` cells in both conditions.

## Workflow

The runner splits control and IFNγ cells separately within each cell type
before generating pairs. The default train, validation, and test fractions are
70%, 15%, and 15%.

1. **Population pretraining:** randomly samples control inputs and IFNγ
   targets with replacement. It draws 2,000 pseudo-pairs per cell type across
   the three partitions.
2. **CINEMA-OT fine-tuning:** runs CINEMA-OT separately in each partition,
   using a partition-specific principal component analysis (PCA)
   representation to construct a control counterfactual for each treated
   profile. Fine-tuning uses matching-confidence weights by default.
3. **Model:** uses 5,000 highly variable genes (HVGs) selected from training
   cells with Scanpy's `seurat` method. The MLP has three hidden layers of
   width 512, with layer normalization, ReLU activation, and dropout of 0.3.
   The same genes are used for validation and test.

## Package modules

The CLI passes the configuration to `pipeline.py`, which coordinates the
workflow. The package files below follow the run sequence:

1. [`pipeline.py`](src/ifn_gamma_mlp/pipeline.py) coordinates the workflow from
   data preparation through evaluation and reporting.
2. [`data.py`](src/ifn_gamma_mlp/data.py) loads AnnData, filters cell types,
   splits source cells, selects genes, and creates both pair sets.
3. [`model.py`](src/ifn_gamma_mlp/model.py) defines the MLP and its full-profile
   and residual prediction behavior.
4. [`train.py`](src/ifn_gamma_mlp/train.py) selects the device and trains both
   stages with validation-based early stopping.
5. [`evaluate.py`](src/ifn_gamma_mlp/evaluate.py) scores each stage by cell type
   and compares the MLP with identity and cell-type-mean baselines.
6. [`report.py`](src/ifn_gamma_mlp/report.py) writes test plots, training
   history plots, and the run summary.

## Expression and prediction

The default `expression.source_scale: log1p` selects genes from `.raw`.
Because `expression.n_hvg` is set to `5000`, Scanpy ranks genes on training
cells only. If `expression.n_hvg` is unset, the workflow uses the existing
`.var['highly_variable']` annotation instead. Selecting a new number of HVGs
requires `log1p` mode and a `.raw` matrix.

For each profile, the workflow applies `expm1` to the selected genes, closes
their linear-space sum to `expression.target_sum` (2,856 by default), then
applies `log1p`. These values are normalized expression, not raw UMI counts.
Reclosing the selected-gene subset changes its values relative to the full
`.raw` profile.

`model.prediction_target` supports two modes:

- `full` predicts the stimulated profile directly.
- `residual` predicts a bounded expression change and adds it to the input.

In both modes, the model produces a full stimulated profile for loss and
evaluation. In `log1p` mode, it bounds the profile, converts it to linear space,
and recloses it to `expression.target_sum`. The default
`expression.scale_factor` of 8 scales model inputs and outputs; residual
changes use `tanh`.

## Evaluation and outputs

The workflow evaluates each model stage on its held-out pairs and reports
metrics by cell type. It compares the MLP with an identity (no-change)
prediction and a cell-type mean computed from training targets. Metrics include
mean and median per-profile root mean square error (RMSE) and mean per-profile
Pearson correlation.

The test plots display three complete profiles per cell type by default.
`evaluation.n_sample_plot` sets the number, and
`evaluation.n_sample_plot_seed` makes the sampling reproducible. Plot
annotations describe the displayed profiles; `test_metrics.csv` includes all
held-out profiles.

The default output directory is
`results/ifn_gamma_mlp_log1p_residual_5000hvg_dropout0.3/`. Choose an empty
directory or set `outputs.overwrite: true` before rerunning into a non-empty
directory. Outputs include:

- `resolved_config.yaml`, `cell_counts.csv`, `source_partitions.csv`, and
  `gene_names.csv`
- `population_pairs.npz` and `cinema_ot_pairs.npz`, with pair metadata CSVs
  for each partition
- `checkpoints/population_pretrained.pt` and
  `checkpoints/cinema_finetuned.pt`
- `population_training_history.csv` and `cinema_training_history.csv`
- `test_metrics.csv`, `test_predictions.npz`, and test-pair metadata CSVs
- `stage1_population_test.png`, `stage2_cinema_test.png`,
  `training_history.png`, and `run_summary.md`

## Interpretation limits

Random pseudo-pairs are not before-and-after measurements from the same cells;
they estimate population-level responses. CINEMA-OT pairs are inferred
counterfactuals, not observed paired measurements. The dataset has no donor or
replicate labels, so held-out results are internal cell-level evaluations, not
independent biological validation.

In `scaled` mode, `.X` was scaled before the source split, making evaluation
transductive with respect to that preprocessing. In `log1p` mode, each cell's
selected `.raw` genes are independently reclosed before the split.
