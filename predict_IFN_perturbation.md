# Predicting IFNγ-stimulated GEPs from unperturbed PBMCs

## Objective

This project predicts an IFNγ-stimulated gene-expression profile (GEP) from a
control-cell profile using a multilayer perceptron (MLP) with three hidden
layers. The hypothesis is that broad control expression can identify cell
state, while CINEMA-OT-inferred counterfactual pairs can teach the model how
IFNγ changes that state. This proposal focuses only on IFNγ.

## Data and training pairs

The analysis uses the Dong et al. PBMC dataset distributed as
`pertpy.data.dong_2023()`; the default configuration reads
`data/dong_2023.h5ad`. Five cell types meet the minimum cell-count threshold in both conditions, with 2,249 control cells and 1,921 IFNγ-treated cells. The
current workflow selects 5,000 highly variable genes (HVGs) from the training
partition.

The workflow partitions source cells by cell type and condition into training,
validation, and test sets (70%, 15%, and 15%) before generating either pair
set. It then trains in two stages:

1. **Population pretraining.** Within each cell type and partition, the
   workflow independently samples control and IFNγ profiles with replacement.
   It draws 2,000 pseudo-pairs per cell type: 1,400 for training, 300 for
   validation, and 300 for testing. Across five cell types, this gives 10,000
   pair draws. The two profiles in a pseudo-pair are not measured before and
   after in the same cell; repeated draws do not create new biological
   observations.
2. **Counterfactual fine-tuning.** CINEMA-OT runs separately within each source partition and constructs a control counterfactual for each treated profile. It recomputes PCA and estimates transport matches within each partition, so held-out cells do not fit the training counterfactual map. The current run
   produces 1,921 pairs overall, including 289 test pairs. Fine-tuning weights
   profile losses by CINEMA-OT matching confidence.

## MLP and evaluation

The current configuration uses 5,000 genes in a consistent order for each
input and target. It selects highly variable genes from the training partition
with Scanpy's `seurat` method. For each profile, the workflow converts the
selected `.raw` values to linear space with `expm1`, closes the selected-gene
sum to 2,856, and applies `log1p`. These are normalized expression values, not
raw UMI counts.

The MLP has three hidden layers of width 512, followed by a 5,000-gene output
layer. Its architecture is `5000 → 512 → 512 → 512 → 5000`. Each hidden
block contains a linear layer, layer normalization, ReLU, and dropout
(`p = 0.3`).
The residual model divides each input by a scale factor of 8. Because each
nonnegative log1p feature is at most `log1p(2856)`, scaling places input
features in `[0, 1)`. The `tanh` activation produces values in `(-1, 1)`;
multiplying by 8 gives residuals in `(-8, 8)` in log1p units. The model adds
each residual to its input profile. Before loss and evaluation, it clips the
resulting log1p profiles to `[0, 8]`, converts them to linear space, recloses
them to 2,856, and applies `log1p` again. Training minimizes mean squared
error on the resulting full stimulated profiles. CINEMA-OT fine-tuning
weights profile losses by matching confidence. Both stages use AdamW, weight
decay, gradient clipping, and validation-based early stopping.

Evaluation scores the pretrained model on held-out pseudo-pairs and the
fine-tuned model on held-out CINEMA-OT pairs. For each cell type and stage, it
reports mean and median profile-level root mean square error (RMSE) and mean
profile-level Pearson correlation. The baselines are an identity/no-change
prediction and a cell-type mean profile computed from training targets. Test
metrics use all held-out profiles; plots show three reproducibly sampled
profiles per cell type by default.

## Scope and limitations

The 10,000 pseudo-pairs are resampled from 4,170 observed control and IFNγ
cells; they are not 10,000 independent examples. CINEMA-OT pairs are inferred
counterfactuals, not observed before-and-after measurements of the same cell.
The dataset has one batch and no donor or replicate identifiers, so these
cell-level test sets do not establish generalization to independent
biological samples. This analysis models a profile over 5,000 HVGs; expanding
to all genes would increase model capacity and is outside this first study.

## Reference

Dong M, et al. Causal identification of single-cell experimental perturbation
effects with CINEMA-OT. *Nature Methods*. 2023;20:1769-1779.
https://doi.org/10.1038/s41592-023-02040-5.
