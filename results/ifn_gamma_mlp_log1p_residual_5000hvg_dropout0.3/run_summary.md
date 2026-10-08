# IFNγ GEP prediction

- Prediction target: residual
- Expression source scale: log1p
- Model input scale factor: 8
- Linear-space profile target sum: 2856
- Selected genes: 5000
- Population test pseudo-pairs: 1,500
- CINEMA-OT test pairs: 289
- Eligible cell types: 5
- Test profile rows across both stages: 1,789

## Test metrics

| stage                  | cell_type   | method         |   n_profiles |   mean_profile_rmse |   median_profile_rmse |   mean_profile_pearson_r |
|:-----------------------|:------------|:---------------|-------------:|--------------------:|----------------------:|-------------------------:|
| population_pretraining | B           | mlp            |          300 |            0.384186 |              0.382606 |                 0.580521 |
| population_pretraining | B           | identity       |          300 |            0.496294 |              0.494559 |                 0.411077 |
| population_pretraining | B           | cell_type_mean |          300 |            0.352889 |              0.354426 |                 0.639655 |
| population_pretraining | CD4 T       | mlp            |          300 |            0.394777 |              0.394912 |                 0.526615 |
| population_pretraining | CD4 T       | identity       |          300 |            0.519755 |              0.51975  |                 0.346973 |
| population_pretraining | CD4 T       | cell_type_mean |          300 |            0.368419 |              0.367527 |                 0.585519 |
| population_pretraining | CD8 T       | mlp            |          300 |            0.418582 |              0.418301 |                 0.506567 |
| population_pretraining | CD8 T       | identity       |          300 |            0.550751 |              0.551322 |                 0.326866 |
| population_pretraining | CD8 T       | cell_type_mean |          300 |            0.387946 |              0.387571 |                 0.574329 |
| population_pretraining | Monocyte    | mlp            |          300 |            0.243435 |              0.231494 |                 0.82813  |
| population_pretraining | Monocyte    | identity       |          300 |            0.351798 |              0.349475 |                 0.665515 |
| population_pretraining | Monocyte    | cell_type_mean |          300 |            0.217115 |              0.203472 |                 0.860077 |
| population_pretraining | NK          | mlp            |          300 |            0.429934 |              0.42807  |                 0.489441 |
| population_pretraining | NK          | identity       |          300 |            0.559461 |              0.55953  |                 0.314551 |
| population_pretraining | NK          | cell_type_mean |          300 |            0.394564 |              0.394641 |                 0.565822 |
| cinema_finetuning      | B           | mlp            |           19 |            0.363933 |              0.360882 |                 0.621037 |
| cinema_finetuning      | B           | identity       |           19 |            0.415152 |              0.410913 |                 0.518866 |
| cinema_finetuning      | B           | cell_type_mean |           19 |            0.352462 |              0.354237 |                 0.638736 |
| cinema_finetuning      | CD4 T       | mlp            |           57 |            0.380568 |              0.373122 |                 0.564144 |
| cinema_finetuning      | CD4 T       | identity       |           57 |            0.414994 |              0.407307 |                 0.489905 |
| cinema_finetuning      | CD4 T       | cell_type_mean |           57 |            0.369249 |              0.366735 |                 0.583995 |
| cinema_finetuning      | CD8 T       | mlp            |           49 |            0.404192 |              0.40385  |                 0.544389 |
| cinema_finetuning      | CD8 T       | identity       |           49 |            0.45302  |              0.451852 |                 0.454626 |
| cinema_finetuning      | CD8 T       | cell_type_mean |           49 |            0.389255 |              0.387429 |                 0.57272  |
| cinema_finetuning      | Monocyte    | mlp            |           71 |            0.219067 |              0.207115 |                 0.857595 |
| cinema_finetuning      | Monocyte    | identity       |           71 |            0.256492 |              0.246047 |                 0.801096 |
| cinema_finetuning      | Monocyte    | cell_type_mean |           71 |            0.211277 |              0.198285 |                 0.868093 |
| cinema_finetuning      | NK          | mlp            |           93 |            0.412586 |              0.410545 |                 0.533908 |
| cinema_finetuning      | NK          | identity       |           93 |            0.466022 |              0.464484 |                 0.436466 |
| cinema_finetuning      | NK          | cell_type_mean |           93 |            0.395473 |              0.394984 |                 0.566847 |

## Interpretation

Random pseudo-pairs test population-average prediction and do not represent observed before/after measurements of the same cell. CINEMA-OT pairs are inferred counterfactuals. This dataset has no donor or replicate labels, so the test results are internal cell-level evaluations rather than independent biological validation. The selected .raw HVGs are reclosed to the configured linear-space total before log1p modeling, changing their per-gene values relative to the all-gene .raw matrix.

## Figures

- `stage1_population_test.png`
- `stage2_cinema_test.png`
- `training_history.png`
- `test_pearson_by_cell_type.png`
