# Baseline Configurations With Best Observed Hyperparameters

These YAML files contain the selected baseline hyperparameters used by the
manuscript. For each `(method, dataset)` pair, the retained configuration is
the candidate with the highest validation score over the `p1 x p2` grid. Grid
indices are decoded into the real hyperparameters by `model/baseline_tri.py`.

## Grids (model/baseline_tri.py)

| Method         | p1 axis                              | p2 axis                                       |
|----------------|--------------------------------------|-----------------------------------------------|
| t-SNE / densNE | `perplexity` [15,30,50,80,120,200]   | `early_exaggeration` [4,8,12,18,24,32]        |
| UMAP / densMAP | `n_neighbors` [10,15,20,40,80,120]   | `min_dist` [0.001,0.01,0.05,0.08,0.15,0.3]    |
| PaCMAP         | `n_neighbors` [10,15,20,40,80,120]   | `(mn_ratio,fp_ratio)` pair-ratio grid         |
| PHATE          | `knn` [5,10,15,20,40,80]             | `decay` [10,20,40,80,120,160]                 |

The sweep used indices `p1, p2 ∈ {0..4}` (a 5×5 = 25-point grid per cell).

## Scope

- **70 retained configurations** cover the six primary baseline methods plus
  the density-weight refinements used in the manuscript.
- The datasets are `act, epi, tree, emnist, gast10k, hcl, mca, ng20, mnist`.
- **Missing `densne/emnist`**: den-SNE has no EMNIST run in the sweep (it exceeds
  the 24 h budget — the OOT cell in the paper's Table 1).
- **No AQC**: the sweep contains no AQC runs.
- Every cell except `densne/emnist` has the full 25-point grid; that single gap is
  intentional.

## Data-path caveat

The resolved files retain the data roots used for the reported experiments.
Override `--data.init_args.data_path` at launch to point to the public dataset
layout documented in `docs/DATA.md`.

## densNE note

densNE shares the t-SNE `(perplexity, early_exaggeration)` grid. Confirm that the
native backend forwards `early_exaggeration` before treating the p2 axis as fully
tuned for densNE.

## Run

```bash
python main.py fit \
  --config configs/baseline_with_best_hyperparameter/umap/hcl.yaml \
  --data.init_args.data_path /data/dmt-dens
```
