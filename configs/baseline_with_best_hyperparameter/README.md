# Baseline Configurations With Best Observed Hyperparameters

These YAML files select, for every baseline method on every dataset, the
hyperparameter combination with the highest **`val_score`** observed in the
2026-06-30 W&B `hp_sweep` export:

```text
D:/ruizhe/result/baseline/hp_sweep/wandb_export_2026-06-30T17_13_41.932+08_00.csv   (act, epi, tree)
D:/ruizhe/result/baseline/hp_sweep/wandb_export_2026-06-30T17_14_58.949+08_00.csv   (emnist, gast10k, hcl, mca, ng20, mnist)
```

Selection metric: the W&B `val_score` column directly (higher is better). For
each `(method, dataset)` the run maximizing `val_score` over the `p1 x p2` grid
is chosen, and its grid indices are decoded into the real hyperparameters via the
grids in `model/baseline_tri.py`.

## Grids (model/baseline_tri.py)

| Method         | p1 axis                              | p2 axis                                       |
|----------------|--------------------------------------|-----------------------------------------------|
| t-SNE / densNE | `perplexity` [15,30,50,80,120,200]   | `early_exaggeration` [4,8,12,18,24,32]        |
| UMAP / densMAP | `n_neighbors` [10,15,20,40,80,120]   | `min_dist` [0.001,0.01,0.05,0.08,0.15,0.3]    |
| PaCMAP         | `n_neighbors` [10,15,20,40,80,120]   | `(mn_ratio,fp_ratio)` pair-ratio grid         |
| PHATE          | `knn` [5,10,15,20,40,80]             | `decay` [10,20,40,80,120,160]                 |

The sweep used indices `p1, p2 ∈ {0..4}` (a 5×5 = 25-point grid per cell).

## Scope

- **53 method-dataset configurations** written: 6 baselines × 9 datasets
  (`act, epi, tree, emnist, gast10k, hcl, mca, ng20, mnist`), minus `densne/emnist`.
- **Missing `densne/emnist`**: den-SNE has no EMNIST run in the sweep (it exceeds
  the 24 h budget — the OOT cell in the paper's Table 1).
- **No AQC**: the sweep contains no AQC runs.
- Every cell except `densne/emnist` has the full 25-point grid; that single gap is
  intentional.

## Data-block provenance and caveat

- For `emnist, gast10k, hcl, mca, ng20, mnist` the data block is reused from the
  prior configs in this directory (path convention `data_path: /zangzelin/data`,
  and `data: data` for mnist).
- For `act, epi, tree` the data block is taken verbatim from the swept base configs
  in `configs/dmtme_dataset_baselines/`. **These use a different mount**
  (`/usr/storage/ruizhe/...`). Adjust `data_path` to your environment before
  running if it differs from the other files here.

## densNE note

densNE shares the t-SNE `(perplexity, early_exaggeration)` grid. Confirm that the
native backend forwards `early_exaggeration` before treating the p2 axis as fully
tuned for densNE.

## Run

```bash
wandb sweep configs/baseline_with_best_hyperparameter/baseline_with_best_hyperparameter_sweep.yaml
wandb agent <entity>/DiffTree_rz/<sweep_id>
```
