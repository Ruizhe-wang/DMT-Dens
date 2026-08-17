# baseline_with_new_metrics

Re-run of the final **5-seed** experiments (6 baselines + DiffTree) so that the two new
**independent density metrics** — Area–Spread Recovery (ASR) and Fixed-Radius Density
Correlation (FR-DC) — are logged alongside the existing metrics. See
`new_density_metrics_spec.md` (handed to the metrics-implementation agent) for the metric
definitions; this folder only contains the run configs.

## Folder contents

```
baseline_with_new_metrics/
├── README.md
├── select_best_hparams.py                  # how best baseline (p1,p2) are chosen
├── apply_best_hparams.py                    # writes the winners into the per-dataset configs
├── best_hparams.csv                         # current winners (5 datasets done; tree/epi/act pending)
├── baseline_hparam_sweep_tree_epi_act.yaml  # top-up SEARCH for the 3 missing datasets
├── baseline_5seed_new_metrics_sweep.yaml   # 6 methods × 8 datasets × 5 seeds
├── difftree_5seed_new_metrics_sweep.yaml   # DiffTree × 8 datasets × 5 seeds
├── tsne|umap|pacmap|phate|densmap|denssne/ # per-dataset baseline configs (best hp baked in)
│   └── {tree,hcl,gast10k,mca,epi,emnist,ng20,act}.yaml
└── difftree/
    └── {tree,hcl,gast10k,mca,epi,emnist,ng20,act}.yaml
```

Datasets are the paper's 8: `tree`(ArtificialTree), `hcl`, `gast10k`, `mca`, `epi`,
`emnist`, `ng20`, `act`. AQC, MNIST, and JAX are intentionally excluded (AQC removed from
the paper; MNIST/JAX not in the main comparison). `denssne` is den-SNE.

## How the best hyperparameters were selected

We do **not** re-tune the baselines against the new metrics. Selecting a baseline's config
by ASR/FR-DC would make the new comparison circular (in reverse) and unfair. Instead:

> **IMPORTANT — the configs in this folder currently hold per-method DEFAULT `p1/p2`,
> identical across datasets.** The source configs
> `configs/dmtme_dataset_baselines/<method>/<dataset>.yaml` were sweep *templates* that all
> carry one default `(p1,p2)` per method (t-SNE 2,2; UMAP 1,2; …), and the original Table-1
> runs used those defaults — i.e. baselines were **not** tuned per dataset. The per-dataset
> sweep winners were never written back and are **not** recoverable from the local CSVs
> (`codex.csv` has no `p1/p2` columns). They must be re-derived from wandb. Until you do the
> two steps below, this folder is NOT per-dataset-tuned.

1. **Recover the per-dataset best config from the sweep:**
   - Re-export the baseline sweep from wandb **including the `model.init_args.p1` and
     `model.init_args.p2` columns** (codex.csv lacks them).
   - `python select_best_hparams.py SWEEP_EXPORT.csv` → `best_hparams.csv`.
   - `python apply_best_hparams.py best_hparams.csv` → writes each dataset's winning
     `(p1,p2)` into the per-dataset configs here.

2. **Selection rule (per method × dataset)** — the rule used to pick the best config from the
   sweep, reproduced in `select_best_hparams.py` so it can be re-derived or audited:
   - Within a (method, dataset) cell, min–max normalize **density correlation** and
     **SVC accuracy** across the swept configs to `[0, 1]`.
   - `score = 0.5 · norm(density_corr) + 0.5 · norm(svc_acc)` (balanced density vs.
     separability — the two axes the paper compares on; this gives every baseline its
     fairest single operating point on the trade-off).
   - Pick `argmax(score)`; ties broken by higher density correlation.
   - The selection uses **only the pre-existing metrics**, never ASR/FR-DC, so the new
     metrics remain an independent test.

3. The grid that `p1`/`p2` index (from the sweep header) is:
   - t-SNE / den-SNE: `p1=perplexity[15,30,50,80,120,200]`, `p2=early_exaggeration[4,8,12,18,24,32]`
   - UMAP / densMAP: `p1=n_neighbors[10,15,20,40,80,120]`, `p2=min_dist[0.001,0.01,0.05,0.08,0.15,0.3]`
   - PaCMAP: `p1=n_neighbors[...]`, `p2=(MN_ratio,FP_ratio) pairs`
   - PHATE: `p1=knn[5,10,15,20,40,80]`, `p2=decay[10,20,40,80,120,160]`

## Current tuning status (as of the 2026-06-25 export)

The 2026-06-25 wandb export (5×5 grid, 25 configs/cell) covered only **5 of 8** datasets,
so `apply_best_hparams.py` has tuned those: **hcl, gast10k, mca, ng20, emnist** (all 6
methods; den-SNE/emnist is OOT). Still on per-method DEFAULT and needing a sweep:
**tree, epi, act** (all 6 methods). To finish:

```bash
# 1. run the top-up search for the 3 missing datasets (6 × 3 × 25 = 450 runs)
wandb sweep configs/baseline_with_new_metrics/baseline_hparam_sweep_tree_epi_act.yaml
wandb agent <entity>/DiffTree_rz/<sweep_id>
# 2. export those runs WITH model.init_args.p1/p2 columns, then re-select across BOTH exports
python select_best_hparams.py main_export.csv tree_epi_act_export.csv   # merges -> best_hparams.csv
python apply_best_hparams.py best_hparams.csv                            # patches all 47 cells
```

## Known gaps in the original baseline sweep (audit of result/baseline/codex.csv)

- **den-SNE × EMNIST**: sweep not done (0 runs) — den-SNE is OOT on the 698K-point EMNIST
  (24 h budget). It is kept in the sweep file for completeness but is expected to OOT; mark
  it OOT in the table as before. Consider dropping `denssne/emnist.yaml` from the sweep if
  you do not want the wasted runs.
- **den-SNE × EPI**: only one sweep batch (~5 configs) vs. two (~10) elsewhere — its best
  config is selected from a smaller grid. If you want parity, complete the second batch
  before the final run; otherwise its current best is used.
- All other 46 (method × dataset) cells completed the full sweep.

## DiffTree hyperparameters

The DiffTree sweep fixes the paper's final density configuration
`density_k=12, density_num_anchors=512, density_weight=0.0018` (= λ_d=1.8e-3, k=12,
A₀=512), matching `dmtme_dataset_weighted_sweep_12_512_0018_5seed.yaml`.

## Required code change before running

The new metrics are computed in `callbacks.Eval_density.FidelityEvalCallback`. The configs
here pass these extra `init_args`, which the callback **must be extended to accept**
(implement per `new_density_metrics_spec.md`):

```yaml
down_sample: 10000        # raised from 3000 so FR-DC has enough eval points
compute_asr: true
asr_min_class_size: 20
asr_min_classes: 8
compute_frdc: true
frdc_target_count: 50
frdc_n_eval: 10000
frdc_m_ref: 50000
```

If the callback is not yet extended, jsonargparse will error on the unknown keys — extend
the callback first, or temporarily remove the new keys.

## How to run

```bash
# 0. FIRST tune baselines per dataset (see "How the best hyperparameters were selected"):
python select_best_hparams.py SWEEP_EXPORT.csv     # -> best_hparams.csv
python apply_best_hparams.py best_hparams.csv        # patch the 48 configs

# baselines (6 × 8 × 5 = up to 240 runs)
wandb sweep configs/baseline_with_new_metrics/baseline_5seed_new_metrics_sweep.yaml
wandb agent <entity>/DiffTree_rz/<sweep_id>

# DiffTree (8 × 5 = 40 runs)
wandb sweep configs/baseline_with_new_metrics/difftree_5seed_new_metrics_sweep.yaml
wandb agent <entity>/DiffTree_rz/<sweep_id>
```

After the runs finish, export the logged `fidelity/*` metrics (including ASR and FR-DC) and
build the new comparison tables exactly as for Table 1.
