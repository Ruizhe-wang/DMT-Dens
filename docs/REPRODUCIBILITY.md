# Manuscript reproducibility map

Run every command from the repository root. Historical YAML files preserve the
resolved experiment settings, including the original machine path. Replace only
the data root at launch:

```bash
python main.py fit --config CONFIG.yaml \
  --data.init_args.data_path /absolute/path/to/dmt-dens-data
```

## Final DMT-Dens architecture

The manuscript configuration uses 32 latent tokens, latent rank 16, token width
224, two pre-normalized Transformer blocks, four attention heads, feed-forward
ratio 4, no dropout, final layer normalization, mean pooling, and a 40-dimensional
representation before the two-dimensional projection head.

The main optimization settings are 1000 epochs, AdamW, learning rate 0.001,
cosine annealing with warmup, batch size 4096, mixed precision, `density_k=12`,
`density_num_anchors=512`, and `density_weight=0.0018`.

## Configuration source of truth

| Manuscript analysis | Resolved configuration family |
| --- | --- |
| Main benchmark, seeds 42–46 | `configs/encoder_bench/sweep_e18/runs/<dataset>_latent_bn_seed<seed>.yaml` |
| Corrected ACT batch-4096 runs | `configs/encoder_bench/act_bs4096_5seed_1000/runs/` |
| Selected baselines | `configs/baseline_with_best_hyperparameter/` |
| Density-weight refinements | `configs/baseline_with_best_hyperparameter/{densmap_lambda,densne_lambda}/` |
| Single-factor ablations, seeds 42–44 | `configs/encoder_bench/ablation_single_factor_loo_3seed_1000/runs/` |
| CELEGAN and dyngen five-seed runs | `configs/encoder_bench/case_study_latent_5seed/runs/` |
| Runtime scaling | `configs/runtime/latent_transformer_150epoch/runs/` |

Main benchmark dataset slugs are `tree`, `act`, `emnist`, `mnist`, `ng20`,
`epi`, `gast10k`, `hcl`, and `mca`. Use the corrected ACT directory instead of
the earlier E18 ACT files for manuscript reproduction.

## Examples

```bash
# HCL, seed 42
python main.py fit \
  --config configs/encoder_bench/sweep_e18/runs/hcl_latent_bn_seed42.yaml \
  --data.init_args.data_path /data/dmt-dens

# ACT corrected batch-size run, seed 42
python main.py fit \
  --config configs/encoder_bench/act_bs4096_5seed_1000/runs/act_latent_bn_bs4096_seed42_c90_20260814.yaml \
  --data.init_args.data_path /data/dmt-dens
```

W&B is used for experiment aggregation in resolved configs. To run without an
online account, override the logger or set `WANDB_MODE=offline`. Outputs and
checkpoints are written below `outputs/`, `results/`, and `wandb/`; all are
ignored by Git.

## Verification levels

1. **Asset test:** validates public metadata, YAML, and generated toy data.
2. **CPU smoke test:** performs two short optimization batches for two epochs.
3. **Paper run:** uses a resolved five-seed GPU configuration and public data.

Record the Git commit, environment export, dataset checksums, GPU model, CUDA
version, and seed with every paper-scale execution.
