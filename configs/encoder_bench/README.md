# TopoBranch encoder benchmark configs

This directory contains standalone YAML configs for the encoder-only
benchmark. Each config inherits the current recommended settings of its
dataset and changes only the encoder selection, provisional learning rate,
run/output names, and benchmark logging callbacks.

## Encoders

- `mlp`: unchanged current DMT/MLP baseline, `lr=5e-3`.
- `resmlp`: width 512, three residual blocks, LayerNorm and GELU.
- `ft_transformer`: feature tokenizer, two layers, four heads,
  `d_token=32`.
- `latent_transformer`: protocol model with 16 latents, `d_token=64`,
  and low-rank `r=4` tokenization.
- `latent_transformer_full`: full `D -> M*d_token` capacity control. It
  must not be presented as the protocol low-rank model.

All new encoders initially use `lr=1e-3`. Run the NG20 learning-rate
sweep first, then replace this provisional value in ACT/MNIST/HCL configs
with the selected per-architecture value before formal training.

## Formal first-round matrix

| Dataset | Formal configs | Notes |
|---|---|---|
| NG20 | MLP, ResMLP, FT; latent variants optional | Select learning rate here |
| ACT | MLP, ResMLP, FT | Start with the baseline batch 5000 |
| MNIST | MLP, ResMLP, FT | Start with the baseline batch 4096 |
| HCL | MLP, ResMLP, latent, latent-full | HCL FT YAML is diagnostic only |
| MCA | latent variants only, conditionally | Run only after acceptable HCL results |

The repository provides every dataset/encoder combination so individual
smoke tests are reproducible. The existence of HCL/MCA FT configs does not
make them part of the formal matrix.

Every encoder starts with the dataset's original batch size: NG20 4096,
ACT 5000, MNIST 4096, HCL 4096, and MCA 4096. Run the required two-batch
full-pipeline smoke test on the remote 11 GB GPU. Reduce batch size only
after a real OOM, and record the final value. Do not use gradient
accumulation as a substitute for the manifold-loss batch.

## Logged outputs

| Category | Logged keys |
|---|---|
| Core | `val_visible_density_correlation`, `val_local_density_correlation`, `val_svc_acc` |
| Manifold | `val_trustworthiness`, `val_knn_preservation`, `val_distance_correlation`, `val_distance_correlation_dcor` |
| Training | `train_status/final_manifold_loss`, `train_status/final_density_loss`, non-finite and oscillation flags |
| Collapse | `val_embedding_collapsed`, `embedding/std_min`, `embedding/axis_ratio`, `val_embedding_nonfinite_fraction` |
| Capacity | `engineering/encoder_params`, total parameters, baseline ratio and in-band flag |
| Runtime | `runtime_peak_cuda_memory_mb`, `runtime_mean_epoch_time_sec`, `runtime_train_time_sec`, `runtime_fit_wall_time_sec`, `runtime_batch_size` |

The fidelity callback also logs the complete metric set under `fidelity/*`.
Runtime values are written to `results/encoder_bench/runtime_runs.csv`.
GPU/CPU/software and log-path metadata are written as JSON under
`outputs/encoder_bench/runtime_info/` and copied to the W&B run summary.
Final embeddings are saved as CSV, and the paper callback writes a 300-DPI
PNG using `final_annotation`, a deterministic `tab20` color mapping,
point size 6, and alpha 0.85 for every dataset and encoder.

## Commands

Two-batch full-pipeline smoke test:

```bash
python main.py fit \
  -c configs/encoder_bench/ng20_ft_transformer.yaml \
  --trainer.fast_dev_run=2
```

Encoder-only smoke test:

```bash
python scripts/smoke_encoder.py \
  --dataset ng20 \
  --encoder ft_transformer
```

NG20 learning-rate sweep:

```bash
wandb sweep configs/encoder_bench/sweep/encoder_bench_ng20_lr_seed42.yaml
wandb agent <entity/project/sweep-id>
```

Regenerate all standalone configs after changing a source baseline:

```bash
python configs/encoder_bench/generate_configs.py
```

Every run writes isolated checkpoints and embeddings under
`outputs/encoder_bench/`, appends runtime/VRAM results to
`results/encoder_bench/runtime_runs.csv`, and logs fidelity metrics through
the unchanged `FidelityEvalCallback`.
