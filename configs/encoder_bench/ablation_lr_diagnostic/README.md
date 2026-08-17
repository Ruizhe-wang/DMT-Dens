# Ablation collapse learning-rate diagnostic

This matrix tests whether the collapse/non-finite events in the latent-encoder
mechanism ablation are caused by an overly aggressive optimizer step.

- Seed: 42
- Learning rates: `1e-3` and `3e-4`
- Epochs: 1000
- Total: 11 conditions x 2 learning rates = 22 runs
- W&B project: `TopoBranch_latent_ablation_lr_diagnostic`
- Machine: c89 for both members of every pair

The historical detached-statistics embedding standardization is explicitly
kept enabled (`stable_embedding_standardization: false`).  Do not combine this
matrix with the separate standardization fix: that would confound the cause of
any collapse reduction.

Targets are the six seed-42 collapse conditions, three stable `full` controls,
and the two EPI non-finite sentinels (`r1_distance`, `r2_unidirectional`).

Generate and validate:

```bash
python configs/encoder_bench/ablation_lr_diagnostic/generate_configs.py
python configs/encoder_bench/ablation_lr_diagnostic/validate_configs.py
```

Pass criteria for `3e-4`: no collapse/non-finite event in the historical
failure conditions, no systematic degradation in the three full controls, and
no core-metric decrease larger than 0.03 absolute or 5% relative.
