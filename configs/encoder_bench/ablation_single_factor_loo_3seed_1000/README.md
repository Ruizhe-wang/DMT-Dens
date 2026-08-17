# Strict single-factor leave-one-out ablation: three seeds, 1000 epochs

This family contains the matched `full` control and the five leave-one-out
variants `r1` through `r5`. It covers HCL, EPI, and MNIST with seeds 42, 43,
and 44: 54 runs total. The original 45 leave-one-out runs are retained; the
nine `full` controls are launched separately by the supplement launcher.

The scientific configuration is copied from the stabilized v2 seed-42 runs.
Only `seed_everything`, run identity, W&B metadata, and output paths differ.
P row normalization remains disabled.

Generate and validate:

```bash
python configs/encoder_bench/ablation_single_factor_loo_3seed_1000/generate_configs.py
python configs/encoder_bench/ablation_single_factor_loo_3seed_1000/validate_configs.py
```

W&B project: `TopoBranch_single_factor_loo_3seed_1000`.

The launcher runs one dataset/seed comparison group at a time, with its five
variants in parallel:

```bash
bash scripts/run_single_factor_loo_3seed_1000.sh 0,1,2,3,4 dry-run
bash scripts/run_single_factor_loo_3seed_1000.sh 0,1,2,3,4 train
```

Run only the nine matched `full` controls (three seeds in parallel per
dataset):

```bash
bash scripts/run_single_factor_full_3seed_1000.sh 0,1,2 dry-run
bash scripts/run_single_factor_full_3seed_1000.sh 0,1,2 train
```

W&B credentials are runtime environment variables and must not be committed.
