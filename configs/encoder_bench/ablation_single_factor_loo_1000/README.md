# Strict single-factor leave-one-out ablation (1000 epochs)

This family reruns only the five strict leave-one-out variants from the full
model on HCL, EPI, and MNIST with seed 42 and 1000 epochs. Each dataset's five
runs must stay on one remote machine so framework-version differences do not
enter a within-dataset comparison.

| Variant | Rank | Bidirectional | Hard-pair | Single density | Multi density |
| --- | :---: | :---: | :---: | :---: | :---: |
| `r1_distance` |  | yes | yes |  | yes |
| `r2_unidirectional` | yes |  | yes |  | yes |
| `r3_allpair` | yes | yes |  |  | yes |
| `r4_single_density` | yes | yes | yes | yes |  |
| `r5_no_density` | yes | yes | yes |  |  |

Unchecked mechanisms use the corresponding baseline choice: distance affinity,
unidirectional affinity, all-pair loss, or no density loss.

Generate the 15 committed run configs with:

```bash
python configs/encoder_bench/ablation_single_factor_loo_1000/generate_configs.py
python configs/encoder_bench/ablation_single_factor_loo_1000/validate_configs.py
```

W&B project: `TopoBranch_single_factor_loo_1000`.

The W&B API key and base URL are runtime environment variables and must never be
stored in these configs or committed to the repository.

After setting those runtime variables, launch five runs per dataset in parallel
and process the three dataset groups sequentially with:

```bash
bash scripts/run_single_factor_loo_1000.sh 0,1,3,4,5
```

Before the real launch, run the same 15-config schedule for two batches each
without W&B or checkpointing:

```bash
bash scripts/run_single_factor_loo_1000.sh 0,1,3,4,5 dry-run
```
