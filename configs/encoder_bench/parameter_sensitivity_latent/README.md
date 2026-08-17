# Latent-Transformer parameter sensitivity

This family regenerates the parameter-sensitivity experiment for the final
latent-Transformer paper model on GAST10K, MCA, HCL, and NG20.

Each seed contains 20 unique settings per dataset. The five-seed matrix uses
seeds 42--46 and therefore contains 400 unique dataset/config/seed combinations:

- density anchors: 128, 256, 512, 768, 1024, 1536;
- training density neighborhood k: 5, 10, 12, 15, 20, 25, 30, 40;
- density weight: 0.0001, 0.0005, 0.001, 0.0018, 0.003, 0.006, 0.01, 0.02.

The paper operating point `anchors=512, k=12, lambda_d=0.0018` is generated
once and shared by all three sensitivity axes. Evaluation always uses
`FidelityEvalCallback(density_k=15, knn_k=12)`, so changing the training k does
not change the metric definition.

All other architecture and training settings match the E18 latent-bn paper
configuration. In particular, this stage explicitly records the historical
embedding standardization (`stable_embedding_standardization=false`) used by
the current five-seed paper results. Labels are used only by post-training
evaluation.

## Execution split

The original `*_seed42.yaml` sweeps are retained for provenance. To avoid
silently duplicating valid runs, the five-seed completion is launched as:

- four `*_seeds43_46.yaml` sweeps: 80 runs per dataset, 320 total;
- `gast10k_seed42_retry.yaml`: the crashed `k=40` setting;
- `hcl_seed42_retry.yaml`: the 12 settings whose first run OOMed;
- `ng20_seed42_retry.yaml`: the 18 settings whose first run OOMed.

MCA completed all 20 unique seed-42 settings and therefore receives no retry
sweep. The completed result matrix is the union of valid original seed-42 runs,
isolated seed-42 retries, and the seed-43--46 sweeps. HCL/NG20 remain on c82 and
MCA/GAST10K remain on c90.

Generate and validate with:

```bash
python configs/encoder_bench/parameter_sensitivity_latent/generate_configs.py
python configs/encoder_bench/parameter_sensitivity_latent/validate_configs.py
```
