# Latent Transformer mechanism ablation

This family applies the existing TopoBranch mechanism-ablation design to the
current latent-bn encoder on HCL, EPI and MNIST.  The encoder and all settings
outside the named mechanism switches come from the seed-42 paper reruns.

The 12 variants are unchanged from
`configs/ablation/experiment1_mechanism/_generate.py`:

| Variant | Affinity | Direction | Pair selection | Density |
|---|---|---|---|---|
| `full` | rank | bidirectional | hard Top-100 | multi-scale |
| `r1_distance` | distance | bidirectional | hard Top-100 | multi-scale |
| `r2_unidirectional` | rank | unidirectional | hard Top-100 | multi-scale |
| `r3_allpair` | rank | bidirectional | all pairs | multi-scale |
| `r4_single_density` | rank | bidirectional | hard Top-100 | single-scale |
| `r5_no_density` | rank | bidirectional | hard Top-100 | off |
| `base` | distance | unidirectional | all pairs | off |
| `b1_rank` | rank | unidirectional | all pairs | off |
| `b2_bidirectional` | distance | bidirectional | all pairs | off |
| `b3_hardpair` | distance | unidirectional | hard Top-100 | off |
| `b4_single_density` | distance | unidirectional | all pairs | single-scale |
| `b5_multi_density` | distance | unidirectional | all pairs | multi-scale |

Generate and validate with:

```bash
python configs/encoder_bench/ablation_mechanism_latent/generate_configs.py
python configs/encoder_bench/ablation_mechanism_latent/validate_configs.py
```

The first-round sweep contains 36 runs: three datasets, 12 variants and seed
42.  Density-on variants use `density_weight=0.0018`, `density_k=12` and 512
anchors.  Multi-scale density evaluates k=6 and k=12; single-scale uses k=12.
