# Global-standardization collapse diagnostic

This seven-run seed-42 matrix implements the advisor-requested causal check
before any formal multi-seed ablation is resumed.

| Dataset | Conditions |
|---|---|
| EPI | `full`, `base`, `b2_bidirectional`, `b5_multi_density` |
| HCL | `full`, `b4_single_density` |
| MNIST | `full` |

All scientific settings are copied from the matching latent-Transformer v2
config. The only implementation changes are:

- differentiable centering and one RMS scale shared by both embedding axes;
- manifold and density loss use the exact same normalized embedding;
- kNN distance and Pearson denominator are floored at `1e-4`;
- raw axis standard deviations, axis ratio, global scale, final BatchNorm gamma,
  and unscaled gamma gradients are logged;
- the first non-finite loss, gradient, or parameter is recorded and terminates
  the run immediately.

Regenerate and validate with:

```bash
python configs/encoder_bench/ablation_global_standardization_diagnostic/generate_configs.py
python configs/encoder_bench/ablation_global_standardization_diagnostic/validate_configs.py
```

After the mean-square-before-sqrt and zero-density-skip numerical fix, generate
isolated rerun identities and artifact paths with:

```bash
python configs/encoder_bench/ablation_global_standardization_diagnostic/generate_postfix_configs.py
```

The resulting configs live under `runs_postfix/`. Their scientific settings
match the seven configs above; only run names, tags, and output destinations
change so earlier diagnostic artifacts remain intact.
