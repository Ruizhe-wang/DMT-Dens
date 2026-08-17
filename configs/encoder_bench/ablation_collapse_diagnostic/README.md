# Collapse causal diagnostic

This targeted matrix must finish before the full five-seed mechanism ablation.
It separates collapse reproducibility from the known detached-standardization
gradient pathology while keeping the encoder, projection head, data,
augmentation, optimizer, losses and all mechanism settings unchanged.

| Group | Seeds | Runs | Purpose |
|---|---:|---:|---|
| `current` | 43, 44 | 20 | Measure collapse reproducibility under the historical backward pass |
| `fixed` | 42 | 10 | Test differentiable float32 standardization with a `1e-4` std floor |

Machine assignment is encoded as four sweeps so each dataset's current/fixed
comparison remains on one host:

- EPI current/fixed: `d_c82_vm`
- HCL and MNIST current/fixed: `d_c90_vm`

Generate and validate:

```bash
python configs/encoder_bench/ablation_collapse_diagnostic/generate_configs.py
python configs/encoder_bench/ablation_collapse_diagnostic/validate_configs.py
```
