# Latent mechanism ablation v2

This is a full seed-42 rerun of the 36-config legacy mechanism ablation after
numerical-stability fixes. The data, encoder, projection head, objective,
mechanism combinations, optimizer, learning rate, batch size, epochs and
callbacks are identical to v1.

The implementation fixes do not change the scientific objective:

- row-normalized Student-t affinity is evaluated in float32 under AMP;
- kNN log-density is evaluated in float32 so its `1e-8` radius floor remains
  representable;
- the mathematically zero diagonal negative-BCE term is masked before `log1p`;
- paper plots use an explicit fixed square canvas instead of a tight bounding
  box that can fail on collapsed coordinates.

Generate and validate:

```bash
python configs/encoder_bench/ablation_mechanism_latent_v2/generate_configs.py
python configs/encoder_bench/ablation_mechanism_latent_v2/validate_configs.py
```
