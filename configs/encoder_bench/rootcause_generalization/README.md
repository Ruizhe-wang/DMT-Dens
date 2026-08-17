# Distance-P row-mass root-cause generalization

This suite independently tests whether the EPI collapse mechanism generalizes to
the two pre-registered counterexample targets:

- HCL `b2_bidirectional`, seed 42;
- MNIST `b5_multi_density`, seed 42.

For each dataset, `distance_allpair` and `distance_p_rownorm_allpair` have the
same seed, data module/order, encoder, projection head, density settings,
optimizer, scheduler, FP32 precision, batch size, and 1,000 epochs. Their only
scientific difference is `distance_p_row_normalize: false | true`.

The optional `rank_allpair` and `distance_hardpair` runs are one-factor mechanism
controls. Geometry is computed from all coordinates every 50 epochs before the
legacy 3,000-point quality-metric sample. Each validation also records norm
tails, projection-row cosine, and a fixed-batch BN train/eval gap.

Regenerate and validate configs with:

```bash
python configs/encoder_bench/rootcause_generalization/generate_configs.py
```

The active 1,000-epoch configs are written to `runs_epoch1000/` and their
outputs to `outputs/rootcause_generalization_1000/`. The committed `runs/`
directory is retained as the immutable 300-epoch pilot definition.
