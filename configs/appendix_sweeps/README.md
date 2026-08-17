# appendix_sweeps

Self-contained folder of wandb **sweep** files for the paper appendix. Point
`scripts/run_wandb_multidata.sh`'s `YAML_DIR` at this folder and run it; the
script's `*.yaml` loop will create + launch each sweep below. (Base configs are
referenced by repo-relative path and deliberately live elsewhere, so the loop
does not try to `wandb sweep` them.)

| file | what it sweeps | runs |
|------|----------------|------|
| `sweep_baselines_structural_celegan.yaml` | all 6 baselines on celegan, structural params (p1 x p2) | 216 (CPU) |
| `sweep_baselines_denslambda_celegan.yaml` | densNE/densMAP on celegan, density strength (p1 x dens_lambda) | 72 (CPU) |
| `sweep_difftree_runtime_5seed.yaml` | DiffTree across datasets/sizes x 5 seeds, records hardware/GPU/CPU/peak-mem/runtime/log-paths | 135 (GPU) |

### Celegan case-study final comparison (5 seeds 41-45, best config per method)
**One sweep** covers the whole comparison: DiffTree + all 8 baselines, each at its
own best config, x 5 seeds = 45 runs.

| file | what it runs | runs |
|------|--------------|------|
| `sweep_all_methods_best_5seed_celegan.yaml` | DiffTree + 8 baselines at best config x seeds 41-45 | 45 (DiffTree GPU, baselines CPU) |

A single `seed_everything: [41..45]` axis drives every method:
- DiffTree is seeded directly by `seed_everything`.
- The baked baseline configs set `random_state: null`; `model/baseline_tri.py`
  resolves a null `random_state` to `PL_GLOBAL_SEED` (exported by Lightning's
  seed_everything), so baselines follow the same seed axis -- as an int, which the
  native densNE backend requires.

Each method's hyperparameters + run name are baked into its `*_best5seed` base
config (in `configs/case_study_hyperparameter_sweep/`), so the sweep only varies
the seed (no per-method params, which would break methods that don't accept them):

`difftree_celegan_best5seed.yaml` -- density_k=12, anchors=1000, weight=0.01.
`baseline_<method>_celegan_best5seed.yaml` -- phate p1=3,p2=0 · tsne p1=3,p2=0 ·
umap p1=0,p2=0 · pacmap p1=0,p2=5 · densmap p1=5,p2=0 · densne p1=3,p2=4 ·
densmap_denslambda p1=5,λ=4.0 · densne_denslambda p1=4,λ=4.0.

## Notes
- The two baseline sweeps log to project `DiffTree_rz`; the runtime sweep logs to
  `DiffTree_runtime` (matching its base configs). Adjust if you want them unified.
- The runtime sweep appends `RuntimeInfoCallback` via `--trainer.callbacks+=...`
  so the 27 runtime base configs need no edits. It complements the
  `RuntimeProfilerCallback` already in those configs (timing + CUDA peak); the
  appended callback adds hardware, CPU peak RSS, software versions, and log paths.
- `dens_lambda` requires the `model/baseline_tri.py` change already on `main`.
