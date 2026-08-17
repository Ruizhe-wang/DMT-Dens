# DMT-Dens

**Density-preserving manifold visualization for biological data**

DMT-Dens is a parametric two-dimensional manifold-visualization method built
around a latent-token Transformer. It combines bidirectional rank-affinity
alignment, hard-pair optimization, and a multi-scale density-consistency loss.
The repository accompanies the manuscript *DMT-Dens: Density-preserving
manifold visualization for biological data*.

The public entry point is `main.py`. Experiments are configured with YAML files
through LightningCLI. Labels are used for evaluation and hyperparameter
selection, never as inputs to representation training.

## Quick start

Linux with Python 3.10 is the primary supported environment. A CUDA-capable GPU
is recommended for paper-scale runs; the smoke test runs on CPU.

```bash
conda env create -f environment.yml
conda activate dmt-dens

python scripts/generate_toy_data.py
python main.py fit --config configs/publication/toy_cpu.yaml
```

The first command creates `examples/data/toy_branches.npz`. The smoke
configuration deliberately uses a smaller encoder and only two epochs so that
it checks installation, data loading, neighbor construction, and optimization
without reproducing a paper result.

To inspect all CLI overrides:

```bash
python main.py fit --help
```

## Use your own matrix

The publication data module accepts either:

- an `.npz` file containing `X` (observations by features) and optionally `y`;
- an `.h5ad` file, with labels optionally stored in `adata.obs`.

```bash
python main.py fit --config configs/publication/toy_cpu.yaml \
  --data.init_args.data_path /path/to/dataset.npz \
  --model.init_args.num_input_dim 2000 \
  --model.init_args.num_train_data 12000 \
  --trainer.accelerator gpu \
  --trainer.precision 16-mixed \
  --trainer.max_epochs 1000
```

`num_input_dim` and `num_train_data` must match the processed matrix. The
paper-scale architecture and optimizer settings are documented in
`docs/REPRODUCIBILITY.md`; do not use the reduced smoke-test architecture for a
scientific comparison.

## Reproduce the manuscript

Resolved configurations for the nine-dataset benchmark, five-seed case
studies, baselines, and three-seed ablations are already included. Start with:

- `docs/REPRODUCIBILITY.md` — exact configuration families and launch commands;
- `docs/DATA.md` — accepted public-data layout and expected filenames;
- `configs/publication/toy_cpu.yaml` — self-contained functional check;
- `configs/encoder_bench/sweep_e18/runs/` — final latent-token benchmark runs;
- `configs/encoder_bench/ablation_single_factor_loo_3seed_1000/runs/` — final
  single-factor ablations.

Machine-specific paths in historical experiment configurations are not needed:
override them with `--data.init_args.data_path /your/data/root`. New public
configs should use relative paths.

## Repository map

| Path | Purpose |
| --- | --- |
| `model/DiffTreeVQ_density.py` | DMT-Dens objective and Lightning module |
| `model/encoder_transformer.py` | latent-token Transformer encoder |
| `model/encoder_factory.py` | encoder construction and capacity accounting |
| `data_model/public_data.py` | path-neutral `.npz` and `.h5ad` loader |
| `callbacks/` | embedding export, evaluation, and runtime callbacks |
| `eval/` | density and manifold-quality metrics |
| `configs/` | resolved training, baseline, ablation, and case-study configs |
| `tests/` | numerical and regression tests |

Older exploratory configurations remain in the repository for auditability. The
paths listed in `docs/REPRODUCIBILITY.md` are the manuscript source of truth.

## Validation

```bash
python -m pytest -q tests/test_publication_assets.py
python -m pytest -q
python -m compileall -q main.py model data_model callbacks eval
```

The full test suite imports PyTorch and Lightning. The publication-asset test is
lightweight and also checks that the public smoke configuration contains no
private absolute path.

## Citation

Citation metadata are provided in `CITATION.cff`. Until the journal article has
a DOI, cite the software repository and the accompanying DMT-Dens manuscript.

## License

DMT-Dens is released under the MIT License. See `LICENSE`.

## Contact

Zelin Zang — <zangzelin@gmail.com>
