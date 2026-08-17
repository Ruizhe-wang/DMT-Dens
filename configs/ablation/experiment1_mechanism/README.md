# Experiment 1 — Mechanism-decomposition ablation

Goal: show that DMT-Dens is an **independent parametric manifold-learning
objective** whose mechanisms each improve a **different** structural axis — *not*
"densMAP with a different density loss". Each mechanism is toggled independently
and every variant is evaluated on the **full metric suite** (density +
neighborhood + global + separability, plus trajectory metrics on trajectory
datasets), not on density correlation alone.

Mechanism → contribution mapping (per advisor framing):

| Mechanism (toggle) | Paper role | What its ablation must show |
|---|---|---|
| Bidirectional rank affinity (`manifold_affinity`, `manifold_symmetric`) | **Contribution 1** — the main body; the fundamental difference from densMAP's UMAP graph objective | robust neighborhood ordering: kNN preservation, continuity, SPIR, global distance |
| Hard-pair BCE with Student-t (`manifold_hardpair`) | **Key mechanism 2** | fewer pseudo-neighbors, better boundaries/continuity: SPIR, continuity, kNN preservation, SVC — **not density alone** |
| Multi-scale density consistency (`density_scale_mode`, `density_weight`) | **Contribution 3** — reformulation, not a brand-new density loss | density corr, local density corr; multi > single scale |
| Joint objective (full ↔ base spectrum) | **Contribution 4** | best *joint* balance across all axes (mean rank / Pareto) + reusable mapping |

Total objective: L = L_rank-manifold + λ_d · L_multi-scale-density.

## Mechanisms (config knobs)

Added to `model/DiffTreeVQ_density.py` (`DMTEVT_model.__init__`); defaults
reproduce the original hard-coded behaviour, so the `full` variant matches the
existing `configs/ablation/component/*_full.yaml` runs.

| Knob | Values | Factor |
|---|---|---|
| `manifold_affinity` | `rank` \| `distance` | M1-a: rank kernel vs absolute-distance Gaussian kernel |
| `manifold_symmetric` | `bidirectional` \| `unidirectional` | M1-b: geometric-mean two-view symmetrization vs single view |
| `manifold_hardpair` | `true` \| `false` | M2: hard-pair mining vs mean over all pairs |
| `hardpair_k` | int (default 100) | M2: number of hardest pairs kept per row |
| `density_weight` / `density_scale_mode` | `0.0` / `single` / `multi` | M3: no / single-scale / multi-scale density |

`distance` affinity = per-row bandwidth-adaptive Gaussian on the (squared)
distance, `exp(-d_ij / mean_j d_ij)` — the counterfactual to the rank kernel
(`cal_dis_to_p_distance`).

## Variant matrix (12 per dataset)

Design = **remove-one-from-full** (necessity) + **add-one-to-base** (sufficiency).
The density axis carries three levels (no / single / multi) at both ends so the
multi-scale claim (contribution 3) is testable.

| Variant | affinity | symmetric | hardpair | density | Role |
|---|---|---|---|---|---|
| `full` | rank | bidirectional | ✓ | multi | full model (anchor) |
| `r1_distance` | distance | bidirectional | ✓ | multi | −rank |
| `r2_unidirectional` | rank | unidirectional | ✓ | multi | −bidirectional |
| `r3_allpair` | rank | bidirectional | ✗ | multi | −hard-pair |
| `r4_single_density` | rank | bidirectional | ✓ | single | multi→single (multi-scale test) |
| `r5_no_density` | rank | bidirectional | ✓ | no | −density |
| `base` | distance | unidirectional | ✗ | no | minimal base (anchor) |
| `b1_rank` | rank | unidirectional | ✗ | no | base +rank |
| `b2_bidirectional` | distance | bidirectional | ✗ | no | base +bidirectional |
| `b3_hardpair` | distance | unidirectional | ✓ | no | base +hard-pair |
| `b4_single_density` | distance | unidirectional | ✗ | single | base +single-scale density |
| `b5_multi_density` | distance | unidirectional | ✗ | multi | base +multi-scale density |

Density axis is clean at both ends: `full`(multi)/`r4`(single)/`r5`(no) and
`base`(no)/`b4`(single)/`b5`(multi).

Datasets (10): `act`, `aqc`, `emnist`, `epi`, `gast10k`, `hcl`, `mca`, `mnist`, `ng20`, `tree`
→ **120 configs**. (`mnist` uses `data_model.M1datamodel_mnist_v2.DMTBaseDataModule`;
its `mnist_full` template lives in `../component/mnist_full.yaml`.)

Trajectory metrics (topology fidelity, DEMaP, pseudotime, trajectory continuity)
are NOT produced by these 9 datasets' `FidelityEvalCallback`; run the same
variants on `dyngen` / `celegan` with `dyngen_gt_metrics_callback` /
`celegan_gt_metrics_callback` to obtain them (separate config set).

Expected reading (must be supported by data): removing **hard-pair** should drop
neighborhood/separability metrics (kNN preservation, continuity, SVC) while
leaving density roughly unchanged; removing **density** should drop the density
metrics while leaving neighborhood metrics roughly unchanged — a double
dissociation demonstrating the mechanisms are orthogonal.

## Regenerating

Configs are produced from the per-dataset `../component/<ds>_full.yaml`
templates (which carry the correct data settings):

```bash
python configs/ablation/experiment1_mechanism/_generate.py
```

## Running one config

```bash
python main.py fit --config configs/ablation/experiment1_mechanism/gast10k_full.yaml
```

Outputs are written under `outputs/ablation/experiment1/{checkpoints,plots}/<ds>/<variant>/`
and logged to W&B run `exp1_<ds>_<variant>` (project `DiffTree_rz`).

## Reporting

For every variant report, per dataset, mean±sd over 5 seeds:
density corr., local density corr., kNN preservation, continuity, SPIR/global
distance corr., SVC; plus trajectory metrics (topology fidelity, pseudotime
corr., ordering acc., DEMaP) on `tree`/trajectory datasets. Add a per-variant
cross-metric **mean rank** row.
