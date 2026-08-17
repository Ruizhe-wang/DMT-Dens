# dyngen — developmental-path-preservation case study

Working spec for the dynGen fate case study. The point of this dataset is that it
ships **ground truth** (true tree topology, true pseudotime, true fate
probabilities), so "our method preserves the developmental path" can be turned
into something **quantifiable, comparable, and visually provable** — not just
"the plot looks nice".

This folder is the planning/spec home. The runnable training config currently
lives at [`configs/case_study/diftree/dyngen_fate.yaml`](../case_study/diftree/dyngen_fate.yaml);
new dyngen-specific configs (baselines, ablations, multi-seed sweeps) should land
here as they are written.

---

## 1. Dataset (verified on c82_vm)

- Path (c82_vm): `/usr/storage/zelin181data/sclineage/single-cell-lineage1/data/figure3_fate/dyngen/`
- File: `fig3_fate_dyngen_shared_dynGenZL10k_hyperbranch_v2_seed42.h5ad` (0.21 GB, the 10k shared subset — **not** the 12.6 GB raw file)
- Shape: **10000 cells × 3327 genes**; `X` is CSR sparse **int64 raw counts** (range 0–237733, mean ≈ 8235, 52% nonzero)
- 8 terminal classes (imbalanced): `multipotent` 3383, `sEndG` 3143, `sEndF` 1069, `sEndK` 952, `sEndJ` 622, `sP` 395, `sEndL` 363, `sN` 73
- Sibling datasets with identical schema (swap `h5ad_file` only): `…multibranch…`, `…ultrabranch_v1…`
- Loader: [`data_model/M1datamodel_dyngen_fate.py`](../../data_model/M1datamodel_dyngen_fate.py) — verified to load this file end-to-end (3327 → padded 3328, MinMax → [0,1], 8 classes, all annotations carried into `adata.obs`). `top_genes` stays `null` (dyngen's gene panel is the designed regulatory network; HVG selection would discard fate-driving signal).

### Ground-truth assets (the whole reason to use this dataset)

| asset | what it gives | used for |
|---|---|---|
| `…gt.json` → `edge_table` | true tree: root `sA` → 6 branchpoints (`sCmid…sMmid`) → 7 terminals | C1 / tree overlay |
| `obs["from"]`, `obs["to"]` | per-cell true tree edge | C1 / tree overlay |
| `obs["fig3_true_time"]` (`pseudotime`) | true pseudotime, continuous | C2 |
| `…true_fate_probabilities.csv` / `obs["fig3_fate_prob_*"]` | true P(reach terminal); early cells are uniform 1/7 | C3 |
| `obs["true_branch"]` | true branch segment label | C1 |
| `…marker_genes.json` | per-terminal marker genes | optional marker panels |

---

## 2. The claim, split into 3 provable sub-claims

Each sub-claim gets one figure **and** one number. A vague "preserves trajectory"
carries no weight with reviewers.

| # | sub-claim | ground truth | figure | metric |
|---|-----------|--------------|--------|--------|
| **C1** | **topology preserved** — same #branches/#terminals, same connectivity | `gt.json` edge_table, `true_branch` | true-tree backbone overlay (§4) | PAGA-style graph F1 vs true edges; branch kNN consistency |
| **C2** | **order preserved** — pseudotime monotone & smooth along each branch | `fig3_true_time` | viridis pseudotime scatter | per-branch Spearman ρ(arclength, pseudotime), averaged |
| **C3** | **fate commitment** — embedding spreads the commitment process geometrically | `fate_prob_*` | 7× magma fate-prob panels | corr / KL of inferred vs true fate probs |

---

## 3. Headline figure (qualitative, one multi-panel)

- **(A)** reference true tree schematic from `gt.json` edge_table (target topology)
- **(B)** our embedding colored by `fig3_true_terminal` (tab20, `multipotent` grey) — should show 7 arms + 1 hub
- **(C)** same embedding colored by `pseudotime` (viridis + colorbar) — monotone hub→tips gradient
- **(D)** true-tree backbone overlaid on the embedding (§4) — the visual clincher
- **(E)** 7× `fate_prob_*` panels (magma, low values drawn first) — the most persuasive row for a fate paper
- **(F)** same row for baselines (UMAP / t-SNE / **PHATE** / **DiffusionMap** / PCA), all colored by fate

Pick **PHATE / diffusion maps** as the headline rivals: they explicitly claim
trajectory preservation, so beating them is what matters. Beating only UMAP/t-SNE
is not enough.

---

## 4. Killer move — true-tree backbone overlay (C1's visual proof)

1. For each milestone node (`sA`, `sCmid`, …, `sEndG`), take the **centroid** of its cells in embedding space (node membership from `obs["from"]`/`obs["to"]`).
2. Connect centroids by the `gt.json` `edge_table`.
3. Draw the resulting "embedded true tree" over the pseudotime-colored scatter.

If topology is preserved, the overlaid tree grows cleanly along the data arms with
**no edge crossings**; on baselines the same tree twists and self-intersects.
Caption line: *"the ground-truth lineage tree embeds without edge crossings only
under our method."*

---

## 5. Quantitative table (figures alone = "just looks nice")

Same 2D embedding per method, multi-seed (reuse the 5-seed sweep pattern), report mean ± std:

1. **DEMaP** (PHATE's trajectory-preservation gold standard): Spearman(geodesic dist in ambient, Euclidean dist in embedding) — the single most direct "path preservation" number.
2. **Per-branch pseudotime monotonicity** (C2 metric above).
3. **Topology preservation**: PAGA-style graph built on embedding milestones vs true `edge_table` → edge F1 / graph edit distance.
4. **Fate-probability recovery** (C3 metric above).
5. **Separability**: fate-label kNN / SVC accuracy (the existing "combined density+SVC" metric applies directly).
6. (suppl.) trustworthiness / continuity / kNN-overlap.

---

## 6. Ablations (reviewers will ask "what makes it work")

Reuse the existing ablation configs (`*_no_density.yaml`, `*_single_scale.yaml`):

- **no density term** → arms collapse/overlap, DEMaP drops ⇒ density term is what keeps continuous trajectories.
- **single-scale** → local fragmentation or global distortion ⇒ multi-scale is necessary.

Turns the case study from "our figure is pretty" into "we know *why* it works".

---

## 7. Paper narrative order

> synthetic dyngen tree → known ground truth (A) → our embedding satisfies
> C1/C2/C3 (B–E) → true-tree backbone overlays without crossings (D) → quantitative
> table wins across the board, esp. DEMaP + fate-prob recovery → ablation shows
> density + multi-scale is the mechanism (F) → transition to real data (mouse HSPC
> etc.), the synthetically-verified properties reproduce.

Standard playbook for a synthetic case study: **prove the property on ground-truth
synthetic data, then show the same property reproduces on real data.** dyngen is the
first half; the existing bio datasets are the second.

---

## 8. TODO checklist

Status legend: ✅ done · 🔶 partial / needs work · ⬜ not started

- ✅ Data loader matches this file, verified end-to-end — `data_model/M1datamodel_dyngen_fate.py`
- ✅ Runnable training config — `configs/case_study/diftree/dyngen_fate.yaml`
- ✅ Continuous coloring for pseudotime (fix 坑1) — `callbacks/paper_embedding_plot_callback.py` (`_is_continuous` → viridis+colorbar)
- ✅ Fate-argmax commitment threshold (fix 坑2) — `callbacks/case_study_fateprob_callback.py` (`fate_commit_threshold`, grey `uncommitted`)
- ✅ **True-tree backbone overlay** (§4) — `callbacks/case_study_truetree_callback.py` (`TrueTreeOverlayVisualizationCallback`); rebuilds the tree from `obs["from"]`/`obs["to"]`, milestone centroids + true edges over the embedding. Verified on real obs: 27 edges, 1 root / 6 branchpoints / 7 terminals. Test config: `configs/dyngen/dyngen_fate_truetree_test.yaml`
- ✅ **Quantitative metrics script** (§5) — `tools/dyngen_case_study/compute_metrics.py`; DEMaP + per-branch monotonicity + topology F1 + fate-prob recovery + kNN label accuracy, batched over embedding CSVs (`cell_id,method,layer,x,y`) into one appended table. Verified end-to-end (random embedding scores at chance). Run: `python -m tools.dyngen_case_study.compute_metrics --h5ad <src> --embedding <emb.csv> --out outputs/dyngen_metrics/metrics.csv`
- ⬜ **Baseline embeddings** (§3F) — run UMAP / t-SNE / PHATE / DiffusionMap / PCA on the same 3328-dim input; save embeddings for the metrics script
- ⬜ **Ablation runs** (§6) — `no_density` / `single_scale` variants on dyngen, same eval
- ⬜ **Multi-seed sweep** — DiffTree + baselines × seeds for mean±std (reuse `scripts/run_wandb_multidata.sh` pattern)
- ⬜ **Headline multi-panel assembly** (§3) — compose A–F into the paper figure
- ⬜ (optional) marker-gene panels from `…marker_genes.json`
- ⬜ (optional) extend to `multibranch` / `ultrabranch` siblings for robustness

---

## 9. How to run (c82_vm)

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate mldr
python main.py fit -c configs/case_study/diftree/dyngen_fate.yaml
```

`dyngen_fate.yaml` is the all-in-one run: it emits every case-study panel
(categorical + continuous pseudotime, fate-probability with commitment threshold,
and the true-tree backbone overlay) to `outputs/case_study/dyngen_fate/`, plus the
embedding CSV to `outputs/embeddings/` via `SaveConsolidatedEmbeddingsCallback`.
Feed that CSV straight into the metrics script (§5):

```bash
python -m tools.dyngen_case_study.compute_metrics \
    --h5ad <source.h5ad> \
    --embedding outputs/embeddings/dyngen_fate_topobranch_embeddings.csv \
    --out outputs/dyngen_metrics/metrics.csv
```

`dyngen_fate_truetree_test.yaml` is the fast (60-epoch) smoke variant for checking
the plotting callbacks; it does not save embeddings.
