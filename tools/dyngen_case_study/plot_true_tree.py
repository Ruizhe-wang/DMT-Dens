"""Standalone true-tree overlay plotter for the dynGen fate case study.

Renders the ground-truth tree backbone over a saved embedding WITHOUT retraining.
Reuses the callback's drawing code (``TrueTreeOverlayVisualizationCallback``), so
the figure matches what training would produce — handy for iterating on the
overlay (e.g. medoid vs mean node placement) straight from the embedding CSV.

Usage
-----
    python -m tools.dyngen_case_study.plot_true_tree \
        --h5ad  /usr/storage/.../fig3_fate_dyngen_shared_..._seed42.h5ad \
        --embedding outputs/embeddings/dyngen_fate_topobranch_embeddings.csv \
        --layer 0 --node-placement medoid \
        --output-dir outputs/case_study/dyngen_fate/topobranch_replot
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd

from callbacks.case_study_truetree_callback import TrueTreeOverlayVisualizationCallback


def build_adata(h5ad_path, from_key="from", to_key="to", time_key="fig3_true_time"):
    """Light AnnData carrying only the obs the overlay needs, keyed by obs_names."""
    src = ad.read_h5ad(h5ad_path, backed="r")
    obs = pd.DataFrame(index=np.asarray(src.obs_names, dtype=str))
    for k in (from_key, to_key):
        if k in src.obs:
            obs[k] = src.obs[k].astype(str).to_numpy()
    if time_key in src.obs:
        obs["pseudotime"] = pd.to_numeric(src.obs[time_key], errors="coerce").to_numpy(float)
    a = ad.AnnData(X=np.zeros((obs.shape[0], 1), dtype=np.float32), obs=obs)
    return a


def load_embedding(csv_path, obs_names, layer):
    df = pd.read_csv(csv_path)
    df["cell_id"] = df["cell_id"].astype(str)
    if "layer" in df and layer is not None:
        df = df[df["layer"] == layer]
    df = df.drop_duplicates(subset="cell_id").set_index("cell_id").reindex(obs_names)
    return df[["x", "y"]].to_numpy(float)


def main():
    ap = argparse.ArgumentParser(description="Standalone dynGen true-tree overlay")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--embedding", required=True, help="embedding CSV (cell_id,method,layer,x,y)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--node-placement", choices=["mean", "medoid"], default="mean")
    ap.add_argument("--min-cells-per-node", type=int, default=10)
    ap.add_argument("--background", default="pseudotime")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--output-dir", default="outputs/case_study/dyngen_fate/replot")
    args = ap.parse_args()

    adata = build_adata(args.h5ad)
    emb = load_embedding(args.embedding, np.asarray(adata.obs_names, dtype=str), args.layer)
    keep = np.isfinite(emb).all(axis=1)
    adata, emb = adata[keep].copy(), emb[keep]
    print(f"[plot] {keep.sum()} cells aligned; node_placement={args.node_placement}")

    cb = TrueTreeOverlayVisualizationCallback(
        output_dir=args.output_dir,
        save_formats=["png", "pdf"],
        dpi=args.dpi,
        node_placement=args.node_placement,
        min_cells_per_node=args.min_cells_per_node,
        background_color_by=args.background,
        panel_prefix="replot",
    )
    edges = cb._edges_from_obs(adata)
    positions, counts = cb._node_positions(adata, emb)
    cb.plot_true_tree(adata, emb, positions, counts, edges,
                      text=f"_layer{args.layer}_{args.node_placement}", log_to_wandb=False)
    out = os.path.join(args.output_dir, f"case_study_truetree_layer{args.layer}_{args.node_placement}.png")
    print(f"[plot] wrote {out} ({len(edges)} edges, {len(positions)} nodes placed)")


if __name__ == "__main__":
    main()
