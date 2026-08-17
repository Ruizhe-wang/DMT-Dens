"""Run the baseline embeddings on the saved branch3d dataset and (optionally)
drop in a trained DiffTree/TopoBranch embedding as an extra panel.

This loads the EXACT dataset that DiffTree trains on (the .npz saved by
data_model/data_tree_gen_density_diverse3d.py), so every method -- DiffTree and
all baselines -- is compared on identical points and identical ground-truth
density.

Usage
-----
  # baselines only (no DiffTree):
  python scripts/branch3d_compare.py

  # include a trained DiffTree embedding exported by scripts/export_embeddings.py:
  python scripts/branch3d_compare.py \
      --difftree-embedding outputs/embeddings/branch3d_topobranch_embeddings.npz

Output: <out>_compare_2d.png + <out>_compare_report.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_model.data_tree_gen_density_diverse3d import (  # noqa: E402
    TreeDensityDemo,
    _standardize_xy,
    density_preservation_score,
    local_radius,
    run_baselines,
    save_baselines_figure,
)


def load_dataset(npz_path: Path) -> TreeDensityDemo:
    d = np.load(npz_path, allow_pickle=True)
    coords = np.asarray(d["data"], dtype=np.float64)
    branch_id = np.asarray(d["clusters"]).reshape(-1)
    density = np.asarray(d["density"], dtype=np.float64).reshape(-1)
    radius = (np.asarray(d["local_radius"], dtype=np.float64).reshape(-1)
              if "local_radius" in d else local_radius(coords))
    names = (np.asarray(d["branch_names"]).reshape(-1)
             if "branch_names" in d else branch_id.astype(str))
    n = coords.shape[0]
    zeros = np.zeros(n, dtype=np.float64)
    return TreeDensityDemo(
        coords=coords, branch_id=branch_id, branch_names=names,
        pseudotime=zeros, branch_progress=zeros,
        local_density=density, local_radius=radius,
        centerlines=[], branch_specs=[],
    )


def load_difftree_embedding(path: Path) -> np.ndarray:
    """Load a 2D embedding from an export_embeddings .npz (layer_0) or a .csv."""
    if path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        layer = df[df["layer"] == df["layer"].min()]
        return layer[["x", "y"]].to_numpy(dtype=np.float64)
    arr = np.load(path, allow_pickle=True)
    if "layer_0" in arr:
        return np.asarray(arr["layer_0"], dtype=np.float64)
    # fall back to the first (N, 2) array we can find
    for key in arr.files:
        a = np.asarray(arr[key])
        if a.ndim == 2 and a.shape[1] == 2:
            return a.astype(np.float64)
    raise KeyError(f"No 2D embedding found in {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=_REPO_ROOT / "data" / "synthetic_branch3d"
                   / "branch3d_diverse_density.npz")
    p.add_argument("--difftree-embedding", type=Path, default=None,
                   help="export_embeddings .npz/.csv from a trained DiffTree run")
    p.add_argument("--difftree-name", default="DiffTree")
    p.add_argument("--output", type=Path,
                   default=_REPO_ROOT / "outputs" / "branch3d" / "branch3d")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsne-perplexity", type=float, default=45.0)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--dens-lambda", type=float, default=0.1)
    p.add_argument("--knn-k", type=int, default=18)
    p.add_argument("--skip", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.dataset)
    print(f"loaded {args.dataset}: {data.coords.shape[0]} points, "
          f"{len(np.unique(data.branch_id))} branches")

    print("running baselines:")
    baseline_emb = run_baselines(
        data, seed=args.seed, tsne_perplexity=args.tsne_perplexity,
        umap_neighbors=args.umap_neighbors, dens_lambda=args.dens_lambda,
        skip=tuple(args.skip),
    )

    embeddings: dict[str, np.ndarray] = {}
    if args.difftree_embedding is not None:
        emb = _standardize_xy(load_difftree_embedding(args.difftree_embedding))
        embeddings[args.difftree_name] = emb        # DiffTree first column
        print(f"  [{args.difftree_name}] loaded {args.difftree_embedding}")
    embeddings.update(baseline_emb)

    fig_path, scores = save_baselines_figure(data, embeddings, args.output, args.knn_k)
    print(f"saved comparison figure: {fig_path}")

    lines = ["branch3d: DiffTree vs baselines -- density preservation (Spearman rho)"]
    for name, rho in sorted(scores.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:10s} rho = {rho:+.3f}")
    report = "\n".join(lines)
    (args.output.parent / f"{args.output.name}_compare_report.txt").write_text(
        report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
