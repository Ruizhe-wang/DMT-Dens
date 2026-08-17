"""Offline validation of the C. elegans trajectory/time metrics.

Runs the metrics in compute_metrics.py on the already-saved embeddings
(outputs/embeddings/celegan_<method>_embeddings.csv) so the numbers can be
sanity-checked before wiring the live callback into training.

The saved embeddings carry a ``cell_id`` column = the original row index into
the full 86,024-cell tsv (unannotated cells were dropped, so cell_id is not
contiguous). We align ground-truth time by indexing the tsv with cell_id.

Usage:
  python -m tools.celegan_case_study.validate_celegan_traj_metrics \
      --emb_dir outputs/embeddings \
      --data_dir D:/ruizhe/data/celegan \
      [--demap]   # also compute DEMaP (loads celegan.h5ad; slower)
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from tools.celegan_case_study.compute_metrics import (
    compute_celegan_traj_metrics,
    embryo_time_to_float,
)


def load_time(data_dir):
    ti = pd.read_csv(os.path.join(data_dir, "celegan_embryo_time.tsv"),
                     sep="\t", header=None).iloc[:, 0].astype(str)
    return ti.map(embryo_time_to_float).to_numpy(float)  # len = full 86,024


def load_ambient(data_dir, pca_dim=50):
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    adata = sc.read(os.path.join(data_dir, "celegan.h5ad"))
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    n_comp = min(pca_dim, X.shape[0], X.shape[1])
    return PCA(n_components=n_comp, random_state=0).fit_transform(X)  # len = full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="outputs/embeddings")
    ap.add_argument("--data_dir", default="D:/ruizhe/data/celegan")
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--demap", action="store_true")
    args = ap.parse_args()

    time_full = load_time(args.data_dir)
    ambient_full = load_ambient(args.data_dir) if args.demap else None

    files = sorted(glob.glob(os.path.join(args.emb_dir, "celegan_*_embeddings.csv")))
    rows = []
    for f in files:
        method = re.sub(r"^celegan_|_embeddings\.csv$", "", os.path.basename(f))
        df = pd.read_csv(f)
        cid = df["cell_id"].to_numpy()
        emb = df[["x", "y"]].to_numpy(dtype=float)
        time = time_full[cid]
        ambient = ambient_full[cid] if ambient_full is not None else None
        m = compute_celegan_traj_metrics(emb, time, ambient=ambient, knn=args.knn)
        m["method"] = method
        m["n"] = len(df)
        rows.append(m)
        print(f"[done] {method:12s} n={len(df)}  "
              f"pt_corr={m['pseudotime_corr']:.3f} (reach {m['pseudotime_reachable_frac']:.2f})  "
              f"order_acc={m['time_ordering_acc']:.3f}  continuity={m['time_continuity']:.3f}"
              + (f"  demap={m.get('demap', float('nan')):.3f}" if ambient is not None else ""))

    cols = ["method", "n", "pseudotime_corr", "pseudotime_reachable_frac",
            "time_ordering_acc", "time_continuity"] + (["demap"] if args.demap else [])
    out = pd.DataFrame(rows)[cols].sort_values("pseudotime_corr", ascending=False)
    print("\n==== C. elegans trajectory/time metrics (higher is better) ====")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
