"""
Mellon ground-truth density-preservation comparison across methods.

Uses Mellon (Otto et al., 2024) as an INDEPENDENT high-dimensional density
estimator (the "ground truth") and reports how well each method's embedding
density tracks it, via Spearman and Pearson correlation, under two
embedding-density definitions:
  - Mellon-2D : Mellon log-density recomputed on the 2-D embedding
  - kNN       : dimension-free kNN log-density (the density definition the
                TopoBranch loss / paper metric already uses)
For reference we also print the paper's legacy kNN-HD vs kNN-2D
``density_correlation`` (Spearman).

Inputs
------
Each method is one ``.npz`` file with keys:
    hd     : (n, D) high-dimensional data   (only the first file's `hd` is used
             as the shared ground truth; all embeddings must be row-aligned to it)
    emb    : (n, d) the method's embedding
    labels : optional
Produce these with `callbacks.dump_embedding.DumpEmbeddingCallback` (for
TopoBranch / model runs) or `--make-baselines` (t-SNE/UMAP/densMAP/PaCMAP/PHATE
computed here from the shared `hd`).

Usage
-----
    python scripts/run_mellon_eval.py \
        --emb TopoBranch=dumps/epi_topobranch.npz \
        --make-baselines \
        --out results/mellon_epi

    python scripts/run_mellon_eval.py \
        --emb TopoBranch=a.npz --emb densMAP=b.npz --out results/mellon_epi
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.stats as ss

from eval.mellon_density import mellon_available, mellon_log_density, knn_log_density

DENSITY_K = 15
MELLON_PCA = 20
SEED = 42


def _corr(a, b):
    sp = ss.spearmanr(a, b).correlation
    pe = ss.pearsonr(a, b)[0]
    return float(sp if not np.isnan(sp) else 0.0), float(pe if not np.isnan(pe) else 0.0)


def make_baselines(hd, seed=SEED):
    out = {}
    def _try(name, fn):
        try:
            t0 = time.time()
            out[name] = np.asarray(fn(hd))
            print(f"  {name:9s} {out[name].shape} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  {name:9s} FAILED: {type(e).__name__}: {e}")
    _try("t-SNE",   lambda X: __import__("openTSNE").TSNE(n_jobs=-1, random_state=seed).fit(X))
    _try("UMAP",    lambda X: __import__("umap").UMAP(random_state=seed).fit_transform(X))
    _try("densMAP", lambda X: __import__("umap").UMAP(densmap=True, random_state=seed).fit_transform(X))
    _try("PaCMAP",  lambda X: __import__("pacmap").PaCMAP(random_state=seed).fit_transform(X))
    _try("PHATE",   lambda X: __import__("phate").PHATE(random_state=seed, verbose=0).fit_transform(X))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", action="append", default=[],
                    help="NAME=path.npz (repeatable). First file supplies the shared `hd`.")
    ap.add_argument("--make-baselines", action="store_true",
                    help="Also compute t-SNE/UMAP/densMAP/PaCMAP/PHATE from the shared hd.")
    ap.add_argument("--density-k", type=int, default=DENSITY_K)
    ap.add_argument("--mellon-pca", type=int, default=MELLON_PCA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="mellon_eval",
                    help="Output prefix (writes <out>.json and <out>_scatter.png).")
    args = ap.parse_args()

    if not mellon_available():
        raise SystemExit("mellon is not installed. `pip install mellon` (JAX-based).")
    if not args.emb:
        raise SystemExit("Provide at least one --emb NAME=path.npz")

    # load method embeddings; shared HD comes from the first file
    embeddings, hd, labels = {}, None, None
    for spec in args.emb:
        name, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        embeddings[name] = np.asarray(d["emb"])
        if hd is None:
            hd = np.asarray(d["hd"], dtype=np.float32)
            labels = d["labels"] if "labels" in d else None
    print(f"HD {hd.shape}; methods from files: {list(embeddings)}")

    if args.make_baselines:
        print("Generating baselines from shared HD:")
        embeddings.update(make_baselines(hd, seed=args.seed))

    # ground truth: Mellon HD log-density (once), + legacy kNN-HD log-density
    t0 = time.time()
    rho_hd = mellon_log_density(hd, pca_dim=args.mellon_pca, seed=args.seed)
    hd_knn = knn_log_density(hd, k=args.density_k)
    print(f"Mellon HD density in {time.time()-t0:.1f}s")

    rows = []
    for name, emb in embeddings.items():
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[0] != hd.shape[0]:
            print(f"  skip {name}: {emb.shape[0]} rows != hd {hd.shape[0]} (must be aligned)")
            continue
        emb_knn = knn_log_density(emb, k=args.density_k)
        emb_mel = mellon_log_density(emb, pca_dim=None, seed=args.seed)
        s_mm, p_mm = _corr(rho_hd, emb_mel)
        s_mk, p_mk = _corr(rho_hd, emb_knn)
        s_kk, _ = _corr(hd_knn, emb_knn)
        rows.append([name, s_mm, p_mm, s_mk, p_mk, s_kk])

    rows.sort(key=lambda r: r[1], reverse=True)
    hdr = ["Method", "Mel->Mel rho", "Mel->Mel r", "Mel->kNN rho", "Mel->kNN r", "kNN->kNN rho(paper)"]
    print("\n" + "=" * 92)
    print(f"{hdr[0]:<12}{hdr[1]:>14}{hdr[2]:>13}{hdr[3]:>14}{hdr[4]:>13}{hdr[5]:>21}")
    print("-" * 92)
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>14.3f}{r[2]:>13.3f}{r[3]:>14.3f}{r[4]:>13.3f}{r[5]:>21.3f}")
    print("=" * 92)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"config": vars(args), "columns": hdr, "rows": rows}, f, indent=2)
    print("wrote", args.out + ".json")


if __name__ == "__main__":
    main()
