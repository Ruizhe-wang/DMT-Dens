"""Quantitative case-study metrics for the dynGen fate dataset (README §5).

Turns "our method preserves the developmental path" into numbers, using the
ground truth shipped with the dynGen fate h5ad. For every (method, layer)
embedding it reports:

  1. DEMaP            — Spearman(ambient geodesic dist, embedding Euclidean dist)
                        on a landmark subsample (PHATE's trajectory-preservation
                        gold standard).  Higher = better.
  2. branch_monotonicity — mean |Spearman(pseudotime, embedding PC1)| within each
                        true branch (C2: order preserved along each branch).
  3. topology_f1      — F1 of the MST over milestone-node centroids (in embedding)
                        vs the true (from->to) edge set (C1: topology preserved).
  4. fate_corr        — mean Pearson corr between true fate probabilities and the
                        distance-softmax fate probabilities read off the embedding
                        terminal centroids (C3: fate commitment).
  5. knn_label_acc    — 5-fold kNN accuracy of the terminal-fate label in the
                        embedding (separability).

Embedding input is the flat CSV written by
``callbacks.xc_save_consolidated_embeddings`` (columns: cell_id, method, layer,
x, y); cells are aligned to the h5ad by ``cell_id`` == ``adata.obs_names``.

Usage
-----
    python -m tools.dyngen_case_study.compute_metrics \
        --h5ad  /path/to/fig3_fate_dyngen_shared_dynGenZL10k_hyperbranch_v2_seed42.h5ad \
        --embedding outputs/embeddings/dyngen_fate_topobranch_embeddings.csv \
        --out outputs/dyngen_metrics/metrics.csv

Pass ``--embedding`` multiple times (one per method) to build a comparison table;
rows are appended to ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# ---------------------------------------------------------------- ground truth


def load_ground_truth(h5ad_path, time_key="fig3_true_time", from_key="from",
                      to_key="to", terminal_key="fig3_true_terminal",
                      fate_prefix="fig3_fate_prob_", pca_dim=50, log1p=True):
    """Read the source h5ad and assemble everything the metrics need."""
    a = ad.read_h5ad(h5ad_path)
    obs = a.obs
    gt = {
        "obs_names": np.asarray(a.obs_names, dtype=str),
        "pseudotime": pd.to_numeric(obs[time_key], errors="coerce").to_numpy(float)
        if time_key in obs else None,
        "from": obs[from_key].astype(str).to_numpy() if from_key in obs else None,
        "to": obs[to_key].astype(str).to_numpy() if to_key in obs else None,
        "terminal_label": obs[terminal_key].astype(str).to_numpy() if terminal_key in obs else None,
    }
    fate_cols = sorted(c for c in obs.columns if str(c).startswith(fate_prefix))
    if fate_cols:
        gt["fate_names"] = [c[len(fate_prefix):] for c in fate_cols]
        gt["fate_probs"] = np.column_stack(
            [pd.to_numeric(obs[c], errors="coerce").to_numpy(float) for c in fate_cols]
        )
    else:
        gt["fate_names"], gt["fate_probs"] = [], None

    # Ambient representation for the geodesic (PCA on log1p-normalized counts).
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    X = np.asarray(X, dtype=np.float32)
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    n_comp = min(pca_dim, X.shape[0], X.shape[1])
    gt["ambient"] = PCA(n_components=n_comp, random_state=0).fit_transform(X)
    return gt


# ------------------------------------------------------------------- metrics


def demap(ambient, emb, n_landmarks=1000, knn=15, seed=0):
    """Spearman between ambient geodesic and embedding Euclidean distances."""
    n = ambient.shape[0]
    rng = np.random.RandomState(seed)
    land = rng.choice(n, min(n_landmarks, n), replace=False)

    nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(ambient)
    knn_graph = nn.kneighbors_graph(mode="distance")
    knn_graph = knn_graph.maximum(knn_graph.T)  # symmetrize

    geo = dijkstra(knn_graph, directed=False, indices=land)[:, land]
    iu = np.triu_indices(len(land), k=1)
    g = geo[iu]
    e = pdist(emb[land])
    finite = np.isfinite(g) & np.isfinite(e)
    if finite.sum() < 10:
        return float("nan")
    return float(spearmanr(g[finite], e[finite]).correlation)


def branch_monotonicity(emb, pseudotime, branch_ids, min_cells=20):
    """Mean |Spearman(pseudotime, embedding PC1)| over branches."""
    if pseudotime is None or branch_ids is None:
        return float("nan")
    scores, weights = [], []
    for b in pd.unique(branch_ids):
        m = branch_ids == b
        if m.sum() < min_cells:
            continue
        pt = pseudotime[m]
        if np.nanstd(pt) == 0:
            continue
        pc1 = PCA(n_components=1, random_state=0).fit_transform(emb[m]).ravel()
        rho = spearmanr(pt, pc1, nan_policy="omit").correlation
        if np.isfinite(rho):
            scores.append(abs(rho))
            weights.append(int(m.sum()))
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def _node_centroids(emb, from_nodes, to_nodes, min_cells=10):
    nodes = set(from_nodes.tolist()) | set(to_nodes.tolist())
    nodes.discard("nan")
    cents, counts = {}, {}
    for nname in nodes:
        m = (from_nodes == nname) | (to_nodes == nname)
        c = int(m.sum())
        if c >= min_cells:
            cents[nname] = emb[m].mean(axis=0)
            counts[nname] = c
    return cents, counts


def topology_f1(emb, from_nodes, to_nodes, min_cells=10):
    """F1 of MST-over-centroids edges vs true (undirected) edges."""
    if from_nodes is None or to_nodes is None:
        return float("nan"), float("nan"), float("nan")
    cents, _ = _node_centroids(emb, from_nodes, to_nodes, min_cells)
    names = sorted(cents)
    if len(names) < 2:
        return float("nan"), float("nan"), float("nan")
    idx = {n: i for i, n in enumerate(names)}
    P = np.vstack([cents[n] for n in names])

    # true undirected edges among placeable nodes
    pairs = pd.DataFrame({"u": from_nodes, "v": to_nodes})
    pairs = pairs[(pairs.u != "nan") & (pairs.v != "nan") & (pairs.u != pairs.v)]
    true_edges = {
        frozenset((u, v)) for u, v in set(map(tuple, pairs.values.tolist()))
        if u in idx and v in idx
    }
    if not true_edges:
        return float("nan"), float("nan"), float("nan")

    # MST over complete euclidean graph of centroids
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    mst = minimum_spanning_tree(sp.csr_matrix(D))
    mst = mst + mst.T
    pred_edges = {
        frozenset((names[i], names[j])) for i, j in zip(*mst.nonzero()) if i < j
    }
    tp = len(pred_edges & true_edges)
    prec = tp / len(pred_edges) if pred_edges else 0.0
    rec = tp / len(true_edges) if true_edges else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def fate_recovery(emb, from_nodes, to_nodes, fate_names, fate_probs, min_cells=10):
    """Mean Pearson corr between true fate probs and distance-softmax probs."""
    if fate_probs is None or not fate_names:
        return float("nan")
    cents, _ = _node_centroids(emb, from_nodes, to_nodes, min_cells)
    term_cents, cols = [], []
    for j, name in enumerate(fate_names):
        if name in cents:
            term_cents.append(cents[name])
            cols.append(j)
    if len(term_cents) < 2:
        return float("nan")
    T = np.vstack(term_cents)
    d = np.linalg.norm(emb[:, None, :] - T[None, :, :], axis=2)  # n x k
    scale = np.median(pdist(T)) or 1.0
    pred = np.exp(-d / scale)
    pred /= pred.sum(axis=1, keepdims=True)

    corrs = []
    for k, j in enumerate(cols):
        truth = fate_probs[:, j]
        ok = np.isfinite(truth) & np.isfinite(pred[:, k])
        if ok.sum() > 10 and np.std(truth[ok]) > 0 and np.std(pred[ok, k]) > 0:
            corrs.append(pearsonr(truth[ok], pred[ok, k])[0])
    return float(np.mean(corrs)) if corrs else float("nan")


def knn_label_acc(emb, labels, k=15, seed=0):
    if labels is None:
        return float("nan")
    mask = pd.notna(labels)
    if np.unique(labels[mask]).size < 2:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=min(k, mask.sum() - 1))
    scores = cross_val_score(clf, emb[mask], labels[mask], cv=5)
    return float(scores.mean())


# ------------------------------------------------------------------- driver


def align_embedding(df_layer, obs_names):
    """Return embedding array aligned to obs_names order (NaN rows dropped)."""
    df_layer = df_layer.drop_duplicates(subset="cell_id").set_index("cell_id")
    df_layer = df_layer.reindex(obs_names)
    emb = df_layer[["x", "y"]].to_numpy(float)
    keep = np.isfinite(emb).all(axis=1)
    return emb, keep


def compute_for_embedding(csv_path, gt, method_override=None, layer=None):
    df = pd.read_csv(csv_path)
    df["cell_id"] = df["cell_id"].astype(str)  # obs_names may look numeric; keep as str to align
    rows = []
    layers = [layer] if layer is not None else sorted(df["layer"].unique())
    for L in layers:
        sub = df[df["layer"] == L]
        if sub.empty:
            continue
        method = method_override or (sub["method"].iloc[0] if "method" in sub else os.path.basename(csv_path))
        emb, keep = align_embedding(sub, gt["obs_names"])

        amb = gt["ambient"][keep]
        e = emb[keep]
        pt = gt["pseudotime"][keep] if gt["pseudotime"] is not None else None
        fr = gt["from"][keep] if gt["from"] is not None else None
        to = gt["to"][keep] if gt["to"] is not None else None
        lab = gt["terminal_label"][keep] if gt["terminal_label"] is not None else None
        fp = gt["fate_probs"][keep] if gt["fate_probs"] is not None else None
        branch = (
            pd.Series(fr).astype(str) + "->" + pd.Series(to).astype(str)
        ).to_numpy() if fr is not None and to is not None else None

        prec, rec, f1 = topology_f1(e, fr, to)
        rows.append({
            "method": method,
            "layer": int(L),
            "n_cells": int(keep.sum()),
            "demap": demap(amb, e),
            "branch_monotonicity": branch_monotonicity(e, pt, branch),
            "topology_precision": prec,
            "topology_recall": rec,
            "topology_f1": f1,
            "fate_corr": fate_recovery(e, fr, to, gt["fate_names"], fp),
            "knn_label_acc": knn_label_acc(e, lab),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="dynGen fate case-study metrics (README §5)")
    ap.add_argument("--h5ad", required=True, help="source dynGen fate .h5ad with ground truth")
    ap.add_argument("--embedding", action="append", default=[],
                    help="embedding CSV (cell_id,method,layer,x,y); repeatable")
    ap.add_argument("--embedding-glob", default=None, help="glob for multiple embedding CSVs")
    ap.add_argument("--method", default=None, help="override method name (single embedding)")
    ap.add_argument("--layer", type=int, default=None, help="only this layer (default: all)")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--no-log1p", action="store_true", help="skip log1p before ambient PCA")
    ap.add_argument("--out", default="outputs/dyngen_metrics/metrics.csv")
    args = ap.parse_args()

    paths = list(args.embedding)
    if args.embedding_glob:
        paths += sorted(glob.glob(args.embedding_glob))
    if not paths:
        ap.error("provide --embedding and/or --embedding-glob")

    print(f"[gt] loading {args.h5ad}")
    gt = load_ground_truth(args.h5ad, pca_dim=args.pca_dim, log1p=not args.no_log1p)
    print(f"[gt] {len(gt['obs_names'])} cells, {len(gt['fate_names'])} fate terminals, "
          f"ambient PCA {gt['ambient'].shape[1]}d")

    all_rows = []
    for p in paths:
        print(f"[metrics] {p}")
        all_rows += compute_for_embedding(p, gt, method_override=args.method, layer=args.layer)

    table = pd.DataFrame(all_rows)
    cols = ["method", "layer", "n_cells", "demap", "branch_monotonicity",
            "topology_precision", "topology_recall", "topology_f1", "fate_corr", "knn_label_acc"]
    table = table[[c for c in cols if c in table.columns]]
    pd.set_option("display.width", 200, "display.max_columns", None)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = not os.path.exists(args.out)
    table.to_csv(args.out, mode="a", header=header, index=False)
    print(f"\n[out] appended {len(table)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()
