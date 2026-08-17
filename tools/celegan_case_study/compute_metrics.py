"""Trajectory / developmental-time preservation metrics for the C. elegans
case study.

Unlike dynGen, C. elegans has no simulated lineage backbone (no from/to tree,
no fate probabilities) -- the only ground truth is the per-cell embryonic time
(``embryo_time``, given as coarse bins that the datamodule converts to a numeric
``embryo_time_numeric``) plus the cell-type label. We therefore measure how well
a 2D embedding preserves the *developmental-time* structure, using three metrics
that need only (embedding, ground-truth time):

  * ``pseudotime_corr``  -- Spearman(embedding pseudotime, GT time). The embedding
    pseudotime is the geodesic distance, on the embedding kNN graph, from the
    earliest-time cells (multi-source). A faithful embedding orders cells along
    development, so this correlation is high and positive.
  * ``time_ordering_acc`` -- tie-aware concordance: over sampled cell pairs with
    *different* GT time, the fraction whose embedding-pseudotime order matches the
    GT-time order. (Pairs sharing a time bin are excluded, since GT time is
    binned ~14% of random pairs tie.)
  * ``time_continuity``  -- 1 - mean|Δtime| over embedding-kNN edges divided by
    mean|Δtime| over random pairs. ~1 means neighbouring cells in the embedding
    have very similar developmental time (smooth, continuous trajectory); ~0
    means neighbours are no more time-similar than random (fragmented).

``demap`` (ambient-geodesic vs embedding-Euclidean Spearman) is reused from the
dynGen module for an optional manifold-preservation number.
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# Reuse the ambient-geodesic manifold metric (no GT tree needed).
from tools.dyngen_case_study.compute_metrics import demap  # noqa: F401


def _knn_graph(emb, knn=15):
    """Symmetric distance-weighted kNN graph on the embedding."""
    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(emb)
    g = nn.kneighbors_graph(mode="distance")
    return g.maximum(g.T)  # symmetrize


def embedding_pseudotime(emb, time, knn=15, root_quantile=0.0):
    """Geodesic distance on the embedding kNN graph from the earliest-time cells.

    Roots are all cells whose GT time equals the minimum time (or, if
    ``root_quantile`` > 0, cells at or below that time quantile). Returns an
    array of shape (n,) with np.inf for cells unreachable from any root.
    """
    time = np.asarray(time, dtype=float)
    graph = _knn_graph(emb, knn=knn)
    if root_quantile > 0:
        thr = np.quantile(time, root_quantile)
        roots = np.where(time <= thr)[0]
    else:
        roots = np.where(time == np.nanmin(time))[0]
    if roots.size == 0:
        return np.full(emb.shape[0], np.nan)
    # multi-source: distance to the nearest root.
    pt = dijkstra(graph, directed=False, indices=roots, min_only=True)
    return pt


def pseudotime_corr(emb, time, knn=15):
    """Spearman correlation between embedding pseudotime and GT time."""
    time = np.asarray(time, dtype=float)
    pt = embedding_pseudotime(emb, time, knn=knn)
    finite = np.isfinite(pt) & np.isfinite(time)
    if finite.sum() < 10:
        return float("nan"), float(finite.mean())
    rho = spearmanr(pt[finite], time[finite]).correlation
    return float(rho), float(finite.mean())  # (correlation, reachable fraction)


def time_ordering_acc(emb, time, knn=15, n_pairs=200000, seed=0):
    """Tie-aware concordance between embedding pseudotime and GT time.

    Samples ``n_pairs`` random cell pairs, keeps those with different GT time,
    and reports the fraction whose pseudotime order agrees with the time order.
    """
    time = np.asarray(time, dtype=float)
    pt = embedding_pseudotime(emb, time, knn=knn)
    ok = np.isfinite(pt) & np.isfinite(time)
    idx = np.where(ok)[0]
    if idx.size < 10:
        return float("nan")
    rng = np.random.RandomState(seed)
    i = rng.choice(idx, n_pairs)
    j = rng.choice(idx, n_pairs)
    dt = time[i] - time[j]
    dp = pt[i] - pt[j]
    nontie = dt != 0
    if nontie.sum() == 0:
        return float("nan")
    concordant = (np.sign(dt[nontie]) == np.sign(dp[nontie]))
    return float(concordant.mean())


def time_continuity(emb, time, knn=15, n_pairs=200000, seed=0):
    """Smoothness of GT time over the embedding kNN graph.

    1 - mean|Δtime| over kNN edges / mean|Δtime| over random pairs. ~1 means
    embedding neighbours share developmental time (continuous); ~0 means no
    better than random (fragmented).
    """
    time = np.asarray(time, dtype=float)
    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(emb)
    _, nbr = nn.kneighbors(emb)
    nbr = nbr[:, 1:]  # drop self
    src = np.repeat(np.arange(n), nbr.shape[1])
    dst = nbr.ravel()
    edge_dt = np.abs(time[src] - time[dst])
    edge_dt = edge_dt[np.isfinite(edge_dt)]
    rng = np.random.RandomState(seed)
    a = rng.randint(0, n, n_pairs)
    b = rng.randint(0, n, n_pairs)
    rand_dt = np.abs(time[a] - time[b])
    rand_dt = rand_dt[np.isfinite(rand_dt)]
    if edge_dt.size == 0 or rand_dt.size == 0 or rand_dt.mean() == 0:
        return float("nan")
    return float(1.0 - edge_dt.mean() / rand_dt.mean())


def compute_celegan_traj_metrics(emb, time, ambient=None, knn=15,
                                 demap_landmarks=1000, demap_knn=15, seed=0):
    """All trajectory/time metrics for one embedding. ``ambient`` optional (demap)."""
    rho, cover = pseudotime_corr(emb, time, knn=knn)
    out = {
        "pseudotime_corr": rho,
        "pseudotime_reachable_frac": cover,
        "time_ordering_acc": time_ordering_acc(emb, time, knn=knn, seed=seed),
        "time_continuity": time_continuity(emb, time, knn=knn, seed=seed),
    }
    if ambient is not None:
        out["demap"] = demap(ambient, emb, n_landmarks=demap_landmarks,
                             knn=demap_knn, seed=seed)
    return out


def embryo_time_to_float(value):
    """Match data_model.M1datamodel_celegan: bin -> numeric (upper bound)."""
    text = str(value).strip()
    if "-" in text:
        return float(text.split("-")[-1].strip())
    if text.startswith("<"):
        return float(text[1:].strip()) - 50.0
    if text.startswith(">"):
        return float(text[1:].strip()) + 100.0
    return float(text)
