"""
Mellon-based density-preservation metrics.

Motivation
----------
The existing ``compute_density_correlation`` in :mod:`eval.fidelity_eval` uses a
kNN radius as the high-dimensional "ground-truth" density and correlates it with
the same kNN radius measured in the embedding.  Because the TopoBranch density
loss is itself built on a kNN log-density, a reviewer can argue that metric is
self-referential.

Mellon (Otto et al., 2024, *Nature Methods*) is an **independent** cell-state
density estimator.  It fits a nonparametric Gaussian-process log-density to the
cells' nearest-neighbour distances (a k-th neighbour distance is a local density
proxy; Mellon turns the whole distance field into a smooth, calibrated
log-density function).  Using Mellon's high-dimensional log-density as the
ground truth and correlating it against each method's embedding density removes
the circularity concern.

Mellon is numerically unstable above ~50 dimensions, and in the single-cell
literature it is run on a reduced representation (PCA / diffusion components).
We therefore PCA-reduce the high-dimensional data to ``mellon_pca_dim`` before
fitting.  The embedding (2-D) is used as-is.

This module has **no hard dependency** on Mellon: if the package is not
installed, :func:`mellon_available` returns ``False`` and the metric helpers
return empty results, so the rest of the eval suite keeps working.
"""
from __future__ import annotations

import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:  # optional dependency (JAX-based); only needed when Mellon metrics are requested
    import mellon as _mellon
    _MELLON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    _mellon = None
    _MELLON_IMPORT_ERROR = exc


def mellon_available() -> bool:
    """True if the ``mellon`` package can be imported in this environment."""
    return _mellon is not None


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sp = scipy.stats.spearmanr(a, b).correlation
    pe = scipy.stats.pearsonr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
    sp = float(sp) if not np.isnan(sp) else 0.0
    pe = float(pe) if not np.isnan(pe) else 0.0
    return sp, pe


def knn_log_density(X, k=15):
    """Dimension-free kNN log-density  -log(mean k-NN radius); higher = denser.

    This mirrors the embedding density used by :func:`compute_density_correlation`
    (which correlates radii directly); we negate/​log so it reads as a density.
    """
    X = np.asarray(X, dtype=np.float32)
    k = min(k, len(X) - 1)
    if k < 1:
        return np.zeros(len(X))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    r = nn.kneighbors(X, return_distance=True)[0][:, 1:].mean(axis=1)
    return -np.log(r + 1e-12)


def mellon_log_density(X, pca_dim=20, seed=42):
    """Mellon log-density per row of ``X``.

    If ``X`` has more than ``pca_dim`` columns it is PCA-reduced first (Mellon is
    unstable in high dimensions).  ``pca_dim=None`` disables the reduction.
    Returns ``None`` if Mellon is not installed.
    """
    if not mellon_available():
        return None
    X = np.asarray(X, dtype=np.float32)
    if pca_dim is not None and X.shape[1] > pca_dim:
        n_comp = min(pca_dim, X.shape[0], X.shape[1])
        X = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    est = _mellon.DensityEstimator()
    return np.asarray(est.fit_predict(X), dtype=np.float64)


def compute_mellon_density_correlation(
    hd_data,
    emb_data,
    density_k=15,
    mellon_pca_dim=20,
    seed=42,
    hd_log_density=None,
):
    """Correlate Mellon's HD ground-truth density with the embedding density.

    Parameters
    ----------
    hd_data : (n, D) array — high-dimensional data.
    emb_data : (n, d) array — the low-dimensional embedding (typically d=2).
    density_k : neighbourhood size for the kNN embedding-density definition.
    mellon_pca_dim : PCA dimensionality used before fitting Mellon on ``hd_data``.
    hd_log_density : optional precomputed Mellon HD log-density (length n).  Pass
        this to avoid refitting Mellon on the same HD data across methods/epochs.

    Returns
    -------
    dict with (empty if Mellon unavailable):
        mellon_density_correlation            : Spearman(HD-Mellon, kNN-emb-density)  [primary]
        mellon_density_correlation_pearson    : Pearson(HD-Mellon, kNN-emb-density)
        mellon_density_correlation_mellon2d           : Spearman(HD-Mellon, 2D-Mellon-density)
        mellon_density_correlation_mellon2d_pearson   : Pearson(HD-Mellon, 2D-Mellon-density)
    """
    if not mellon_available():
        return {}

    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)

    rho_hd = (
        np.asarray(hd_log_density, dtype=np.float64)
        if hd_log_density is not None
        else mellon_log_density(hd_data, pca_dim=mellon_pca_dim, seed=seed)
    )
    if rho_hd is None:
        return {}

    # embedding density, two definitions
    emb_knn = knn_log_density(emb_data, k=density_k)
    rho_emb = mellon_log_density(emb_data, pca_dim=None, seed=seed)

    sp_k, pe_k = _safe_corr(rho_hd, emb_knn)
    out = {
        "mellon_density_correlation": sp_k,
        "mellon_density_correlation_pearson": pe_k,
    }
    if rho_emb is not None:
        sp_m, pe_m = _safe_corr(rho_hd, rho_emb)
        out["mellon_density_correlation_mellon2d"] = sp_m
        out["mellon_density_correlation_mellon2d_pearson"] = pe_m
    return out
