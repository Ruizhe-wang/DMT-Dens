"""
Scaling Consistency Metrics for Multi-Scale Dimensionality Reduction

Provides scale-SENSITIVE metrics for evaluating whether global and local
views exhibit consistent contraction behavior across scales.

Unlike Procrustes-based fidelity (which normalizes away scale), these
metrics explicitly measure how embeddings contract at different scales.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple


def compute_local_scale_distortion(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample local scale and local distortion.

    local_scale uses the mean high-dimensional kNN distance for each point.
    distortion compares the distances to those same high-dimensional neighbors
    after projection into the low-dimensional embedding.
    """

    high_dim = np.asarray(high_dim, dtype=np.float32)
    low_dim = np.asarray(low_dim, dtype=np.float32)

    if high_dim.ndim != 2 or low_dim.ndim != 2:
        raise ValueError("high_dim and low_dim must both be 2D arrays")
    if len(high_dim) != len(low_dim):
        raise ValueError("high_dim and low_dim must have the same number of samples")

    n_samples = len(high_dim)
    if n_samples == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if n_samples == 1:
        ones = np.ones((1,), dtype=np.float32)
        return np.zeros((1,), dtype=np.float32), ones

    k_effective = min(max(int(k), 1), n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k_effective + 1, metric="euclidean")
    nn.fit(high_dim)
    hd_distances, hd_indices = nn.kneighbors(high_dim, return_distance=True)
    hd_distances = hd_distances[:, 1:]
    hd_indices = hd_indices[:, 1:]

    low_neighbors = low_dim[hd_indices]
    low_centered = low_neighbors - low_dim[:, None, :]
    low_distances = np.linalg.norm(low_centered, axis=2)

    local_scale = hd_distances.mean(axis=1).astype(np.float32)
    distortion = (
        low_distances / np.maximum(hd_distances, 1e-12)
    ).mean(axis=1).astype(np.float32)
    distortion = np.where(np.isfinite(distortion), distortion, 1.0).astype(np.float32)
    return local_scale, distortion


def percentile_limits(
    values: np.ndarray,
    low_q: float = 0.02,
    high_q: float = 0.98,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 1.0

    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if np.isclose(low, high):
        high = low + 1e-6
    return low, high


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix

    upper = matrix[np.triu_indices_from(matrix, k=1)]
    if upper.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)

    scale = np.quantile(upper, 0.95)
    return (matrix / max(float(scale), 1e-12)).astype(np.float32)


def compute_distance_matrix_triplet(
    cluster_high: np.ndarray,
    cluster_low: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hd_matrix = pairwise_distances(cluster_high, metric="euclidean").astype(np.float32)
    ld_matrix = pairwise_distances(cluster_low, metric="euclidean").astype(np.float32)
    hd_norm = normalize_matrix(hd_matrix)
    ld_norm = normalize_matrix(ld_matrix)
    diff = (ld_norm - hd_norm).astype(np.float32)
    return hd_norm, ld_norm, diff


def _sampled_mean_distance(
    data: np.ndarray, max_pairs: int = 5000, seed: int = 42
) -> float:
    """Estimate mean pairwise distance by sampling random pairs.

    For small data (N*(N-1)/2 <= max_pairs), computes exact mean.
    Otherwise samples max_pairs random pairs for O(max_pairs) cost
    instead of O(N^2).
    """
    n = len(data)
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        triu = np.triu_indices(n, k=1)
        diffs = data[triu[0]] - data[triu[1]]
        return float(np.mean(np.sqrt(np.sum(diffs ** 2, axis=1))))

    rng = np.random.RandomState(seed)
    row_idx = rng.randint(0, n, size=max_pairs)
    col_idx = rng.randint(0, n - 1, size=max_pairs)
    col_idx[col_idx >= row_idx] += 1  # avoid self-pairs

    diffs = data[row_idx] - data[col_idx]
    return float(np.mean(np.sqrt(np.sum(diffs ** 2, axis=1))))


def _sample_pair_indices(n: int, max_pairs: int = 5000, seed: int = 42):
    """Sample pair indices without materializing all N*(N-1)/2 pairs."""
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        triu = np.triu_indices(n, k=1)
        return triu[0], triu[1]

    rng = np.random.RandomState(seed)
    row_idx = rng.randint(0, n, size=max_pairs)
    col_idx = rng.randint(0, n - 1, size=max_pairs)
    col_idx[col_idx >= row_idx] += 1
    return row_idx, col_idx

def knn_preservation_rate(
    high_dim_data: np.ndarray,
    layer_embeddings: List[np.ndarray],
    k: int = 12,
) -> Dict[str, object]:
    """
    Metric 3: k-NN Preservation Rate

    For each scale, measure how well the embedding preserves the
    k-nearest-neighbor structure from the high-dimensional space.

    Recall(s) = mean_i[ |N_hd(i,k) ∩ N_emb(i,k,s)| / k ]

    Args:
        high_dim_data: Original high-dim data, shape (n_samples, n_features)
        layer_embeddings: List of embeddings, one per scale
        k: Number of neighbors to compare

    Returns:
        Dict with:
        - per_scale_recall: k-NN recall per scale
        - per_scale_precision: k-NN precision per scale
    """
    n_samples = len(high_dim_data)
    if n_samples < 2:
        return {
            "per_scale_recall": [1.0 for _ in layer_embeddings],
            "per_scale_precision": [1.0 for _ in layer_embeddings],
        }

    k_safe = min(k, n_samples - 1)
    if k_safe <= 0:
        return {
            "per_scale_recall": [1.0 for _ in layer_embeddings],
            "per_scale_precision": [1.0 for _ in layer_embeddings],
        }

    # Query k+1 neighbors on the same fitted dataset and drop self-neighbor.
    nn_hd = NearestNeighbors(n_neighbors=k_safe + 1).fit(high_dim_data)
    hd_neighbors = nn_hd.kneighbors(high_dim_data, return_distance=False)[:, 1:]

    per_scale_recall = []
    per_scale_precision = []

    for emb in layer_embeddings:
        nn_emb = NearestNeighbors(n_neighbors=k_safe + 1).fit(emb)
        emb_neighbors = nn_emb.kneighbors(emb, return_distance=False)[:, 1:]

        recalls = []
        for i in range(n_samples):
            hd_set = set(hd_neighbors[i])
            emb_set = set(emb_neighbors[i])
            overlap = len(hd_set & emb_set)
            recalls.append(overlap / k_safe)

        per_scale_recall.append(float(np.mean(recalls)))
        per_scale_precision.append(float(np.mean(recalls)))  # symmetric for k-NN

    return {
        "per_scale_recall": per_scale_recall,
        "per_scale_precision": per_scale_precision,
    }
