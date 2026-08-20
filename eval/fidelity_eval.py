import numpy as np
import scipy
import scipy.stats
try:
    import torch
except ImportError:  # torch only needed by eval_normalize; metrics work without it
    torch = None
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def _neighbours_and_ranks(distances, k):
    """Extract k-neighbourhoods and full ranks from a distance matrix."""
    indices = np.argsort(distances, axis=-1, kind="stable")
    neighbourhood = indices[:, 1 : k + 1]
    ranks = indices.argsort(axis=-1, kind="stable")
    return neighbourhood, ranks


def _rank_based_trustworthiness(
    x_neighbourhood,
    x_ranks,
    z_neighbourhood,
    n,
    k,
):
    """Lee & Verleysen rank-based trustworthiness kernel."""
    if n < 2 or k < 1:
        return 0.0

    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return 0.0

    penalty = 0.0
    for row in range(n):
        missing_neighbours = np.setdiff1d(z_neighbourhood[row], x_neighbourhood[row])
        for neighbour in missing_neighbours:
            penalty += x_ranks[row, neighbour] - k

    return float(1 - 2 * penalty / denom)


def _resolve_neighbor_k(n_samples, requested_k, for_sklearn_trust=False):
    """Resolve a safe neighborhood size for small-sample robustness."""
    if n_samples < 2:
        return 0

    k_cap = n_samples - 2
    if for_sklearn_trust:
        # sklearn trustworthiness requires n_neighbors < n_samples / 2
        k_cap = min(k_cap, (n_samples - 1) // 2)

    return max(0, min(int(requested_k), int(k_cap)))


def eval_normalize(emb):
    if torch is not None and isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    emb = np.nan_to_num(emb, nan=0.0, posinf=1e6, neginf=-1e6)
    return emb - emb.mean(axis=0, keepdims=True)


def compute_knn_preservation(hd_data, emb_data, k=12):
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    k = min(k, len(hd_data) - 1)
    if k < 1:
        return 0.0
    nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hd_data)
    hd_neighbors = nn_hd.kneighbors(hd_data, return_distance=False)[:, 1:]
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
    emb_neighbors = nn_emb.kneighbors(emb_data, return_distance=False)[:, 1:]
    overlap = sum(
        len(np.intersect1d(hd_neighbors[i], emb_neighbors[i])) for i in range(len(hd_data))
    )
    return float(overlap / (len(hd_data) * k))


def compute_distance_correlation(hd_data, emb_data, n_pairs=5000, seed=42):
    """Also known as Shepard Diagram Correlation (Spearman Rank Correlation of Distances)."""
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < 2:
        return 0.0
    rng = np.random.RandomState(seed)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    if len(idx_a) == 0:
        return 0.0
    dist_hd = np.linalg.norm(hd_data[idx_a] - hd_data[idx_b], axis=1)
    dist_emb = np.linalg.norm(emb_data[idx_a] - emb_data[idx_b], axis=1)
    corr = scipy.stats.spearmanr(dist_hd, dist_emb).correlation
    return float(corr if not np.isnan(corr) else 0.0)


def compute_true_distance_correlation(hd_data, emb_data, n_samples=2000, seed=42):
    """
    Székely et al. Distance Correlation (dCor).
    Computed on a subsample for O(N^2) computational efficiency.
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < 2:
        return 0.0
        
    if n > n_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(n, size=n_samples, replace=False)
        X = hd_data[indices]
        Y = emb_data[indices]
    else:
        X = hd_data
        Y = emb_data
        
    n_sub = len(X)
    
    from scipy.spatial.distance import cdist
    a = cdist(X, X, metric='euclidean')
    b = cdist(Y, Y, metric='euclidean')
    
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    
    dCov2_XY = (A * B).sum() / (n_sub ** 2)
    dCov2_XX = (A * A).sum() / (n_sub ** 2)
    dCov2_YY = (B * B).sum() / (n_sub ** 2)
    
    if dCov2_XX > 0 and dCov2_YY > 0:
        val = dCov2_XY / np.sqrt(dCov2_XX * dCov2_YY)
        return float(np.sqrt(max(val, 0.0)))
    return 0.0


def compute_random_triplet_accuracy(hd_data, emb_data, n_triplets=10000, seed=42):
    """Random Triplet Accuracy."""
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < 3:
        return 0.0
    rng = np.random.RandomState(seed)
    
    idx_A = rng.randint(0, n, size=n_triplets)
    idx_B = rng.randint(0, n, size=n_triplets)
    idx_C = rng.randint(0, n, size=n_triplets)
    
    mask = (idx_A != idx_B) & (idx_B != idx_C) & (idx_A != idx_C)
    idx_A, idx_B, idx_C = idx_A[mask], idx_B[mask], idx_C[mask]
    
    if len(idx_A) == 0:
        return 0.0
        
    dist_hd_AB = np.linalg.norm(hd_data[idx_A] - hd_data[idx_B], axis=1)
    dist_hd_AC = np.linalg.norm(hd_data[idx_A] - hd_data[idx_C], axis=1)
    
    dist_emb_AB = np.linalg.norm(emb_data[idx_A] - emb_data[idx_B], axis=1)
    dist_emb_AC = np.linalg.norm(emb_data[idx_A] - emb_data[idx_C], axis=1)
    
    hd_closer_B = dist_hd_AB < dist_hd_AC
    emb_closer_B = dist_emb_AB < dist_emb_AC
    
    matches = (hd_closer_B == emb_closer_B)
    return float(np.mean(matches))


def compute_continuity(hd_data, emb_data, k=12):
    """Compute continuity as reverse-neighbourhood trustworthiness."""
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < 3:
        return 0.0

    k = _resolve_neighbor_k(n, k, for_sklearn_trust=False)
    if k < 1:
        return 0.0

    from scipy.spatial.distance import cdist
    hd_dist = cdist(hd_data, hd_data, metric='euclidean')
    emb_dist = cdist(emb_data, emb_data, metric='euclidean')

    hd_neighbourhood, hd_ranks = _neighbours_and_ranks(hd_dist, k)
    emb_neighbourhood, emb_ranks = _neighbours_and_ranks(emb_dist, k)

    return _rank_based_trustworthiness(
        emb_neighbourhood,
        emb_ranks,
        hd_neighbourhood,
        n,
        k,
    )


def compute_expansion_compression(hd_data, emb_data, k=15):
    """
    Computes a neighborhood Expansion/Compression (Distortion) metric.
    Compares the mean distance of k-nearest neighbors in HD vs LD.
    Returns the mean absolute log distortion. Higher is worse (0 is perfect).
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    k = min(k, n - 1)
    if k < 1:
        return 0.0
        
    nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hd_data)
    hd_dists = nn_hd.kneighbors(hd_data, return_distance=True)[0][:, 1:]
    
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
    emb_dists = nn_emb.kneighbors(emb_data, return_distance=True)[0][:, 1:]
    
    hd_dists_norm = hd_dists / (hd_dists.mean() + 1e-8)
    emb_dists_norm = emb_dists / (emb_dists.mean() + 1e-8)
    
    ratio = emb_dists_norm.mean(axis=1) / (hd_dists_norm.mean(axis=1) + 1e-8)
    distortion = np.abs(np.log(ratio + 1e-8)).mean()
    return float(distortion)

def compute_density_correlation(hd_data, emb_data, k=15):
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    k = min(k, len(hd_data) - 1)
    if k < 1:
        return 0.0
    nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hd_data)
    hd_radius = nn_hd.kneighbors(hd_data, return_distance=True)[0][:, 1:].mean(axis=1)
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
    emb_radius = nn_emb.kneighbors(emb_data, return_distance=True)[0][:, 1:].mean(axis=1)
    corr = scipy.stats.spearmanr(hd_radius, emb_radius).correlation
    return float(corr if not np.isnan(corr) else 0.0)


def compute_ldc(hd_data, emb_data, k=15):
    """
    Local Density Correlation (LDC).
    Spearman rank correlation of distances to the k-th nearest neighbor.
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data) 
    k = min(k, n - 1)
    if k < 1:
        return 0.0
        
    nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hd_data)
    hd_k_dist = nn_hd.kneighbors(hd_data, return_distance=True)[0][:, k]
    
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
    emb_k_dist = nn_emb.kneighbors(emb_data, return_distance=True)[0][:, k]
    
    corr = scipy.stats.spearmanr(hd_k_dist, emb_k_dist).correlation
    return float(corr if not np.isnan(corr) else 0.0)


def _deterministic_choice(n, size, seed):
    """Seeded subsample of `size` distinct indices from range(n)."""
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=size, replace=False)


def _mean_pairwise_euclidean(X):
    """Mean of the pairwise Euclidean distances of the rows of X."""
    from scipy.spatial.distance import pdist
    X = np.asarray(X, dtype=np.float32)
    if len(X) < 2:
        return 0.0
    return float(pdist(X, metric="euclidean").mean())


# High-dimensional reference quantities (S_c for ASR, n_HD for FR-DC) depend
# only on (hd_data, labels) and are identical across methods/seeds for a given
# dataset. Cache them per HD array so the multi-layer loop (and repeat calls on
# the same object) does not recompute them.
_ASR_SPREAD_CACHE = {}
_FRDC_REF_CACHE = {}


def compute_asr(
    emb_data,
    hd_data,
    labels,
    n_min=20,
    c_min=8,
    trim_frac=0.05,
    sub_cap=1000,
    rng_seed=0,
):
    """Area-Spread Recovery (ASR) -- a density-fidelity metric independent of
    the training loss.

    For each eligible class, compare the 2D convex-hull area A_c it occupies to
    its true high-dimensional spread S_c (mean pairwise distance in the HD
    reference space). A density-faithful embedding gives small area to compact
    (low-spread) classes and large area to diffuse ones, so A_c and S_c should
    rank-correlate. Methods that inflate dense clusters score low.

    Independence: it is class-level (not per-cell), uses withheld `labels`
    (absent from the loss), and tests 2D area vs HD spread rather than the
    kNN-radius density the model optimizes.

    Returns (asr, asr_mpd) where asr_mpd replaces the hull area with the class's
    mean pairwise 2D distance. Returns (nan, nan) when fewer than `c_min`
    classes have at least `n_min` points, or when labels are unavailable.
    Higher is better; range [-1, 1].
    """
    try:
        from scipy.spatial import ConvexHull
        try:
            from scipy.spatial import QhullError
        except ImportError:  # older scipy
            from scipy.spatial.qhull import QhullError
    except Exception:  # pragma: no cover - scipy.spatial always present here
        ConvexHull = None
        QhullError = Exception

    emb_data = np.asarray(emb_data, dtype=np.float32)
    hd_data = np.asarray(hd_data, dtype=np.float32)
    if labels is None:
        return float("nan"), float("nan")
    labels = np.asarray(labels).ravel()
    if len(labels) != len(emb_data) or len(labels) != len(hd_data):
        return float("nan"), float("nan")

    classes, counts = np.unique(labels, return_counts=True)
    eligible = classes[counts >= n_min]
    if len(eligible) < c_min:
        return float("nan"), float("nan")

    cache_key = (id(hd_data), hd_data.shape, n_min, sub_cap, rng_seed)
    spread = _ASR_SPREAD_CACHE.get(cache_key)
    compute_spread = spread is None
    if compute_spread:
        spread = {}

    areas, areas_mpd, spreads = [], [], []
    for c in eligible:
        idx = np.where(labels == c)[0]
        zc = emb_data[idx]

        # --- 2D hull area A_c (per method), after trimming the farthest tail ---
        centroid = zc.mean(axis=0)
        d = np.linalg.norm(zc - centroid, axis=1)
        zc_trim = zc[d <= np.quantile(d, 1.0 - trim_frac)]
        area = 0.0
        if ConvexHull is not None and len(zc_trim) >= 3:
            try:
                area = float(ConvexHull(zc_trim).volume)  # .volume == area in 2D
            except QhullError:  # collinear / degenerate class
                area = 0.0
        areas.append(area)

        # --- robustness variant: mean pairwise 2D distance (sub_cap capped) ---
        zc_sub = zc
        if len(zc_sub) > sub_cap:
            zc_sub = zc_sub[_deterministic_choice(len(zc_sub), sub_cap, rng_seed)]
        areas_mpd.append(_mean_pairwise_euclidean(zc_sub))

        # --- HD spread S_c (shared reference; computed once, cached) ---
        if compute_spread:
            xc = hd_data[idx]
            if len(xc) > sub_cap:
                xc = xc[_deterministic_choice(len(xc), sub_cap, rng_seed)]
            spread[c] = _mean_pairwise_euclidean(xc)
        spreads.append(spread[c])

    if compute_spread:
        _ASR_SPREAD_CACHE[cache_key] = spread

    def _spearman(a, b):
        if len(a) < 2:
            return float("nan")
        r = scipy.stats.spearmanr(a, b).correlation
        return float(r) if not np.isnan(r) else float("nan")

    return _spearman(areas, spreads), _spearman(areas_mpd, spreads)


def compute_frdc(
    hd_data,
    emb_data,
    n_eval=10000,
    m_ref=50000,
    target_count=50,
    rng_seed=0,
):
    """Fixed-Radius Density Correlation (FR-DC) -- a density-fidelity metric
    independent of the kNN-radius training loss.

    Density is estimated by counting neighbours within a *fixed radius* eps (a
    uniform-kernel / eps-ball estimator), the dual of the kNN-radius estimator
    the model optimizes: kNN fixes the count and measures the radius; FR-DC
    fixes the radius and measures the count. The radius eps is one value per
    space, the median distance to the `target_count`-th neighbour, held fixed
    across all evaluation points. Spearman-correlates HD and LD counts.

    The HD counts `n_HD` depend only on `hd_data` and are cached per HD array.
    Higher is better; range [-1, 1].
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < 3 or len(emb_data) != n:
        return 0.0
    tc = min(int(target_count), n - 1)
    if tc < 1:
        return 0.0

    e_idx = np.arange(n) if n <= n_eval else _deterministic_choice(n, n_eval, rng_seed)
    r_idx = np.arange(n) if n <= m_ref else _deterministic_choice(n, m_ref, rng_seed + 1)

    def _fixed_radius_counts(ref, query):
        tree = NearestNeighbors(algorithm="auto").fit(ref)
        dk = tree.kneighbors(query, n_neighbors=tc, return_distance=True)[0][:, -1]
        eps = float(np.median(dk))  # one fixed radius for all query points
        neigh = tree.radius_neighbors(query, radius=eps, return_distance=False)
        return np.array([len(a) for a in neigh], dtype=np.float64)

    cache_key = (id(hd_data), hd_data.shape, n_eval, m_ref, target_count, rng_seed)
    n_hd = _FRDC_REF_CACHE.get(cache_key)
    if n_hd is None:
        n_hd = _fixed_radius_counts(hd_data[r_idx], hd_data[e_idx])
        _FRDC_REF_CACHE[cache_key] = n_hd

    n_ld = _fixed_radius_counts(emb_data[r_idx], emb_data[e_idx])

    corr = scipy.stats.spearmanr(n_hd, n_ld).correlation
    return float(corr if not np.isnan(corr) else 0.0)


def compute_spir(hd_data, emb_data, k=12, noise_quantile=0.9):
    """
    Scattered Point Intrusion Rate (SPIR).
    Measures the proportion of high-density cluster points in the low-dimensional 
    neighborhood of noise points identified in high-dimensional space.
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    if n < k + 1:
        return 0.0
        
    # Identify noise points in HD by thresholding k-th neighbor distance
    nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(hd_data)
    hd_k_dist = nn_hd.kneighbors(hd_data, return_distance=True)[0][:, k]
    threshold = np.quantile(hd_k_dist, noise_quantile)
    is_noise = hd_k_dist >= threshold
    noise_indices = np.where(is_noise)[0]
    
    if len(noise_indices) == 0:
        return 0.0
        
    # Find k-NN in embedding space for these noise points
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
    emb_neighbors = nn_emb.kneighbors(emb_data[noise_indices], return_distance=False)[:, 1:]
    
    # Proportion of neighbors that are NOT noise points
    intrusion_counts = [np.sum(~is_noise[neighbors]) for neighbors in emb_neighbors]
    spir = np.mean(intrusion_counts) / k
    return float(spir)


def compute_msdp_auc(hd_data, emb_data, k_list=(5, 10, 15, 25, 50, 100, 200)):
    """
    Multi-Scale Density Preservation AUC (MSDP-AUC).
    Computes density_correlation at each scale k in k_list and returns
    the mean (AUC under the uniform scale axis).
    """
    hd_data = np.asarray(hd_data, dtype=np.float32)
    emb_data = np.asarray(emb_data, dtype=np.float32)
    n = len(hd_data)
    scores = []
    for k in k_list:
        k_safe = min(int(k), n - 1)
        if k_safe < 1:
            continue
        scores.append(compute_density_correlation(hd_data, emb_data, k=k_safe))
    return float(np.mean(scores)) if scores else 0.0


def compute_svc_accuracy(emb_data, labels, cv=5, seed=42):
    """
    Linear-kernel SVC classification accuracy on the embedding via stratified
    k-fold cross-validation. Higher is better (1.0 is perfect).
    Requires class labels.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    emb_data = np.asarray(emb_data, dtype=np.float32)
    labels = np.asarray(labels).ravel()
    n = len(emb_data)
    if n != len(labels):
        return 0.0

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes, class_counts = np.unique(y, return_counts=True)

    # Drop classes too rare for stratified CV (e.g. singletons produced by the
    # down_sample step) instead of collapsing the whole metric to 0. Rare
    # classes are a tiny fraction of cells; the well-populated classes still
    # yield a fair, comparable score. Previously a single <2-sample class
    # forced SVC accuracy to 0.0 -- the root cause of baseline svc_acc == 0 on
    # datasets with rare cell types (e.g. CELEGAN: 36 types, rarest 25 cells).
    keep_classes = classes[class_counts >= 2]
    if len(keep_classes) < 2:
        return 0.0
    keep_mask = np.isin(y, keep_classes)
    emb_data = emb_data[keep_mask]
    y = np.unique(y[keep_mask], return_inverse=True)[1]
    class_counts = np.bincount(y)

    cv_safe = min(int(cv), int(class_counts.min()))
    if cv_safe < 2:
        return 0.0

    cv_split = StratifiedKFold(n_splits=cv_safe, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", max_iter=900000),
    )
    scores = cross_val_score(
        clf,
        emb_data,
        y,
        cv=cv_split,
        scoring="accuracy",
        n_jobs=1,
        error_score=0.0,
    )
    return float(np.mean(scores))


def summarize_embedding_metrics(
    hd_data,
    emb_data,
    knn_k=12,
    density_k=15,
    seed=42,
    labels=None,
):
    emb_eval = eval_normalize(emb_data)
    hd_eval = np.asarray(hd_data, dtype=np.float32)
    trust_k = _resolve_neighbor_k(len(hd_eval), knn_k, for_sklearn_trust=True)
    trust_value = float(trustworthiness(hd_eval, emb_eval, n_neighbors=trust_k)) if trust_k >= 1 else 0.0

    summary = {
        "knn_preservation": compute_knn_preservation(hd_eval, emb_eval, k=knn_k),
        "trustworthiness": trust_value,
        "shepard_spearman_correlation": compute_distance_correlation(hd_eval, emb_eval, seed=seed),
        "distance_correlation_dcor": compute_true_distance_correlation(hd_eval, emb_eval, seed=seed),
        "random_triplet_accuracy": compute_random_triplet_accuracy(hd_eval, emb_eval, seed=seed),
        "density_correlation": compute_density_correlation(hd_eval, emb_eval, k=density_k),
        "local_density_correlation": compute_ldc(hd_eval, emb_eval, k=density_k),
        "scattered_point_intrusion_rate": compute_spir(hd_eval, emb_eval, k=knn_k),
        "continuity": compute_continuity(hd_eval, emb_eval, k=knn_k),
        "neighborhood_distortion": compute_expansion_compression(hd_eval, emb_eval, k=knn_k),
        "msdp_auc": compute_msdp_auc(hd_eval, emb_eval),
    }

    if labels is not None:
        summary["svc_accuracy"] = compute_svc_accuracy(emb_eval, labels, seed=seed)
        summary["svc_acc"] = summary["svc_accuracy"]

    # Backward-compatible aliases for existing callers.
    return {
        **summary,
        "knn": summary["knn_preservation"],
        "trust": summary["trustworthiness"],
        "dist_corr": summary["shepard_spearman_correlation"],
        "den_corr": summary["density_correlation"],
        "ldc": summary["local_density_correlation"],
        "spir": summary["scattered_point_intrusion_rate"],
    }
