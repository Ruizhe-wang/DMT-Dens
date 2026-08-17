"""
Local-Global Consistency Analysis Module

Core analysis engine for comparing global and local dimensionality reduction.
Provides metrics computation and embedding alignment functionality.
"""

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE, trustworthiness
import umap
from typing import Optional, List, Dict, Tuple

try:
    import phate
except ImportError:
    phate = None


# Color palette for visualization
D3_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
] * 100


class LocalGlobalConsistencyVisualizer:
    """
    Visualizer for comparing local and global dimensionality reduction results.

    Provides core analysis functionality: embedding computation, alignment, metrics.
    Plotting functions are separated to consistency_callback.py (Plotly for WandB)
    and consistency_util.py (matplotlib for local saving).
    """

    def __init__(
        self,
        global_data: np.ndarray,
        global_embedding: np.ndarray,
        labels: np.ndarray,
        method: str = 'tsne',
        random_state: int = 42
    ):
        self.global_data = global_data
        self.global_embedding = global_embedding
        self.labels = labels
        self.method = method
        self.random_state = random_state

        # Removed: O(N^2) full distance matrices are too expensive for large N.
        # Metrics in compute_metrics() now use sampled pairs instead.
        self._global_data = global_data
        self._global_embedding = global_embedding

    def select_subset(
        self,
        selected_labels: Optional[List] = None,
        indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select a subset of data based on labels or indices."""
        if indices is not None:
            mask = np.zeros(len(self.labels), dtype=bool)
            mask[indices] = True
        elif selected_labels is not None:
            mask = np.isin(self.labels, selected_labels)
        else:
            raise ValueError("Either selected_labels or indices must be provided")

        subset_data = self.global_data[mask]
        subset_global_embedding = self.global_embedding[mask]
        subset_labels = self.labels[mask]

        return subset_data, subset_global_embedding, subset_labels, mask

    def compute_local_embedding(self, subset_data: np.ndarray) -> np.ndarray:
        """Compute local dimensionality reduction on subset data."""
        if self.method == 'tsne':
            perplexity = min(30, len(subset_data) - 1)
            reducer = TSNE(
                n_components=2,
                perplexity=max(5, perplexity),
                random_state=self.random_state
            )
            return reducer.fit_transform(subset_data)

        if self.method == 'umap':
            n_neighbors = min(15, len(subset_data) - 1)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, n_neighbors),
                random_state=self.random_state
            )
            return reducer.fit_transform(subset_data)

        if self.method == 'densmap':
            n_neighbors = min(15, len(subset_data) - 1)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, n_neighbors),
                densmap=True,
                output_dens=False,
                random_state=self.random_state
            )
            return reducer.fit_transform(subset_data)

        if self.method == 'phate':
            if phate is None:
                raise ImportError("PHATE consistency analysis requires the `phate` package.")
            reducer = phate.PHATE(
                n_components=2,
                knn=min(15, max(5, len(subset_data) - 1)),
                random_state=self.random_state,
                verbose=0,
            )
            return reducer.fit_transform(subset_data)

        raise ValueError(f"Unsupported method: {self.method}")

    def align_embeddings(
        self,
        reference: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align two embeddings using Procrustes analysis."""
        ref_centered = reference - reference.mean(axis=0)
        target_centered = target - target.mean(axis=0)

        ref_normalized = ref_centered / np.std(ref_centered)
        target_normalized = target_centered / np.std(target_centered)

        aligned_ref, aligned_target, disparity = procrustes(
            ref_normalized, target_normalized
        )

        return aligned_ref, aligned_target, disparity

    def compute_metrics(
        self,
        subset_data: np.ndarray,
        global_subset_embedding: np.ndarray,
        local_embedding: np.ndarray,
        max_pairs: int = 10000,
    ) -> Dict[str, float]:
        """Compute consistency metrics between global and local embeddings.

        Uses sampled pairs instead of full pairwise distance matrices
        to avoid O(N^2) memory and compute for large subsets.
        """
        n = len(subset_data)
        total_pairs = n * (n - 1) // 2

        if total_pairs <= max_pairs:
            triu = np.triu_indices(n, k=1)
            row_idx, col_idx = triu[0], triu[1]
        else:
            rng = np.random.RandomState(42)
            row_idx = rng.randint(0, n, size=max_pairs)
            col_idx = rng.randint(0, n - 1, size=max_pairs)
            col_idx[col_idx >= row_idx] += 1

        hd_flat = np.sqrt(np.sum((subset_data[row_idx] - subset_data[col_idx]) ** 2, axis=1))
        global_flat = np.sqrt(np.sum((global_subset_embedding[row_idx] - global_subset_embedding[col_idx]) ** 2, axis=1))
        local_flat = np.sqrt(np.sum((local_embedding[row_idx] - local_embedding[col_idx]) ** 2, axis=1))

        global_hd_corr, _ = pearsonr(hd_flat, global_flat)
        local_hd_corr, _ = pearsonr(hd_flat, local_flat)
        global_local_corr, _ = pearsonr(global_flat, local_flat)

        global_hd_spearman, _ = spearmanr(hd_flat, global_flat)
        local_hd_spearman, _ = spearmanr(hd_flat, local_flat)

        k = min(12, len(subset_data) - 2)
        if k > 0:
            global_trust = trustworthiness(subset_data, global_subset_embedding, n_neighbors=k)
            local_trust = trustworthiness(subset_data, local_embedding, n_neighbors=k)
        else:
            global_trust = local_trust = 1.0

        _, _, disparity = self.align_embeddings(global_subset_embedding, local_embedding)

        return {
            'global_hd_correlation': global_hd_corr,
            'local_hd_correlation': local_hd_corr,
            'global_local_correlation': global_local_corr,
            'global_hd_spearman': global_hd_spearman,
            'local_hd_spearman': local_hd_spearman,
            'global_trustworthiness': global_trust,
            'local_trustworthiness': local_trust,
            'procrustes_disparity': disparity,
            'fidelity_score': 1 - disparity
        }

    # def compute_per_point_distortion(
    #     self,
    #     global_subset_embedding: np.ndarray,
    #     local_embedding: np.ndarray,
    #     aligned: bool = True
    # ) -> np.ndarray:
    #     """Compute per-point distortion between global and local embeddings."""
    #     if aligned:
    #         aligned_global, aligned_local, _ = self.align_embeddings(
    #             global_subset_embedding, local_embedding
    #         )
    #     else:
    #         aligned_global = global_subset_embedding
    #         aligned_local = local_embedding

    #     distortion = np.linalg.norm(aligned_global - aligned_local, axis=1)
    #     return distortion


def quick_consistency_analysis(
    data: np.ndarray,
    labels: np.ndarray,
    selected_labels: List,
    method: str = 'tsne',
    save_dir: str = './consistency_plots',
    use_wandb: bool = False
) -> Dict:
    """Quick consistency analysis using core visualizer."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    elif method == 'densmap':
        reducer = umap.UMAP(n_components=2, densmap=True, output_dens=False, random_state=42)
    elif method == 'phate':
        if phate is None:
            raise ImportError("PHATE consistency analysis requires the `phate` package.")
        reducer = phate.PHATE(n_components=2, random_state=42, verbose=0)
    else:
        raise ValueError(f"Unsupported method: {method}")

    global_embedding = reducer.fit_transform(data)

    viz = LocalGlobalConsistencyVisualizer(
        global_data=data,
        global_embedding=global_embedding,
        labels=labels,
        method=method
    )

    subset_data, subset_global_emb, subset_labels, mask = viz.select_subset(
        selected_labels=selected_labels
    )

    local_emb = viz.compute_local_embedding(subset_data)
    metrics = viz.compute_metrics(subset_data, subset_global_emb, local_emb)

    results = {
        'metrics': metrics,
        'subset_size': len(subset_data),
        'selected_labels': selected_labels
    }

    if use_wandb:
        import wandb
        log_dict = {
            f'consistency/fidelity': metrics['fidelity_score'],
            f'consistency/trustworthiness': metrics['local_trustworthiness'],
            f'consistency/disparity': metrics['procrustes_disparity']
        }
        wandb.log(log_dict)

    return results
