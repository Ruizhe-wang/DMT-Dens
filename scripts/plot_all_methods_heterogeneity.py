"""
Generate cluster heterogeneity figures for UMAP, t-SNE, and DiffTree v3 side by side,
with consistent color scales across all three methods.

All kNN metrics use the original high-dimensional data (e.g. 784-dim pixels) as baseline,
making the results directly comparable.

Outputs per method (in <output-dir>/<method>/):
  1. Embedding colored by high-dimensional local scale
  2. Embedding colored by local distortion ratio
  3. Within-cluster pairwise distance matrix comparison
  4. Ordered sample strip along the cluster direction

Examples:
  python scripts/plot_all_methods_heterogeneity.py \\
      --ckpt path/to/difftree.ckpt --labels 1 7 --target-label 1

  python scripts/plot_all_methods_heterogeneity.py \\
      --ckpt path/to/difftree.ckpt --target-label 1 --max-samples 6000

  # Skip DiffTree if no checkpoint available
  python scripts/plot_all_methods_heterogeneity.py \\
      --methods umap tsne --target-label 1
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.datasets import MNIST

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cluster heterogeneity across UMAP, t-SNE, and DiffTree v3."
    )
    # --- Methods ---
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=["umap", "tsne", "difftree"],
        help="Which methods to run. Default: umap tsne difftree.",
    )
    # --- Data ---
    parser.add_argument(
        "--input-npz", type=str, default=None,
        help="Optional .npz file with `data` and `labels` arrays.",
    )
    parser.add_argument("--data-key", type=str, default="data")
    parser.add_argument("--labels-key", type=str, default="labels")
    parser.add_argument("--images-key", type=str, default="images")
    parser.add_argument(
        "--labels", type=int, nargs="*", default=None,
        help="Optional label subset.",
    )
    parser.add_argument(
        "--target-label", type=int, default=None,
        help="Label for within-cluster views (figures 3 and 4).",
    )
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--cluster-max-samples", type=int, default=120)
    parser.add_argument("--strip-samples", type=int, default=12)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="results/all_methods_heterogeneity",
    )
    # --- UMAP ---
    parser.add_argument("--umap-neighbors", type=int, default=15)
    # --- t-SNE ---
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-lr", type=float, default=200.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    # --- DiffTree ---
    parser.add_argument("--ckpt", type=str, default=None,
                        help="DiffTree v3 checkpoint (.ckpt).")
    parser.add_argument("--head-index", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist_subset(
    label_subset: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = MNIST(root="data", train=True, download=False, transform=transform)

    data_list, image_list, label_list = [], [], []
    for img, label in train_data:
        arr = img.numpy().squeeze()
        image_list.append(arr)
        data_list.append(arr.reshape(-1))
        label_list.append(label)

    data = np.stack(data_list).astype(np.float32)
    images = np.stack(image_list).astype(np.float32)
    labels = np.asarray(label_list)

    if label_subset is not None:
        mask = np.isin(labels, label_subset)
        data, images, labels = data[mask], images[mask], labels[mask]
    return data, labels, images


def load_npz_dataset(
    path: str, data_key: str, labels_key: str, images_key: str,
    label_subset: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    loaded = np.load(path, allow_pickle=True)
    if data_key not in loaded or labels_key not in loaded:
        raise KeyError(f"{path} must contain `{data_key}` and `{labels_key}`.")

    raw_data = np.asarray(loaded[data_key])
    labels = np.asarray(loaded[labels_key])
    images = np.asarray(loaded[images_key]) if images_key in loaded else None

    if raw_data.ndim > 2:
        data = raw_data.reshape(raw_data.shape[0], -1).astype(np.float32)
        if images is None and raw_data.ndim == 3:
            images = raw_data
    else:
        data = raw_data.astype(np.float32)

    if label_subset is not None:
        mask = np.isin(labels, label_subset)
        data, labels = data[mask], labels[mask]
        if images is not None:
            images = images[mask]
    return data, labels, images


def stratified_downsample(
    data: np.ndarray, labels: np.ndarray, images: Optional[np.ndarray],
    max_samples: Optional[int], seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if max_samples is None or len(data) <= max_samples:
        return data, labels, images

    rng = np.random.RandomState(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_to_target = {}
    remaining = max_samples

    for label, count in zip(unique_labels, counts):
        target = max(1, int(round(max_samples * (count / len(labels)))))
        label_to_target[label] = min(target, count)
        remaining -= label_to_target[label]

    if remaining > 0:
        for label, count in sorted(
            zip(unique_labels, counts), key=lambda item: item[1], reverse=True
        ):
            extra = min(remaining, count - label_to_target[label])
            if extra > 0:
                label_to_target[label] += extra
                remaining -= extra
            if remaining == 0:
                break

    indices = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        take = label_to_target[label]
        if take >= len(label_idx):
            indices.append(label_idx)
        else:
            indices.append(np.sort(rng.choice(label_idx, size=take, replace=False)))

    keep = np.concatenate(indices)
    keep.sort()
    return data[keep], labels[keep], images[keep] if images is not None else None


# ---------------------------------------------------------------------------
# Embedding methods
# ---------------------------------------------------------------------------

def run_umap(data: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(data) - 1),
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(data)


def run_tsne(
    data: np.ndarray, perplexity: float, lr: float, n_iter: int, seed: int,
) -> np.ndarray:
    from openTSNE import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(data) - 1),
        learning_rate=lr,
        n_iter=n_iter,
        random_state=seed,
        initialization="pca",
        negative_gradient_method="fft",
    )
    return np.asarray(tsne.fit(data))


def run_difftree(
    data: np.ndarray, ckpt_path: str, head_index: int, batch_size: int,
    device: torch.device,
) -> np.ndarray:
    from model.DiffTreeVQ_v3 import DMTEVT_model

    model = DMTEVT_model.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            x = torch.from_numpy(data[start:end]).float().to(device)
            _, _, lat_vis_best, lat_vis_list = model(x, tau=1.0)

            if 0 <= head_index < len(lat_vis_list):
                emb = lat_vis_list[head_index]
            else:
                best_idx = (
                    model.best_head_index
                    if 0 <= model.best_head_index < len(lat_vis_list)
                    else -1
                )
                emb = lat_vis_list[best_idx]
            all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_knn_metrics(
    high_dim: np.ndarray, low_dim: np.ndarray, k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k_effective = min(k, len(high_dim) - 1)
    nn = NearestNeighbors(n_neighbors=k_effective + 1, metric="euclidean")
    nn.fit(high_dim)
    hd_distances, hd_indices = nn.kneighbors(high_dim, return_distance=True)
    hd_distances = hd_distances[:, 1:]
    hd_indices = hd_indices[:, 1:]

    low_neighbors = low_dim[hd_indices]
    low_centered = low_neighbors - low_dim[:, None, :]
    low_distances = np.linalg.norm(low_centered, axis=2)

    local_scale = hd_distances.mean(axis=1)
    distortion = (low_distances / np.maximum(hd_distances, 1e-12)).mean(axis=1)
    return local_scale, distortion


def percentile_limits(
    values: np.ndarray, low_q: float = 0.02, high_q: float = 0.98,
) -> Tuple[float, float]:
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if math.isclose(low, high):
        high = low + 1e-6
    return low, high


def choose_target_label(labels: np.ndarray, requested: Optional[int]) -> int:
    if requested is not None:
        if requested not in set(np.unique(labels).tolist()):
            raise ValueError(f"target-label {requested} not in subset.")
        return requested
    unique_labels, counts = np.unique(labels, return_counts=True)
    return int(unique_labels[np.argmax(counts)])


def ordered_cluster_subset(
    cluster_high: np.ndarray, cluster_low: np.ndarray,
    cluster_s1: np.ndarray, cluster_s2: np.ndarray,
    cluster_images: Optional[np.ndarray], max_samples: int,
):
    axis_coord = PCA(n_components=1).fit_transform(cluster_low).reshape(-1)
    order = np.argsort(axis_coord)
    if len(order) > max_samples:
        positions = np.linspace(0, len(order) - 1, num=max_samples, dtype=int)
        order = order[positions]
    return (
        cluster_high[order], cluster_low[order],
        cluster_s1[order], cluster_s2[order],
        cluster_images[order] if cluster_images is not None else None,
        axis_coord[order],
    )


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    scale = np.quantile(upper, 0.95) if len(upper) else 1.0
    return matrix / max(float(scale), 1e-12)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "umap": ("UMAP", "UMAP 1", "UMAP 2"),
    "tsne": ("t-SNE", "t-SNE 1", "t-SNE 2"),
    "difftree": ("DiffTree v3", "DiffTree Dim 1", "DiffTree Dim 2"),
}


def render_preview(ax: plt.Axes, sample: np.ndarray) -> None:
    if sample.ndim == 2:
        ax.imshow(sample, cmap="gray")
    else:
        side = int(round(math.sqrt(sample.size)))
        if side * side == sample.size:
            ax.imshow(sample.reshape(side, side), cmap="gray")
        else:
            ax.plot(sample[: min(128, sample.size)], color="#1f77b4", linewidth=1.5)
            ax.set_ylim(sample.min(), sample.max())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_metric_scatter(
    embedding: np.ndarray, labels: np.ndarray, metric: np.ndarray,
    title: str, colorbar_label: str, output_path: Path,
    method: str, vmin: float, vmax: float,
) -> None:
    _, xlabel, ylabel = METHOD_LABELS[method]
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=metric, s=4, cmap="viridis_r", alpha=0.86, linewidths=0,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.5)

    for lbl in np.unique(labels):
        mask = labels == lbl
        center = embedding[mask].mean(axis=0)
        ax.text(center[0], center[1], str(lbl), fontsize=10, weight="bold",
                ha="center", va="center")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_distance_matrix(
    cluster_high: np.ndarray, cluster_low: np.ndarray,
    target_label: int, method: str, output_path: Path,
) -> None:
    method_name = METHOD_LABELS[method][0]
    hd_matrix = pairwise_distances(cluster_high, metric="euclidean")
    ld_matrix = pairwise_distances(cluster_low, metric="euclidean")
    hd_norm = normalize_matrix(hd_matrix)
    ld_norm = normalize_matrix(ld_matrix)
    diff = ld_norm - hd_norm

    vmax = max(float(hd_norm.max()), float(ld_norm.max()), 1.0)
    diff_lim = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    panels = [
        (hd_norm, "High-D Pairwise Distance", "magma", 0.0, vmax),
        (ld_norm, f"{method_name} Pairwise Distance", "magma", 0.0, vmax),
        (diff, "Low - High (normalized)", "coolwarm", -diff_lim, diff_lim),
    ]
    for ax, (matrix, title, cmap, v0, v1) in zip(axes, panels):
        img = ax.imshow(matrix, cmap=cmap, vmin=v0, vmax=v1, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ordered samples")
        ax.set_ylabel("Ordered samples")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Within-cluster distance comparison for label {target_label} ({method_name})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_strip(
    cluster_low: np.ndarray, axis_coord: np.ndarray,
    local_scale: np.ndarray, distortion: np.ndarray,
    images: Optional[np.ndarray], target_label: int,
    strip_samples: int, method: str, output_path: Path,
) -> None:
    method_name, xlabel, ylabel = METHOD_LABELS[method]
    selected = np.linspace(
        0, len(cluster_low) - 1, num=min(strip_samples, len(cluster_low)), dtype=int,
    )

    fig = plt.figure(figsize=(18, 7.6))
    grid = GridSpec(3, max(len(selected), 1), figure=fig, height_ratios=[1.6, 1.0, 1.1])

    ax_sc = fig.add_subplot(grid[0, :])
    ax_sc.scatter(cluster_low[:, 0], cluster_low[:, 1], s=16, c="#d3d3d3",
                  alpha=0.65, linewidths=0)
    ax_sc.scatter(
        cluster_low[selected, 0], cluster_low[selected, 1],
        c=np.linspace(0, 1, len(selected)), cmap="plasma", s=70,
        edgecolors="black", linewidths=0.6,
    )
    for rank, idx in enumerate(selected):
        ax_sc.text(cluster_low[idx, 0], cluster_low[idx, 1], str(rank + 1),
                   fontsize=8, ha="center", va="center")
    ax_sc.set_title(f"Ordered traversal across {method_name} cluster {target_label}",
                    fontsize=13)
    ax_sc.set_xlabel(xlabel)
    ax_sc.set_ylabel(ylabel)
    ax_sc.grid(alpha=0.2, linewidth=0.5)

    ax_m = fig.add_subplot(grid[1, :])
    x_axis = np.arange(len(axis_coord))
    ln1 = ax_m.plot(x_axis, local_scale, color="#1f77b4", linewidth=2,
                    label="High-D local scale")
    ax_m.scatter(selected, local_scale[selected], color="#1f77b4", s=24, zorder=3)
    ax_m.set_xlabel("Ordered samples along the cluster direction")
    ax_m.set_ylabel("High-D local scale", color="#1f77b4")
    ax_m.tick_params(axis="y", labelcolor="#1f77b4")
    ax_m.grid(alpha=0.2, linewidth=0.5)

    ax_d = ax_m.twinx()
    ln2 = ax_d.plot(x_axis, distortion, color="#d62728", linewidth=2,
                    label="Distortion ratio")
    ax_d.scatter(selected, distortion[selected], color="#d62728", s=24, zorder=3)
    ax_d.set_ylabel("Distortion ratio (low/high)", color="#d62728")
    ax_d.tick_params(axis="y", labelcolor="#d62728")

    lines = ln1 + ln2
    ax_m.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=False)

    for pi, si in enumerate(selected):
        ax_img = fig.add_subplot(grid[2, pi])
        if images is not None:
            render_preview(ax_img, images[si])
        else:
            ax_img.text(0.5, 0.5, f"idx={si}", ha="center", va="center", fontsize=10)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
        ax_img.set_title(
            f"{pi+1}\nscale={local_scale[si]:.3f}\ndist={distortion[si]:.3f}",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    methods = [m.lower() for m in args.methods]
    if "difftree" in methods and args.ckpt is None:
        print("WARNING: --ckpt not provided, skipping difftree.")
        methods = [m for m in methods if m != "difftree"]

    # --- Load data (shared across all methods) ---
    label_subset = np.asarray(args.labels) if args.labels else None
    if args.input_npz:
        data, labels, images = load_npz_dataset(
            args.input_npz, args.data_key, args.labels_key,
            args.images_key, label_subset,
        )
        source_name = Path(args.input_npz).stem
    else:
        data, labels, images = load_mnist_subset(label_subset)
        source_name = "mnist"

    data, labels, images = stratified_downsample(
        data, labels, images, args.max_samples, args.seed,
    )
    if len(data) < 10:
        raise ValueError("Too few samples after filtering.")

    print(f"Data: {len(data)} samples, labels={np.unique(labels).tolist()}")

    target_label = choose_target_label(labels, args.target_label)
    cluster_mask = labels == target_label
    if int(cluster_mask.sum()) < 8:
        raise ValueError(f"Label {target_label} has too few samples for within-cluster views.")

    # --- Run embeddings ---
    embeddings: Dict[str, np.ndarray] = {}
    for method in methods:
        print(f"Computing {method} embedding ...")
        if method == "umap":
            embeddings[method] = run_umap(data, args.umap_neighbors, args.seed)
        elif method == "tsne":
            embeddings[method] = run_tsne(
                data, args.perplexity, args.tsne_lr, args.tsne_n_iter, args.seed,
            )
        elif method == "difftree":
            embeddings[method] = run_difftree(
                data, args.ckpt, args.head_index, args.batch_size, device,
            )

    # --- Compute kNN metrics (all use original data as high-dim) ---
    metrics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        print(f"Computing kNN metrics for {method} ...")
        scale, dist = compute_knn_metrics(data, embeddings[method], args.knn_k)
        metrics[method] = (scale, dist)

    # --- Compute GLOBAL color limits across all methods ---
    all_scales = np.concatenate([metrics[m][0] for m in methods])
    all_distortions = np.concatenate([metrics[m][1] for m in methods])

    scale_vmin, scale_vmax = percentile_limits(all_scales)
    dist_vmin, dist_vmax = percentile_limits(all_distortions)

    print(f"Global color limits:")
    print(f"  local_scale:  [{scale_vmin:.4f}, {scale_vmax:.4f}]")
    print(f"  distortion:   [{dist_vmin:.4f}, {dist_vmax:.4f}]")

    # --- Generate figures per method ---
    all_outputs = {}
    for method in methods:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        emb = embeddings[method]
        local_scale, distortion = metrics[method]
        method_name = METHOD_LABELS[method][0]

        # Cluster subset
        cluster_high, cluster_low, c_scale, c_dist, c_images, axis_coord = \
            ordered_cluster_subset(
                data[cluster_mask], emb[cluster_mask],
                local_scale[cluster_mask], distortion[cluster_mask],
                images[cluster_mask] if images is not None else None,
                args.cluster_max_samples,
            )

        # Figure paths
        f1 = method_dir / "01_local_scale.png"
        f2 = method_dir / "02_local_distortion.png"
        f3 = method_dir / f"03_label_{target_label}_distance_matrix.png"
        f4 = method_dir / f"04_label_{target_label}_sample_strip.png"

        # Plot with GLOBAL color limits
        plot_metric_scatter(
            emb, labels, local_scale,
            f"{method_name} colored by high-dimensional local scale",
            "mean high-d distance to kNN", f1,
            method, vmin=scale_vmin, vmax=scale_vmax,
        )
        plot_metric_scatter(
            emb, labels, distortion,
            f"{method_name} colored by local distortion ratio",
            "mean low/high distance ratio over kNN", f2,
            method, vmin=dist_vmin, vmax=dist_vmax,
        )
        plot_distance_matrix(cluster_high, cluster_low, target_label, method, f3)
        plot_sample_strip(
            cluster_low, axis_coord, c_scale, c_dist, c_images,
            target_label, args.strip_samples, method, f4,
        )

        all_outputs[method] = [
            str(f.relative_to(PROJECT_ROOT)) for f in [f1, f2, f3, f4]
        ]

        print(f"  {method_name} figures saved to {method_dir.relative_to(PROJECT_ROOT)}/")

    # --- Summary ---
    summary = {
        "source": source_name,
        "methods": methods,
        "n_samples": int(len(data)),
        "high_dim_for_knn": int(data.shape[1]),
        "label_subset": label_subset.tolist() if label_subset is not None else None,
        "target_label": int(target_label),
        "knn_k": int(min(args.knn_k, len(data) - 1)),
        "global_scale_range": [scale_vmin, scale_vmax],
        "global_distortion_range": [dist_vmin, dist_vmax],
        "cluster_samples_for_matrix": int(len(
            list(ordered_cluster_subset(
                data[cluster_mask], embeddings[methods[0]][cluster_mask],
                metrics[methods[0]][0][cluster_mask],
                metrics[methods[0]][1][cluster_mask],
                None, args.cluster_max_samples,
            ))[0]
        )),
        "per_method_stats": {},
        "outputs": all_outputs,
    }

    for method in methods:
        scale, dist = metrics[method]
        summary["per_method_stats"][method] = {
            "scale_mean": float(scale.mean()),
            "scale_std": float(scale.std()),
            "distortion_mean": float(dist.mean()),
            "distortion_std": float(dist.std()),
            "distortion_cv": float(dist.std() / max(dist.mean(), 1e-12)),
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("All figures saved:")
    for method, paths in all_outputs.items():
        for p in paths:
            print(f"  [{method}] {p}")
    print(f"  summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
