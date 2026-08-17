"""
Generate four MNIST visualizations that expose hidden intra-cluster heterogeneity in UMAP.

The four outputs are:
  1. UMAP + local heterogeneity coloring
  2. Single-class pairwise distance heatmap
  3. Shepard-like plot between high-dimensional and UMAP distances
  4. Prototype-to-boundary sample gallery
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.datasets import MNIST


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four MNIST UMAP cluster heterogeneity visualizations."
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional label subset. Example: --labels 4 5 9",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=None,
        help="Label used for figures 2-4. Defaults to the largest class in the subset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Maximum number of MNIST samples used overall.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of same-class neighbors used in the local heterogeneity score.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.05,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="If positive and smaller than 784, fit UMAP and distance metrics in PCA space.",
    )
    parser.add_argument(
        "--heatmap-max-samples",
        type=int,
        default=120,
        help="Maximum samples shown in the pairwise heatmap.",
    )
    parser.add_argument(
        "--pair-samples",
        type=int,
        default=12000,
        help="Number of random within-class pairs for the Shepard-like plot.",
    )
    parser.add_argument(
        "--gallery-samples",
        type=int,
        default=12,
        help="Number of images shown from prototype to boundary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MNIST if it is missing under ./data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis/outputs/mnist_umap_cluster_heterogeneity",
        help="Directory for exported figures.",
    )
    return parser.parse_args()


def load_mnist_subset(
    label_subset: Optional[np.ndarray],
    download: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root=str(PROJECT_ROOT / "data"), train=True, download=download, transform=transform)

    data_list = []
    image_list = []
    label_list = []
    for image, label in dataset:
        arr = image.numpy().squeeze().astype(np.float32)
        image_list.append(arr)
        data_list.append(arr.reshape(-1))
        label_list.append(label)

    data = np.stack(data_list)
    images = np.stack(image_list)
    labels = np.asarray(label_list, dtype=np.int64)

    if label_subset is not None:
        mask = np.isin(labels, label_subset)
        data = data[mask]
        images = images[mask]
        labels = labels[mask]

    return data, labels, images


def stratified_downsample(
    data: np.ndarray,
    labels: np.ndarray,
    images: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples is None or len(data) <= max_samples:
        return data, labels, images

    rng = np.random.RandomState(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    target_per_label = {}
    remaining = max_samples

    for label, count in zip(unique_labels, counts):
        target = max(1, int(round(max_samples * (count / len(labels)))))
        target_per_label[label] = min(target, count)
        remaining -= target_per_label[label]

    if remaining > 0:
        for label, count in sorted(zip(unique_labels, counts), key=lambda item: item[1], reverse=True):
            extra = min(remaining, count - target_per_label[label])
            if extra > 0:
                target_per_label[label] += extra
                remaining -= extra
            if remaining == 0:
                break

    keep_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        take = target_per_label[label]
        if take >= len(label_indices):
            keep_indices.append(label_indices)
        else:
            sampled = np.sort(rng.choice(label_indices, size=take, replace=False))
            keep_indices.append(sampled)

    keep = np.concatenate(keep_indices)
    keep.sort()
    return data[keep], labels[keep], images[keep]


def prepare_feature_space(
    data: np.ndarray,
    pca_components: int,
    seed: int,
) -> Tuple[np.ndarray, int, bool]:
    if pca_components <= 0 or pca_components >= data.shape[1]:
        return data.copy(), data.shape[1], False

    n_components = min(pca_components, data.shape[0], data.shape[1])
    feature_space = PCA(n_components=n_components, random_state=seed).fit_transform(data)
    return feature_space.astype(np.float32), n_components, True


def fit_umap_embedding(
    feature_space: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(feature_space) - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(feature_space)


def compute_same_class_local_scale(
    feature_space: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    local_scale = np.zeros(len(feature_space), dtype=np.float32)

    for label in np.unique(labels):
        mask = labels == label
        class_features = feature_space[mask]
        if len(class_features) <= 1:
            continue

        k_effective = min(k, len(class_features) - 1)
        neighbors = NearestNeighbors(n_neighbors=k_effective + 1, metric="euclidean")
        neighbors.fit(class_features)
        distances, _ = neighbors.kneighbors(class_features, return_distance=True)
        local_scale[mask] = distances[:, 1:].mean(axis=1)

    return local_scale


def percentile_limits(values: np.ndarray, low_q: float = 0.02, high_q: float = 0.98) -> Tuple[float, float]:
    lower = float(np.quantile(values, low_q))
    upper = float(np.quantile(values, high_q))
    if math.isclose(lower, upper):
        upper = lower + 1e-6
    return lower, upper


def choose_target_label(labels: np.ndarray, requested: Optional[int]) -> int:
    if requested is not None:
        if requested not in set(np.unique(labels).tolist()):
            raise ValueError(f"target label {requested} is not present in the current subset.")
        return requested

    unique_labels, counts = np.unique(labels, return_counts=True)
    return int(unique_labels[np.argmax(counts)])


def compute_medoid_distances(class_features: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    pairwise = pairwise_distances(class_features, metric="euclidean")
    medoid_idx = int(np.argmin(pairwise.mean(axis=1)))
    distances_to_medoid = pairwise[medoid_idx]
    return pairwise, medoid_idx, distances_to_medoid


def order_target_class(distances_to_medoid: np.ndarray, class_embedding: np.ndarray) -> np.ndarray:
    axis_coord = PCA(n_components=1).fit_transform(class_embedding).reshape(-1)
    order = np.lexsort((axis_coord, distances_to_medoid))
    return order


def select_even_positions(length: int, count: int) -> np.ndarray:
    return np.linspace(0, length - 1, num=min(length, count), dtype=int)


def render_digit(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_umap_local_heterogeneity(
    embedding: np.ndarray,
    labels: np.ndarray,
    local_scale: np.ndarray,
    target_label: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    vmin, vmax = percentile_limits(local_scale)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=local_scale,
        s=7,
        cmap="viridis",
        alpha=0.82,
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
    )

    for label in np.unique(labels):
        mask = labels == label
        center = embedding[mask].mean(axis=0)
        ax.text(center[0], center[1], str(label), fontsize=10, weight="bold", ha="center", va="center")

    target_mask = labels == target_label
    ax.scatter(
        embedding[target_mask, 0],
        embedding[target_mask, 1],
        s=10,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.2,
        alpha=0.35,
    )

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("mean same-class kNN distance in high-dimensional space")
    ax.set_title("UMAP + local heterogeneity coloring")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_heatmap(
    class_pairwise: np.ndarray,
    distances_to_medoid: np.ndarray,
    class_embedding: np.ndarray,
    target_label: int,
    max_samples: int,
    output_path: Path,
) -> int:
    order = order_target_class(distances_to_medoid, class_embedding)
    if len(order) > max_samples:
        keep = select_even_positions(len(order), max_samples)
        order = order[keep]

    ordered_matrix = class_pairwise[np.ix_(order, order)]
    vmax = float(np.quantile(ordered_matrix, 0.98))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    image = ax.imshow(ordered_matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="pairwise distance")
    ax.set_title(f"Digit {target_label}: pairwise distance heatmap")
    ax.set_xlabel("Samples ordered by distance to medoid")
    ax.set_ylabel("Samples ordered by distance to medoid")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return len(order)


def plot_shepard_like(
    class_features: np.ndarray,
    class_embedding: np.ndarray,
    target_label: int,
    pair_samples: int,
    seed: int,
    output_path: Path,
) -> int:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(class_features), size=pair_samples)
    right = rng.integers(0, len(class_features), size=pair_samples)
    valid = left != right
    left = left[valid]
    right = right[valid]

    if len(left) == 0:
        raise ValueError("Not enough samples to draw within-class pairs.")

    high_dist = np.linalg.norm(class_features[left] - class_features[right], axis=1)
    low_dist = np.linalg.norm(class_embedding[left] - class_embedding[right], axis=1)
    corr = float(np.corrcoef(high_dist, low_dist)[0, 1])

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    hexbin = ax.hexbin(high_dist, low_dist, gridsize=48, cmap="Blues", mincnt=1)
    fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04, label="pair count")
    ax.set_title(f"Digit {target_label}: Shepard-like plot")
    ax.set_xlabel("distance in original / PCA feature space")
    ax.set_ylabel("distance in UMAP space")
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return len(left)


def plot_prototype_gallery(
    class_embedding: np.ndarray,
    class_images: np.ndarray,
    medoid_idx: int,
    distances_to_medoid: np.ndarray,
    target_label: int,
    gallery_samples: int,
    output_path: Path,
) -> int:
    order = np.argsort(distances_to_medoid)
    chosen_positions = select_even_positions(len(order), gallery_samples)
    chosen = order[chosen_positions]

    fig = plt.figure(figsize=(2.2 * len(chosen), 7.2))
    grid = fig.add_gridspec(2, len(chosen), height_ratios=[1.2, 1.0])

    ax_scatter = fig.add_subplot(grid[0, :])
    scatter = ax_scatter.scatter(
        class_embedding[:, 0],
        class_embedding[:, 1],
        c=distances_to_medoid,
        s=12,
        cmap="magma",
        alpha=0.8,
        linewidths=0,
    )
    ax_scatter.scatter(
        class_embedding[medoid_idx, 0],
        class_embedding[medoid_idx, 1],
        c="#00d5ff",
        s=90,
        marker="X",
        edgecolors="black",
        linewidths=0.8,
        label="medoid",
    )
    ax_scatter.scatter(
        class_embedding[chosen, 0],
        class_embedding[chosen, 1],
        facecolors="none",
        edgecolors="white",
        s=90,
        linewidths=1.0,
    )
    for rank, idx in enumerate(chosen):
        ax_scatter.text(
            class_embedding[idx, 0],
            class_embedding[idx, 1],
            str(rank + 1),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
        )

    fig.colorbar(scatter, ax=ax_scatter, fraction=0.03, pad=0.02, label="distance to medoid")
    ax_scatter.set_title(f"Digit {target_label}: prototype-to-boundary gallery")
    ax_scatter.set_xlabel("UMAP 1")
    ax_scatter.set_ylabel("UMAP 2")
    ax_scatter.legend(loc="best", frameon=False)
    ax_scatter.grid(alpha=0.18, linewidth=0.5)

    for column, idx in enumerate(chosen):
        ax_img = fig.add_subplot(grid[1, column])
        render_digit(ax_img, class_images[idx])
        ax_img.set_title(
            f"#{column + 1}\nd={distances_to_medoid[idx]:.3f}",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return len(chosen)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    label_subset = np.asarray(args.labels, dtype=np.int64) if args.labels else None
    data, labels, images = load_mnist_subset(label_subset=label_subset, download=args.download)
    data, labels, images = stratified_downsample(
        data=data,
        labels=labels,
        images=images,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if len(data) < 10:
        raise ValueError("Too few samples remain after filtering; adjust --labels or --max-samples.")

    feature_space, feature_dim, used_pca = prepare_feature_space(
        data=data,
        pca_components=args.pca_components,
        seed=args.seed,
    )
    embedding = fit_umap_embedding(
        feature_space=feature_space,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )
    local_scale = compute_same_class_local_scale(
        feature_space=feature_space,
        labels=labels,
        k=args.knn_k,
    )

    target_label = choose_target_label(labels, args.target_label)
    class_mask = labels == target_label
    class_features = feature_space[class_mask]
    class_embedding = embedding[class_mask]
    class_images = images[class_mask]

    if len(class_features) < 8:
        raise ValueError(f"Label {target_label} has too few samples for within-class views.")

    class_pairwise, medoid_idx, distances_to_medoid = compute_medoid_distances(class_features)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_1 = output_dir / "01_umap_local_heterogeneity.png"
    figure_2 = output_dir / f"02_label_{target_label}_pairwise_heatmap.png"
    figure_3 = output_dir / f"03_label_{target_label}_shepard_plot.png"
    figure_4 = output_dir / f"04_label_{target_label}_prototype_gallery.png"

    plot_umap_local_heterogeneity(
        embedding=embedding,
        labels=labels,
        local_scale=local_scale,
        target_label=target_label,
        output_path=figure_1,
    )
    heatmap_samples = plot_pairwise_heatmap(
        class_pairwise=class_pairwise,
        distances_to_medoid=distances_to_medoid,
        class_embedding=class_embedding,
        target_label=target_label,
        max_samples=args.heatmap_max_samples,
        output_path=figure_2,
    )
    pair_count = plot_shepard_like(
        class_features=class_features,
        class_embedding=class_embedding,
        target_label=target_label,
        pair_samples=args.pair_samples,
        seed=args.seed,
        output_path=figure_3,
    )
    gallery_count = plot_prototype_gallery(
        class_embedding=class_embedding,
        class_images=class_images,
        medoid_idx=medoid_idx,
        distances_to_medoid=distances_to_medoid,
        target_label=target_label,
        gallery_samples=args.gallery_samples,
        output_path=figure_4,
    )

    summary = {
        "dataset": "MNIST train",
        "n_samples": int(len(data)),
        "label_subset": None if label_subset is None else label_subset.tolist(),
        "target_label": int(target_label),
        "target_class_size": int(len(class_features)),
        "feature_dim": int(feature_dim),
        "used_pca": bool(used_pca),
        "knn_k": int(min(args.knn_k, max(len(class_features) - 1, 1))),
        "umap_neighbors": int(min(args.umap_neighbors, len(data) - 1)),
        "umap_min_dist": float(args.umap_min_dist),
        "heatmap_samples": int(heatmap_samples),
        "shepard_pairs": int(pair_count),
        "gallery_samples": int(gallery_count),
        "outputs": [
            str(figure_1.relative_to(PROJECT_ROOT)),
            str(figure_2.relative_to(PROJECT_ROOT)),
            str(figure_3.relative_to(PROJECT_ROOT)),
            str(figure_4.relative_to(PROJECT_ROOT)),
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Saved MNIST UMAP cluster heterogeneity figures:")
    for rel_path in summary["outputs"]:
        print(f"  - {rel_path}")
    print(f"  - {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
