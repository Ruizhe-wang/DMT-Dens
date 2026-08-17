"""
Generate four figures that expose hidden heterogeneity inside visually smooth DiffTree v3 clusters.

Instead of UMAP, uses a trained DiffTree v3 model to produce the 2D embedding, then runs
the same heterogeneity analysis:
  1. Embedding colored by high-dimensional local scale
  2. Embedding colored by local distortion ratio (low-dim / high-dim)
  3. Within-cluster pairwise distance matrix comparison
  4. Ordered sample strip along the cluster direction

Usage:
  # Train first, then analyze
  python main.py fit -c configs/label_scaling_v3/diff_s2_1_7.yaml
  python scripts/plot_difftree_cluster_heterogeneity.py \
      --ckpt path/to/checkpoint.ckpt \
      --labels 1 7 --target-label 1

  # Or specify a config to train from scratch, then analyze
  python scripts/plot_difftree_cluster_heterogeneity.py \
      --config configs/label_scaling_v3/diff_s2_1_7.yaml \
      --labels 1 7 --target-label 1
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
        description="Visualize hidden heterogeneity inside DiffTree v3 clusters."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a trained DiffTree v3 checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config for training DiffTree v3 from scratch (ignored if --ckpt is given).",
    )
    parser.add_argument(
        "--head-index",
        type=int,
        default=-1,
        help="Which projection head to use for the 2D embedding. -1 = best or last.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional label subset. Defaults to all labels.",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=None,
        help="Label used for within-cluster views (figures 3 and 4). Defaults to the largest class.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Maximum samples used for analysis.",
    )
    parser.add_argument(
        "--cluster-max-samples",
        type=int,
        default=120,
        help="Maximum ordered samples in the distance-matrix comparison.",
    )
    parser.add_argument(
        "--strip-samples",
        type=int,
        default=12,
        help="Number of evenly spaced samples in the ordered sample strip.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of neighbors for local scale and distortion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/difftree_cluster_heterogeneity",
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for model inference.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist_subset(
    label_subset: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST train set, flatten images, optionally filter by labels."""
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


def stratified_downsample(
    data: np.ndarray,
    labels: np.ndarray,
    images: Optional[np.ndarray],
    max_samples: Optional[int],
    seed: int,
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
            picked = np.sort(rng.choice(label_idx, size=take, replace=False))
            indices.append(picked)

    keep = np.concatenate(indices)
    keep.sort()
    return data[keep], labels[keep], images[keep] if images is not None else None


# ---------------------------------------------------------------------------
# DiffTree v3 model loading & inference
# ---------------------------------------------------------------------------

def load_difftree_model(ckpt_path: str, device: torch.device):
    """Load a trained DiffTree v3 model from checkpoint."""
    from model.DiffTreeVQ_v3 import DMTEVT_model

    model = DMTEVT_model.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model,
    data: np.ndarray,
    head_index: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run data through the DiffTree v3 model and extract:
      - high-dim encoder output  (n, t_output_dim)
      - 2D embedding from the selected head  (n, 2)
    """
    all_high_dim = []
    all_emb_2d = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            x = torch.from_numpy(data[start:end]).float().to(device)

            _, lat_high_dim, lat_vis_best, lat_vis_list = model(x, tau=1.0)

            # Choose head
            if 0 <= head_index < len(lat_vis_list):
                emb_2d = lat_vis_list[head_index]
            else:
                # Use model's best_head_index, or last head
                best_idx = (
                    model.best_head_index
                    if 0 <= model.best_head_index < len(lat_vis_list)
                    else -1
                )
                emb_2d = lat_vis_list[best_idx]

            all_high_dim.append(lat_high_dim.cpu().numpy())
            all_emb_2d.append(emb_2d.cpu().numpy())

    high_dim = np.concatenate(all_high_dim, axis=0)
    embedding = np.concatenate(all_emb_2d, axis=0)
    return high_dim, embedding


# ---------------------------------------------------------------------------
# Metrics (same as UMAP version)
# ---------------------------------------------------------------------------

def compute_knn_metrics(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_effective = min(k, len(high_dim) - 1)
    if k_effective < 1:
        raise ValueError("Need at least 2 samples to compute local neighborhood metrics.")

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
    return local_scale, distortion, hd_indices


def percentile_limits(
    values: np.ndarray, low_q: float = 0.02, high_q: float = 0.98
) -> Tuple[float, float]:
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if math.isclose(low, high):
        high = low + 1e-6
    return low, high


def choose_target_label(labels: np.ndarray, requested: Optional[int]) -> int:
    if requested is not None:
        if requested not in set(np.unique(labels).tolist()):
            raise ValueError(f"target-label {requested} is not present in the selected subset.")
        return requested
    unique_labels, counts = np.unique(labels, return_counts=True)
    return int(unique_labels[np.argmax(counts)])


def ordered_cluster_subset(
    cluster_high: np.ndarray,
    cluster_low: np.ndarray,
    cluster_scores_1: np.ndarray,
    cluster_scores_2: np.ndarray,
    cluster_images: Optional[np.ndarray],
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    axis_coord = PCA(n_components=1).fit_transform(cluster_low).reshape(-1)
    order = np.argsort(axis_coord)

    if len(order) > max_samples:
        positions = np.linspace(0, len(order) - 1, num=max_samples, dtype=int)
        order = order[positions]

    return (
        cluster_high[order],
        cluster_low[order],
        cluster_scores_1[order],
        cluster_scores_2[order],
        cluster_images[order] if cluster_images is not None else None,
        axis_coord[order],
    )


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    scale = np.quantile(upper, 0.95) if len(upper) else 1.0
    scale = max(float(scale), 1e-12)
    return matrix / scale


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def plot_embedding_metric(
    embedding: np.ndarray,
    labels: np.ndarray,
    metric: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    vmin, vmax = percentile_limits(metric)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=metric,
        s=4,
        cmap="viridis_r",
        alpha=0.86,
        linewidths=0,
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("DiffTree Dim 1")
    ax.set_ylabel("DiffTree Dim 2")
    ax.grid(alpha=0.2, linewidth=0.5)

    for lbl in np.unique(labels):
        mask = labels == lbl
        center = embedding[mask].mean(axis=0)
        ax.text(center[0], center[1], str(lbl), fontsize=10, weight="bold", ha="center", va="center")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_distance_matrix_comparison(
    cluster_high: np.ndarray,
    cluster_low: np.ndarray,
    target_label: int,
    output_path: Path,
) -> None:
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
        (ld_norm, "DiffTree Pairwise Distance", "magma", 0.0, vmax),
        (diff, "Low - High (normalized)", "coolwarm", -diff_lim, diff_lim),
    ]

    for ax, (matrix, title, cmap, vmin, vmax_panel) in zip(axes, panels):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax_panel, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ordered samples")
        ax.set_ylabel("Ordered samples")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Within-cluster distance comparison for label {target_label} (DiffTree v3)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_strip(
    cluster_low: np.ndarray,
    axis_coord: np.ndarray,
    local_scale: np.ndarray,
    distortion: np.ndarray,
    images: Optional[np.ndarray],
    target_label: int,
    strip_samples: int,
    output_path: Path,
) -> None:
    selected_positions = np.linspace(
        0, len(cluster_low) - 1, num=min(strip_samples, len(cluster_low)), dtype=int
    )

    fig = plt.figure(figsize=(18, 7.6))
    grid = GridSpec(3, max(len(selected_positions), 1), figure=fig, height_ratios=[1.6, 1.0, 1.1])

    # --- Top: scatter with numbered traversal ---
    ax_scatter = fig.add_subplot(grid[0, :])
    ax_scatter.scatter(
        cluster_low[:, 0], cluster_low[:, 1], s=16, c="#d3d3d3", alpha=0.65, linewidths=0
    )
    ax_scatter.scatter(
        cluster_low[selected_positions, 0],
        cluster_low[selected_positions, 1],
        c=np.linspace(0, 1, len(selected_positions)),
        cmap="plasma",
        s=70,
        edgecolors="black",
        linewidths=0.6,
    )
    for rank, idx in enumerate(selected_positions):
        ax_scatter.text(
            cluster_low[idx, 0], cluster_low[idx, 1], str(rank + 1),
            fontsize=8, ha="center", va="center",
        )
    ax_scatter.set_title(f"Ordered traversal across DiffTree cluster {target_label}", fontsize=13)
    ax_scatter.set_xlabel("DiffTree Dim 1")
    ax_scatter.set_ylabel("DiffTree Dim 2")
    ax_scatter.grid(alpha=0.2, linewidth=0.5)

    # --- Middle: dual-axis metric curves ---
    ax_metric = fig.add_subplot(grid[1, :])
    x_axis = np.arange(len(axis_coord))
    ln1 = ax_metric.plot(x_axis, local_scale, color="#1f77b4", linewidth=2, label="High-D local scale")
    ax_metric.scatter(selected_positions, local_scale[selected_positions], color="#1f77b4", s=24, zorder=3)
    ax_metric.set_xlabel("Ordered samples along the cluster direction")
    ax_metric.set_ylabel("High-D local scale", color="#1f77b4")
    ax_metric.tick_params(axis="y", labelcolor="#1f77b4")
    ax_metric.grid(alpha=0.2, linewidth=0.5)

    ax_dist = ax_metric.twinx()
    ln2 = ax_dist.plot(x_axis, distortion, color="#d62728", linewidth=2, label="Distortion ratio")
    ax_dist.scatter(selected_positions, distortion[selected_positions], color="#d62728", s=24, zorder=3)
    ax_dist.set_ylabel("Distortion ratio (low/high)", color="#d62728")
    ax_dist.tick_params(axis="y", labelcolor="#d62728")

    lines = ln1 + ln2
    ax_metric.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=False)

    # --- Bottom: image previews ---
    for panel_idx, sample_idx in enumerate(selected_positions):
        ax_img = fig.add_subplot(grid[2, panel_idx])
        if images is not None:
            render_preview(ax_img, images[sample_idx])
        else:
            ax_img.text(0.5, 0.55, f"idx={sample_idx}", ha="center", va="center", fontsize=10)
            ax_img.text(0.5, 0.35, "no preview", ha="center", va="center", fontsize=9, color="#666666")
            ax_img.set_xticks([])
            ax_img.set_yticks([])

        ax_img.set_title(
            f"{panel_idx + 1}\nscale={local_scale[sample_idx]:.3f}\ndist={distortion[sample_idx]:.3f}",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_checkpoint(ckpt_path: Optional[str], config_path: Optional[str]) -> str:
    """Resolve checkpoint path, or train from scratch if only config is given."""
    if ckpt_path is not None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if config_path is not None:
        print(f"No checkpoint provided. Training from scratch using: {config_path}")
        import subprocess

        cmd = [sys.executable, "main.py", "fit", "-c", config_path]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

        # Find the latest checkpoint in lightning_logs
        ckpt_dir = PROJECT_ROOT / "lightning_logs"
        if not ckpt_dir.exists():
            raise FileNotFoundError("Training finished but no lightning_logs directory found.")

        ckpt_files = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpt_files:
            raise FileNotFoundError("Training finished but no .ckpt file found in lightning_logs/.")

        print(f"Using checkpoint: {ckpt_files[0]}")
        return str(ckpt_files[0])

    raise ValueError("Must provide either --ckpt or --config.")


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Resolve checkpoint and load model ---
    ckpt_path = find_checkpoint(args.ckpt, args.config)
    model = load_difftree_model(ckpt_path, device)
    print(f"Model loaded from {ckpt_path}")
    print(f"  Encoder output dim: {model.hparams.t_output_dim}")
    print(f"  Projection heads: {len(model.hparams.num_use_mlevel_list)} "
          f"(scales={model.hparams.num_use_mlevel_list})")
    print(f"  best_head_index: {model.best_head_index}")

    # --- Load data ---
    label_subset = np.asarray(args.labels) if args.labels else None
    data, labels, images = load_mnist_subset(label_subset=label_subset)
    data, labels, images = stratified_downsample(
        data=data,
        labels=labels,
        images=images,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if len(data) < 10:
        raise ValueError("Too few samples after filtering.")

    print(f"Data: {len(data)} samples, labels={np.unique(labels).tolist()}")

    # --- Extract embeddings ---
    high_dim, embedding = extract_embeddings(
        model=model,
        data=data,
        head_index=args.head_index,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Embeddings: high_dim={high_dim.shape}, 2d={embedding.shape}")

    # --- Compute kNN metrics ---
    # Use original input data (e.g. 784-dim pixels) as the high-dim reference
    # so that metrics are comparable across UMAP / t-SNE / DiffTree.
    local_scale, distortion, _ = compute_knn_metrics(
        high_dim=data,
        low_dim=embedding,
        k=args.knn_k,
    )

    # --- Within-cluster analysis ---
    target_label = choose_target_label(labels, args.target_label)
    cluster_mask = labels == target_label
    cluster_count = int(cluster_mask.sum())
    if cluster_count < 8:
        raise ValueError(
            f"Label {target_label} only has {cluster_count} samples; not enough for within-cluster views."
        )

    (
        cluster_high, cluster_low, cluster_scale, cluster_distortion,
        cluster_images, axis_coord,
    ) = ordered_cluster_subset(
        cluster_high=data[cluster_mask],
        cluster_low=embedding[cluster_mask],
        cluster_scores_1=local_scale[cluster_mask],
        cluster_scores_2=distortion[cluster_mask],
        cluster_images=images[cluster_mask] if images is not None else None,
        max_samples=args.cluster_max_samples,
    )

    # --- Generate figures ---
    fig1_path = output_dir / "01_difftree_local_scale.png"
    fig2_path = output_dir / "02_difftree_local_distortion.png"
    fig3_path = output_dir / f"03_label_{target_label}_distance_matrix.png"
    fig4_path = output_dir / f"04_label_{target_label}_sample_strip.png"

    plot_embedding_metric(
        embedding=embedding,
        labels=labels,
        metric=local_scale,
        title="DiffTree v3 colored by high-dimensional local scale",
        colorbar_label="mean high-d distance to kNN",
        output_path=fig1_path,
    )
    plot_embedding_metric(
        embedding=embedding,
        labels=labels,
        metric=distortion,
        title="DiffTree v3 colored by local distortion ratio",
        colorbar_label="mean low/high distance ratio over kNN",
        output_path=fig2_path,
    )
    plot_distance_matrix_comparison(
        cluster_high=cluster_high,
        cluster_low=cluster_low,
        target_label=target_label,
        output_path=fig3_path,
    )
    plot_sample_strip(
        cluster_low=cluster_low,
        axis_coord=axis_coord,
        local_scale=cluster_scale,
        distortion=cluster_distortion,
        images=cluster_images,
        target_label=target_label,
        strip_samples=args.strip_samples,
        output_path=fig4_path,
    )

    # --- Summary ---
    summary = {
        "source": "difftree_v3",
        "checkpoint": ckpt_path,
        "head_index": args.head_index,
        "best_head_index": int(model.best_head_index),
        "n_samples": int(len(data)),
        "label_subset": None if label_subset is None else label_subset.tolist(),
        "target_label": int(target_label),
        "knn_k": int(min(args.knn_k, len(data) - 1)),
        "high_dim_for_knn": int(data.shape[1]),
        "encoder_dim": int(high_dim.shape[1]),
        "num_heads": len(model.hparams.num_use_mlevel_list),
        "scales": model.hparams.num_use_mlevel_list,
        "cluster_samples_for_matrix": int(len(cluster_high)),
        "outputs": [
            str(fig1_path.relative_to(PROJECT_ROOT)),
            str(fig2_path.relative_to(PROJECT_ROOT)),
            str(fig3_path.relative_to(PROJECT_ROOT)),
            str(fig4_path.relative_to(PROJECT_ROOT)),
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DiffTree v3 cluster heterogeneity figures saved:")
    for rel_path in summary["outputs"]:
        print(f"  - {rel_path}")
    print(f"  - {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
