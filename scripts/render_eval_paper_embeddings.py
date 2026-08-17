"""Render paper-style PNGs from saved eval-mode epoch-1000 embeddings.

Coordinates come from VisualizationCallback's final validation CSV, avoiding
the BatchNorm train-mode bug in the old on_train_end render.  The visual style
is unchanged from PaperEmbeddingCallback: transparent background, no axes,
equal aspect, tab20/gist_ncar, s=1.5, alpha=0.5, 4 inches, 300 DPI, all points.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = (
    "act",
    "emnist",
    "epi",
    "gast10k",
    "hcl",
    "mca",
    "mnist",
    "ng20",
    "tree",
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def coordinate_path(repo: Path, dataset: str) -> Path:
    current = repo / (
        "outputs/paper_general_exact_rerun/latent_bn/"
        f"{dataset}/seed42/diagnostics/latent-bn_epoch1000_layer0.csv"
    )
    if current.exists():
        return current
    historical = repo / (
        f"outputs/encoder_tuning/E18/{dataset}_latent_bn_seed42/"
        "plots/latent-bn_epoch1000_layer0.csv"
    )
    if historical.exists():
        return historical
    raise FileNotFoundError(f"No eval-mode epoch-1000 CSV for {dataset}")


def labels_from_npz(repo: Path, dataset: str):
    root = repo / (
        "outputs/paper_general_exact_rerun/latent_bn/"
        f"{dataset}/seed42/final_embedding"
    )
    matches = sorted(root.glob("*_embeddings.npz"))
    if not matches:
        return None
    archive = np.load(matches[0], allow_pickle=True)
    return (
        archive["cell_ids"].astype(str),
        archive["labels"].astype(str),
        matches[0],
    )


def labels_from_datamodule(repo: Path, dataset: str):
    config_path = repo / (
        "configs/encoder_bench/paper_general_seed42/runs/"
        f"paper_exact_latent-bn_{dataset}_seed42.yaml"
    )
    config = load_yaml(config_path)
    data_config = config["data"]
    datamodule = import_class(data_config["class_path"])(**data_config["init_args"])
    datamodule.setup("fit")
    adata = datamodule.adata
    if "final_annotation" not in adata.obs:
        raise KeyError(f"{dataset} has no final_annotation in adata.obs")
    return (
        adata.obs_names.astype(str).to_numpy(),
        adata.obs["final_annotation"].astype(str).to_numpy(),
        config_path,
    )


def aligned_labels(repo: Path, dataset: str, frame: pd.DataFrame):
    label_data = labels_from_npz(repo, dataset)
    if label_data is None:
        label_data = labels_from_datamodule(repo, dataset)
    source_ids, source_labels, source_path = label_data
    target_ids = frame["cell_id"].astype(str).to_numpy()
    if np.array_equal(source_ids, target_ids):
        return source_labels, source_path
    lookup = dict(zip(source_ids, source_labels))
    missing = [cell_id for cell_id in target_ids if cell_id not in lookup]
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} embedding cell IDs have no label; "
            f"first={missing[0]!r}"
        )
    return np.asarray([lookup[cell_id] for cell_id in target_ids]), source_path


def categorical_palette(n_classes: int):
    """Match PaperEmbeddingCallback's tab20/gist_ncar palette exactly."""
    cmap = plt.get_cmap("tab20")
    capacity = getattr(cmap, "N", 0)
    if n_classes <= capacity:
        return [cmap(i) for i in range(n_classes)]
    overflow = plt.get_cmap("gist_ncar")
    return [overflow(i / max(n_classes - 1, 1)) for i in range(n_classes)]


def save_paper_embedding(xy: np.ndarray, labels: np.ndarray, path: Path):
    """Pure-Matplotlib copy of PaperEmbeddingCallback._save_clean."""
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    classes = np.unique(labels)
    palette = categorical_palette(len(classes))
    for index, category in enumerate(classes):
        mask = labels == category
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=1.5,
            color=palette[index],
            alpha=0.5,
            linewidths=0,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.margins(0.02)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(fig)


def render_one(repo: Path, output_root: Path, dataset: str) -> dict:
    coords_path = coordinate_path(repo, dataset)
    frame = pd.read_csv(coords_path, usecols=["cell_id", "x", "y"])
    xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError(f"{dataset}: non-finite eval coordinates")
    labels, label_source = aligned_labels(repo, dataset, frame)
    if labels.shape[0] != xy.shape[0]:
        raise ValueError(f"{dataset}: coordinate/label row mismatch")

    output_dir = output_root / dataset / "seed42" / "paper"
    image_path = (
        output_dir / "latent-bn_layer0_final_annotation_eval_epoch1000.png"
    )
    save_paper_embedding(xy, labels, image_path)
    image_path = image_path.resolve()
    metadata = {
        "dataset": dataset,
        "seed": 42,
        "n_samples": int(xy.shape[0]),
        "n_labels": int(np.unique(labels).size),
        "coordinate_source": str(coords_path.resolve()),
        "label_source": str(Path(label_source).resolve()),
        "image_path": str(image_path),
        "style": {
            "point_size": 1.5,
            "alpha": 0.5,
            "cmap": "tab20",
            "overflow_cmap": "gist_ncar",
            "figsize": 4.0,
            "dpi": 300,
            "transparent": True,
            "axes": False,
            "aspect": "equal",
            "sampling": "all",
        },
    }
    metadata_path = output_dir / "render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[paper-eval] {dataset}: n={xy.shape[0]} -> {image_path}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paper_general_eval_fixed/latent_bn"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = [render_one(repo, output_root, dataset) for dataset in args.datasets]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "dataset",
            "seed",
            "n_samples",
            "n_labels",
            "coordinate_source",
            "label_source",
            "image_path",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})
    print(f"[paper-eval] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
