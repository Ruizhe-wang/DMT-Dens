"""Diagnose thin filament structures in a saved 2D embedding.

A 2D embedding can look wrong for reasons that are invisible in the aggregate
metrics: this run sits at parity with the MLP on density yet shows a thin,
one-dimensional arc among otherwise blob-shaped clusters. This script decides
what that arc actually is, using only saved artifacts - no retraining.

It answers three questions in order:

1. **Is the arc one class or a mixture?** Per label it reports the 2D
   elongation (ratio of PCA singular values). A genuine data property shows up
   as one label with extreme elongation; an optimization artifact shows up as a
   thin structure of mixed labels.
2. **Are those points unusual in the input space?** For the most elongated
   label it compares the input-space norm and k-th nearest-neighbour distance
   (the same density estimator the training loss uses) against the global
   distribution. Low-density outliers extruded into a chain look very different
   from a compact class that happens to be mapped to a curve.
3. **When did it form?** Elongation is recomputed across saved epochs, which
   separates "present from the start" from "emerged late in training".

Usage:
    python scripts/diagnose_embedding_filament.py \\
        --plots-dir outputs/encoder_tuning/E18/emnist_latent_bn_seed44/plots \\
        --config configs/encoder_bench/sweep_e18/runs/emnist_latent_bn_seed44.yaml
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_embedding(csv_path):
    """Returns (cell_id-sorted) xy array."""
    import csv as _csv

    ids, xs, ys = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            ids.append(int(row["cell_id"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    order = np.argsort(np.asarray(ids))
    return np.stack([np.asarray(xs)[order], np.asarray(ys)[order]], axis=1)


def load_labels_and_inputs(config_path, want_inputs=True):
    """Iterates the val dataloader exactly as the callbacks do, so the row order
    matches the saved embedding's cell_id."""
    import torch
    import yaml
    from lightning.pytorch.cli import instantiate_class

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    datamodule = instantiate_class(tuple(), cfg["data"])
    datamodule.setup("fit")

    labels, inputs = [], []
    for batch in datamodule.val_dataloader():
        if "label" in batch:
            labels.append(torch.as_tensor(batch["label"]).cpu().numpy())
        if want_inputs:
            inputs.append(batch["data_input_item"].float().cpu().numpy())
    labels = np.concatenate(labels) if labels else None
    inputs = np.concatenate(inputs) if inputs else None
    return labels, inputs


def elongation(points):
    """Ratio of the two PCA singular values; 1.0 is isotropic, large is a line."""
    if len(points) < 3:
        return np.nan
    centred = points - points.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[1] <= 1e-12:
        return np.inf
    return float(sv[0] / sv[1])


def per_label_shape(xy, labels, min_count=30):
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        n = int(mask.sum())
        if n < min_count:
            continue
        pts = xy[mask]
        rows.append({
            "label": label,
            "n": n,
            "elongation": elongation(pts),
            "x": float(pts[:, 0].mean()),
            "y": float(pts[:, 1].mean()),
            "spread": float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean()),
        })
    rows.sort(key=lambda r: -r["elongation"])
    return rows


def input_space_stats(inputs, mask, k=15, sample=4000, seed=0):
    """Norms and k-th NN distance for the selected points versus a random
    reference sample, using the same kNN-radius density estimator as the loss."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(inputs), size=min(sample, len(inputs)), replace=False)
    reference = inputs[ref_idx]

    sel_idx = np.flatnonzero(mask)
    if len(sel_idx) > sample:
        sel_idx = rng.choice(sel_idx, size=sample, replace=False)
    selected = inputs[sel_idx]

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(reference))).fit(reference)
    d_sel = nn.kneighbors(selected, return_distance=True)[0][:, -1]
    d_ref = nn.kneighbors(reference, return_distance=True)[0][:, -1]

    return {
        "norm_selected": float(np.linalg.norm(selected, axis=1).mean()),
        "norm_reference": float(np.linalg.norm(reference, axis=1).mean()),
        "knn_radius_selected": float(np.median(d_sel)),
        "knn_radius_reference": float(np.median(d_ref)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--density-k", type=int, default=15)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.plots_dir, "*_epoch*_layer0.csv")))
    if not files:
        print("no embedding csv found", file=sys.stderr)
        return 2
    final = files[-1]
    print(f"final embedding: {os.path.basename(final)}")

    xy = load_embedding(final)
    print(f"points: {len(xy)}")

    labels, inputs = load_labels_and_inputs(args.config)
    if labels is None:
        print("datamodule provided no labels; cannot attribute the structure")
        return 1
    if len(labels) != len(xy):
        print(f"WARNING length mismatch: labels {len(labels)} vs embedding {len(xy)}")
        n = min(len(labels), len(xy))
        labels, xy = labels[:n], xy[:n]
        inputs = inputs[:n] if inputs is not None else None

    print("\n== 1. per-label 2D shape (most elongated first) ==")
    rows = per_label_shape(xy, labels)
    print(f"{'label':>6} {'n':>7} {'elongation':>11} {'spread':>8}   centre")
    for row in rows[: args.top]:
        print(
            f"{row['label']:>6} {row['n']:>7} {row['elongation']:>11.2f} "
            f"{row['spread']:>8.2f}   ({row['x']:.0f}, {row['y']:.0f})"
        )
    median_elong = float(np.nanmedian([r["elongation"] for r in rows]))
    print(f"median elongation across {len(rows)} labels: {median_elong:.2f}")

    worst = rows[0]
    print(
        f"\n-> most elongated: label {worst['label']}, {worst['n']} points, "
        f"{worst['elongation']:.1f}x vs a median of {median_elong:.1f}x"
    )

    if inputs is not None:
        print("\n== 2. input-space character of that label ==")
        stats = input_space_stats(inputs, labels == worst["label"], k=args.density_k)
        print(f"  mean input norm      selected {stats['norm_selected']:.3f} "
              f"vs reference {stats['norm_reference']:.3f}")
        print(f"  median kNN radius    selected {stats['knn_radius_selected']:.3f} "
              f"vs reference {stats['knn_radius_reference']:.3f}")
        ratio = stats["knn_radius_selected"] / max(stats["knn_radius_reference"], 1e-12)
        verdict = (
            "sparser than typical (low-density outliers)" if ratio > 1.3
            else "denser than typical" if ratio < 0.77
            else "typical density"
        )
        print(f"  -> {verdict} (radius ratio {ratio:.2f})")

    print("\n== 3. when did it form ==")
    print(f"{'epoch':>7} {'elongation':>11}")
    for path in files:
        match = re.search(r"epoch(\d+)", os.path.basename(path))
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch % 200 and epoch != int(re.search(r"epoch(\d+)", os.path.basename(files[-1])).group(1)):
            continue
        pts = load_embedding(path)
        n = min(len(pts), len(labels))
        e = elongation(pts[:n][labels[:n] == worst["label"]])
        print(f"{epoch:>7} {e:>11.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
