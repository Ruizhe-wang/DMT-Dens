"""Generate a small branching dataset for the DMT-Dens CPU smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate(n_samples=600, n_features=32, seed=42):
    if n_samples < 30:
        raise ValueError("n_samples must be at least 30")
    if n_features < 3:
        raise ValueError("n_features must be at least 3")

    rng = np.random.default_rng(seed)
    counts = [n_samples // 3] * 3
    counts[0] += n_samples - sum(counts)
    angles = np.deg2rad([20.0, 140.0, 260.0])

    latent_parts = []
    label_parts = []
    for branch, (count, angle) in enumerate(zip(counts, angles)):
        # Squaring a uniform coordinate creates a dense shared root and sparse
        # distal branch, giving the smoke test a real density gradient.
        radius = np.sort(rng.uniform(0.0, 1.0, count) ** 2)
        bend = 0.18 * np.sin(2.5 * np.pi * radius + branch)
        x = radius * np.cos(angle) - bend * np.sin(angle)
        y = radius * np.sin(angle) + bend * np.cos(angle)
        z = 0.25 * radius + 0.08 * np.sin(4.0 * np.pi * radius)
        latent = np.column_stack([x, y, z])
        latent += rng.normal(scale=0.012 + 0.025 * radius[:, None], size=latent.shape)
        latent_parts.append(latent)
        label_parts.append(np.full(count, branch, dtype=np.int64))

    latent = np.vstack(latent_parts).astype(np.float32)
    labels = np.concatenate(label_parts)
    projection = rng.normal(size=(3, n_features)).astype(np.float32)
    data = latent @ projection
    data += 0.03 * rng.normal(size=data.shape).astype(np.float32)
    data = (data - data.mean(axis=0)) / np.maximum(data.std(axis=0), 1e-6)
    return data.astype(np.float32), labels, latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/data/toy_branches.npz")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data, labels, latent = generate(args.samples, args.features, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=data, y=labels, latent=latent)
    print(f"Wrote {output} with X={data.shape}, y={labels.shape}")


if __name__ == "__main__":
    main()
