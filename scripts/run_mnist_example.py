"""Download MNIST, train a small DMT-Dens demo, and save a 2D PNG.

The example uses labels only to color the final figure. They are never used to
construct neighbors or optimize the representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "publication" / "mnist_example.yaml"
RAW_DATA_DIR = ROOT / "examples" / "data" / "mnist_raw"
OUTPUT_IMAGE = (
    ROOT
    / "outputs"
    / "mnist_example"
    / "dmt_dens_mnist_layer0_final_annotation_final.png"
)


def _balanced_indices(labels, n_samples: int, seed: int):
    import numpy as np

    classes = np.unique(labels)
    base, remainder = divmod(n_samples, len(classes))
    rng = np.random.default_rng(seed)
    selected = []
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        count = base + int(position < remainder)
        if count > len(candidates):
            raise ValueError(f"Class {label} contains only {len(candidates)} samples")
        selected.append(rng.choice(candidates, size=count, replace=False))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices


def prepare_mnist(n_samples: int, seed: int, force: bool = False) -> Path:
    import numpy as np

    if not 100 <= n_samples <= 60_000:
        raise ValueError("--samples must be between 100 and 60000")

    prepared = ROOT / "examples" / "data" / f"mnist_{n_samples}_seed{seed}.npz"
    if prepared.is_file() and not force:
        with np.load(prepared, allow_pickle=False) as archive:
            if archive["X"].shape == (n_samples, 784) and archive["y"].shape == (
                n_samples,
            ):
                print(f"Reusing prepared data: {prepared}")
                return prepared

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required. Install the repository environment first: "
            "conda env create -f environment.yml"
        ) from exc

    dataset = MNIST(root=RAW_DATA_DIR, train=True, download=True)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    indices = _balanced_indices(labels, n_samples=n_samples, seed=seed)
    features = images[indices].reshape(n_samples, -1).astype(np.float32) / 255.0
    selected_labels = labels[indices].astype(np.int64)

    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared, X=features, y=selected_labels)
    print(f"Prepared {prepared} with X={features.shape}, y={selected_labels.shape}")
    return prepared


def build_training_command(
    data_path: Path,
    n_samples: int,
    epochs: int,
    accelerator: str,
    seed: int,
):
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    return [
        sys.executable,
        str(ROOT / "main.py"),
        "fit",
        "--config",
        str(CONFIG),
        f"--seed_everything={seed}",
        f"--data.init_args.data_path={data_path}",
        f"--data.init_args.seed={seed}",
        f"--model.init_args.num_train_data={n_samples}",
        f"--model.init_args.max_epochs={epochs}",
        f"--trainer.max_epochs={epochs}",
        f"--trainer.accelerator={accelerator}",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Download MNIST, train a compact DMT-Dens demo, and save a 2D PNG."
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Lightning accelerator; auto uses a GPU when one is available.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Recreate the sampled NPZ even when it already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download and prepare MNIST without training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare MNIST and print the training command without running it.",
    )
    args = parser.parse_args()

    data_path = prepare_mnist(args.samples, args.seed, force=args.force_prepare)
    if args.prepare_only:
        return

    command = build_training_command(
        data_path=data_path,
        n_samples=args.samples,
        epochs=args.epochs,
        accelerator=args.accelerator,
        seed=args.seed,
    )
    print("Running:", shlex.join(command), flush=True)
    if args.dry_run:
        return

    if OUTPUT_IMAGE.exists():
        OUTPUT_IMAGE.unlink()
    subprocess.run(command, cwd=ROOT, check=True)
    if not OUTPUT_IMAGE.is_file() or OUTPUT_IMAGE.stat().st_size == 0:
        raise RuntimeError(f"Training finished but the expected plot was not created: {OUTPUT_IMAGE}")
    print(f"MNIST embedding plot: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
