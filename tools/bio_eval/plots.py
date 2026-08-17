from __future__ import annotations

from pathlib import Path

import numpy as np

from .embeddings import MethodEmbedding
from .signals import BioSignals


def plot_all_panels(
    embedding: MethodEmbedding,
    signals: BioSignals,
    output_dir: str | Path,
    enabled_panels: tuple[str, ...] = (),
) -> list[Path]:
    output_path = Path(output_dir)
    enabled = set(enabled_panels)
    written: list[Path] = []

    if _enabled(enabled, "celltype") and "label" in signals.categorical:
        written.append(
            plot_categorical(
                embedding,
                signals.categorical["label"],
                output_path / "celltype" / f"{embedding.method}_label.png",
                title=f"{embedding.method} - cell type",
            )
        )
    if _enabled(enabled, "pseudotime") and "pseudotime" in signals.continuous:
        written.append(
            plot_continuous(
                embedding,
                signals.continuous["pseudotime"],
                output_path / "pseudotime" / f"{embedding.method}_pseudotime.png",
                title=f"{embedding.method} - pseudotime",
                cmap="viridis",
            )
        )
    if _enabled(enabled, "transition") and "transition" in signals.masks:
        written.append(
            plot_mask(
                embedding,
                signals.masks["transition"],
                output_path / "transition" / f"{embedding.method}_transition.png",
                title=f"{embedding.method} - transition cells",
                color="#d7301f",
            )
        )
    if _enabled(enabled, "rare") and "rare" in signals.masks:
        written.append(
            plot_mask(
                embedding,
                signals.masks["rare"],
                output_path / "rare" / f"{embedding.method}_rare.png",
                title=f"{embedding.method} - rare cells",
                color="#6a51a3",
            )
        )
    for name, values in signals.continuous.items():
        if _enabled(enabled, "marker") and name.startswith("marker_"):
            written.append(
                plot_continuous(
                    embedding,
                    values,
                    output_path / "markers" / f"{embedding.method}_{name}.png",
                    title=f"{embedding.method} - {name}",
                    cmap="magma",
                )
            )
        elif (
            (_enabled(enabled, "branch_probability") and name.startswith("branch_prob_"))
            or (_enabled(enabled, "diffusion_potential") and name == "diffusion_potential")
            or (_enabled(enabled, "branch_probability") and name in {"branch_entropy", "max_branch_prob"})
        ):
            written.append(
                plot_continuous(
                    embedding,
                    values,
                    output_path / "lineage" / f"{embedding.method}_{name}.png",
                    title=f"{embedding.method} - {name}",
                    cmap="YlOrRd",
                )
            )
    return written


def _enabled(enabled: set[str], panel: str) -> bool:
    return not enabled or panel in enabled


def plot_categorical(embedding: MethodEmbedding, values: np.ndarray, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    coords = embedding.coords
    labels = np.asarray(values).astype(str)
    categories = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", max(len(categories), 1))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for index, category in enumerate(categories):
        mask = labels == category
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.8, color=cmap(index), label=category)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if len(categories) <= 15:
        ax.legend(markerscale=2, fontsize=6, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_continuous(embedding: MethodEmbedding, values: np.ndarray, path: Path, title: str, cmap: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    coords = embedding.coords
    vals = np.asarray(values, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, s=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mask(embedding: MethodEmbedding, mask: np.ndarray, path: Path, title: str, color: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    coords = embedding.coords
    selected = np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(coords[~selected, 0], coords[~selected, 1], s=6, color="#d9d9d9", alpha=0.45, linewidths=0)
    ax.scatter(coords[selected, 0], coords[selected, 1], s=12, color=color, alpha=0.9, linewidths=0)
    ax.set_title(f"{title} (n={int(selected.sum())})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
