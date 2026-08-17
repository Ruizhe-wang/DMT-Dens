"""Create transparent paper-ready C. elegans method comparison plots.

Example:
  python tools/celegan_paper_comparison.py \
      --data-dir /usr/storage/ruizhe/zangzelin/data \
      --embedding outputs/embeddings/celegan_topobranch_embeddings.csv \
      --embedding outputs/embeddings/celegan_umap_embeddings.csv \
      --method TopoBranch \
      --method UMAP \
      --layer 0 \
      --output-dir outputs/paper_figures/celegan_comparison
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_COLOR_SCALE = [
    "#fff7bc",
    "#fec44f",
    "#fd8d3c",
    "#e31a1c",
    "#800026",
    "#3f007d",
]
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "embryo_time_vivid",
    TIME_COLOR_SCALE,
)


@dataclass
class EmbeddingPanel:
    method: str
    frame: pd.DataFrame


def embryo_time_to_float(value: object) -> float:
    text = str(value).strip()
    if "-" in text:
        return float(text.split("-")[-1].strip())
    if text.startswith("<"):
        return float(text[1:].strip()) - 50.0
    if text.startswith(">"):
        return float(text[1:].strip()) + 100.0
    return float(text)


def category_color_map(categories: Iterable[str]) -> dict[str, object]:
    colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(np.linspace(0, 1, cmap.N)))
    return {
        cat: colors[i % len(colors)]
        for i, cat in enumerate(sorted(str(cat) for cat in categories))
    }


def resolve_data_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir,
        data_dir / "celegan",
        data_dir / "difftreedata" / "data",
        data_dir / "difftreedata" / "data" / "celegan",
    ]
    required = {
        "celegan.h5ad",
        "celegan_celltype_2.tsv",
        "celegan_embryo_time.tsv",
    }
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(f"Could not find celegan files under {data_dir}")


def load_celegan_obs(data_dir: Path, exclude_unannotated: bool = True) -> pd.DataFrame:
    import anndata as ad

    data_dir = resolve_data_dir(data_dir)
    adata = ad.read_h5ad(data_dir / "celegan.h5ad", backed="r")
    try:
        cell_ids = adata.obs_names.astype(str).to_numpy()
    finally:
        close = getattr(getattr(adata, "file", None), "close", None)
        if close is not None:
            close()

    celltype = pd.read_csv(data_dir / "celegan_celltype_2.tsv", sep="\t", header=None).iloc[:, 0].astype(str)
    embryo_time = pd.read_csv(data_dir / "celegan_embryo_time.tsv", sep="\t", header=None).iloc[:, 0].astype(str)
    if len(cell_ids) != len(celltype) or len(cell_ids) != len(embryo_time):
        raise ValueError(
            "celegan label lengths do not match h5ad observations: "
            f"adata={len(cell_ids)}, celltype={len(celltype)}, embryo_time={len(embryo_time)}"
        )

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "final_annotation": celltype.to_numpy(),
            "embryo_time": embryo_time.to_numpy(),
        }
    )
    obs["embryo_time_numeric"] = obs["embryo_time"].map(embryo_time_to_float).astype(float)
    if exclude_unannotated:
        obs = obs[obs["final_annotation"].str.strip().str.lower() != "unannotated"].copy()
    return obs.set_index("cell_id", drop=False)


def load_embedding(path: Path, method: str | None, layer: int) -> EmbeddingPanel:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"]) if "num_layers" in data else len([key for key in data if key.startswith("layer_")])
        resolved_layer = num_layers + layer if layer < 0 else layer
        coords = np.asarray(data[f"layer_{resolved_layer}"], dtype=float)
        cell_ids = np.asarray(data["cell_ids"], dtype=str)
        if method is not None:
            resolved_method = method
        elif "method" in data:
            resolved_method = str(np.asarray(data["method"]).item())
        else:
            resolved_method = path.stem.replace("_embeddings", "")
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        if "layer" in frame.columns:
            resolved_layer = int(frame["layer"].max()) if layer < 0 else layer
            frame = frame[frame["layer"].astype(int) == resolved_layer].copy()
        if method is None and "method" in frame.columns and not frame["method"].dropna().empty:
            resolved_method = str(frame["method"].dropna().iloc[0])
        else:
            resolved_method = method or path.stem.replace("_embeddings", "")
        required = {"cell_id", "x", "y"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        cell_ids = frame["cell_id"].astype(str).to_numpy()
        coords = frame[["x", "y"]].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unsupported embedding format: {path}")

    if coords.shape[0] != len(cell_ids):
        raise ValueError(f"{path} has {coords.shape[0]} coordinates but {len(cell_ids)} cell ids")
    panel = pd.DataFrame({"cell_id": cell_ids, "x": coords[:, 0], "y": coords[:, 1]})
    return EmbeddingPanel(method=str(resolved_method).lstrip("_"), frame=panel.set_index("cell_id", drop=False))


def align_panels(panels: list[EmbeddingPanel], obs: pd.DataFrame) -> list[EmbeddingPanel]:
    aligned = []
    for panel in panels:
        frame = panel.frame.join(
            obs[["final_annotation", "embryo_time", "embryo_time_numeric"]],
            how="inner",
        )
        if frame.empty:
            raise ValueError(f"No C. elegans labels matched embedding {panel.method!r}")
        aligned.append(EmbeddingPanel(panel.method, frame))
    return aligned


def subset_common_cells(panels: list[EmbeddingPanel], max_points: int, seed: int) -> list[EmbeddingPanel]:
    if max_points <= 0:
        return panels
    common = set(panels[0].frame.index)
    for panel in panels[1:]:
        common.intersection_update(panel.frame.index)
    common_ids = np.array(sorted(common), dtype=str)
    if len(common_ids) <= max_points:
        return panels
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(common_ids, size=max_points, replace=False))
    return [
        EmbeddingPanel(panel.method, panel.frame.loc[selected].copy())
        for panel in panels
    ]


def style_axis(ax):
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.02)
    ax.patch.set_alpha(0.0)


def plot_celltype_grid(
    panels: list[EmbeddingPanel],
    output_dir: Path,
    formats: list[str],
    ncols: int,
    panel_size: float,
    point_size: float,
    alpha: float,
    show_titles: bool,
) -> None:
    categories = sorted({label for panel in panels for label in panel.frame["final_annotation"].astype(str)})
    palette = category_color_map(categories)
    fig, axes = make_grid(len(panels), ncols, panel_size)
    for ax, panel in zip(axes, panels):
        style_axis(ax)
        labels = panel.frame["final_annotation"].astype(str).to_numpy()
        for category in categories:
            mask = labels == category
            if not np.any(mask):
                continue
            ax.scatter(
                panel.frame.loc[mask, "x"],
                panel.frame.loc[mask, "y"],
                c=[palette[category]],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
            )
        if show_titles:
            ax.set_title(panel.method, fontsize=10, pad=2)
    hide_unused_axes(axes, len(panels))
    save_figure(fig, output_dir / "celegan_final_annotation_comparison", formats)
    plt.close(fig)
    save_celltype_legend(categories, palette, output_dir, formats)


def plot_time_grid(
    panels: list[EmbeddingPanel],
    output_dir: Path,
    formats: list[str],
    ncols: int,
    panel_size: float,
    point_size: float,
    alpha: float,
    show_titles: bool,
    time_vmin: float | None,
    time_vmax: float | None,
) -> None:
    all_time = np.concatenate([panel.frame["embryo_time_numeric"].to_numpy(dtype=float) for panel in panels])
    norm = mcolors.Normalize(
        vmin=float(np.nanmin(all_time)) if time_vmin is None else time_vmin,
        vmax=float(np.nanmax(all_time)) if time_vmax is None else time_vmax,
    )
    fig, axes = make_grid(len(panels), ncols, panel_size)
    scatter = None
    for ax, panel in zip(axes, panels):
        style_axis(ax)
        values = panel.frame["embryo_time_numeric"].to_numpy(dtype=float)
        order = np.argsort(values, kind="mergesort")
        scatter = ax.scatter(
            panel.frame["x"].to_numpy()[order],
            panel.frame["y"].to_numpy()[order],
            c=values[order],
            cmap=TIME_CMAP,
            norm=norm,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        if show_titles:
            ax.set_title(panel.method, fontsize=10, pad=2)
    hide_unused_axes(axes, len(panels))
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes[: len(panels)].tolist(), fraction=0.028, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.ax.patch.set_alpha(0.0)
        cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
        cbar.set_label("Embryo time", fontsize=9)
    save_figure(fig, output_dir / "celegan_embryo_time_comparison", formats)
    plt.close(fig)


def make_grid(n_panels: int, ncols: int, panel_size: float):
    ncols = max(1, min(ncols, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        squeeze=False,
    )
    fig.patch.set_alpha(0.0)
    return fig, axes.ravel()


def hide_unused_axes(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)
        ax.patch.set_alpha(0.0)


def save_figure(fig, stem: Path, formats: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(
            stem.with_suffix(f".{fmt}"),
            format=fmt,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=600 if fmt.lower() == "png" else None,
        )


def save_celltype_legend(categories, palette, output_dir: Path, formats: list[str]) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=palette[category],
            markeredgecolor="none",
            label=category,
        )
        for category in categories
    ]
    ncols = 2 if len(categories) > 18 else 1
    fig = plt.figure(figsize=(4.8 * ncols, max(2.0, 0.22 * math.ceil(len(categories) / ncols))))
    fig.patch.set_alpha(0.0)
    fig.legend(handles=handles, loc="center", frameon=False, ncol=ncols, fontsize=7)
    save_figure(fig, output_dir / "celegan_final_annotation_legend", formats)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/usr/storage/ruizhe/zangzelin/data"))
    parser.add_argument("--embedding", action="append", required=True, type=Path)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures/celegan_comparison"))
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--panel-size", type=float, default=2.2)
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-vmin", type=float, default=50.0)
    parser.add_argument("--time-vmax", type=float, default=750.0)
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--keep-unannotated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method and len(args.method) != len(args.embedding):
        raise ValueError("--method must be provided once per --embedding, or omitted")

    formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]
    obs = load_celegan_obs(args.data_dir, exclude_unannotated=not args.keep_unannotated)
    panels = [
        load_embedding(path, args.method[i] if args.method else None, args.layer)
        for i, path in enumerate(args.embedding)
    ]
    panels = align_panels(panels, obs)
    panels = subset_common_cells(panels, args.max_points, args.seed)

    plot_celltype_grid(
        panels,
        args.output_dir,
        formats,
        args.ncols,
        args.panel_size,
        args.point_size,
        args.alpha,
        not args.no_titles,
    )
    plot_time_grid(
        panels,
        args.output_dir,
        formats,
        args.ncols,
        args.panel_size,
        args.point_size,
        args.alpha,
        not args.no_titles,
        args.time_vmin,
        args.time_vmax,
    )


if __name__ == "__main__":
    main()
