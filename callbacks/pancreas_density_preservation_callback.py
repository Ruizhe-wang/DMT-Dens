from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import lightning as pl
except ModuleNotFoundError:  # pragma: no cover - keeps helper tests lightweight.
    from types import SimpleNamespace

    class _Callback:
        pass

    pl = SimpleNamespace(Callback=_Callback)

from callbacks.pancreas_paper_plot_callback import PANCREAS_LABEL_COLORS, PANCREAS_LABEL_ORDER


PANCREAS_TERMINAL_LABELS = {"Alpha", "Beta", "Delta", "Epsilon"}
PANCREAS_EARLY_LABELS = {"Ngn3 low EP"}
PANCREAS_TRANSITION_LABELS = {"Ngn3 high EP", "Fev+"}
PANCREAS_DEFAULT_MARKERS = [
    "Neurog3",
    "Fev",
    "Ins1",
    "Ins2",
    "Gcg",
    "Sst",
    "Ghrl",
    "Pdx1",
]
STATE_GROUP_ORDER = ["early", "transition", "terminal", "other"]
STATE_GROUP_COLORS = {
    "early": "#4e79a7",
    "transition": "#e15759",
    "terminal": "#59a14f",
    "other": "#bab0ac",
}


@dataclass(frozen=True)
class EmbeddingInput:
    path: str
    method: str | None = None
    layer: int | None = None


def infer_pancreas_state_group(
    obs: pd.DataFrame,
    *,
    final_key: str = "final_annotation",
    fine_key: str = "clusters_fine",
) -> pd.Series:
    """Infer early/transition/terminal groups for pancreas case-study labels."""
    final = _obs_string_series(obs, [final_key, "clusters", "celltype", "cell_type"])
    fine = _obs_string_series(obs, [fine_key, final_key, "clusters_fine", "clusters"])

    groups = pd.Series("other", index=obs.index, dtype=object)
    groups[final.isin(PANCREAS_TERMINAL_LABELS)] = "terminal"
    groups[final.isin(PANCREAS_EARLY_LABELS)] = "early"

    fine_transition = (
        fine.isin(PANCREAS_TRANSITION_LABELS)
        | final.isin(PANCREAS_TRANSITION_LABELS)
        | fine.str.contains(r"(?:^|[^A-Za-z])Pre-", regex=True, na=False)
        | fine.str.contains(r"Fev\+", regex=True, na=False)
    )
    groups[fine_transition] = "transition"
    return groups


def _obs_string_series(obs: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for key in candidates:
        if key in obs:
            return obs[key].astype(str)
    return pd.Series("", index=obs.index, dtype=object)


def local_inverse_radius_density(coords: np.ndarray, k: int = 15) -> np.ndarray:
    """Log inverse mean kNN radius. Higher values mean locally denser points."""
    coords = np.asarray(coords, dtype=float)
    density = np.full(coords.shape[0], np.nan, dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 2:
        return np.nan_to_num(density, nan=0.0)

    valid = coords[finite]
    n_neighbors = max(2, min(int(k) + 1, valid.shape[0]))
    try:
        from sklearn.neighbors import NearestNeighbors

        distances = NearestNeighbors(n_neighbors=n_neighbors).fit(valid).kneighbors(valid)[0]
    except Exception:
        diff = valid[:, None, :] - valid[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        distances.sort(axis=1)
        distances = distances[:, :n_neighbors]

    mean_radius = distances[:, 1:].mean(axis=1)
    density[finite] = np.log1p(1.0 / np.maximum(mean_radius, 1.0e-8))
    return np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)


class PancreasDensityPreservationCallback(pl.Callback):
    """Post-hoc subplot generator for pancreas density-preservation figures.

    The callback is designed for paper assembly: it writes independent panels
    for each method and signal, plus CSV tables with density/abundance metrics.
    It can be called directly through ``run()`` or attached to a Lightning run.
    """

    def __init__(
        self,
        adata_path: str | None = None,
        embedding_paths: list[str] | None = None,
        output_dir: str = "outputs/paper_figures/pancreas_density_preservation",
        method_names: list[str] | None = None,
        layers: list[int | None] | None = None,
        final_key: str = "final_annotation",
        fine_key: str = "clusters_fine",
        pseudotime_key: str = "pseudotime",
        reference_obsm_key: str | None = "X_pca",
        marker_genes: list[str] | None = None,
        panel_keys: list[str] | None = None,
        save_formats: list[str] | None = None,
        density_k: int = 15,
        abundance_k: int = 15,
        point_size: float = 5.0,
        alpha: float = 0.82,
        dpi: int = 300,
        log_to_wandb: bool = False,
        skip_missing_embeddings: bool = False,
    ):
        super().__init__()
        self.adata_path = adata_path
        self.embedding_paths = embedding_paths or []
        self.output_dir = Path(output_dir)
        self.method_names = method_names or []
        self.layers = layers or []
        self.final_key = final_key
        self.fine_key = fine_key
        self.pseudotime_key = pseudotime_key
        self.reference_obsm_key = reference_obsm_key
        self.marker_genes = marker_genes or PANCREAS_DEFAULT_MARKERS
        self.panel_keys = panel_keys or ["pseudotime", "state_group", "transition_density", "markers"]
        self.save_formats = save_formats or ["png", "pdf", "svg"]
        self.density_k = int(density_k)
        self.abundance_k = int(abundance_k)
        self.point_size = float(point_size)
        self.alpha = float(alpha)
        self.dpi = int(dpi)
        self.log_to_wandb = log_to_wandb
        self.skip_missing_embeddings = bool(skip_missing_embeddings)
        self._has_run = False

    def run(self) -> dict[str, list[Path] | Path]:
        if not self.adata_path:
            raise ValueError("adata_path is required for post-hoc pancreas density plotting.")
        if not self.embedding_paths:
            raise ValueError("embedding_paths must contain at least one embedding CSV/NPZ.")

        import anndata as ad

        adata = ad.read_h5ad(self.adata_path)
        obs = self._normalized_obs(adata.obs)
        high_dim = self._high_dimensional_reference(adata)

        tables = []
        metric_rows = []
        figure_paths: list[Path] = []
        for index, embedding_path in enumerate(self.embedding_paths):
            if self.skip_missing_embeddings and not Path(embedding_path).exists():
                continue
            method_name = self.method_names[index] if index < len(self.method_names) else None
            layer = self.layers[index] if index < len(self.layers) else None
            method, cell_ids, coords = self._load_embedding(embedding_path, method_name=method_name, layer=layer)
            aligned_obs, aligned_coords, aligned_high_dim = self._align_arrays(obs, high_dim, cell_ids, coords)
            table = self.build_plot_table(method, aligned_obs, aligned_coords, adata=adata)
            table = table.assign(high_dim_density=local_inverse_radius_density(aligned_high_dim, self.density_k))
            table = table.assign(embedding_density=local_inverse_radius_density(aligned_coords, self.density_k))
            tables.append(table)
            metric_rows.append(self.compute_method_metrics(table, aligned_high_dim))
            figure_paths.extend(self.save_method_panels(table, method))

        if not tables:
            raise FileNotFoundError("No configured pancreas embedding files were available for plotting.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_tables = pd.concat(tables, ignore_index=True)
        table_path = self.output_dir / "pancreas_density_preservation_aligned_tables.csv"
        metrics_path = self.output_dir / "pancreas_density_preservation_metrics.csv"
        all_tables.to_csv(table_path, index=False)
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        self._has_run = True
        return {"figures": figure_paths, "table": table_path, "metrics": metrics_path}

    def on_fit_end(self, trainer, pl_module):
        if self._has_run or not self.embedding_paths:
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        self.run()

    def _normalized_obs(self, obs: pd.DataFrame) -> pd.DataFrame:
        out = obs.copy()
        if "final_annotation" not in out:
            for key in (self.final_key, "clusters", "celltype", "cell_type"):
                if key in out:
                    out["final_annotation"] = out[key].astype(str)
                    break
        if "clusters_fine" not in out and self.fine_key in out:
            out["clusters_fine"] = out[self.fine_key].astype(str)
        if "pseudotime" not in out and self.pseudotime_key in out:
            out["pseudotime"] = pd.to_numeric(out[self.pseudotime_key], errors="coerce")
        return out

    def _high_dimensional_reference(self, adata) -> np.ndarray:
        if self.reference_obsm_key and self.reference_obsm_key in adata.obsm:
            return np.asarray(adata.obsm[self.reference_obsm_key], dtype=np.float32)
        data = adata.X
        if hasattr(data, "toarray"):
            data = data.toarray()
        return np.asarray(data, dtype=np.float32)

    def _load_embedding(
        self,
        path: str,
        *,
        method_name: str | None = None,
        layer: int | None = None,
    ) -> tuple[str, np.ndarray, np.ndarray]:
        from tools.bio_eval.embeddings import load_embedding

        embedding = load_embedding(path, method=method_name, layer=layer)
        return embedding.method, embedding.cell_ids.astype(str), embedding.coords.astype(np.float32)

    @staticmethod
    def _align_arrays(
        obs: pd.DataFrame,
        high_dim: np.ndarray,
        cell_ids: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        lookup = {str(cell_id): idx for idx, cell_id in enumerate(cell_ids.astype(str))}
        obs_ids = obs.index.astype(str)
        keep_obs = [i for i, cell_id in enumerate(obs_ids) if cell_id in lookup]
        if not keep_obs:
            raise ValueError("Embedding has no cell IDs in common with AnnData.obs_names.")
        emb_order = [lookup[str(obs_ids[i])] for i in keep_obs]
        return obs.iloc[keep_obs].copy(), coords[emb_order], high_dim[keep_obs]

    def build_plot_table(self, method: str, obs: pd.DataFrame, coords: np.ndarray, adata=None) -> pd.DataFrame:
        if coords.shape[0] != obs.shape[0]:
            raise ValueError(f"coords rows ({coords.shape[0]}) do not match obs rows ({obs.shape[0]}).")

        normalized = self._normalized_obs(obs)
        table = pd.DataFrame(
            {
                "cell_id": normalized.index.astype(str),
                "method": str(method),
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        )
        for key in ("final_annotation", "clusters_fine", "clusters_coarse", "pseudotime"):
            if key in normalized:
                table[key] = normalized[key].to_numpy()

        table["state_group"] = infer_pancreas_state_group(
            normalized,
            final_key="final_annotation",
            fine_key="clusters_fine",
        ).to_numpy()
        table["is_transition"] = table["state_group"].eq("transition").to_numpy()
        table["is_terminal"] = table["state_group"].eq("terminal").to_numpy()
        table["local_density"] = local_inverse_radius_density(coords, self.density_k)

        for gene in self.marker_genes:
            values = self._marker_values(adata, normalized, gene)
            if values is not None:
                table[f"marker_{gene}"] = values
        return table

    @staticmethod
    def _marker_values(adata, obs: pd.DataFrame, gene: str) -> np.ndarray | None:
        if adata is None or not hasattr(adata, "var_names") or gene not in adata.var_names:
            return None
        gene_index = adata.var_names.get_loc(gene)
        values = adata.X[:, gene_index]
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values, dtype=float).reshape(-1)
        positions = adata.obs_names.get_indexer(obs.index.astype(str))
        if np.any(positions < 0):
            return None
        return values[positions]

    def compute_method_metrics(self, table: pd.DataFrame, high_dim: np.ndarray) -> dict[str, float | str]:
        coords = table[["x", "y"]].to_numpy(dtype=float)
        high_density = local_inverse_radius_density(high_dim, self.density_k)
        low_density = local_inverse_radius_density(coords, self.density_k)
        states = table["state_group"].astype(str).to_numpy()
        transition = states == "transition"
        terminal = states == "terminal"

        high_ratio = _density_ratio(high_density, transition, terminal)
        low_ratio = _density_ratio(low_density, transition, terminal)
        return {
            "method": str(table["method"].iloc[0]) if "method" in table and len(table) else "",
            "n_cells": float(len(table)),
            "transition_cell_count": float(transition.sum()),
            "terminal_cell_count": float(terminal.sum()),
            "density_reference_correlation": _safe_corr(high_density, low_density),
            "high_dim_transition_density_ratio": high_ratio,
            "transition_density_ratio": low_ratio,
            "transition_density_error": abs(low_ratio - high_ratio) if np.isfinite(high_ratio + low_ratio) else np.nan,
            "neighbor_abundance_jsd": self._neighbor_abundance_jsd(high_dim, coords, states),
            "pseudotime_bin_abundance_error": self._pseudotime_bin_abundance_error(table),
        }

    def _neighbor_abundance_jsd(self, high_dim: np.ndarray, coords: np.ndarray, states: np.ndarray) -> float:
        high_neighbors = _knn_indices(high_dim, self.abundance_k)
        low_neighbors = _knn_indices(coords, self.abundance_k)
        categories = STATE_GROUP_ORDER
        high_dist = _neighbor_state_distribution(states, high_neighbors, categories)
        low_dist = _neighbor_state_distribution(states, low_neighbors, categories)
        return float(np.mean([_jensen_shannon(high_dist[i], low_dist[i]) for i in range(len(states))]))

    @staticmethod
    def _pseudotime_bin_abundance_error(table: pd.DataFrame) -> float:
        if "pseudotime" not in table:
            return np.nan
        pseudotime = pd.to_numeric(table["pseudotime"], errors="coerce")
        finite = pseudotime.notna()
        if finite.sum() < 3:
            return np.nan
        work = table.loc[finite, ["state_group"]].copy()
        work["bin"] = pd.qcut(pseudotime[finite], q=min(5, finite.sum()), duplicates="drop")
        counts = pd.crosstab(work["bin"], work["state_group"], normalize="index")
        for state in STATE_GROUP_ORDER:
            if state not in counts:
                counts[state] = 0.0
        terminal = counts["terminal"].to_numpy(dtype=float)
        transition = counts["transition"].to_numpy(dtype=float)
        return float(np.mean(np.abs(np.diff(terminal) + np.diff(transition))))

    def save_method_panels(self, table: pd.DataFrame, method: str) -> list[Path]:
        panels = self._panel_keys(table)
        paths: list[Path] = []
        for panel in panels:
            if panel == "transition_density" or panel in table:
                fig = self._plot_single_panel(table, method, panel)
                try:
                    paths.extend(self._save_figure(fig, f"pancreas__{_slug(method)}__{panel}"))
                finally:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
        return paths

    def _panel_keys(self, table: pd.DataFrame) -> list[str]:
        panels: list[str] = []
        for key in self.panel_keys:
            if key == "markers":
                panels.extend([column for column in table.columns if column.startswith("marker_")])
            elif key == "transition_density" or key in table:
                panels.append(key)
        return list(dict.fromkeys(panels))

    def _plot_single_panel(self, table: pd.DataFrame, method: str, panel: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.1, 3.0), constrained_layout=True)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{method}: {panel.replace('marker_', '').replace('_', ' ')}", fontsize=8)

        if panel == "transition_density":
            values = pd.Series(np.nan, index=table.index, dtype=float)
            mask = table["is_transition"].astype(bool)
            values.loc[mask] = pd.to_numeric(table.loc[mask, "local_density"], errors="coerce")
            self._scatter_continuous(ax, table, values.to_numpy(dtype=float), cmap="magma")
        elif panel in {"final_annotation", "state_group"}:
            self._scatter_categorical(ax, table, panel)
        else:
            values = pd.to_numeric(table[panel], errors="coerce").to_numpy(dtype=float)
            self._scatter_continuous(ax, table, values, cmap="viridis")
        return fig

    def _scatter_categorical(self, ax, table: pd.DataFrame, panel: str) -> None:
        labels = table[panel].astype(str)
        if panel == "state_group":
            order = [label for label in STATE_GROUP_ORDER if label in set(labels)]
            colors = STATE_GROUP_COLORS
        else:
            order = [label for label in PANCREAS_LABEL_ORDER if label in set(labels)]
            order += sorted(set(labels) - set(order))
            colors = PANCREAS_LABEL_COLORS
        for label in order:
            mask = labels == label
            ax.scatter(
                table.loc[mask, "x"],
                table.loc[mask, "y"],
                s=self.point_size,
                c=colors.get(label, "#bab0ac"),
                alpha=self.alpha,
                linewidths=0,
                rasterized=True,
            )

    def _scatter_continuous(self, ax, table: pd.DataFrame, values: np.ndarray, cmap: str):
        finite = np.isfinite(values)
        if finite.any():
            vmin, vmax = np.nanpercentile(values[finite], [1, 99])
            if np.isclose(vmin, vmax):
                vmin, vmax = float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))
        else:
            vmin, vmax = None, None
        background = ~finite
        if background.any():
            ax.scatter(
                table.loc[background, "x"],
                table.loc[background, "y"],
                s=self.point_size,
                c="#d1d5db",
                alpha=0.28,
                linewidths=0,
                rasterized=True,
            )
        scatter = ax.scatter(
            table.loc[finite, "x"],
            table.loc[finite, "y"],
            s=self.point_size,
            c=values[finite],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=self.alpha,
            linewidths=0,
            rasterized=True,
        )
        if finite.any():
            colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.01)
            colorbar.outline.set_visible(False)

    def _save_figure(self, fig, stem: str) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in self.save_formats:
            path = self.output_dir / f"{stem}.{fmt}"
            fig.savefig(
                path,
                dpi=self.dpi,
                bbox_inches="tight",
                transparent=True,
                facecolor="none",
                edgecolor="none",
            )
            paths.append(path)
        return paths


def _density_ratio(density: np.ndarray, transition: np.ndarray, terminal: np.ndarray) -> float:
    if not transition.any() or not terminal.any():
        return np.nan
    return float(np.nanmean(density[transition]) / max(np.nanmean(density[terminal]), 1.0e-8))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return np.nan
    a = a[finite]
    b = b[finite]
    if np.isclose(np.std(a), 0.0) or np.isclose(np.std(b), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return np.empty((coords.shape[0], 0), dtype=int)
    n_neighbors = max(2, min(int(k) + 1, coords.shape[0]))
    try:
        from sklearn.neighbors import NearestNeighbors

        return NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    except Exception:
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        return np.argsort(distances, axis=1)[:, 1:n_neighbors]


def _neighbor_state_distribution(states: np.ndarray, indices: np.ndarray, categories: list[str]) -> np.ndarray:
    if indices.shape[1] == 0:
        return np.full((len(states), len(categories)), 1.0 / len(categories), dtype=float)
    out = np.zeros((len(states), len(categories)), dtype=float)
    category_to_index = {category: idx for idx, category in enumerate(categories)}
    for row_idx, neighbors in enumerate(indices):
        for state in states[neighbors]:
            out[row_idx, category_to_index.get(str(state), category_to_index["other"])] += 1.0
        total = out[row_idx].sum()
        if total > 0:
            out[row_idx] /= total
    return out


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1.0e-12)
    q = q / max(q.sum(), 1.0e-12)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1.0e-12))))


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
