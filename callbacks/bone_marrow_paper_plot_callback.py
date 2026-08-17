from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - plotting helpers can be unit-tested without torch.
    torch = None

try:
    import lightning as pl
except ModuleNotFoundError:  # pragma: no cover - keeps unit tests lightweight.
    from types import SimpleNamespace

    class _Callback:
        pass

    pl = SimpleNamespace(Callback=_Callback)


BONE_MARROW_LABEL_ORDER = [
    "HSC_1",
    "HSC_2",
    "Precursors",
    "Ery_1",
    "Ery_2",
    "Mono_1",
    "Mono_2",
    "CLP",
    "DCs",
    "Mega",
]

BONE_MARROW_LABEL_COLORS = {
    "HSC_1": "#f28e2b",
    "HSC_2": "#edc948",
    "Precursors": "#e15759",
    "Ery_1": "#59a14f",
    "Ery_2": "#8cd17d",
    "Mono_1": "#4e79a7",
    "Mono_2": "#76b7b2",
    "CLP": "#b07aa1",
    "DCs": "#9c755f",
    "Mega": "#ff9da7",
}

BONE_MARROW_STATE_GROUPS = {
    "HSC_1": "early",
    "HSC_2": "early",
    "Precursors": "transition",
    "Ery_1": "terminal",
    "Ery_2": "terminal",
    "Mono_1": "terminal",
    "Mono_2": "terminal",
    "CLP": "terminal",
    "DCs": "terminal",
    "Mega": "terminal",
}

STATE_GROUP_ORDER = ["early", "transition", "terminal"]

STATE_GROUP_COLORS = {
    "early": "#4e79a7",
    "transition": "#e15759",
    "terminal": "#59a14f",
}


class BoneMarrowPaperPlotCallback(pl.Callback):
    """Publication-focused plots for CellRank/Palantir bone marrow embeddings.

    The callback is intentionally dataset-specific. It uses a fixed label order,
    fixed colors, shared continuous color scaling, and the Palantir metadata
    preserved by ``M1datamodel_cellrank_bone_marrow``. It handles both DiffTree
    models and baseline models that expose ``validation_step_outputs_vis``.
    """

    def __init__(
        self,
        output_dir: str = "outputs/paper_figures/bone_marrow",
        every_n_epochs: int | None = None,
        method_name: str | None = None,
        layer: int = -1,
        signal_keys: list[str] | None = None,
        branch_probability_keys: list[str] | None = None,
        marker_genes: list[str] | None = None,
        dataset_slug: str = "bone_marrow",
        label_order: list[str] | None = None,
        label_colors: dict[str, str] | None = None,
        state_group_map: dict[str, str] | None = None,
        state_group_order: list[str] | None = None,
        state_group_colors: dict[str, str] | None = None,
        comparison_embedding_paths: list[str] | None = None,
        save_formats: list[str] | None = None,
        save_table: bool = True,
        log_to_wandb: bool = True,
        point_size: float = 4.0,
        alpha: float = 0.78,
        dpi: int = 300,
        random_state: int = 42,
        max_plot_cells: int | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.method_name = method_name
        self.layer = int(layer)
        self.signal_keys = signal_keys or [
            "final_annotation",
            "state_group",
            "local_density",
            "pseudotime",
            "diffusion_potential",
            "terminal_state",
        ]
        self.branch_probability_keys = branch_probability_keys or [
            "branch_prob_Mono",
            "branch_prob_Ery",
            "branch_prob_Mega",
            "branch_prob_CLP",
            "branch_prob_cDC",
            "branch_prob_pDC",
        ]
        self.marker_genes = marker_genes or []
        self.dataset_slug = dataset_slug
        self.label_order = label_order or BONE_MARROW_LABEL_ORDER
        self.label_colors = label_colors or BONE_MARROW_LABEL_COLORS
        self.state_group_map = state_group_map or BONE_MARROW_STATE_GROUPS
        self.state_group_order = state_group_order or STATE_GROUP_ORDER
        self.state_group_colors = state_group_colors or STATE_GROUP_COLORS
        self.comparison_embedding_paths = comparison_embedding_paths or []
        self.save_formats = save_formats or ["png", "pdf", "svg"]
        self.save_table = save_table
        self.log_to_wandb = log_to_wandb
        self.point_size = float(point_size)
        self.alpha = float(alpha)
        self.dpi = int(dpi)
        self.random_state = int(random_state)
        self.max_plot_cells = max_plot_cells
        self._saved_epochs: set[int] = set()

    def _is_baseline_model(self, pl_module) -> bool:
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _method_name(self, pl_module) -> str:
        if self.method_name:
            return self.method_name
        if hasattr(pl_module, "method"):
            return str(pl_module.method).lstrip("_")
        return "difftree"

    def _get_wandb_run(self, trainer):
        if not self.log_to_wandb:
            return None
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None
        return getattr(logger, "experiment", None)

    def _wandb_log_step(self, trainer, run) -> int:
        trainer_step = int(getattr(trainer, "global_step", 0) or 0)
        run_step = getattr(run, "step", None)
        if run_step is None:
            return trainer_step
        try:
            return max(trainer_step, int(run_step))
        except (TypeError, ValueError):
            return trainer_step

    def _normalize_embedding_layers(self, embedding) -> list[np.ndarray]:
        if torch is not None and isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        if isinstance(embedding, (list, tuple)):
            return [self._as_2d_array(item) for item in embedding]

        arr = self._as_array(embedding)
        if arr.ndim == 2:
            return [self._as_2d_array(arr)]
        if arr.ndim == 3:
            return [self._as_2d_array(arr[i]) for i in range(arr.shape[0])]
        raise ValueError(f"Expected 2D or 3D embedding array, got shape {arr.shape}")

    @staticmethod
    def _as_array(value) -> np.ndarray:
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _as_2d_array(value) -> np.ndarray:
        arr = BoneMarrowPaperPlotCallback._as_array(value).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected an embedding with shape [n_cells, 2], got {arr.shape}")
        return arr

    def _select_layer(self, layers: list[np.ndarray]) -> tuple[int, np.ndarray]:
        layer = self.layer if self.layer >= 0 else len(layers) + self.layer
        if layer < 0 or layer >= len(layers):
            raise IndexError(f"Layer {self.layer} is out of range for {len(layers)} embedding layer(s).")
        return layer, layers[layer]

    def _reorder_by_index(self, layers: list[np.ndarray], indices, n_obs: int) -> list[np.ndarray]:
        if indices is None:
            return layers
        order = self._as_array(indices).astype(int).reshape(-1)
        if len(order) != layers[0].shape[0] or len(order) != n_obs:
            return layers
        reordered = []
        for layer in layers:
            out = np.full((n_obs, 2), np.nan, dtype=np.float32)
            out[order] = layer
            reordered.append(out)
        return reordered

    def _full_dataset_input(self, trainer, n_obs: int):
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "dataset", None)
        data = getattr(dataset, "data", None)
        if data is None:
            return None

        data = self._as_array(data)
        if data.shape[0] != n_obs:
            return None
        return data.astype(np.float32, copy=False)

    def _fit_baseline_full_embedding_layers(self, trainer, pl_module, n_obs: int) -> list[np.ndarray] | None:
        data = self._full_dataset_input(trainer, n_obs)
        if data is None:
            return None

        data_input = data.reshape(data.shape[0], -1)
        if torch is not None:
            device = getattr(pl_module, "device", torch.device("cpu"))
            model_input = torch.as_tensor(data_input, dtype=torch.float32, device=device)
        else:
            model_input = data_input

        with torch.inference_mode() if torch is not None else nullcontext():
            model_output = pl_module(model_input)
        return self._normalize_embedding_layers(model_output)

    def _extract_embedding_layers(self, trainer, pl_module, adata) -> tuple[list[np.ndarray], pd.DataFrame]:
        if self._is_baseline_model(pl_module):
            full_layers = self._fit_baseline_full_embedding_layers(trainer, pl_module, adata.n_obs)
            if full_layers is not None:
                return full_layers, adata.obs

            cached = getattr(pl_module, "validation_step_outputs_vis", None)
            if cached is not None:
                layers = self._normalize_embedding_layers(cached)
                indices = getattr(pl_module, "validation_step_outputs_index", None)
                if indices is not None:
                    order = self._as_array(indices).astype(int).reshape(-1)
                    if len(order) == layers[0].shape[0]:
                        if len(order) == adata.n_obs:
                            return self._reorder_by_index(layers, order, adata.n_obs), adata.obs
                        return layers, adata.obs.iloc[order].copy()
                if layers[0].shape[0] == adata.n_obs:
                    return layers, adata.obs
                return layers, adata.obs.iloc[: layers[0].shape[0]].copy()

        return self._infer_embedding_layers_from_val_loader(trainer, pl_module, adata.n_obs), adata.obs

    def _infer_embedding_layers_from_val_loader(self, trainer, pl_module, n_obs: int) -> list[np.ndarray]:
        import inspect

        forward_params = inspect.signature(pl_module.forward).parameters
        supports_tau = "tau" in forward_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
        )

        chunks: list[list[np.ndarray]] | None = None
        indices: list[np.ndarray] = []
        if torch is None:
            raise ModuleNotFoundError("torch is required to infer embeddings from a Lightning model.")
        device = getattr(pl_module, "device", torch.device("cpu"))

        with torch.inference_mode():
            for batch in trainer.datamodule.val_dataloader():
                data_input = batch["data_input_item"].to(device)
                if supports_tau:
                    tau = getattr(getattr(pl_module, "hparams", None), "tau", 1.0)
                    model_output = pl_module(data_input, tau=tau)
                else:
                    model_output = pl_module(data_input)

                if isinstance(model_output, tuple) and len(model_output) >= 4:
                    layers = self._normalize_embedding_layers(model_output[3])
                elif isinstance(model_output, tuple) and len(model_output) >= 3:
                    layers = self._normalize_embedding_layers(model_output[2])
                else:
                    layers = self._normalize_embedding_layers(model_output)

                if chunks is None:
                    chunks = [[] for _ in layers]
                for layer_index, layer in enumerate(layers):
                    chunks[layer_index].append(layer)

                if "index" in batch:
                    indices.append(self._as_array(batch["index"]).astype(int))

        if chunks is None:
            raise RuntimeError("Validation dataloader produced no batches; cannot plot bone marrow embedding.")

        merged = [np.concatenate(layer_chunks, axis=0) for layer_chunks in chunks]
        if indices:
            merged = self._reorder_by_index(merged, np.concatenate(indices, axis=0), n_obs)
        return merged

    def _marker_values(self, adata, obs: pd.DataFrame, gene: str) -> np.ndarray | None:
        if adata is None or gene not in adata.var_names:
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

    def _build_plot_table(self, obs: pd.DataFrame, embedding: np.ndarray, adata=None) -> pd.DataFrame:
        if embedding.shape[0] != obs.shape[0]:
            raise ValueError(
                f"Embedding rows ({embedding.shape[0]}) do not match AnnData obs rows ({obs.shape[0]})."
            )
        table = pd.DataFrame(
            {
                "cell_id": obs.index.astype(str),
                "x": embedding[:, 0],
                "y": embedding[:, 1],
            }
        )
        for key in dict.fromkeys([*self.signal_keys, *self.branch_probability_keys]):
            if key in obs:
                table[key] = obs[key].to_numpy()
        for gene in self.marker_genes:
            values = self._marker_values(adata, obs, gene)
            if values is not None:
                table[f"marker_{gene}"] = values
        return table

    def _add_derived_signals(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()
        if "final_annotation" in table and "state_group" not in table:
            labels = table["final_annotation"].astype(str)
            table["state_group"] = labels.map(self.state_group_map).fillna("other")
        if "local_density" not in table:
            table["local_density"] = self._local_density(table[["x", "y"]].to_numpy(dtype=float))
        return table

    @staticmethod
    def _local_density(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        finite = np.isfinite(coords).all(axis=1)
        density = np.full(coords.shape[0], np.nan, dtype=float)
        if finite.sum() < 2:
            return np.nan_to_num(density, nan=0.0)

        valid = coords[finite]
        try:
            from sklearn.neighbors import NearestNeighbors

            k = min(16, valid.shape[0])
            distances, _ = NearestNeighbors(n_neighbors=k).fit(valid).kneighbors(valid)
            mean_radius = distances[:, 1:].mean(axis=1) if k > 1 else distances[:, 0]
        except Exception:
            diff = valid[:, None, :] - valid[None, :, :]
            distances = np.sqrt(np.sum(diff * diff, axis=2))
            distances.sort(axis=1)
            k = min(16, valid.shape[0])
            mean_radius = distances[:, 1:k].mean(axis=1)

        valid_density = np.log1p(1.0 / np.maximum(mean_radius, 1.0e-8))
        density[finite] = valid_density
        return np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_signal_keys(self, table: pd.DataFrame) -> list[str]:
        keys = []
        for key in self.signal_keys:
            if key in table:
                keys.append(key)
        for key in self.branch_probability_keys:
            if key in table and key not in keys:
                keys.append(key)
        for gene in self.marker_genes:
            key = f"marker_{gene}"
            if key in table and key not in keys:
                keys.append(key)
        return keys

    def _categorical_order_and_colors(self, signal: str, labels: pd.Series) -> tuple[list[str], dict[str, str]]:
        if signal == "state_group":
            order = [item for item in self.state_group_order if item in set(labels)]
            unknown = sorted(set(labels) - set(order))
            return order + unknown, self.state_group_colors

        unknown = sorted(set(labels) - set(self.label_order))
        order = [item for item in self.label_order if item in set(labels)] + unknown
        return order, self.label_colors

    def _maybe_downsample(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.max_plot_cells is None or len(table) <= int(self.max_plot_cells):
            return table
        return table.sample(
            n=int(self.max_plot_cells),
            random_state=self.random_state,
            replace=False,
        ).sort_index()

    def _save_table(self, table: pd.DataFrame, method: str, epoch_num: int, layer: int) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.dataset_slug}__{method}__epoch_{epoch_num:04d}__layer_{layer}.csv"
        table.to_csv(path, index=False)
        return path

    def _plot_signal(self, ax, table: pd.DataFrame, signal: str, title: str):
        import matplotlib.pyplot as plt

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_facecolor("none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9, pad=5)

        if signal == "final_annotation" or table[signal].dtype.name in ("object", "category"):
            labels = table[signal].astype(str)
            order, colors = self._categorical_order_and_colors(signal, labels)
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
                    label=label,
                )
            if signal in {"final_annotation", "state_group"}:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                    fontsize=6,
                    markerscale=2.2,
                    handletextpad=0.2,
                    borderaxespad=0.0,
                )
            return None

        values = pd.to_numeric(table[signal], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            vmin, vmax = np.nanpercentile(values[finite], [1, 99])
            if np.isclose(vmin, vmax):
                vmin, vmax = float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))
        else:
            vmin, vmax = None, None
        sc = ax.scatter(
            table["x"],
            table["y"],
            s=self.point_size,
            c=values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=self.alpha,
            linewidths=0,
            rasterized=True,
        )
        return sc

    def _plot_panel(self, table: pd.DataFrame, method: str, signals: list[str]):
        import matplotlib.pyplot as plt

        plot_table = self._maybe_downsample(table)
        n_cols = min(3, len(signals))
        n_rows = int(np.ceil(len(signals) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.2 * n_cols, 3.15 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.patch.set_alpha(0.0)
        colorbars = []
        for ax, signal in zip(axes.ravel(), signals):
            title = signal.replace("branch_prob_", "P(").replace("_", " ")
            if signal.startswith("branch_prob_"):
                title = f"{title})"
            artist = self._plot_signal(ax, plot_table, signal, title)
            if artist is not None:
                colorbars.append((artist, ax))
        for ax in axes.ravel()[len(signals):]:
            ax.axis("off")
            ax.set_facecolor("none")
        fig.suptitle(method, fontsize=11, y=1.02)
        for artist, ax in colorbars:
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.01)
            colorbar.ax.set_facecolor("none")
            colorbar.outline.set_visible(False)
        return fig

    def _plot_single_figure(self, table: pd.DataFrame, method: str, signal: str):
        import matplotlib.pyplot as plt

        plot_table = self._maybe_downsample(table)
        fig, ax = plt.subplots(figsize=(3.3, 3.25), constrained_layout=True)
        fig.patch.set_alpha(0.0)
        title = signal.replace("branch_prob_", "P(").replace("marker_", "").replace("_", " ")
        if signal.startswith("branch_prob_"):
            title = f"{title})"
        artist = self._plot_signal(ax, plot_table, signal, title)
        ax.text(
            0.02,
            0.98,
            method,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        if artist is not None:
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.01)
            colorbar.ax.set_facecolor("none")
            colorbar.outline.set_visible(False)
        return fig

    def _save_individual_signal_figures(
        self,
        table: pd.DataFrame,
        method: str,
        signals: list[str],
        epoch_num: int,
        layer: int,
    ) -> list[Path]:
        saved = []
        for signal in signals:
            fig = self._plot_single_figure(table, method, signal)
            try:
                saved.extend(
                    self._save_figure(
                        fig,
                        f"{self.dataset_slug}__{method}__epoch_{epoch_num:04d}__layer_{layer}__{signal}",
                    )
                )
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)
        return saved

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

    def _load_external_embedding_table(self, path: str, obs: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        source = Path(path)
        if source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
            method = str(frame["method"].iloc[0]) if "method" in frame and len(frame) else source.stem
            if "layer" in frame:
                layer = self.layer if self.layer >= 0 else int(frame["layer"].max())
                frame = frame[frame["layer"].astype(int) == layer]
            embedding = frame[["x", "y"]].to_numpy(dtype=np.float32)
            if "cell_id" in frame:
                indexed = frame.set_index(frame["cell_id"].astype(str))
                aligned_obs = obs.loc[indexed.index]
                table = self._build_plot_table(aligned_obs, embedding)
            else:
                table = self._build_plot_table(obs.iloc[: embedding.shape[0]], embedding)
            metadata_columns = {"cell_id", "method", "layer", "x", "y"}
            for column in frame.columns:
                if column not in metadata_columns and column not in table:
                    table[column] = frame[column].to_numpy()
            return method, table

        data = np.load(source, allow_pickle=True)
        method = str(data["method"].tolist()) if "method" in data else source.stem
        layers = [data[key] for key in sorted(data.files) if key.startswith("layer_")]
        layer, embedding = self._select_layer(self._normalize_embedding_layers(layers))
        cell_ids = data["cell_ids"].astype(str) if "cell_ids" in data else obs.index.astype(str)
        aligned_obs = obs.loc[cell_ids]
        return method, self._build_plot_table(aligned_obs, embedding)

    def _plot_comparison_for_signal(self, records: list[tuple[str, pd.DataFrame]], signal: str):
        import matplotlib.pyplot as plt

        n_cols = len(records)
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(3.15 * n_cols, 3.1),
            squeeze=False,
            constrained_layout=True,
        )
        fig.patch.set_alpha(0.0)
        artists = []
        for ax, (method, table) in zip(axes.ravel(), records):
            plot_table = self._maybe_downsample(table)
            artist = self._plot_signal(ax, plot_table, signal, method)
            if artist is not None:
                artists.append((artist, ax))
        if artists:
            colorbar = fig.colorbar(artists[-1][0], ax=[ax for _, ax in artists], fraction=0.025, pad=0.01)
            colorbar.ax.set_facecolor("none")
            colorbar.outline.set_visible(False)
        fig.suptitle(signal.replace("_", " "), fontsize=11, y=1.05)
        return fig

    def _save_comparison_figures(
        self,
        current_method: str,
        current_table: pd.DataFrame,
        obs: pd.DataFrame,
        epoch_num: int,
        layer: int,
        signals: list[str],
    ) -> list[Path]:
        records = [(current_method, current_table)]
        for path in self.comparison_embedding_paths:
            if Path(path).exists():
                records.append(self._load_external_embedding_table(path, obs))
        if len(records) <= 1:
            return []

        saved = []
        for signal in signals:
            available_records = [(method, table) for method, table in records if signal in table]
            if len(available_records) <= 1:
                continue
            fig = self._plot_comparison_for_signal(available_records, signal)
            try:
                saved.extend(
                    self._save_figure(
                        fig,
                        f"comparison__{signal}__epoch_{epoch_num:04d}__layer_{layer}",
                    )
                )
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)
        return saved

    def _log_wandb_images(self, run, trainer, image_paths: Iterable[Path]):
        if run is None:
            return
        try:
            import wandb
        except ModuleNotFoundError:
            return
        log_dict = {
            f"{self.dataset_slug}_paper/{path.stem}": wandb.Image(str(path))
            for path in image_paths
            if path.suffix.lower() == ".png"
        }
        if log_dict:
            run.log(log_dict, step=self._wandb_log_step(trainer, run))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        epoch_num = int(getattr(trainer, "current_epoch", 0)) + 1
        if (
            self.every_n_epochs is not None
            and self.every_n_epochs > 0
            and epoch_num % self.every_n_epochs != 0
        ):
            return
        if epoch_num in self._saved_epochs:
            return

        adata = trainer.datamodule.adata
        method = self._method_name(pl_module)
        layers, obs = self._extract_embedding_layers(trainer, pl_module, adata)
        layer, embedding = self._select_layer(layers)
        table = self._add_derived_signals(self._build_plot_table(obs, embedding, adata=adata))
        signals = self._resolve_signal_keys(table)
        if not signals:
            raise ValueError("No configured bone marrow plotting signals are present in AnnData.obs.")

        saved_paths = []
        if self.save_table:
            saved_paths.append(self._save_table(table, method, epoch_num, layer))

        saved_paths.extend(
            self._save_individual_signal_figures(
                table=table,
                method=method,
                signals=signals,
                epoch_num=epoch_num,
                layer=layer,
            )
        )
        self._log_wandb_images(self._get_wandb_run(trainer), trainer, saved_paths)
        self._saved_epochs.add(epoch_num)

    def on_fit_end(self, trainer, pl_module):
        # Baseline runs have one epoch and are often easier to consume at fit end.
        if not self._saved_epochs:
            self.on_validation_epoch_end(trainer, pl_module)
