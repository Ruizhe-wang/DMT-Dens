"""Paper-ready 2D embedding callback: clean, no-background, no-axes scatter.

Drop this into any Lightning config (DiffTree or a baseline_tri method) and it
renders the model's 2D embedding as a transparent, frame-less figure suitable
for dropping straight into a paper figure -- no axes, ticks, title, legend, or
background. One panel coloured by branch label, and (if a density .npz is given)
one coloured by ground-truth local density.

It always renders at the end of training (``on_train_end`` -- the same hook the
SaveConsolidatedEmbeddingsCallback uses, so it is guaranteed to fire for both the
1-epoch baselines and the multi-epoch DiffTree run) and, optionally, every
``every_n_epochs`` validation epochs. Embedding extraction reuses
``VisualizationCallback.get_our_visualization`` so it works for every method.
"""

from __future__ import annotations

import os
import traceback

import lightning as pl
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from callbacks.xc_plot_callback import VisualizationCallback  # noqa: E402
from callbacks.wandb_utils import safe_wandb_step  # noqa: E402

try:  # W&B is optional for local/offline rendering.
    import wandb
except ImportError:  # pragma: no cover - exercised only without W&B installed
    wandb = None


class PaperEmbeddingCallback(VisualizationCallback):
    def __init__(
        self,
        output_dir: str = "outputs/paper_embeddings",
        method_name: str | None = None,
        color_key: str | None = None,
        every_n_epochs: int | None = None,
        point_size: float = 6.0,
        alpha: float = 0.85,
        cmap: str = "tab10",
        overflow_cmap: str = "gist_ncar",
        density_cmap: str = "magma",
        density_npz: str | None = None,   # .npz with a 'density' array (N,) for density panel
        figsize: float = 4.0,
        dpi: int = 300,
        formats: tuple[str, ...] = ("png", "pdf"),
        log_to_wandb: bool = False,
        wandb_key: str = "paper_embedding/figure",
    ):
        super().__init__(
            output_dir=output_dir,
            every_n_epochs=every_n_epochs,
            save_embeddings=False,
            embedding_method_name=method_name,
        )
        self.point_size = point_size
        self.alpha = alpha
        self.cmap = cmap
        self.overflow_cmap = overflow_cmap
        self.color_key = color_key
        self.density_cmap = density_cmap
        self.density_npz = density_npz
        self.figsize = figsize
        self.dpi = dpi
        self.formats = tuple(formats)
        self.log_to_wandb = log_to_wandb
        self.wandb_key = wandb_key

    # ------------------------------------------------------------------ #
    def _load_density(self, n: int) -> np.ndarray | None:
        if not self.density_npz or not os.path.exists(self.density_npz):
            return None
        d = np.load(self.density_npz, allow_pickle=True)
        if "density" not in d:
            return None
        dens = np.asarray(d["density"], dtype=float).reshape(-1)
        return dens if dens.shape[0] == n else None

    def _categorical_palette(self, n_classes):
        """One distinct colour per class.

        A qualitative map such as tab20 only holds N colours; indexing it with
        ``i % N`` silently paints classes 0, N, 2N ... the same colour, which is
        unusable on datasets with dozens of cell types. Beyond the map's
        capacity, colours are sampled from a continuous map instead so every
        class stays distinguishable.
        """
        cmap_obj = plt.get_cmap(self.cmap)
        capacity = getattr(cmap_obj, "N", 0)
        if n_classes <= capacity:
            return [cmap_obj(i) for i in range(n_classes)]
        overflow = plt.get_cmap(self.overflow_cmap)
        return [overflow(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    def _save_clean(self, xy, colors, *, categorical, fname):
        fig = plt.figure(figsize=(self.figsize, self.figsize))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])   # full-bleed, zero margins
        if categorical:
            classes = np.unique(colors)
            palette = self._categorical_palette(len(classes))
            for i, c in enumerate(classes):
                m = colors == c
                ax.scatter(xy[m, 0], xy[m, 1], s=self.point_size,
                           color=palette[i], alpha=self.alpha,
                           linewidths=0)
        else:
            order = np.argsort(colors)
            ax.scatter(xy[order, 0], xy[order, 1], c=np.log1p(colors[order]),
                       cmap=self.density_cmap, s=self.point_size,
                       alpha=self.alpha, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.margins(0.02)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.0)

        os.makedirs(self.output_dir, exist_ok=True)
        saved = []
        for ext in self.formats:
            path = os.path.join(self.output_dir, f"{fname}.{ext}")
            fig.savefig(
                path,
                dpi=self.dpi,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.0,
            )
            saved.append(path)
        plt.close(fig)
        return saved

    def _categorical_colors(self, trainer, adata):
        candidates = []
        if self.color_key:
            candidates.append(self.color_key)
        candidates.extend(
            [
                "branch_id",
                "final_annotation",
                "cell_type",
                "celltype",
                "label_id",
                "batch",
            ]
        )
        candidates.extend(getattr(trainer.datamodule, "info_list", []) or [])
        for key in candidates:
            if key in adata.obs:
                return key, np.asarray(adata.obs[key]).astype(str)
        return None, None

    def _log_saved_images(self, trainer, saved_paths):
        """Upload the exact files written by the historical paper renderer."""
        if not self.log_to_wandb or wandb is None:
            return
        logger = getattr(trainer, "logger", None)
        run = getattr(logger, "experiment", None) if logger is not None else None
        if run is None or not callable(getattr(run, "log", None)):
            return

        png_paths = [path for path in saved_paths if path.lower().endswith(".png")]
        if not png_paths:
            return
        payload = {}
        for path in png_paths:
            key = (
                f"{self.wandb_key}_density"
                if "_density_" in os.path.basename(path)
                else self.wandb_key
            )
            payload[key] = wandb.Image(path)
        run.log(payload, step=safe_wandb_step(trainer, run))

    def _render(self, trainer, pl_module, tag: str) -> None:
        if not trainer.is_global_zero:
            return
        try:
            adata = trainer.datamodule.adata
            method = self._embedding_method_name(pl_module)
            lat_vis, _ = self.get_our_visualization(trainer, pl_module)
            n = lat_vis.shape[1]

            color_key, categorical_colors = self._categorical_colors(trainer, adata)
            density = self._load_density(n)

            any_saved = []
            for layer in range(lat_vis.shape[0]):
                xy = lat_vis[layer].detach().cpu().numpy()
                if categorical_colors is not None and categorical_colors.shape[0] == n:
                    any_saved += self._save_clean(
                        xy, categorical_colors, categorical=True,
                        fname=f"{method}_layer{layer}_{color_key}_{tag}")
                if density is not None:
                    any_saved += self._save_clean(
                        xy, density, categorical=False,
                        fname=f"{method}_layer{layer}_density_{tag}")
            if any_saved:
                self._log_saved_images(trainer, any_saved)
                print(f"[PaperEmbeddingCallback] saved {len(any_saved)} files to "
                      f"{os.path.abspath(self.output_dir)} (e.g. {os.path.basename(any_saved[0])})")
            else:
                print(f"[PaperEmbeddingCallback] WARNING: nothing saved "
                      f"(color_key={color_key}, density={density is not None}, n={n})")
        except Exception:  # noqa: BLE001 - never crash training over a figure
            print("[PaperEmbeddingCallback] render failed:")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    def on_validation_epoch_end(self, trainer, pl_module):
        # optional intermediate snapshots; the guaranteed render is on_train_end
        if self.every_n_epochs is None or self.every_n_epochs <= 0:
            return
        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
            and not self._is_baseline_model(pl_module)
        ):
            return
        epoch_num = trainer.current_epoch + 1
        if epoch_num % self.every_n_epochs != 0:
            return
        self._render(trainer, pl_module, tag=f"epoch{epoch_num:04d}")

    def on_train_end(self, trainer, pl_module):
        # always produce the final paper figure
        self._render(trainer, pl_module, tag="final")
