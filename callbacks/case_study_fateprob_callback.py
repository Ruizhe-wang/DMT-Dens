"""Case-study visualization callback with fate-probability plotting support.

Copy of ``callbacks/case_study_callback.py`` that adds a dedicated plotting
schema for continuous fate-probability columns (``fate_prob_*`` in ``adata.obs``,
produced by ``M1datamodel_dyngen_fate._add_fate_probability_columns``).

Two new panel types are produced, one per embedding layer:
  * one continuous-colored scatter per ``fate_prob_<terminal>`` column, colored
    by P(reach that terminal state) with a sequential colormap + colorbar;
  * an optional composite "predicted fate" panel coloring each cell by the
    arg-max terminal state across all fate-probability columns.

The categorical / pseudotime ``info_list`` panels are intentionally left to the
original ``case_study_callback.VisualizationCallback`` so the two callbacks can
run side by side. This one only renders the fate-probability schema.
"""

import os

import lightning as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


def _category_color_map(categories):
    colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(np.linspace(0, 1, cmap.N)))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(sorted(categories))}


class FateProbabilityVisualizationCallback(pl.Callback):
    def __init__(
        self,
        output_dir="outputs/case_study",
        every_n_epochs=None,
        dataset_name="",
        method_name="",
        max_plot_samples=40000,
        save_formats=None,
        dpi=800,
        point_size=4.5,
        alpha=0.9,
        panel_prefix="case_study",
        show_colorbar=True,
        save_legend=True,
        fate_prob_prefix="fate_prob_",
        fate_cmap="magma",
        fate_vmin=0.0,
        fate_vmax=1.0,
        plot_fate_argmax=True,
        fate_commit_threshold=0.5,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.max_plot_samples = int(max_plot_samples)
        self.save_formats = save_formats if save_formats is not None else ["png"]
        self.dpi = dpi
        self.point_size = point_size
        self.alpha = alpha
        self.panel_prefix = panel_prefix
        self.show_colorbar = show_colorbar
        self.save_legend = save_legend
        self.fate_prob_prefix = fate_prob_prefix
        self.fate_cmap = fate_cmap
        self.fate_vmin = fate_vmin
        self.fate_vmax = fate_vmax
        self.plot_fate_argmax = plot_fate_argmax
        self.fate_commit_threshold = fate_commit_threshold

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def get_our_visualization(self, trainer, pl_module):
        data_input = []
        data_list = []

        with torch.inference_mode():
            import inspect

            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in forward_params.values()
            )

            for batch in trainer.datamodule.val_dataloader():
                data_input_item = batch["data_input_item"].to(pl_module.device)

                if supports_tau:
                    tau = getattr(pl_module.hparams, "tau", 1.0) if hasattr(pl_module, "hparams") else 1.0
                    model_output = pl_module(data_input_item, tau=tau)
                else:
                    model_output = pl_module(data_input_item)

                if isinstance(model_output, tuple) and len(model_output) >= 4:
                    lat_vis_list = model_output[3]
                elif isinstance(model_output, tuple) and len(model_output) >= 3:
                    lat_vis_list = [model_output[2]]
                else:
                    lat_vis = torch.as_tensor(
                        model_output,
                        dtype=torch.float32,
                        device=data_input_item.device,
                    )
                    lat_vis_list = [lat_vis]

                lat_vis_list_stack = torch.stack([lv.detach().cpu() for lv in lat_vis_list], dim=0)
                data_list.append(lat_vis_list_stack.float())
                data_input.append(data_input_item.detach().float().cpu())

        data = torch.cat(data_list, dim=1)
        data_input = torch.cat(data_input, dim=0)
        return data, data_input

    def _style_axis(self, fig, ax):
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.02)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    def _save_fig(self, fig, path, fmt):
        fig.savefig(
            path,
            format=fmt,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=self.dpi,
        )

    def _fate_prob_columns(self, adata):
        cols = [
            str(c)
            for c in adata.obs.columns
            if str(c).startswith(self.fate_prob_prefix)
        ]
        return sorted(cols)

    def _fate_label(self, col):
        return col[len(self.fate_prob_prefix):] if col.startswith(self.fate_prob_prefix) else col

    def plot_fate_prob_panel(self, adata, col):
        x = adata.obsm["X_dmtevt"][:, 0]
        y = adata.obsm["X_dmtevt"][:, 1]
        values = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 8))
        self._style_axis(fig, ax)

        # Draw low-probability cells first so high-probability cells sit on top.
        order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="mergesort")
        scatter = ax.scatter(
            x[order],
            y[order],
            c=values[order],
            cmap=self.fate_cmap,
            norm=mcolors.Normalize(vmin=self.fate_vmin, vmax=self.fate_vmax),
            s=self.point_size,
            alpha=self.alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        if self.show_colorbar:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.01)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0.0)
            cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
            cbar.set_label(f"P({self._fate_label(col)})", fontsize=9)
        return fig

    def plot_fate_argmax_panel(self, adata, cols):
        x = adata.obsm["X_dmtevt"][:, 0]
        y = adata.obsm["X_dmtevt"][:, 1]
        probs = np.stack(
            [pd.to_numeric(adata.obs[c], errors="coerce").to_numpy(dtype=float) for c in cols],
            axis=1,
        )
        names = [self._fate_label(c) for c in cols]
        # Cells with all-NaN rows fall back to a sentinel category.
        all_nan = np.all(np.isnan(probs), axis=1)
        filled = np.where(np.isnan(probs), -np.inf, probs)
        argmax_idx = np.argmax(filled, axis=1)
        max_prob = np.max(filled, axis=1)
        labels = np.array([names[i] for i in argmax_idx], dtype=object)
        labels[all_nan] = "unassigned"
        # Cells whose top fate probability is below the commitment threshold are
        # not confidently assigned (e.g. uniform multipotent cells where argmax
        # is arbitrary noise) — mark them as a neutral "uncommitted" category.
        uncommitted = (~all_nan) & (max_prob < self.fate_commit_threshold)
        labels[uncommitted] = "uncommitted"

        fig, ax = plt.subplots(figsize=(8, 8))
        self._style_axis(fig, ax)

        categories = sorted(set(labels.tolist()))
        cat_to_color = _category_color_map(categories)
        # Neutral greys for the non-committed sentinels so they recede visually.
        sentinel_grey = {"uncommitted": (0.82, 0.82, 0.82, 1.0), "unassigned": (0.6, 0.6, 0.6, 1.0)}
        for sentinel, grey in sentinel_grey.items():
            if sentinel in cat_to_color:
                cat_to_color[sentinel] = grey
        # Draw sentinels first so confidently-committed cells sit on top.
        draw_order = [c for c in categories if c in sentinel_grey] + [
            c for c in categories if c not in sentinel_grey
        ]
        for cat in draw_order:
            mask = labels == cat
            ax.scatter(
                x[mask],
                y[mask],
                c=[cat_to_color[cat]],
                s=self.point_size,
                alpha=self.alpha,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
            )
        return fig, categories, cat_to_color

    def plot_fate_argmax_legend(self, categories, cat_to_color):
        if not self.save_legend:
            return None
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=4,
                markerfacecolor=cat_to_color[category],
                markeredgecolor="none",
                label=category,
            )
            for category in categories
        ]
        ncols = 2 if len(categories) > 18 else 1
        height = max(2.0, 0.22 * np.ceil(len(categories) / ncols))
        fig = plt.figure(figsize=(4.8 * ncols, height))
        fig.patch.set_alpha(0.0)
        fig.legend(handles=handles, loc="center", frameon=False, ncol=ncols, fontsize=7)
        return fig

    def plot_dmt_fate(self, adata, data_input, lat_vis, text=""):
        down_sample = self.max_plot_samples
        if data_input.shape[0] > down_sample:
            indices_t = torch.randperm(data_input.shape[0])[:down_sample]
            idx_np = indices_t.detach().cpu().numpy()
            if isinstance(lat_vis, torch.Tensor):
                lat_vis = lat_vis[indices_t]
            else:
                lat_vis = lat_vis[idx_np]
            adata = adata[idx_np, :].copy()

        adata.obsm["X_dmtevt"] = lat_vis.detach().cpu().numpy()

        fate_cols = self._fate_prob_columns(adata)
        if not fate_cols:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        for col in fate_cols:
            name = self._fate_label(col)
            fig = self.plot_fate_prob_panel(adata, col)
            for fmt in self.save_formats:
                path = os.path.join(self.output_dir, f"case_study_fateprob_{name}{text}.{fmt}")
                self._save_fig(fig, path, fmt)
            plt.close(fig)

        if self.plot_fate_argmax and len(fate_cols) >= 2:
            fig, categories, cat_to_color = self.plot_fate_argmax_panel(adata, fate_cols)
            for fmt in self.save_formats:
                path = os.path.join(self.output_dir, f"case_study_fate_argmax{text}.{fmt}")
                self._save_fig(fig, path, fmt)
            plt.close(fig)

            legend_fig = self.plot_fate_argmax_legend(categories, cat_to_color)
            if legend_fig is not None:
                for fmt in self.save_formats:
                    path = os.path.join(self.output_dir, f"case_study_fate_argmax_legend{text}.{fmt}")
                    self._save_fig(legend_fig, path, fmt)
                plt.close(legend_fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
            and not self._is_baseline_model(pl_module)
        ):
            return

        epoch_num = trainer.current_epoch + 1
        if (
            self.every_n_epochs is not None
            and self.every_n_epochs > 0
            and epoch_num % self.every_n_epochs != 0
        ):
            return

        adata = trainer.datamodule.adata
        if not self._fate_prob_columns(adata):
            return

        sc.set_figure_params(dpi=100, facecolor="white", frameon=False)
        lat_vis, data_input = self.get_our_visualization(trainer, pl_module)
        for i in range(lat_vis.shape[0]):
            self.plot_dmt_fate(
                adata,
                data_input,
                lat_vis[i],
                text=f"_layer{i}",
            )
