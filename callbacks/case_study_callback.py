import os

import lightning as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


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


def _embryo_time_to_float(value):
    text = str(value).strip()
    if "-" in text:
        return float(text.split("-")[-1].strip())
    if text.startswith("<"):
        return float(text[1:].strip()) - 50.0
    if text.startswith(">"):
        return float(text[1:].strip()) + 100.0
    return float(text)


def _category_color_map(categories):
    colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors.extend(cmap(np.linspace(0, 1, cmap.N)))
    return {cat: colors[i % len(colors)] for i, cat in enumerate(sorted(categories))}


class VisualizationCallback(pl.Callback):
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
        time_vmin=50.0,
        time_vmax=750.0,
        show_colorbar=True,
        save_legend=True,
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
        self.time_vmin = time_vmin
        self.time_vmax = time_vmax
        self.show_colorbar = show_colorbar
        self.save_legend = save_legend

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _method_label(self, pl_module):
        if self.method_name:
            return self.method_name
        if hasattr(pl_module, "method"):
            return str(pl_module.method).lstrip("_")
        return "topobranch"

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

    def _time_values(self, values, info):
        if "time" not in str(info).lower():
            return None
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            return numeric.to_numpy(dtype=float)
        try:
            return np.array([_embryo_time_to_float(value) for value in values], dtype=float)
        except (TypeError, ValueError):
            return None

    def plot_case_study_panel(self, adata, info="batch"):
        x = adata.obsm["X_dmtevt"][:, 0]
        y = adata.obsm["X_dmtevt"][:, 1]
        values = adata.obs[info]

        fig, ax = plt.subplots(figsize=(8, 8))
        self._style_axis(fig, ax)

        time_values = self._time_values(values, info)
        if time_values is not None:
            order = np.argsort(time_values, kind="mergesort")
            scatter = ax.scatter(
                x[order],
                y[order],
                c=time_values[order],
                cmap=TIME_CMAP,
                norm=mcolors.Normalize(vmin=self.time_vmin, vmax=self.time_vmax),
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
                cbar.set_label("Embryo time", fontsize=9)
            return fig

        labels = values.astype(str).values
        categories = sorted(set(labels))
        cat_to_color = _category_color_map(categories)
        for cat in categories:
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
        return fig

    def plot_case_study_legend(self, values, info):
        if not self.save_legend or self._time_values(values, info) is not None:
            return None

        categories = sorted(set(values.astype(str).values))
        palette = _category_color_map(categories)
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
        height = max(2.0, 0.22 * np.ceil(len(categories) / ncols))
        fig = plt.figure(figsize=(4.8 * ncols, height))
        fig.patch.set_alpha(0.0)
        fig.legend(handles=handles, loc="center", frameon=False, ncol=ncols, fontsize=7)
        return fig

    def _save_fig(self, fig, path, fmt):
        fig.savefig(
            path,
            format=fmt,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=self.dpi,
        )

    def plot_dmt(self, adata, data_input, lat_vis, text="", info="batch"):
        down_sample = self.max_plot_samples
        if data_input.shape[0] > down_sample:
            indices_t = torch.randperm(data_input.shape[0])[:down_sample]
            idx_np = indices_t.detach().cpu().numpy()

            if isinstance(data_input, torch.Tensor):
                data_input = data_input[indices_t]
                lat_vis = lat_vis[indices_t]
            else:
                data_input = data_input[idx_np]
                lat_vis = lat_vis[idx_np]
            adata = adata[idx_np, :].copy()

        adata.obsm["X_dmtevt"] = lat_vis.detach().cpu().numpy()

        os.makedirs(self.output_dir, exist_ok=True)
        fig = self.plot_case_study_panel(adata, info=info)
        for fmt in self.save_formats:
            path = os.path.join(self.output_dir, f"case_study_{info}_batch{text}.{fmt}")
            self._save_fig(fig, path, fmt)
        plt.close(fig)

        legend_fig = self.plot_case_study_legend(adata.obs[info], info)
        if legend_fig is not None:
            for fmt in self.save_formats:
                path = os.path.join(self.output_dir, f"case_study_{info}_legend{text}.{fmt}")
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

        sc.set_figure_params(dpi=100, facecolor="white", frameon=False)
        lat_vis, data_input = self.get_our_visualization(trainer, pl_module)
        info_list = trainer.datamodule.info_list
        for i in range(lat_vis.shape[0]):
            for info in info_list:
                self.plot_dmt(
                    adata,
                    data_input,
                    lat_vis[i],
                    text=f"_layer{i}",
                    info=info,
                )
