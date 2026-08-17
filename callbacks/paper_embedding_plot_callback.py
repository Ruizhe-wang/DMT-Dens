import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import lightning as pl
import wandb
import pandas as pd
import plotly.express as px
import plotly.subplots as sp

from callbacks.wandb_utils import safe_wandb_step


class VisualizationCallback(pl.Callback):
    def __init__(
        self,
        output_dir="output",
        every_n_epochs=None,
        dataset_name="",
        method_name="",
        max_plot_samples=40000,
        save_formats=None,
        scale_down_factor=1.0,
        dpi=150,
        base_point_size=2.4,
        min_point_size=0.5,
        alpha=0.6,
        edge_color="none",
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.max_plot_samples = int(max_plot_samples)
        self.save_formats = save_formats if save_formats is not None else ["png"]
        self.scale_down_factor = scale_down_factor
        self.dpi = dpi
        self.base_point_size = base_point_size
        self.min_point_size = min_point_size
        self.alpha = alpha
        self.edge_color = edge_color
        self.upload_dict = False

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None

        experiment = getattr(logger, "experiment", None)
        return experiment if experiment is not None else None

    def get_our_visualization(self, trainer, pl_module, down_sample=10000):
        data_input = []
        data_list = []

        with torch.inference_mode():
            import inspect
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values())

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
                # Move to CPU immediately to free GPU memory
                data_list.append(lat_vis_list_stack.float())
                data_input.append(data_input_item.detach().float().cpu())

        # Concatenate on CPU
        data = torch.cat(data_list, dim=1)
        data_input = torch.cat(data_input, dim=0)
        # data_p = data.permute(1, 0, 2, 3).reshape(lat_vis_list_stack.shape[0], -1, data.shape[-1])

        return data, data_input

    def plot_umap(self, adata, data_input):
        import umap

        umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric="euclidean")
        umap_data = umap_model.fit_transform(data_input)

        umap_fig = px.scatter(
            x=umap_data[:, 0],
            y=umap_data[:, 1],
            color=adata.obs["final_annotation"],
            title="UMAP Projection of Latent Space",
            labels={"x": "UMAP 1", "y": "UMAP 2"},
        )
        return umap_fig

    def _is_continuous(self, adata, info):
        """Decide whether ``info`` should be colored on a continuous colormap.

        Numeric (non-boolean) columns with many distinct values — e.g.
        ``pseudotime`` / ``fig3_true_time`` — are continuous. Boolean flags,
        string categories, and low-cardinality numeric codes stay categorical.
        """
        if info not in adata.obs:
            return False
        col = adata.obs[info]
        if pd.api.types.is_bool_dtype(col):
            return False
        if not pd.api.types.is_numeric_dtype(col):
            return False
        return col.nunique(dropna=True) > 20

    def plot_single_plot(self, adata, info="batch"):
        import numpy as np

        x = adata.obsm["X_dmtevt"][:, 0]
        y = adata.obsm["X_dmtevt"][:, 1]

        fig, ax = plt.subplots(figsize=(8, 8))

        if self._is_continuous(adata, info):
            # 连续字段（如 pseudotime）：viridis + colorbar，低值先画让高值压上层
            values = pd.to_numeric(adata.obs[info], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="mergesort")
            scatter = ax.scatter(
                x[order], y[order],
                c=values[order],
                cmap="viridis",
                s=max(self.base_point_size, self.min_point_size),
                alpha=self.alpha,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.01)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0.0)
            cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
            cbar.set_label(info, fontsize=9)
        else:
            labels = adata.obs[info].astype(str).values
            categories = sorted(set(labels))
            n = len(categories)

            # 类别 ≤ 20 用离散高对比调色板，否则用连续 colormap 均匀采样
            if n <= 20:
                cmap = plt.get_cmap("tab20", n)
            else:
                cmap = plt.get_cmap("turbo", n)
            cat_to_color = {cat: cmap(i) for i, cat in enumerate(categories)}

            for cat in categories:
                mask = labels == cat
                ax.scatter(
                    x[mask], y[mask],
                    c=[cat_to_color[cat]],
                    s=max(self.base_point_size, self.min_point_size),
                    alpha=self.alpha,
                    linewidths=0 if self.edge_color in ("none", "None", None) else 0.3,
                    edgecolors="none" if self.edge_color in ("none", "None", None) else self.edge_color,
                    rasterized=True,
                )

        ax.set_axis_off()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        return fig

    def plot_dmt(
        self,
        adata,
        data_input,
        lat_vis,
        text="",
        info="batch",
        log_to_wandb=True,
    ):
        down_sample = self.max_plot_samples
        if data_input.shape[0] > down_sample:
            indices_t = torch.randperm(data_input.shape[0])[
                :down_sample
            ]  # torch tensor [K]

            idx_np = indices_t.detach().cpu().numpy()

            # 切 data_input（torch 或 numpy 都兼容）
            if isinstance(data_input, torch.Tensor):
                data_input = data_input[indices_t]
                lat_vis = lat_vis[indices_t]
            else:
                data_input = data_input[idx_np]
                lat_vis = lat_vis[idx_np]

            # 切 adata：必须用 numpy 索引或布尔掩码
            adata = adata[idx_np, :].copy()

        adata.obsm["X_dmtevt"] = lat_vis.detach().cpu().numpy()

        fig_dict = {}
        fig = self.plot_single_plot(adata, info=info)

        os.makedirs(self.output_dir, exist_ok=True)
        for fmt in self.save_formats:
            save_path = os.path.join(self.output_dir, f"{info}_batch{text}.{fmt}")
            fig.savefig(save_path, format=fmt, transparent=True, bbox_inches="tight", dpi=self.dpi)

        if log_to_wandb:
            import io
            from PIL import Image as PILImage
            buf = io.BytesIO()
            fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=self.dpi)
            buf.seek(0)
            pil_img = PILImage.open(buf).convert("RGBA")
            fig_dict[f"{info}/batch_{text}"] = wandb.Image(pil_img)

        plt.close(fig)
        return fig_dict

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        # if in training and epoch == 0, return, unless it's a baseline model
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

        run = self._get_wandb_run(trainer)

        up_dict = {}
        adata = trainer.datamodule.adata

        sc.set_figure_params(dpi=100, facecolor="white", frameon=False)

        # if 'PCA' not in adata.uns and :
        #     sc.tl.pca(adata, n_comps=50)
        #     adata.uns['PCA'] = adata.obsm['X_pca']

        lat_vis, data_input = self.get_our_visualization(trainer, pl_module)
        # import pdb; pdb.set_trace()

        # info_list = ['batch', 'cell_type']
        info_list = trainer.datamodule.info_list
        for i in range(lat_vis.shape[0]):
            for info in info_list:
                fig_dict = self.plot_dmt(
                    adata,
                    data_input,
                    lat_vis[i],
                    text=f"_layer{i}",
                    info=info,
                    log_to_wandb=run is not None,
                )
                up_dict.update(fig_dict)

        # data_input = data_input.detach().cpu().numpy()
        # if self.upload_dict == False:
        #     umap_fig = self.plot_umap(adata, data_input)
        #     up_dict["umap_projection"] = wandb.Plotly(umap_fig)
        #     self.upload_dict = True

        if run is not None and up_dict:
            run.log(up_dict, step=safe_wandb_step(trainer, run))
