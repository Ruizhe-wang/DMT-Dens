import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import lightning as pl
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import plotly.subplots as sp


class VisualizationCallback(pl.Callback):
    def __init__(
        self,
        output_dir: str = "output",
        every_n_epochs: int = None,
        save_embeddings: bool = False,
        embedding_method_name: str = None,
    ):
        super().__init__()
        # adata = adata
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.save_embeddings = save_embeddings
        self.embedding_method_name = embedding_method_name
        self.upload_dict = False

    def get_our_visualization(self, trainer, pl_module, down_sample=10000):
        data_input = []
        data_list = []

        # on_train_end does not guarantee eval mode. A BatchNorm encoder run in
        # train mode centers every inference batch independently, so concatenated
        # batches no longer share one global coordinate system.
        was_training = pl_module.training
        pl_module.eval()
        try:
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
        finally:
            pl_module.train(was_training)

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

    def plot_single_plot(self, adata, info="batch", color_map=None):
        df = pd.DataFrame(
            {
                "x": adata.obsm["X_dmtevt"][:, 0],
                "y": adata.obsm["X_dmtevt"][:, 1],
                info: adata.obs[info],
            }
        )

        # 如果没有提供 color_map，就根据全部类别生成一个固定映射
        if is_numeric_dtype(df[info]):
            title = "embryo_time" if info == "embryo_time_numeric" else info
            fig1 = px.scatter(
                df,
                x="x",
                y="y",
                color=info,
                title=title,
                opacity=0.65,
                color_continuous_scale=[
                    (0.0, "#f7fbff"),
                    (0.25, "#c6dbef"),
                    (0.5, "#6baed6"),
                    (0.75, "#2171b5"),
                    (1.0, "#08306b"),
                ],
                labels={info: title},
            )
            fig1.update_traces(marker=dict(size=2))
            fig1.update_layout(height=500, width=1000, showlegend=False)
            return fig1

        if color_map is None:
            categories = df[info].unique()
            px_colors = px.colors.qualitative.Set1  # 可换成其它调色板
            color_map = {
                cat: px_colors[i % len(px_colors)]
                for i, cat in enumerate(sorted(categories))
            }

        fig1 = px.scatter(
            df,
            x="x",
            y="y",
            color=info,
            title=info,
            opacity=0.6,
            color_discrete_map=color_map,
        )
        fig1.update_traces(marker=dict(size=2))
        fig1.update_layout(height=500, width=1000, showlegend=True)
        return fig1

    def plot_dmt(
        self,
        adata,
        data_input,
        lat_vis,
        text="",
        info="batch",
    ):
        down_sample = 10000
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

        fig = self.plot_single_plot(adata, info=info)
        return fig

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _embedding_method_name(self, pl_module=None):
        if self.embedding_method_name:
            return self.embedding_method_name
        if pl_module is not None and hasattr(pl_module, "method"):
            return str(pl_module.method).lstrip("_")
        return "topobranch"

    def _embedding_dataframe(self, adata, lat_vis, layer=0, method=None):
        if isinstance(lat_vis, torch.Tensor):
            coords = lat_vis.detach().cpu().numpy()
        else:
            coords = lat_vis
        coords = pd.DataFrame(coords, columns=["x", "y"]).astype(float)
        if coords.shape[0] != adata.n_obs:
            raise ValueError(
                f"Embedding rows ({coords.shape[0]}) do not match AnnData cells ({adata.n_obs})."
            )
        method = method or self._embedding_method_name()
        return pd.DataFrame(
            {
                "cell_id": adata.obs_names.astype(str),
                "method": method,
                "layer": int(layer),
                "x": coords["x"].to_numpy(),
                "y": coords["y"].to_numpy(),
            }
        )

    def _save_embedding_csv(self, adata, lat_vis, epoch_num, layer, method):
        os.makedirs(self.output_dir, exist_ok=True)
        frame = self._embedding_dataframe(
            adata=adata,
            lat_vis=lat_vis,
            layer=layer,
            method=method,
        )
        path = os.path.join(
            self.output_dir,
            f"{method}_epoch{epoch_num:04d}_layer{layer}.csv",
        )
        frame.to_csv(path, index=False)
        return path

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
            if self.save_embeddings:
                self._save_embedding_csv(
                    adata=adata,
                    lat_vis=lat_vis[i],
                    epoch_num=epoch_num,
                    layer=i,
                    method=self._embedding_method_name(pl_module),
                )
            for info in info_list:
                self.plot_dmt(
                    adata,
                    data_input,
                    lat_vis[i],
                    text=f"_layer{i}",
                    info=info,
                )
