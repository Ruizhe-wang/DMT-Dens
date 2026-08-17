import inspect
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata
import lightning as pl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import torch
from plotly.subplots import make_subplots
from callbacks.wandb_utils import safe_wandb_step

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


MarkerConfig = Optional[Union[List[str], Dict[str, List[str]]]]


class MarkerGeneExpressionCallback(pl.Callback):
    """
    Visualize marker gene expression on the learned low-dimensional embedding.

    Supports two modes:
      1. Manual markers via a list or group-to-genes mapping.
      2. Automatic marker discovery from adata using group labels.
    """

    def __init__(
        self,
        output_dir: str = "outputs/plots/marker_genes",
        vis_index: int = -1,
        every_n_epochs: Optional[int] = None,
        down_sample: int = 3000,
        group_key: str = "auto",
        marker_genes: MarkerConfig = None,
        top_n_genes_per_group: int = 2,
        max_genes: int = 12,
        expression_quantile: float = 0.99,
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.vis_index = vis_index
        self.every_n_epochs = every_n_epochs
        self.down_sample = down_sample
        self.group_key = group_key
        self.marker_genes = marker_genes
        self.top_n_genes_per_group = top_n_genes_per_group
        self.max_genes = max_genes
        self.expression_quantile = expression_quantile
        self.seed = seed
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None
        return getattr(logger, "experiment", None)

    def _resolve_tau(self, pl_module) -> float:
        if hasattr(pl_module, "hparams"):
            return getattr(pl_module.hparams, "tau", 1.0)
        return 1.0

    def _is_plot_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> bool:
        if not trainer.is_global_zero:
            return False
        if getattr(trainer, "sanity_checking", False):
            return False
        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
        ):
            return False

        epoch_num = trainer.current_epoch + 1
        if (
            self.every_n_epochs is not None
            and self.every_n_epochs > 0
            and epoch_num % self.every_n_epochs != 0
        ):
            return False
        return True

    def _extract_embeddings(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vis_batches: List[torch.Tensor] = []
        high_batches: List[torch.Tensor] = []

        with torch.inference_mode():
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
            )

            for batch in trainer.datamodule.val_dataloader():
                data_input_item = batch["data_input_item"].to(pl_module.device)
                if supports_tau:
                    model_output = pl_module(data_input_item, tau=self._resolve_tau(pl_module))
                else:
                    model_output = pl_module(data_input_item)

                if isinstance(model_output, tuple) and len(model_output) >= 4:
                    lat_high = model_output[1]
                    lat_vis_best = model_output[2]
                    lat_vis_list = model_output[3]
                elif isinstance(model_output, tuple) and len(model_output) >= 3:
                    lat_high = model_output[1]
                    lat_vis_best = model_output[2]
                    lat_vis_list = [lat_vis_best]
                else:
                    raise ValueError(
                        "MarkerGeneExpressionCallback expects model forward to return "
                        "high-dimensional and low-dimensional embeddings."
                    )

                if 0 <= self.vis_index < len(lat_vis_list):
                    lat_vis = lat_vis_list[self.vis_index]
                else:
                    lat_vis = lat_vis_best

                vis_batches.append(lat_vis.detach().float().cpu())
                high_batches.append(lat_high.detach().float().cpu())

        return (
            torch.cat(vis_batches, dim=0).numpy(),
            torch.cat(high_batches, dim=0).numpy(),
        )

    def _subsample(
        self,
        data_vis: np.ndarray,
        data_high: np.ndarray,
        adata: anndata.AnnData,
    ) -> Tuple[np.ndarray, np.ndarray, anndata.AnnData]:
        if self.down_sample is None or data_vis.shape[0] <= self.down_sample:
            return data_vis, data_high, adata

        rng = np.random.RandomState(self.seed)
        idx = np.sort(rng.choice(data_vis.shape[0], size=self.down_sample, replace=False))
        return data_vis[idx], data_high[idx], adata[idx, :].copy()

    def _choose_group_key(self, adata: anndata.AnnData) -> str:
        if self.group_key != "auto":
            if self.group_key not in adata.obs:
                raise ValueError(f"group_key '{self.group_key}' not found in adata.obs")
            return self.group_key

        candidates = [
            "cell_type",
            "celltype",
            "final_annotation",
            "stage",
            "time",
            "leiden",
        ]
        for key in candidates:
            if key not in adata.obs:
                continue
            values = adata.obs[key].astype(str)
            if values.nunique() >= 2:
                return key

        raise ValueError("No valid group key found for marker visualization")

    def _to_dense_vector(self, values) -> np.ndarray:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values).reshape(-1)

    def _to_dense_matrix(self, values) -> np.ndarray:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values)

    def _flatten_manual_markers(
        self,
        adata: anndata.AnnData,
    ) -> Tuple[List[str], pd.DataFrame]:
        rows: List[Dict[str, object]] = []
        genes: List[str] = []

        if isinstance(self.marker_genes, dict):
            for source_group, group_genes in self.marker_genes.items():
                for rank, gene in enumerate(group_genes, start=1):
                    if gene in adata.var_names and gene not in genes:
                        genes.append(gene)
                        rows.append(
                            {
                                "gene": gene,
                                "source_group": str(source_group),
                                "rank": rank,
                                "selection_mode": "manual",
                            }
                        )
        elif isinstance(self.marker_genes, list):
            for rank, gene in enumerate(self.marker_genes, start=1):
                if gene in adata.var_names and gene not in genes:
                    genes.append(gene)
                    rows.append(
                        {
                            "gene": gene,
                            "source_group": "manual",
                            "rank": rank,
                            "selection_mode": "manual",
                        }
                    )

        return genes[: self.max_genes], pd.DataFrame(rows)

    def _auto_select_marker_genes(
        self,
        adata: anndata.AnnData,
        group_key: str,
    ) -> Tuple[List[str], pd.DataFrame]:
        rows: List[Dict[str, object]] = []
        genes: List[str] = []

        work_adata = adata.copy()
        work_adata.obs[group_key] = work_adata.obs[group_key].astype("category")

        try:
            sc.tl.rank_genes_groups(
                work_adata,
                groupby=group_key,
                method="wilcoxon",
                n_genes=min(max(self.top_n_genes_per_group * 5, 10), work_adata.n_vars),
            )
            markers_df = sc.get.rank_genes_groups_df(work_adata, group=None)
            if "logfoldchanges" in markers_df.columns:
                markers_df = markers_df[markers_df["logfoldchanges"].fillna(0.0) > 0]

            for source_group, group_df in markers_df.groupby("group", sort=False):
                taken = 0
                for _, row in group_df.iterrows():
                    gene = str(row["names"])
                    if gene not in adata.var_names or gene in genes:
                        continue
                    genes.append(gene)
                    rows.append(
                        {
                            "gene": gene,
                            "source_group": str(source_group),
                            "rank": taken + 1,
                            "score": float(row.get("scores", np.nan)),
                            "logfoldchanges": float(row.get("logfoldchanges", np.nan)),
                            "selection_mode": "auto_rank_genes_groups",
                        }
                    )
                    taken += 1
                    if taken >= self.top_n_genes_per_group or len(genes) >= self.max_genes:
                        break
                if len(genes) >= self.max_genes:
                    break
        except Exception as exc:
            if self.verbose:
                print(f"[MarkerGeneExpression] rank_genes_groups failed, fallback to mean-diff ranking: {exc}")

        if genes:
            return genes[: self.max_genes], pd.DataFrame(rows)

        expr = self._to_dense_matrix(work_adata.X)
        groups = work_adata.obs[group_key].astype(str)
        overall_mean = expr.mean(axis=0)
        for source_group in groups.unique():
            mask = (groups == source_group).to_numpy()
            if mask.sum() == 0:
                continue
            score = expr[mask].mean(axis=0) - overall_mean
            rank_idx = np.argsort(score)[::-1]
            taken = 0
            for gene_idx in rank_idx:
                gene = str(work_adata.var_names[gene_idx])
                if gene in genes:
                    continue
                genes.append(gene)
                rows.append(
                    {
                        "gene": gene,
                        "source_group": str(source_group),
                        "rank": taken + 1,
                        "score": float(score[gene_idx]),
                        "selection_mode": "auto_mean_diff",
                    }
                )
                taken += 1
                if taken >= self.top_n_genes_per_group or len(genes) >= self.max_genes:
                    break
            if len(genes) >= self.max_genes:
                break

        return genes[: self.max_genes], pd.DataFrame(rows)

    def _resolve_marker_genes(
        self,
        adata: anndata.AnnData,
        group_key: str,
    ) -> Tuple[List[str], pd.DataFrame]:
        if self.marker_genes:
            genes, marker_df = self._flatten_manual_markers(adata)
            if genes:
                return genes, marker_df
            if self.verbose:
                print("[MarkerGeneExpression] No valid manual marker genes found, switching to auto mode")
        return self._auto_select_marker_genes(adata, group_key)

    def _make_expression_subplot_figure(
        self,
        adata: anndata.AnnData,
        genes: List[str],
        epoch_num: int,
    ) -> go.Figure:
        n_cols = min(3, len(genes))
        n_rows = int(math.ceil(len(genes) / n_cols))
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=genes,
            horizontal_spacing=0.06,
            vertical_spacing=0.08,
        )
        vis = adata.obsm["X_vis"]

        for idx, gene in enumerate(genes):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            expr = self._to_dense_vector(adata[:, gene].X)
            clip_max = np.quantile(expr, self.expression_quantile) if np.any(np.isfinite(expr)) else 1.0
            clip_max = max(float(clip_max), 1e-6)
            expr_clipped = np.clip(expr, 0.0, clip_max)

            fig.add_trace(
                go.Scatter(
                    x=vis[:, 0],
                    y=vis[:, 1],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=expr_clipped,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=clip_max,
                        opacity=0.85,
                        showscale=True,
                        colorbar=dict(title=gene, len=0.7),
                    ),
                    text=[f"{gene}: {value:.4f}" for value in expr],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(showticklabels=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, row=row, col=col)

        fig.update_layout(
            title=f"Marker Gene Expression on Latent Embedding (Epoch {epoch_num})",
            template="plotly_white",
            width=380 * n_cols,
            height=360 * n_rows,
        )
        return fig

    def _make_group_heatmap(
        self,
        adata: anndata.AnnData,
        genes: List[str],
        group_key: str,
        epoch_num: int,
    ) -> go.Figure:
        groups = adata.obs[group_key].astype(str)
        unique_groups = list(pd.Index(groups).unique())
        heatmap_data: List[List[float]] = []
        for gene in genes:
            expr = self._to_dense_vector(adata[:, gene].X)
            gene_values = []
            for group_name in unique_groups:
                mask = (groups == group_name).to_numpy()
                gene_values.append(float(np.mean(expr[mask])) if mask.any() else np.nan)
            heatmap_data.append(gene_values)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=np.asarray(heatmap_data),
                    x=unique_groups,
                    y=genes,
                    colorscale="Viridis",
                    colorbar=dict(title="Mean expression"),
                )
            ]
        )
        fig.update_layout(
            title=f"Marker Gene Mean Expression by {group_key} (Epoch {epoch_num})",
            template="plotly_white",
            width=max(900, 120 + 80 * len(unique_groups)),
            height=max(500, 120 + 40 * len(genes)),
        )
        return fig

    def _save_html(self, fig: go.Figure, filename: str) -> None:
        path = Path(self.output_dir) / filename
        path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")

    def _save_marker_table(self, marker_df: pd.DataFrame, filename: str) -> None:
        if marker_df.empty:
            return
        path = Path(self.output_dir) / filename
        marker_df.to_csv(path, index=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._is_plot_epoch(trainer, pl_module):
            return

        run = self._get_wandb_run(trainer)
        adata_src = getattr(trainer.datamodule, "adata", None)
        if adata_src is None or adata_src.n_vars == 0:
            return

        data_vis, data_high = self._extract_embeddings(trainer, pl_module)
        finite_mask = np.isfinite(data_vis).all(axis=1) & np.isfinite(data_high).all(axis=1)
        data_vis = data_vis[finite_mask]
        data_high = data_high[finite_mask]
        adata_src = adata_src[finite_mask, :].copy()

        if len(data_vis) < 50:
            return

        data_vis, data_high, adata_src = self._subsample(data_vis, data_high, adata_src)
        adata_src.obsm["X_vis"] = data_vis
        adata_src.obsm["X_latent"] = data_high

        group_key = self._choose_group_key(adata_src)
        genes, marker_df = self._resolve_marker_genes(adata_src, group_key)
        if not genes:
            if self.verbose:
                print("[MarkerGeneExpression] No marker genes available for visualization")
            return

        epoch_num = trainer.current_epoch + 1
        expr_fig = self._make_expression_subplot_figure(adata_src, genes, epoch_num)
        heatmap_fig = self._make_group_heatmap(adata_src, genes, group_key, epoch_num)

        self._save_html(expr_fig, f"marker_expression_epoch_{epoch_num:04d}.html")
        self._save_html(heatmap_fig, f"marker_expression_heatmap_epoch_{epoch_num:04d}.html")
        self._save_marker_table(marker_df, f"marker_genes_epoch_{epoch_num:04d}.csv")

        if run is not None and wandb is not None:
            log_dict = {
                "marker_genes/expression": wandb.Html(
                    expr_fig.to_html(include_plotlyjs="cdn", full_html=False)
                ),
                "marker_genes/heatmap": wandb.Html(
                    heatmap_fig.to_html(include_plotlyjs="cdn", full_html=False)
                ),
                "marker_genes/group_key": group_key,
                "marker_genes/n_genes": int(len(genes)),
            }
            run.log(log_dict, step=safe_wandb_step(trainer, run))

        if self.verbose:
            print(
                f"[MarkerGeneExpression] Saved marker plots for {len(genes)} genes "
                f"using group_key={group_key}"
            )
