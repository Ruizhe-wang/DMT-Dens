import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import lightning as pl
import scanpy as sc
import anndata
import networkx as nx
import plotly.graph_objects as go
from plotly.colors import qualitative

try:
    import wandb
except Exception:
    wandb = None

# A visually appealing color palette suitable for top-tier publications
PUB_COLORS = qualitative.Pastel + qualitative.Set2 + qualitative.D3

class DiffTreeDensityBranchStructureCallback(pl.Callback):
    """
    A biologically meaningful callback to visualize the branching structure of 
    2D embeddings produced by the DiffTree_density model.
    
    This callback uses PAGA (Partition-based Graph Abstraction) to capture 
    the continuous topology and branching lineage, projecting it onto the 
    DiffTree 2D latent space. The visualization adheres to top-conference standards:
    clean, non-cluttered, and highly informative.
    """

    def __init__(
        self,
        output_dir: str = "outputs/plots/difftree_density_branch",
        vis_index: int = -1,
        every_n_epochs: int = 10,
        down_sample: int = 10000,
        n_neighbors: int = 15,
        cluster_key: str = "auto",
        leiden_resolution: float = 0.5,
        paga_threshold: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.vis_index = vis_index
        self.every_n_epochs = every_n_epochs
        self.down_sample = down_sample
        self.n_neighbors = n_neighbors
        self.cluster_key = cluster_key
        self.leiden_resolution = leiden_resolution
        self.paga_threshold = paga_threshold
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

    def _is_plot_epoch(self, trainer: pl.Trainer) -> bool:
        if not trainer.is_global_zero:
            return False
        if getattr(trainer, "sanity_checking", False):
            return False
        epoch = trainer.current_epoch + 1
        return epoch % self.every_n_epochs == 0

    def _extract_embeddings(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> Tuple[np.ndarray, np.ndarray, anndata.AnnData]:
        vis_list = []
        high_list = []
        
        # Try to gather original adata if available from datamodule
        adata_src = None
        if hasattr(trainer.datamodule, "adata"):
            adata_src = trainer.datamodule.adata

        with torch.inference_mode():
            for batch in trainer.datamodule.val_dataloader():
                data_input = batch["data_input_item"].to(pl_module.device)
                
                # Compatible with DiffTree_density output format
                tau = getattr(pl_module.hparams, "tau", 1.0)
                model_out = pl_module(data_input, tau=tau)
                
                if isinstance(model_out, tuple) and len(model_out) >= 4:
                    lat_high = model_out[1]
                    lat_vis_list = model_out[3]
                    lat_vis = lat_vis_list[self.vis_index] if len(lat_vis_list) > 0 else model_out[2]
                else:
                    # Fallback
                    lat_high = model_out[1] if len(model_out) > 1 else data_input
                    lat_vis = model_out[2] if len(model_out) > 2 else model_out[0]

                vis_list.append(lat_vis.detach().cpu().float().numpy())
                high_list.append(lat_high.detach().cpu().float().numpy())

        data_vis = np.concatenate(vis_list, axis=0)
        data_high = np.concatenate(high_list, axis=0)

        # Downsample for cleaner visualization and faster PAGA
        if self.down_sample and data_vis.shape[0] > self.down_sample:
            np.random.seed(self.seed)
            idx = np.random.choice(data_vis.shape[0], self.down_sample, replace=False)
            data_vis = data_vis[idx]
            data_high = data_high[idx]
            if adata_src is not None:
                adata_src = adata_src[idx].copy()
        elif adata_src is not None:
            adata_src = adata_src.copy()

        if adata_src is None:
            adata_src = anndata.AnnData(X=data_high)

        return data_vis, data_high, adata_src

    def _determine_cluster_key(self, adata: anndata.AnnData) -> str:
        if self.cluster_key != "auto" and self.cluster_key in adata.obs:
            return self.cluster_key
            
        candidates = ["cell_type", "celltype", "leiden", "louvain"]
        for key in candidates:
            if key in adata.obs and adata.obs[key].nunique() > 1:
                return key

        # Fallback to Leiden clustering if no valid annotations exist
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep="X_pca" if "X_pca" in adata.obsm else "X")
        sc.tl.leiden(adata, resolution=self.leiden_resolution, key_added="leiden_clusters", random_state=self.seed)
        return "leiden_clusters"

    def _compute_paga(self, data_high: np.ndarray, data_vis: np.ndarray, adata: anndata.AnnData) -> Tuple[anndata.AnnData, str]:
        # High dimensional representation used for neighborhood graph
        adata.obsm["X_latent"] = data_high
        adata.obsm["X_vis"] = data_vis
        
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep="X_latent")
        group_key = self._determine_cluster_key(adata)
        adata.obs[group_key] = adata.obs[group_key].astype("category")

        # Compute PAGA graph
        sc.tl.paga(adata, groups=group_key)
        return adata, group_key

    def _create_paga_plot(self, adata: anndata.AnnData, group_key: str, epoch: int) -> go.Figure:
        categories = list(adata.obs[group_key].cat.categories)
        paga_conn = adata.uns["paga"]["connectivities"]
        conn_dense = paga_conn.toarray() if hasattr(paga_conn, "toarray") else np.asarray(paga_conn)

        # Calculate cluster centroids in the DiffTree density 2D space
        centroids = {}
        for cat in categories:
            mask = adata.obs[group_key] == cat
            centroids[cat] = adata.obsm["X_vis"][mask].mean(axis=0)

        fig = go.Figure()

        # 1. Plot background cells with high transparency (clean scatter)
        for i, cat in enumerate(categories):
            mask = adata.obs[group_key] == cat
            color = PUB_COLORS[i % len(PUB_COLORS)]
            coords = adata.obsm["X_vis"][mask]
            
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.3, line=dict(width=0)),
                name=str(cat),
                showlegend=True,
                hoverinfo="text",
                text=[f"Cluster: {cat}"] * sum(mask)
            ))

        # 2. Plot PAGA edges (branching structure)
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                weight = float(conn_dense[i, j])
                if weight > self.paga_threshold:
                    cat_i, cat_j = categories[i], categories[j]
                    edge_x.extend([centroids[cat_i][0], centroids[cat_j][0], None])
                    edge_y.extend([centroids[cat_i][1], centroids[cat_j][1], None])
                    edge_weights.append(weight)

        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='rgba(50, 50, 50, 0.6)', width=2),
                hoverinfo='none',
                showlegend=False
            ))

        # 3. Plot Cluster Centroids (Nodes)
        node_x = [centroids[cat][0] for cat in categories]
        node_y = [centroids[cat][1] for cat in categories]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=12,
                color=[PUB_COLORS[i % len(PUB_COLORS)] for i in range(len(categories))],
                line=dict(color='white', width=2),
                symbol='circle'
            ),
            text=[str(c) for c in categories],
            textposition="top center",
            textfont=dict(size=12, color="black", family="Arial, sans-serif"),
            showlegend=False,
            hoverinfo="text"
        ))

        # Top-tier publication styling
        fig.update_layout(
            title=dict(
                text=f"DiffTree Density: Biological Branching Structure (Epoch {epoch})",
                font=dict(size=18, family="Arial, sans-serif"),
                x=0.5,
                y=0.95
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                title="", linecolor='black', linewidth=1, mirror=True
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                title="", linecolor='black', linewidth=1, mirror=True
            ),
            legend=dict(
                title="Clusters",
                font=dict(size=10),
                itemsizing='constant',
                borderwidth=0
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            width=800,
            height=600
        )
        
        return fig

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._is_plot_epoch(trainer):
            return

        try:
            data_vis, data_high, adata = self._extract_embeddings(trainer, pl_module)
            
            # Compute PAGA graph representing branch skeleton
            adata, group_key = self._compute_paga(data_high, data_vis, adata)
            
            # Create standard-compliant plot
            fig = self._create_paga_plot(adata, group_key, trainer.current_epoch + 1)
            
            # Save artifacts
            save_path = os.path.join(self.output_dir, f"difftree_density_branch_ep{trainer.current_epoch + 1:04d}.html")
            fig.write_html(save_path)
            
            # Log to wandb if enabled
            logger = getattr(trainer, "logger", None)
            if logger is not None and wandb is not None:
                exp = getattr(logger, "experiment", None)
                if exp is not None and hasattr(exp, "log"):
                    exp.log({"DiffTree_Density_PAGA_Branching": wandb.Html(save_path)}, step=trainer.current_epoch)

        except Exception as e:
            print(f"[DiffTreeDensityBranchStructureCallback] Plotting failed: {e}")
