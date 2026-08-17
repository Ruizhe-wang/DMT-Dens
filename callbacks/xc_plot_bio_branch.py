import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import lightning as pl
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import torch
from plotly.colors import qualitative
from callbacks.wandb_utils import safe_wandb_step

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


BRANCH_COLORS = qualitative.D3 + qualitative.Set2 + qualitative.Bold


class BioBranchTrajectoryCallback(pl.Callback):
    """
    Visualize branch structure for biological datasets using latent representations.
    
    This callback performs trajectory inference and branch detection through:
      1. Neighborhood graph construction from high-dimensional latent features
      2. PAGA-based cluster connectivity analysis with statistical thresholding
      3. Maximum spanning tree extraction for backbone/branch identification
      4. Multi-heuristic root selection (time-based, pseudotime, topology)
      5. Interactive visualization with quality metrics
    
    Key Features:
      - Independent of tree routing (uses latent embeddings only)
      - Robust fallback strategies for edge cases
      - Quality metrics for branch confidence assessment
      - Excludes technical batch effects from temporal inference
      
    Args:
        output_dir: Directory for saving HTML visualizations
        vis_index: Index of visualization layer (-1 for best)
        every_n_epochs: Plotting frequency (None = only at end)
        down_sample: Maximum cells to analyze (for efficiency)
        n_neighbors: Number of neighbors for KNN graph
        cluster_key: Metadata key for cell grouping ('auto' for inference)
        time_key: Metadata key for temporal information ('auto' for inference)
        root_group: Manually specify root cluster (None for automatic)
        max_groups: Maximum allowed clusters (>40 may be noisy)
        min_group_size: Minimum cells per valid cluster
        leiden_resolution: Resolution for Leiden clustering fallback
        paga_threshold: Minimum PAGA connectivity to include edge (0.05 recommended)
        seed: Random seed for reproducibility
        verbose: Print detailed logging information
    """

    def __init__(
        self,
        output_dir: str = "outputs/plots/bio_branch",
        vis_index: int = -1,
        every_n_epochs: Optional[int] = None,
        down_sample: int = 12000,
        n_neighbors: int = 15,
        cluster_key: str = "auto",
        time_key: str = "auto",
        root_group: Optional[str] = None,
        max_groups: int = 40,
        min_group_size: int = 20,
        leiden_resolution: float = 0.5,
        paga_threshold: float = 0.05,
        seed: int = 42,
        verbose: bool = False,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.vis_index = vis_index
        self.every_n_epochs = every_n_epochs
        self.down_sample = down_sample
        self.n_neighbors = n_neighbors
        self.cluster_key = cluster_key
        self.time_key = time_key
        self.root_group = root_group
        self.max_groups = max_groups
        self.min_group_size = min_group_size
        self.leiden_resolution = leiden_resolution
        self.paga_threshold = paga_threshold
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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> Tuple[np.ndarray, np.ndarray]:
        vis_batches: List[torch.Tensor] = []
        high_batches: List[torch.Tensor] = []

        with torch.inference_mode():
            for batch in trainer.datamodule.val_dataloader():
                data_input_item = batch["data_input_item"].to(pl_module.device)
                model_output = pl_module(
                    data_input_item,
                    tau=self._resolve_tau(pl_module),
                )

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
                        "BioBranchTrajectoryCallback expects model forward to return "
                        "latent high-dimensional and low-dimensional embeddings."
                    )

                if 0 <= self.vis_index < len(lat_vis_list):
                    lat_vis = lat_vis_list[self.vis_index]
                else:
                    lat_vis = lat_vis_best

                vis_batches.append(lat_vis.detach().float().cpu())
                high_batches.append(lat_high.detach().float().cpu())

        data_vis = torch.cat(vis_batches, dim=0).numpy()
        data_high = torch.cat(high_batches, dim=0).numpy()
        return data_vis, data_high

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

    def _is_valid_group_series(self, series: pd.Series) -> bool:
        if series is None or len(series) == 0:
            return False
        counts = series.astype(str).value_counts()
        n_groups = len(counts)
        if n_groups < 2 or n_groups > self.max_groups:
            return False
        if counts.max() >= 0.98 * len(series):
            return False
        if (counts >= self.min_group_size).sum() < 2:
            return False
        return True

    def _add_fallback_groups(self, adata: anndata.AnnData) -> str:
        try:
            sc.tl.leiden(
                adata,
                resolution=self.leiden_resolution,
                key_added="latent_leiden",
                random_state=self.seed,
            )
            return "latent_leiden"
        except Exception:
            from sklearn.cluster import KMeans

            n_clusters = int(np.clip(np.sqrt(adata.n_obs / 150.0), 3, 12))
            labels = KMeans(
                n_clusters=n_clusters,
                random_state=self.seed,
                n_init=10,
            ).fit_predict(adata.obsm["X_latent"])
            adata.obs["latent_kmeans"] = pd.Categorical(labels.astype(str))
            return "latent_kmeans"

    def _choose_group_key(self, adata: anndata.AnnData) -> str:
        if self.cluster_key != "auto":
            if self.cluster_key not in adata.obs:
                raise ValueError(f"cluster_key '{self.cluster_key}' not found in adata.obs")
            return self.cluster_key

        candidates = [
            "cell_type",
            "final_annotation",
            "celltype",
            "state_info",
            "leiden",
        ]
        for key in candidates:
            if key in adata.obs and self._is_valid_group_series(adata.obs[key]):
                return key

        return self._add_fallback_groups(adata)

    def _infer_time_values(
        self, adata: anndata.AnnData, group_key: str
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if self.time_key != "auto":
            candidate_keys = [self.time_key]
        else:
            # Explicit biological time keys (excluding 'batch' as it typically represents technical batches)
            explicit = [
                "time_information",
                "time",
                "stage",
                "day",
                "pseudotime",
                "dpt_pseudotime",
            ]
            # Fuzzy matching for time-related columns (excluding batch-related terms)
            fuzzy = [
                col
                for col in adata.obs.columns
                if any(token in col.lower() for token in ["time", "stage", "day", "pseudo"])
                and "batch" not in col.lower()  # Exclude batch-related columns
            ]
            candidate_keys = explicit + [c for c in fuzzy if c not in explicit]

        group_series = adata.obs[group_key].astype(str)
        for key in candidate_keys:
            if key not in adata.obs:
                continue
            series = adata.obs[key]
            if series.astype(str).equals(group_series):
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() < 0.9:
                continue
            if numeric.nunique(dropna=True) < 2:
                continue
            return numeric.to_numpy(dtype=float), key

        return None, None

    def _build_analysis_adata(
        self,
        data_vis: np.ndarray,
        data_high: np.ndarray,
        adata_src: anndata.AnnData,
    ) -> Tuple[anndata.AnnData, str, Optional[np.ndarray], Optional[str]]:
        adata = anndata.AnnData(X=data_high)
        adata.obsm["X_latent"] = data_high
        adata.obsm["X_vis"] = data_vis
        adata.obs = adata_src.obs.copy()

        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep="X_latent")
        group_key = self._choose_group_key(adata)
        adata.obs[group_key] = adata.obs[group_key].astype("category")

        time_values, time_key = self._infer_time_values(adata, group_key)

        if time_values is not None:
            root_idx = int(np.nanargmin(time_values))
            try:
                sc.tl.diffmap(adata)
                adata.uns["iroot"] = root_idx
                sc.tl.dpt(adata)
            except Exception:
                pass

        sc.tl.paga(adata, groups=group_key)
        return adata, group_key, time_values, time_key

    def _build_cluster_graph(
        self,
        adata: anndata.AnnData,
        group_key: str,
    ) -> Tuple[nx.Graph, List[str], np.ndarray]:
        categories = list(adata.obs[group_key].cat.categories)
        group_series = adata.obs[group_key]
        paga_conn = adata.uns["paga"]["connectivities"]
        conn_dense = paga_conn.toarray() if hasattr(paga_conn, "toarray") else np.asarray(paga_conn)

        pos = np.zeros((len(categories), 2), dtype=float)
        counts: Dict[str, int] = {}
        mean_pt: Dict[str, float] = {}
        cell_pt = (
            adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
            if "dpt_pseudotime" in adata.obs
            else None
        )

        graph = nx.Graph()
        for i, cat in enumerate(categories):
            mask = (group_series == cat).to_numpy()
            counts[cat] = int(mask.sum())
            pos[i] = adata.obsm["X_vis"][mask].mean(axis=0)
            mean_pt[cat] = (
                float(np.nanmean(cell_pt[mask]))
                if cell_pt is not None and np.isfinite(cell_pt[mask]).any()
                else np.nan
            )
            graph.add_node(
                cat,
                pos=pos[i],
                count=counts[cat],
                pseudotime=mean_pt[cat],
            )

        # Apply threshold to filter weak connections for better tree quality
        edge_count = 0
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                weight = float(conn_dense[i, j])
                if weight > self.paga_threshold:
                    graph.add_edge(categories[i], categories[j], weight=weight)
                    edge_count += 1
        
        if self.verbose and edge_count > 0:
            print(f"[BioBranch] Created graph with {len(categories)} nodes and {edge_count} edges (threshold={self.paga_threshold})")

        # Fallback: if no edges pass threshold, use distance-based connections
        if graph.number_of_edges() == 0 and len(categories) > 1:
            if self.verbose:
                print(f"[BioBranch] No PAGA edges above threshold, falling back to distance-based graph")
            centroid_graph = nx.Graph()
            for cat in categories:
                centroid_graph.add_node(cat, **graph.nodes[cat])
            dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            # Connect each cluster to its k nearest neighbors
            k_neighbors = min(3, len(categories) - 1)
            for i in range(len(categories)):
                nearest = np.argsort(dists[i])[1 : k_neighbors + 1]
                for j in nearest:
                    if dists[i, j] > 1e-9:  # Avoid self-connections
                        weight = 1.0 / max(dists[i, j], 1e-6)
                        centroid_graph.add_edge(categories[i], categories[j], weight=weight)
            graph = centroid_graph

        return graph, categories, pos

    def _choose_root(
        self,
        tree: nx.Graph,
        time_values: Optional[np.ndarray],
        time_key: Optional[str],
        adata: anndata.AnnData,
        group_key: str,
    ) -> str:
        """Choose root node with multiple heuristics for robustness."""
        nodes = list(tree.nodes)
        
        # 1. User-specified root (highest priority)
        if self.root_group is not None and str(self.root_group) in tree.nodes:
            if self.verbose:
                print(f"[BioBranch] Using user-specified root: {self.root_group}")
            return str(self.root_group)

        # 2. Time-based root (e.g., earliest timepoint)
        if time_values is not None:
            group_means = {}
            for node in nodes:
                mask = (adata.obs[group_key].astype(str) == node).to_numpy()
                if np.isfinite(time_values[mask]).any():
                    group_means[node] = float(np.nanmean(time_values[mask]))
            if group_means:
                root = min(group_means, key=group_means.get)
                if self.verbose:
                    print(f"[BioBranch] Root chosen by {time_key}: {root} (mean_time={group_means[root]:.3f})")
                return root

        # 3. Pseudotime-based root (from DPT if available)
        valid_pt = {
            node: tree.nodes[node].get("pseudotime", np.nan)
            for node in nodes
            if np.isfinite(tree.nodes[node].get("pseudotime", np.nan))
        }
        if valid_pt:
            root = min(valid_pt, key=valid_pt.get)
            if self.verbose:
                print(f"[BioBranch] Root chosen by pseudotime: {root} (pt={valid_pt[root]:.3f})")
            return root

        # 4. Single node case
        if tree.number_of_nodes() == 1:
            return nodes[0]

        # 5. Topological heuristic: choose node with highest weighted degree (most connected)
        weighted_degree = {node: sum(d["weight"] for _, _, d in tree.edges(node, data=True)) for node in nodes}
        if weighted_degree:
            root = max(weighted_degree, key=weighted_degree.get)
            if self.verbose:
                print(f"[BioBranch] Root chosen by weighted degree: {root} (degree={weighted_degree[root]:.3f})")
            return root

        # 6. Fallback: use closeness centrality
        try:
            centrality = nx.closeness_centrality(tree, distance=lambda u, v, d: 1.0 / max(d["weight"], 1e-6))
            root = max(centrality, key=centrality.get)
            if self.verbose:
                print(f"[BioBranch] Root chosen by centrality: {root}")
            return root
        except Exception:
            return nodes[0]

    def _largest_component_tree(self, graph: nx.Graph) -> nx.Graph:
        """Extract maximum spanning tree from largest connected component.
        
        Ensures tree structure for downstream analysis. If graph is disconnected,
        selects the largest component to maximize biological coverage.
        """
        if graph.number_of_nodes() == 0:
            return graph
        if nx.is_connected(graph):
            return nx.maximum_spanning_tree(graph, weight="weight")

        # Handle disconnected graphs: select largest component
        component_trees = []
        components = list(nx.connected_components(graph))
        if self.verbose and len(components) > 1:
            sizes = [len(c) for c in components]
            print(f"[BioBranch] Graph has {len(components)} components (sizes: {sizes}), using largest")
        
        for component in components:
            sub = graph.subgraph(component).copy()
            component_trees.append(nx.maximum_spanning_tree(sub, weight="weight"))

        component_trees.sort(key=lambda g: g.number_of_nodes(), reverse=True)
        return component_trees[0]

    def _compute_graph_pseudotime(self, tree: nx.Graph, root: str) -> Dict[str, float]:
        lengths = nx.single_source_dijkstra_path_length(
            tree,
            root,
            weight=lambda u, v, d: 1.0 / max(d["weight"], 1e-6),
        )
        max_len = max(lengths.values()) if lengths else 1.0
        if max_len <= 0:
            max_len = 1.0
        return {node: dist / max_len for node, dist in lengths.items()}

    def _extract_backbone_and_branches(
        self, tree: nx.Graph, root: str, node_pt: Dict[str, float]
    ) -> Tuple[List[str], List[Dict[str, object]], Dict[str, str]]:
        """Extract main trajectory (backbone) and branching points.
        
        Algorithm:
          1. Identify all leaf nodes (endpoints)
          2. Select leaf with maximum pseudotime as main trajectory endpoint
          3. Shortest path from root to main leaf defines backbone
          4. DFS from each backbone node to identify side branches
          
        Returns:
            backbone: Ordered list of nodes along main trajectory
            branches: List of branch metadata (label, anchor, nodes, subtree)
            branch_assignments: Mapping from node to branch label
        """
        if tree.number_of_nodes() == 1:
            only = next(iter(tree.nodes))
            return [only], [], {only: "Backbone"}

        # Identify leaf nodes (degree 1, excluding root)
        leaves = [n for n in tree.nodes if tree.degree[n] == 1 and n != root]
        if not leaves:
            # Fallback: if no clear leaves (e.g., cycles), use all non-root nodes
            leaves = [n for n in tree.nodes if n != root]
            if self.verbose:
                print(f"[BioBranch] No leaf nodes found, using {len(leaves)} non-root nodes")

        # Select main leaf: furthest in pseudotime
        main_leaf = max(leaves, key=lambda n: node_pt.get(n, 0.0))
        backbone = nx.shortest_path(tree, root, main_leaf)
        backbone_set = set(backbone)
        
        if self.verbose:
            main_leaf_pt = node_pt.get(main_leaf, 0.0)
            print(f"[BioBranch] Backbone: {root} -> {main_leaf} (pt={main_leaf_pt:.3f}), {len(backbone)} nodes")

        branch_assignments: Dict[str, str] = {node: "Backbone" for node in backbone}
        branches: List[Dict[str, object]] = []

        # Identify branches: connected components off backbone
        branch_index = 0
        for anchor in backbone:
            off_backbone_neighbors = [
                nbr for nbr in tree.neighbors(anchor) if nbr not in backbone_set
            ]
            for nbr in off_backbone_neighbors:
                # DFS to collect all nodes in this branch
                component_nodes = set()
                stack = [nbr]
                seen = {anchor}
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    component_nodes.add(cur)
                    for nxt in tree.neighbors(cur):
                        if nxt not in seen and nxt not in backbone_set:
                            stack.append(nxt)

                if not component_nodes:
                    continue

                branch_label = f"Branch {branch_index + 1}"
                branch_index += 1
                for node in component_nodes:
                    branch_assignments[node] = branch_label

                subtree = tree.subgraph(component_nodes | {anchor}).copy()
                
                # Compute branch statistics
                branch_size = len(component_nodes)
                total_cells = sum(tree.nodes[n]["count"] for n in component_nodes)
                
                branches.append(
                    {
                        "label": branch_label,
                        "anchor": anchor,
                        "nodes": component_nodes,
                        "tree": subtree,
                        "size": branch_size,
                        "total_cells": total_cells,
                    }
                )
                
                if self.verbose:
                    print(f"[BioBranch] {branch_label}: {branch_size} clusters, {total_cells} cells, anchor={anchor}")

        return backbone, branches, branch_assignments

    def _make_edge_trace(
        self,
        graph: nx.Graph,
        color: str,
        width: float,
        name: str,
        showlegend: bool = False,
    ) -> go.Scatter:
        x_vals: List[float] = []
        y_vals: List[float] = []
        for u, v in graph.edges:
            ux, uy = graph.nodes[u]["pos"]
            vx, vy = graph.nodes[v]["pos"]
            x_vals.extend([ux, vx, None])
            y_vals.extend([uy, vy, None])

        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            hoverinfo="skip",
            showlegend=showlegend,
        )

    def _make_branch_structure_fig(
        self,
        adata: anndata.AnnData,
        group_key: str,
        tree: nx.Graph,
        backbone: List[str],
        branches: List[Dict[str, object]],
        root: str,
        time_key: Optional[str],
    ) -> go.Figure:
        fig = go.Figure()
        vis = adata.obsm["X_vis"]
        fig.add_trace(
            go.Scatter(
                x=vis[:, 0],
                y=vis[:, 1],
                mode="markers",
                marker=dict(size=3, color="rgba(180,180,180,0.25)"),
                name="Cells",
                hoverinfo="skip",
            )
        )

        backbone_graph = tree.subgraph(backbone).copy()
        fig.add_trace(
            self._make_edge_trace(
                backbone_graph,
                color="rgba(20,20,20,0.95)",
                width=5,
                name="Backbone",
                showlegend=True,
            )
        )

        for i, branch in enumerate(branches):
            color = BRANCH_COLORS[i % len(BRANCH_COLORS)]
            fig.add_trace(
                self._make_edge_trace(
                    branch["tree"],
                    color=color,
                    width=3,
                    name=branch["label"],
                    showlegend=True,
                )
            )

        node_x = []
        node_y = []
        node_size = []
        node_text = []
        node_color = []
        for node in tree.nodes:
            x, y = tree.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
            # Use logarithmic scaling for better size differentiation
            count = tree.nodes[node]["count"]
            node_size.append(max(12, min(35, 10 + 3 * np.log10(count + 1))))
            pt = tree.nodes[node].get("pseudotime", np.nan)
            pt_text = f"{pt:.3f}" if np.isfinite(pt) else "NA"
            # Enhanced hover information
            node_text.append(
                f"<b>{node}</b><br>"
                f"Cells: {count}<br>"
                f"Pseudotime: {pt_text}<br>"
                f"Position: ({x:.2f}, {y:.2f})"
            )
            if node == root:
                node_color.append("gold")
            elif node in backbone:
                node_color.append("black")
            else:
                node_color.append("white")

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(color="black", width=1.5),
                ),
                text=list(tree.nodes),
                textposition="top center",
                textfont=dict(size=10),
                name="States",
                hovertext=node_text,
                hoverinfo="text",
            )
        )

        title_suffix = f" | root={root}"
        if time_key is not None:
            title_suffix += f" | time={time_key}"
        fig.update_layout(
            title=f"Biological Branch Structure ({group_key}){title_suffix}",
            template="plotly_white",
            width=1000,
            height=750,
            showlegend=True,
            annotations=[
                dict(
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    align="left",
                    showarrow=False,
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=1,
                    bgcolor="rgba(255,255,255,0.88)",
                    font=dict(size=11, color="black"),
                    text=(
                        f"<b>Figure Guide</b><br>"
                        f"Groups: {group_key}<br>"
                        f"Root: {root}<br>"
                        f"Time key: {time_key if time_key is not None else 'None'}<br>"
                        f"Gray dots: all cells in 2D embedding<br>"
                        f"Black edge: backbone path<br>"
                        f"Colored edges: side branches<br>"
                        f"Gold node: root state<br>"
                        f"Black nodes: backbone states<br>"
                        f"White nodes: branch states"
                    ),
                )
            ],
        )
        return fig

    def _make_branch_assignment_fig(
        self,
        adata: anndata.AnnData,
        group_key: str,
        tree: nx.Graph,
        backbone: List[str],
        branches: List[Dict[str, object]],
        branch_assignments: Dict[str, str],
    ) -> go.Figure:
        fig = go.Figure()
        groups = adata.obs[group_key].astype(str).to_numpy()
        vis = adata.obsm["X_vis"]
        branch_names = ["Backbone"] + [branch["label"] for branch in branches]
        color_map = {"Backbone": "rgba(30,30,30,0.75)"}
        for i, name in enumerate(branch_names[1:]):
            color_map[name] = BRANCH_COLORS[i % len(BRANCH_COLORS)]
        color_map["Detached"] = "rgba(160,160,160,0.45)"

        plot_names = branch_names + ["Detached"]
        for name in plot_names:
            mask = np.array([branch_assignments.get(g, "Detached") == name for g in groups])
            if mask.sum() == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=vis[mask, 0],
                    y=vis[mask, 1],
                    mode="markers",
                    marker=dict(size=4, color=color_map[name], opacity=0.6),
                    name=name,
                    hoverinfo="skip",
                )
            )

        backbone_graph = tree.subgraph(backbone).copy()
        fig.add_trace(
            self._make_edge_trace(
                backbone_graph,
                color="rgba(0,0,0,0.95)",
                width=4,
                name="Backbone Skeleton",
                showlegend=False,
            )
        )
        for i, branch in enumerate(branches):
            fig.add_trace(
                self._make_edge_trace(
                    branch["tree"],
                    color=BRANCH_COLORS[i % len(BRANCH_COLORS)],
                    width=2.5,
                    name=branch["label"],
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Cell-Level Branch Assignment ({group_key})",
            template="plotly_white",
            width=1000,
            height=750,
            showlegend=True,
            annotations=[
                dict(
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    align="left",
                    showarrow=False,
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=1,
                    bgcolor="rgba(255,255,255,0.88)",
                    font=dict(size=11, color="black"),
                    text=(
                        f"<b>Figure Guide</b><br>"
                        f"Groups: {group_key}<br>"
                        f"Cell colors indicate branch assignment<br>"
                        f"Backbone: main developmental path<br>"
                        f"Branch i: cells inherited from branch-linked groups<br>"
                        f"Detached: groups outside the extracted tree"
                    ),
                )
            ],
        )
        return fig

    def _make_pseudotime_fig(
        self,
        adata: anndata.AnnData,
        group_key: str,
        node_pt: Dict[str, float],
        time_key: Optional[str],
    ) -> Optional[go.Figure]:
        if "dpt_pseudotime" in adata.obs:
            cell_pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
            title = "Diffusion Pseudotime on Latent Embedding"
        else:
            groups = adata.obs[group_key].astype(str).to_numpy()
            cell_pt = np.array([node_pt.get(group, np.nan) for group in groups], dtype=float)
            title = "Graph-Distance Pseudotime on Latent Embedding"

        if not np.isfinite(cell_pt).any():
            return None

        vis = adata.obsm["X_vis"]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=vis[:, 0],
                    y=vis[:, 1],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=cell_pt,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Pseudotime"),
                        opacity=0.7,
                    ),
                    hoverinfo="skip",
                )
            ]
        )
        if time_key is not None:
            title += f" | root inferred from {time_key}"
        fig.update_layout(
            title=title,
            template="plotly_white",
            width=1000,
            height=750,
            showlegend=False,
            annotations=[
                dict(
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top",
                    align="left",
                    showarrow=False,
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=1,
                    bgcolor="rgba(255,255,255,0.88)",
                    font=dict(size=11, color="black"),
                    text=(
                        f"<b>Figure Guide</b><br>"
                        f"Groups: {group_key}<br>"
                        f"Time key: {time_key if time_key is not None else 'None'}<br>"
                        f"Color encodes pseudotime value<br>"
                        f"If DPT exists: cell-level diffusion pseudotime<br>"
                        f"Else: group-level graph-distance pseudotime"
                    ),
                )
            ],
        )
        return fig

    def _save_html(self, fig: go.Figure, filename: str) -> None:
        path = Path(self.output_dir) / filename
        path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._is_plot_epoch(trainer, pl_module):
            return

        run = self._get_wandb_run(trainer)
        adata_src = getattr(trainer.datamodule, "adata", None)
        if adata_src is None:
            return

        data_vis, data_high = self._extract_embeddings(trainer, pl_module)
        finite_mask = np.isfinite(data_vis).all(axis=1) & np.isfinite(data_high).all(axis=1)
        data_vis = data_vis[finite_mask]
        data_high = data_high[finite_mask]
        adata_src = adata_src[finite_mask, :].copy()

        if len(data_vis) < 50:
            return

        data_vis, data_high, adata_src = self._subsample(data_vis, data_high, adata_src)

        analysis_adata, group_key, time_values, time_key = self._build_analysis_adata(
            data_vis=data_vis,
            data_high=data_high,
            adata_src=adata_src,
        )

        graph, _, _ = self._build_cluster_graph(analysis_adata, group_key)
        tree = self._largest_component_tree(graph)
        if tree.number_of_nodes() < 2:
            return

        root = self._choose_root(tree, time_values, time_key, analysis_adata, group_key)
        node_pt = self._compute_graph_pseudotime(tree, root)

        for node, pt in node_pt.items():
            tree.nodes[node]["pseudotime"] = pt

        backbone, branches, branch_assignments = self._extract_backbone_and_branches(
            tree,
            root,
            node_pt,
        )

        structure_fig = self._make_branch_structure_fig(
            analysis_adata,
            group_key,
            tree,
            backbone,
            branches,
            root,
            time_key,
        )
        assignment_fig = self._make_branch_assignment_fig(
            analysis_adata,
            group_key,
            tree,
            backbone,
            branches,
            branch_assignments,
        )
        pseudotime_fig = self._make_pseudotime_fig(
            analysis_adata,
            group_key,
            node_pt,
            time_key,
        )

        epoch_num = trainer.current_epoch + 1
        self._save_html(structure_fig, f"bio_branch_structure_epoch_{epoch_num:04d}.html")
        self._save_html(assignment_fig, f"bio_branch_assignment_epoch_{epoch_num:04d}.html")
        if pseudotime_fig is not None:
            self._save_html(pseudotime_fig, f"bio_branch_pseudotime_epoch_{epoch_num:04d}.html")

        if run is not None and wandb is not None:
            # Compute quality metrics
            n_backbone_nodes = len(backbone)
            n_branch_nodes = sum(len(b["nodes"]) for b in branches)
            backbone_ratio = n_backbone_nodes / tree.number_of_nodes() if tree.number_of_nodes() > 0 else 0
            
            # Compute average edge weight (connectivity strength)
            edge_weights = [d["weight"] for _, _, d in tree.edges(data=True)]
            avg_edge_weight = float(np.mean(edge_weights)) if edge_weights else 0.0
            
            log_dict = {
                "bio_branch/structure": wandb.Html(
                    structure_fig.to_html(include_plotlyjs="cdn", full_html=False)
                ),
                "bio_branch/assignment": wandb.Html(
                    assignment_fig.to_html(include_plotlyjs="cdn", full_html=False)
                ),
                "bio_branch/n_groups": int(tree.number_of_nodes()),
                "bio_branch/n_branches": int(len(branches)),
                "bio_branch/n_backbone_nodes": n_backbone_nodes,
                "bio_branch/n_branch_nodes": n_branch_nodes,
                "bio_branch/backbone_ratio": backbone_ratio,
                "bio_branch/avg_edge_weight": avg_edge_weight,
                "bio_branch/root_group": root,
                "bio_branch/group_key": group_key,
            }
            if time_key is not None:
                log_dict["bio_branch/time_key"] = time_key
            if pseudotime_fig is not None:
                log_dict["bio_branch/pseudotime"] = wandb.Html(
                    pseudotime_fig.to_html(include_plotlyjs="cdn", full_html=False)
                )
            run.log(log_dict, step=safe_wandb_step(trainer, run))
            
            if self.verbose:
                print(f"[BioBranch] Logged metrics: {tree.number_of_nodes()} groups, "
                      f"{len(branches)} branches, backbone_ratio={backbone_ratio:.2f}")
