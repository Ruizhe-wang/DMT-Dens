"""True developmental-tree backbone overlay callback (dynGen fate case study).

Renders the ground-truth lineage tree *on top of* the learned 2D embedding, which
is the visual proof for sub-claim **C1 (topology preservation)** in
``configs/dyngen/README.md``:

  * background scatter of all cells, colored by pseudotime (continuous viridis) or
    by a categorical key;
  * the ground-truth tree drawn as nodes + edges, where each milestone node is
    placed at the centroid of its incident cells in embedding space and edges
    follow the true ``(from -> to)`` topology.

If the method preserves topology, the overlaid tree grows cleanly along the data
arms with no edge crossings; on a method that tangles the trajectory the same tree
self-intersects. The tree is reconstructed **directly from** ``adata.obs['from']`` /
``adata.obs['to']`` (no external gt.json needed); if those are absent it falls back
to parsing an ``"u->v"`` branch column. ``milestone_percentage`` is intentionally
not used — in the shared dynGen files it is all-NaN, so node positions come from
unweighted incident-cell centroids.
"""

import os

import lightning as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


class TrueTreeOverlayVisualizationCallback(pl.Callback):
    def __init__(
        self,
        output_dir="outputs/case_study",
        every_n_epochs=None,
        dataset_name="",
        method_name="",
        max_plot_samples=40000,
        save_formats=None,
        dpi=800,
        point_size=4.0,
        alpha=0.7,
        panel_prefix="case_study",
        background_color_by="pseudotime",
        background_cmap="viridis",
        min_cells_per_node=10,
        node_size=70.0,
        edge_width=1.8,
        edge_color="#222222",
        node_placement="mean",
        show_labels=True,
        label_terminals_only=False,
        from_key="from",
        to_key="to",
        branch_key="branch",
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
        self.background_color_by = background_color_by
        self.background_cmap = background_cmap
        self.min_cells_per_node = int(min_cells_per_node)
        self.node_size = node_size
        self.edge_width = edge_width
        self.edge_color = edge_color
        self.node_placement = node_placement
        self.show_labels = show_labels
        self.label_terminals_only = label_terminals_only
        self.from_key = from_key
        self.to_key = to_key
        self.branch_key = branch_key

    # ------------------------------------------------------------------ utils
    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _style_axis(self, fig, ax):
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.02)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    def _save_fig(self, fig, path, fmt):
        fig.savefig(path, format=fmt, transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=self.dpi)

    def get_our_visualization(self, trainer, pl_module):
        import inspect

        data_list = []
        with torch.inference_mode():
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
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
                    lat_vis_list = [torch.as_tensor(model_output, dtype=torch.float32, device=data_input_item.device)]

                data_list.append(torch.stack([lv.detach().cpu() for lv in lat_vis_list], dim=0).float())

        return torch.cat(data_list, dim=1)  # [n_layers, n_cells, dim]

    # ---------------------------------------------------------------- tree ops
    def _edges_from_obs(self, adata):
        """Return a sorted unique list of (from_node, to_node) ground-truth edges."""
        obs = adata.obs
        if self.from_key in obs and self.to_key in obs:
            pairs = pd.DataFrame(
                {"u": obs[self.from_key].astype(str).values, "v": obs[self.to_key].astype(str).values}
            )
        elif self.branch_key in obs:
            split = obs[self.branch_key].astype(str).str.split("->", n=1, expand=True)
            if split.shape[1] < 2:
                return []
            pairs = pd.DataFrame({"u": split[0].str.strip(), "v": split[1].str.strip()})
        else:
            return []
        pairs = pairs[(pairs["u"] != "nan") & (pairs["v"] != "nan") & (pairs["u"] != pairs["v"])]
        return sorted(set(map(tuple, pairs.values.tolist())))

    @staticmethod
    def _node_point(coords, mode):
        """Single representative point for a node's incident cells.

        ``mean`` is the plain centroid. ``medoid`` returns the actual cell
        nearest the coordinate-wise median (always on data). Note: on the dynGen
        1000-epoch embedding, medoid did NOT shorten edges (total drawn-edge
        length 320->349, max 49->65) — the long cross-tree edges are genuine
        embedding structure on the rare lineages, not a centroid artifact — so
        ``mean`` is the default.
        """
        if mode == "mean" or coords.shape[0] <= 2:
            return coords.mean(axis=0)
        med = np.median(coords, axis=0)
        j = int(np.argmin(((coords - med) ** 2).sum(axis=1)))
        return coords[j]

    def _node_positions(self, adata, emb):
        """Embedding representative point + incident-cell count per milestone node."""
        obs = adata.obs
        members = {}
        if self.from_key in obs and self.to_key in obs:
            f = obs[self.from_key].astype(str).values
            t = obs[self.to_key].astype(str).values
        elif self.branch_key in obs:
            split = obs[self.branch_key].astype(str).str.split("->", n=1, expand=True)
            f = split[0].str.strip().values
            t = (split[1].str.strip().values if split.shape[1] > 1 else np.array(["nan"] * len(obs)))
        else:
            return {}, {}

        nodes = set(f.tolist()) | set(t.tolist())
        nodes.discard("nan")
        positions, counts = {}, {}
        for n in nodes:
            mask = (f == n) | (t == n)
            c = int(mask.sum())
            if c == 0:
                continue
            counts[n] = c
            positions[n] = self._node_point(emb[mask], self.node_placement)
        return positions, counts

    def _node_roles(self, edges):
        out_deg, in_deg = {}, {}
        for u, v in edges:
            out_deg[u] = out_deg.get(u, 0) + 1
            in_deg[v] = in_deg.get(v, 0) + 1
        roles = {}
        for n in set(out_deg) | set(in_deg):
            if in_deg.get(n, 0) == 0:
                roles[n] = "root"
            elif out_deg.get(n, 0) == 0:
                roles[n] = "terminal"
            elif out_deg.get(n, 0) >= 2:
                roles[n] = "branchpoint"
            else:
                roles[n] = "internal"
        return roles

    def _draw_background(self, fig, ax, adata, emb):
        key = self.background_color_by
        if key and key in adata.obs and pd.api.types.is_numeric_dtype(adata.obs[key]) \
                and not pd.api.types.is_bool_dtype(adata.obs[key]):
            values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="mergesort")
            sca = ax.scatter(
                emb[order, 0], emb[order, 1], c=values[order], cmap=self.background_cmap,
                s=self.point_size, alpha=self.alpha, linewidths=0, edgecolors="none", rasterized=True,
            )
            cbar = fig.colorbar(sca, ax=ax, fraction=0.036, pad=0.01)
            cbar.outline.set_visible(False)
            cbar.ax.patch.set_alpha(0.0)
            cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
            cbar.set_label(key, fontsize=9)
        else:
            ax.scatter(
                emb[:, 0], emb[:, 1], c="#cfcfcf",
                s=self.point_size, alpha=self.alpha, linewidths=0, edgecolors="none", rasterized=True,
            )

    _NODE_STYLE = {
        "root": dict(marker="*", s_mul=3.0, c="#2ca02c", edge="black"),
        "branchpoint": dict(marker="D", s_mul=1.3, c="#d62728", edge="black"),
        "terminal": dict(marker="o", s_mul=1.1, c="#1f1f1f", edge="white"),
        "internal": dict(marker="o", s_mul=0.5, c="#777777", edge="none"),
    }

    def plot_true_tree(self, adata_bg, emb_bg, positions, counts, edges, text=""):
        """Draw the ground-truth tree (``positions``/``counts`` from full data) over a
        background scatter (``adata_bg``/``emb_bg``, possibly downsampled)."""
        roles = self._node_roles(edges)

        fig, ax = plt.subplots(figsize=(8, 8))
        self._style_axis(fig, ax)
        self._draw_background(fig, ax, adata_bg, emb_bg)

        def placeable(n):
            return n in positions and counts.get(n, 0) >= self.min_cells_per_node

        for u, v in edges:
            if placeable(u) and placeable(v):
                ax.plot(
                    [positions[u][0], positions[v][0]], [positions[u][1], positions[v][1]],
                    color=self.edge_color, lw=self.edge_width, alpha=0.9, zorder=3, solid_capstyle="round",
                )

        for n, p in positions.items():
            if counts.get(n, 0) < self.min_cells_per_node:
                continue
            st = self._NODE_STYLE.get(roles.get(n, "internal"), self._NODE_STYLE["internal"])
            ax.scatter([p[0]], [p[1]], marker=st["marker"], s=self.node_size * st["s_mul"],
                       c=st["c"], edgecolors=st["edge"], linewidths=0.6, zorder=4)
            if self.show_labels and not (self.label_terminals_only and roles.get(n) == "internal"):
                ax.annotate(
                    n, (p[0], p[1]), textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
                )

        os.makedirs(self.output_dir, exist_ok=True)
        for fmt in self.save_formats:
            self._save_fig(fig, os.path.join(self.output_dir, f"case_study_truetree{text}.{fmt}"), fmt)
        plt.close(fig)

    # ----------------------------------------------------------------- hooks
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
        if self.every_n_epochs and self.every_n_epochs > 0 and epoch_num % self.every_n_epochs != 0:
            return

        adata = trainer.datamodule.adata
        edges = self._edges_from_obs(adata)
        if not edges:
            return

        sc.set_figure_params(dpi=100, facecolor="white", frameon=False)
        lat_vis = self.get_our_visualization(trainer, pl_module)  # [layers, n, dim]

        for i in range(lat_vis.shape[0]):
            emb_full = lat_vis[i].detach().cpu().numpy()
            # node centroids always from full embedding; only background scatter is downsampled
            positions, counts = self._node_positions(adata, emb_full)
            if emb_full.shape[0] > self.max_plot_samples:
                idx = np.random.RandomState(0).choice(emb_full.shape[0], self.max_plot_samples, replace=False)
                adata_bg, emb_bg = adata[idx].copy(), emb_full[idx]
            else:
                adata_bg, emb_bg = adata, emb_full
            self.plot_true_tree(
                adata_bg,
                emb_bg,
                positions,
                counts,
                edges,
                text=f"_layer{i}",
            )
