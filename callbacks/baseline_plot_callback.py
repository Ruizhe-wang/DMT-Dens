"""
Baseline Visualization Callback for t-SNE and UMAP

Generates global embedding visualizations for baseline dimensionality reduction methods.
Designed for simplicity and clarity - shows the full dataset embedding colored by labels.

Usage in config:
    callbacks:
      - class_path: callbacks.baseline_plot_callback.BaselinePlotCallback
        init_args:
          label_key: 'cell_type'
          output_dir: 'outputs/results/baseline'
"""

import os
import numpy as np
import torch
import lightning as pl
import wandb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for consistent visualization
COLORS = px.colors.qualitative.D3 + px.colors.qualitative.Set2


class BaselinePlotCallback(pl.Callback):
    """
    Visualization callback for baseline models (t-SNE/UMAP).

    Generates:
      - Global embedding scatter plot colored by labels
      - Per-class distribution visualization
      - Embedding statistics

    Args:
        label_key: Key for labels in adata.obs (default: 'cell_type')
        output_dir: Directory for saving local copies (default: 'outputs/results/baseline')
        downsample: Max samples to plot for performance (default: 40000)
    """

    def __init__(
        self,
        label_key: str = 'cell_type',
        output_dir: str = 'outputs/results/baseline',
        downsample: int = 40000,
        label_sets: list = None,
    ):
        super().__init__()
        self.label_key = label_key
        self.output_dir = output_dir
        self.downsample = downsample
        self.label_sets = label_sets
        os.makedirs(output_dir, exist_ok=True)

    def _is_baseline_model(self, pl_module) -> bool:
        """Check if model is a baseline (t-SNE/UMAP) model."""
        return (hasattr(pl_module, 'validation_step_outputs_vis') and
                pl_module.validation_step_outputs_vis is not None)

    def _get_method_name(self, pl_module) -> str:
        """Get the method name from model."""
        if hasattr(pl_module, 'method'):
            from model.baseline_tri import DMTEVT_model
            return DMTEVT_model.get_method_display_name(pl_module.method)
        return 'Baseline'

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate visualizations at end of training."""
        if not self._is_baseline_model(pl_module):
            print("[BaselinePlotCallback] Not a baseline model, skipping.")
            return

        print(f"\n[BaselinePlotCallback] Generating visualizations...")

        try:
            log_dict = self._generate_plots(trainer, pl_module)

            if log_dict and trainer.logger is not None:
                trainer.logger.experiment.log(log_dict)
                print(f"[BaselinePlotCallback] Logged {len(log_dict)} plots to WandB")
        except Exception as e:
            import traceback
            print(f"[BaselinePlotCallback] Error: {e}")
            traceback.print_exc()

    def _generate_plots(self, trainer, pl_module) -> dict:
        """Generate all visualization plots."""
        log_dict = {}

        # Get data
        embedding = pl_module.validation_step_outputs_vis
        high_dim = pl_module.validation_step_outputs_high
        adata = getattr(trainer.datamodule, 'adata', None)
        method_name = self._get_method_name(pl_module)

        # Convert to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if isinstance(high_dim, torch.Tensor):
            high_dim = high_dim.cpu().numpy()

        # Get labels
        labels = self._get_labels(pl_module, adata, len(embedding))

        n_samples = len(embedding)
        n_classes = len(np.unique(labels))
        print(f"[BaselinePlotCallback] {method_name}: {n_samples} samples, {n_classes} classes")

        # Downsample if needed
        if n_samples > self.downsample:
            indices = np.random.choice(n_samples, self.downsample, replace=False)
            embedding = embedding[indices]
            labels = labels[indices]
            print(f"[BaselinePlotCallback] Downsampled to {self.downsample} samples")

        # Generate main scatter plot
        main_fig = self._plot_global_embedding(embedding, labels, method_name)
        log_dict[f'{method_name.lower()}/global_embedding'] = wandb.Html(main_fig.to_html(include_plotlyjs='cdn', full_html=False))

        # Generate per-class subplots
        class_fig = self._plot_per_class(embedding, labels, method_name)
        log_dict[f'{method_name.lower()}/per_class_view'] = wandb.Html(class_fig.to_html(include_plotlyjs='cdn', full_html=False))

        return log_dict

    def _get_labels(self, pl_module, adata, n_samples) -> np.ndarray:
        """Extract labels from model or adata."""
        labels = None

        # Try model's stored labels
        if hasattr(pl_module, '_labels') and pl_module._labels is not None:
            labels = pl_module._labels
        # Try adata
        elif adata is not None and self.label_key in adata.obs.columns:
            labels = adata.obs[self.label_key].values[:n_samples]

        if labels is None:
            labels = np.zeros(n_samples, dtype=int)

        # Normalize to integers
        if hasattr(labels, 'dtype') and labels.dtype.kind in ['U', 'S', 'O']:
            try:
                labels = np.array([int(l) for l in labels])
            except:
                unique = sorted(set(labels))
                label_map = {l: i for i, l in enumerate(unique)}
                labels = np.array([label_map[l] for l in labels])

        return labels.astype(int)

    def _plot_global_embedding(self, embedding: np.ndarray, labels: np.ndarray, method_name: str) -> go.Figure:
        """Create main global embedding scatter plot."""
        unique_labels = sorted(np.unique(labels))

        fig = go.Figure()

        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = COLORS[i % len(COLORS)]

            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.7),
                name=f'Class {label}',
                hovertemplate=f'Class {label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>',
            ))

        fig.update_layout(
            title=dict(
                text=f'<b>{method_name} Global Embedding</b><br>'
                     f'<span style="font-size:12px; color:#666;">'
                     f'{len(embedding)} samples | {len(unique_labels)} classes</span>',
                x=0.5,
                xanchor='center',
            ),
            xaxis_title=f'{method_name} 1',
            yaxis_title=f'{method_name} 2',
            width=900,
            height=700,
            plot_bgcolor='white',
            legend=dict(
                orientation='v',
                x=1.02,
                y=0.5,
                yanchor='middle',
                title=dict(text='<b>Classes</b>'),
            ),
            margin=dict(l=60, r=150, t=80, b=60),
        )

        fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', zeroline=False)

        return fig

    def _plot_per_class(self, embedding: np.ndarray, labels: np.ndarray, method_name: str) -> go.Figure:
        """Create per-class subplot grid."""
        
        # Determine items to plot (either label sets or individual classes)
        if self.label_sets:
            # If label_sets provided, each subplot is a set of labels
            plot_items = self.label_sets
            titles = [f'Group {s}' for s in plot_items]
        else:
            # Fallback: one subplot per unique label
            unique_labels = sorted(np.unique(labels))
            plot_items = [[l] for l in unique_labels]
            titles = [f'Class {l}' for l in unique_labels]

        n_items = len(plot_items)
        
        # Determine grid size
        n_cols = min(4, n_items)
        n_rows = (n_items + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles,
            horizontal_spacing=0.03,
            vertical_spacing=0.08,
        )

        # Global range for consistent axes
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05

        for i, item_labels in enumerate(plot_items):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Normalize item_labels to list if not already
            if not isinstance(item_labels, (list, tuple)):
                item_labels = [item_labels]

            # Background: all points in gray
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(size=2, color='lightgray', opacity=0.3),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row, col=col
            )

            # Foreground: highlight classes in this set
            for lbl in item_labels:
                mask = labels == int(lbl) # Ensure type match
                # Color based on label value to be consistent across plots
                color = COLORS[int(lbl) % len(COLORS)]
                
                fig.add_trace(
                    go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode='markers',
                        marker=dict(size=3, color=color, opacity=0.8),
                        name=f'Class {lbl}',
                        showlegend=False,
                        hovertemplate=f'Class {lbl}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>',
                    ),
                    row=row, col=col
                )

            # Set consistent axes
            fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=row, col=col, showticklabels=False)
            fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], row=row, col=col, showticklabels=False)

        fig.update_layout(
            title=dict(
                text=f'<b>{method_name} Per-Group View</b>',
                x=0.5,
                xanchor='center',
            ),
            width=200 * n_cols + 100,
            height=200 * n_rows + 100,
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40),
        )

        return fig
