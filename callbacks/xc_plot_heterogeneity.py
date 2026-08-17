import os
import torch
import numpy as np
import lightning as pl
import plotly.graph_objects as go
from typing import Optional, Any
from sklearn.neighbors import NearestNeighbors
from callbacks.wandb_utils import safe_wandb_step

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class HeterogeneityPlotCallback(pl.Callback):
    """
    Visualizes Fidelity Metrics:
    - Local Density Correlation (LDC) via HD and LD density maps.
    - Scattered Point Intrusion Rate (SPIR) via noise intrusion maps.
    """

    DENSITY_COLORSCALE = "Viridis_r"
    INTRUSION_COLORSCALE = "Reds"

    def __init__(
        self,
        output_dir: str = "outputs/plots/fidelity_vis",
        every_n_epochs: Optional[int] = None,
        target_label: Optional[int] = None,  # kept for yaml backward compatibility
        density_k: int = 15,
        knn_k: int = 12,
        noise_quantile: float = 0.9,
        max_samples: int = 3000,
        seed: int = 42,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.target_label = target_label
        self.density_k = density_k
        self.knn_k = knn_k
        self.noise_quantile = noise_quantile
        self.max_samples = max_samples
        self.seed = seed
        self._temp_data = {"high_dim": [], "low_dim": []}

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None
        return getattr(logger, "experiment", None)

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _is_plot_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> bool:
        if not trainer.is_global_zero:
            return False
        if getattr(trainer, "sanity_checking", False):
            return False
        
        epoch_num = trainer.current_epoch + 1
        
        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
            and not self._is_baseline_model(pl_module)
        ):
            return False

        if (
            self.every_n_epochs is not None
            and self.every_n_epochs > 0
            and epoch_num % self.every_n_epochs != 0
        ):
            return False
        return True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._is_plot_epoch(trainer, pl_module):
            return

        with torch.inference_mode():
            import inspect
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values())

            data_input_item = batch["data_input_item"]
            
            if supports_tau:
                tau = getattr(pl_module.hparams, "tau", 1.0) if hasattr(pl_module, "hparams") else 1.0
                model_output = pl_module(data_input_item, tau=tau)
            else:
                model_output = pl_module(data_input_item)
            
            if isinstance(model_output, tuple) and len(model_output) >= 4:
                # model_output[3] is typically lat_vis_list in the DiffTree codebase
                lat_vis_list = model_output[3]
                lat_vis_best = lat_vis_list[-1]
            elif isinstance(model_output, tuple) and len(model_output) >= 3:
                lat_vis_best = model_output[2]
            else:
                lat_vis_best = torch.as_tensor(model_output, dtype=torch.float32)

            self._temp_data["high_dim"].append(data_input_item.detach().float().cpu())
            self._temp_data["low_dim"].append(lat_vis_best.detach().float().cpu())

    def _compute_metrics_for_vis(self, hd_data: np.ndarray, ld_data: np.ndarray):
        n = len(hd_data)
        
        # 1. Density (LDC)
        dk = min(self.density_k, n - 1)
        nn_hd_d = NearestNeighbors(n_neighbors=dk + 1, algorithm="auto").fit(hd_data)
        hd_density_dist = nn_hd_d.kneighbors(hd_data, return_distance=True)[0][:, dk]
        
        nn_ld_d = NearestNeighbors(n_neighbors=dk + 1, algorithm="auto").fit(ld_data)
        ld_density_dist = nn_ld_d.kneighbors(ld_data, return_distance=True)[0][:, dk]

        # 2. SPIR
        kk = min(self.knn_k, n - 1)
        nn_hd_k = NearestNeighbors(n_neighbors=kk + 1, algorithm="auto").fit(hd_data)
        hd_k_dist = nn_hd_k.kneighbors(hd_data, return_distance=True)[0][:, kk]
        
        threshold = np.quantile(hd_k_dist, self.noise_quantile)
        is_noise = hd_k_dist >= threshold
        
        nn_ld_k = NearestNeighbors(n_neighbors=kk + 1, algorithm="auto").fit(ld_data)
        ld_neighbors = nn_ld_k.kneighbors(ld_data, return_distance=False)[:, 1:]
        
        intrusion_rates = np.zeros(n, dtype=float)
        if np.any(is_noise):
            noise_indices = np.where(is_noise)[0]
            emb_neighbors_of_noise = ld_neighbors[noise_indices]
            # Count non-noise neighbors
            intrusion_counts = np.sum(~is_noise[emb_neighbors_of_noise], axis=1)
            intrusion_rates[noise_indices] = intrusion_counts / kk

        return hd_density_dist, ld_density_dist, is_noise, intrusion_rates

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._is_plot_epoch(trainer, pl_module):
            self._temp_data = {"high_dim": [], "low_dim": []}
            return

        if not self._temp_data["high_dim"]:
            return

        hd_data = torch.cat(self._temp_data["high_dim"], dim=0).numpy()
        ld_data = torch.cat(self._temp_data["low_dim"], dim=0).numpy()
        self._temp_data = {"high_dim": [], "low_dim": []}

        # Remove infinite/NaN values
        finite_mask = np.isfinite(ld_data).all(axis=1) & np.isfinite(hd_data).all(axis=1)
        hd_data = hd_data[finite_mask]
        ld_data = ld_data[finite_mask]

        # Subsample if too large
        if self.max_samples and len(hd_data) > self.max_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(hd_data), size=self.max_samples, replace=False)
            hd_data = hd_data[idx]
            ld_data = ld_data[idx]
            
        if len(hd_data) < 3:
            return

        hd_dens, ld_dens, is_noise, intrusion_rates = self._compute_metrics_for_vis(hd_data, ld_data)

        # Normalize density to Percentile Rank [0, 1] to ensure comparable color scales
        hd_dens_norm = np.argsort(np.argsort(hd_dens)) / float(len(hd_dens))
        ld_dens_norm = np.argsort(np.argsort(ld_dens)) / float(len(ld_dens))
        eps = 1e-6
        distortion_ratio = np.log2((ld_dens_norm + eps) / (hd_dens_norm + eps))

        epoch_num = trainer.current_epoch + 1
        os.makedirs(self.output_dir, exist_ok=True)

        # Plot 1: HD Density (Ground Truth)
        fig_hd_dens = go.Figure(data=[go.Scatter(
            x=ld_data[:, 0], y=ld_data[:, 1], mode="markers",
            marker=dict(
                size=3,
                color=hd_dens_norm,
                colorscale=self.DENSITY_COLORSCALE,
                colorbar=dict(title="HD Sparsity (Percentile)"),
                showscale=True,
                cmin=0.0, cmax=1.0
            )
        )])
        fig_hd_dens.update_layout(
            title=f"Ground Truth Density (Epoch {epoch_num})<br><sup>Dark/Purple=Sparse (Severe), Yellow=Dense</sup>", 
            template="plotly_white",
            width=800, height=600
        )

        # Plot 2: LD Density (Embedded Density)
        fig_ld_dens = go.Figure(data=[go.Scatter(
            x=ld_data[:, 0], y=ld_data[:, 1], mode="markers",
            marker=dict(
                size=3,
                color=ld_dens_norm,
                colorscale=self.DENSITY_COLORSCALE,
                colorbar=dict(title="LD Sparsity (Percentile)"),
                showscale=True,
                cmin=0.0, cmax=1.0
            )
        )])
        fig_ld_dens.update_layout(
            title=f"Embedded Density (Epoch {epoch_num})<br><sup>Ideal: Color distribution matches Ground Truth</sup>", 
            template="plotly_white",
            width=800, height=600
        )

        # Plot 3: SPIR (Scattered Point Intrusion)
        fig_spir = go.Figure()
        # Plot main cluster points in gray
        fig_spir.add_trace(go.Scatter(
            x=ld_data[~is_noise, 0], y=ld_data[~is_noise, 1], mode="markers",
            marker=dict(size=2, color="lightgray"),
            name="Main Clusters (Dense)"
        ))
        # Plot noise points colored by intrusion rate
        fig_spir.add_trace(go.Scatter(
            x=ld_data[is_noise, 0], y=ld_data[is_noise, 1], mode="markers",
            marker=dict(
                size=4,
                color=intrusion_rates[is_noise],
                colorscale=self.INTRUSION_COLORSCALE,
                colorbar=dict(title="Intrusion Rate"),
                showscale=True,
                cmin=0.0, cmax=1.0
            ),
            name="Scattered Points (Noise)"
        ))
        fig_spir.update_layout(
            title=f"Scattered Point Intrusion (Epoch {epoch_num})<br><sup>Dark Red=High Intrusion (Severe), White=Low Intrusion (Good)</sup>", 
            template="plotly_white",
            width=800, height=600
        )

        # Plot 4: Distortion Ratio
        # Determine symmetric color scale range
        max_abs_dist = max(np.max(np.abs(distortion_ratio)), eps)
        fig_distortion = go.Figure(data=[go.Scatter(
            x=ld_data[:, 0], y=ld_data[:, 1], mode="markers",
            marker=dict(
                size=3,
                color=distortion_ratio,
                colorscale="RdBu_r", # Blue for compression (<0), Red for expansion (>0)
                colorbar=dict(title="Log2(LD_norm / HD_norm)"),
                showscale=True,
                cmin=-max_abs_dist,
                cmax=max_abs_dist
            )
        )])
        fig_distortion.update_layout(
            title=f"Distortion Ratio (Epoch {epoch_num})<br><sup>Red=Expanded, Blue=Compressed, White=Preserved</sup>", 
            template="plotly_white",
            width=800, height=600
        )

        # Save HTMLs
        fig_hd_dens.write_html(os.path.join(self.output_dir, f"epoch_{epoch_num:04d}_hd_density.html"))
        fig_ld_dens.write_html(os.path.join(self.output_dir, f"epoch_{epoch_num:04d}_ld_density.html"))
        fig_spir.write_html(os.path.join(self.output_dir, f"epoch_{epoch_num:04d}_spir.html"))
        fig_distortion.write_html(os.path.join(self.output_dir, f"epoch_{epoch_num:04d}_distortion.html"))

        # Log to Wandb
        run = self._get_wandb_run(trainer)
        if run is not None and wandb is not None:
            run.log({
                "fidelity_vis/hd_density_ground_truth": wandb.Plotly(fig_hd_dens),
                "fidelity_vis/ld_density_embedded": wandb.Plotly(fig_ld_dens),
                "fidelity_vis/scattered_point_intrusion": wandb.Plotly(fig_spir),
                "fidelity_vis/distortion_ratio": wandb.Plotly(fig_distortion),
            }, step=safe_wandb_step(trainer, run))
