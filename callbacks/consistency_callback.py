import copy
import gc
import io
import os
import warnings
import numpy as np
import torch
import lightning as pl
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.manifold import TSNE, trustworthiness
import umap as umap_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from callbacks.local_global_consistency import (
    LocalGlobalConsistencyVisualizer,
    D3_COLORS
)
from callbacks.consistency_util import (
    create_subset_datamodule,
    extract_datamodule_init_args,
)
from callbacks.scaling_consistency_metrics import knn_preservation_rate


class ConsistencyCallback(pl.Callback):
    """
    Consistency callback for comparing global vs local embeddings.

    For DiffTree:
        - Global: trained model's embeddings on all data
        - Local: fresh model (same config, same pipeline) trained on subset
        - Compares consistency across selected layers
        - Computes scale-sensitive contraction metrics

    For baselines (t-SNE/UMAP):
        - Global: fitted on all data
        - Local: fresh fit on subset

    Args:
        label_sets: Label sets to analyze, e.g. [[1,7], [4,9]]
        label_key: Key for labels in data (default: 'cell_type')
        layers: Which layers to analyze. None=all, or list e.g. [0, 3]
        local_method: Method for baseline local embedding ('umap', 'tsne', 'densmap', 'densne', 'phate', 'denssne')
        micro_training_epochs: Epochs for training local DiffTree model (default: 50)
        micro_training_strategy: How to handle training budget across subsets:
            - 'fixed_epochs': same epochs for all subsets (original, default)
            - 'proportional': epochs = global_epochs × (subset_size / full_size)
            - 'converge': constant LR (no cosine decay)
    """

    def __init__(
        self,
        label_sets: list = None,
        label_key: str = 'cell_type',
        layers: list = None,
        local_method: str = 'umap',
        micro_training_epochs: int = 50,
        micro_training_strategy: str = 'fixed_epochs',
        save_micro_ckpt: str = None,
        output_dir: str = None,
        eval_every_n_epochs: int = None,
        method: str = None,
        use_wandb: bool = None,
        max_samples: int = None,
        save_local: bool = None,
        down_sample: int = None,
        use_model_embedding: bool = None,
        schemes: list = None,
    ):
        super().__init__()
        self.label_sets = label_sets
        self.label_key = label_key
        self.layers = layers
        self.local_method = local_method
        self.micro_training_epochs = micro_training_epochs
        self.micro_training_strategy = micro_training_strategy
        self.save_micro_ckpt = save_micro_ckpt
        self.output_dir = output_dir
        self.eval_every_n_epochs = eval_every_n_epochs
        self.legacy_method = method
        self.use_wandb = use_wandb
        self.max_samples = max_samples
        self._save_local_explicit = save_local is not None
        self.save_local = bool(output_dir) if save_local is None else save_local
        self.down_sample = down_sample
        self.use_model_embedding = use_model_embedding
        self.schemes = schemes or []

        if method in ('tsne', 'umap', 'densmap', 'densne', 'phate', 'denssne'):
            self.local_method = method

        valid_methods = ('umap', 'tsne', 'densmap', 'densne', 'phate', 'denssne')
        if self.local_method not in valid_methods:
            raise ValueError(
                f"local_method must be one of {valid_methods}, got '{self.local_method}'"
            )

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Will be set during analysis
        self._model_class = None
        self._model_hparams = None
        self._device = None
        self._datamodule_class = None
        self._datamodule_init_args = None
        self._datamodule_adata = None
        self._trainer_fit_params = None
        self._micro_seed = 42
        self._baseline_method = None
        self._baseline_hparams = {}
        self._full_dataset_size = None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Run analysis at end of training."""
        print(f"\n[ConsistencyCallback] Training complete. Running consistency analysis...")
        try:
            self._run_analysis(trainer, pl_module)
        except Exception as e:
            import traceback
            print(f"[ConsistencyCallback] Error: {e}")
            traceback.print_exc()

    def _run_analysis(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Main analysis logic."""
        self._resolve_output_dir(trainer)
        result = self._get_embeddings(trainer, pl_module)
        if result is None:
            print("[ConsistencyCallback] Could not extract embeddings")
            return

        data_high_dim, labels, layer_embeddings, is_difftree = result

        if data_high_dim is None or len(data_high_dim) < 10:
            print("[ConsistencyCallback] Not enough data")
            return

        # Store model info for micro-training
        if is_difftree:
            self._model_class = type(pl_module)
            self._model_hparams = copy.deepcopy(dict(pl_module.hparams)) if hasattr(pl_module, 'hparams') else {}
            self._device = pl_module.device

            dm = trainer.datamodule
            self._datamodule_class = type(dm)
            self._datamodule_init_args = extract_datamodule_init_args(dm)
            self._datamodule_adata = getattr(dm, 'adata', None)
            self._trainer_fit_params = {
                'check_val_every_n_epoch': getattr(trainer, 'check_val_every_n_epoch', 1),
                'num_sanity_val_steps': getattr(trainer, 'num_sanity_val_steps', 0),
            }
            self._micro_seed = int(getattr(dm, 'seed', 42))
            self._full_dataset_size = len(data_high_dim)
            print(
                "[ConsistencyCallback] DataModule replay: "
                f"class={self._datamodule_class.__name__}, "
                f"seed={self._micro_seed}, "
                f"full_dataset_size={self._full_dataset_size}, "
                f"micro_strategy={self.micro_training_strategy}, "
                f"check_val_every_n_epoch={self._trainer_fit_params['check_val_every_n_epoch']}"
            )

        # Filter layers
        if is_difftree and self.layers is not None:
            available = len(layer_embeddings)
            valid_layers = [i for i in self.layers if 0 <= i < available]
            if not valid_layers:
                print(f"[ConsistencyCallback] No valid layers in {self.layers}")
                return
            layer_embeddings = [layer_embeddings[i] for i in valid_layers]
            layer_indices = valid_layers
        else:
            layer_indices = list(range(len(layer_embeddings)))

        print(f"[ConsistencyCallback] Data: {data_high_dim.shape}, "
              f"Labels: {len(np.unique(labels))} unique, "
              f"Model: {'DiffTree' if is_difftree else 'Baseline'}, "
              f"Layers: {layer_indices}")

        label_sets = self._get_label_sets(labels)

        log_dict = {}
        all_metrics = []

        global_best_head = None

        if is_difftree:
            global_best_head = getattr(pl_module, 'best_head_index', None)
            log_dict, all_metrics = self._analyze_difftree(
                data_high_dim, labels, layer_embeddings, label_sets, layer_indices
            )

        else:
            log_dict, all_metrics = self._analyze_baseline(
                data_high_dim, labels, layer_embeddings[0], label_sets
            )

        # Log to WandB
        if (log_dict or all_metrics) and trainer.logger is not None:
            try:
                run = trainer.logger.experiment

                # Separate scalar metrics from Plotly figures
                scalar_dict = {}
                plot_dict = {}
                for key, value in log_dict.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        scalar_dict[key] = float(value)
                    else:
                        plot_dict[key] = value

                # Log Plotly figures to run history (visualizations)
                if plot_dict:
                    run.log(plot_dict)

                # Write scalar metrics to run.summary so they appear
                # alongside hyperparameters in W&B sweep tables
                for key, value in scalar_dict.items():
                    run.summary[key] = value

                # Compute and log aggregated summary metrics across all label sets
                self._log_aggregated_summary(
                    run,
                    all_metrics,
                    is_difftree,
                    global_best_head=global_best_head,
                )

                print(f"[ConsistencyCallback] Logged {len(scalar_dict)} scalar metrics "
                      f"to summary, {len(plot_dict)} plots to history")
            except Exception as e:
                print(f"[ConsistencyCallback] WandB logging error: {e}")

        print("[ConsistencyCallback] Analysis complete.")

    def _safe_mean(self, values: list) -> Optional[float]:
        """Compute mean, returning None if the list is empty."""
        if not values:
            return None
        return float(np.mean(values))

    def _resolve_output_dir(self, trainer: pl.Trainer) -> None:
        """Infer a local output directory when the callback config omitted one."""
        if self.output_dir:
            return

        inferred_dir = None
        for callback in getattr(trainer, 'callbacks', []):
            if callback is self:
                continue
            candidate = getattr(callback, 'output_dir', None)
            if candidate:
                inferred_dir = os.path.join(candidate, "consistency")
                break

        if inferred_dir is None and self.save_micro_ckpt:
            inferred_dir = os.path.join(self.save_micro_ckpt, "consistency")

        if inferred_dir is None:
            logger = getattr(trainer, 'logger', None)
            save_dir = getattr(logger, 'save_dir', None) if logger is not None else None
            run_name = getattr(logger, 'name', None) if logger is not None else None
            if save_dir and run_name:
                inferred_dir = os.path.join(save_dir, "consistency_output", str(run_name))

        if inferred_dir is None:
            # Final fallback: always provide a deterministic local directory.
            default_root_dir = getattr(trainer, 'default_root_dir', None) or os.getcwd()
            inferred_dir = os.path.join(default_root_dir, "consistency_output")

        if inferred_dir:
            self.output_dir = inferred_dir
            if not self._save_local_explicit:
                self.save_local = True
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[ConsistencyCallback] Using local output_dir: {self.output_dir}")

    def _save_plotly_figure(self, fig: go.Figure, relative_name: str) -> None:
        """Persist a Plotly figure locally when output_dir/save_local are enabled."""
        if not self.save_local or not self.output_dir or fig is None:
            return

        base_path = os.path.join(self.output_dir, relative_name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        html_path = f"{base_path}.html"
        fig.write_html(html_path, include_plotlyjs='cdn')

        try:
            png_path = f"{base_path}.png"
            fig.write_image(png_path)
        except Exception as exc:
            print(f"[ConsistencyCallback] PNG export skipped for {html_path}: {exc}")

    def _log_aggregated_summary(
        self,
        run,
        all_metrics: List[Dict],
        is_difftree: bool,
        global_best_head: Optional[int] = None,
    ):
        """Write aggregated metrics to run.summary for sweep table comparison."""
        if not all_metrics:
            return

        if is_difftree:
            all_fidelities = [m['best_fidelity'] for m in all_metrics]
            mean_best_fidelity = self._safe_mean(all_fidelities)
            if global_best_head is not None and global_best_head >= 0:
                run.summary['consistency/global_best_head'] = int(global_best_head)
            if mean_best_fidelity is not None:
                run.log({'consistency/best_fidelity': mean_best_fidelity})

            for metrics in all_metrics:
                label_slug = metrics.get('label_slug')
                local_best_head = metrics.get('local_best_head')
                if label_slug and local_best_head is not None:
                    run.summary[f'consistency/labels_{label_slug}/local_best_head'] = int(local_best_head)

            layer_counts = [len(m.get('layer_fidelities', [])) for m in all_metrics]
            n_layers = max(layer_counts) if layer_counts else 0

            metric_keys = [
                ('layer_fidelities', 'fidelity'),
                ('layer_correlations', 'correlation'),
                ('layer_knn_global', 'knn_global'),
                ('layer_knn_local', 'knn_local'),
                ('layer_trust_global', 'trust_global'),
                ('layer_trust_local', 'trust_local'),
            ]

            for i in range(n_layers):
                valid = [m for m in all_metrics if i < len(m.get('layer_fidelities', []))]
                if not valid:
                    continue
                layer_idx = valid[0]['layer_indices'][i]
                prefix = f"consistency/head{layer_idx}"

                for src_key, summary_name in metric_keys:
                    vals = [m[src_key][i] for m in valid if i < len(m.get(src_key, []))]
                    mean_val = self._safe_mean(vals)
                    if mean_val is not None:
                        run.summary[f"{prefix}_{summary_name}"] = mean_val

            grand_keys = [
                ('layer_fidelities', 'consistency/fidelity_mean'),
                ('layer_correlations', 'consistency/correlation_mean'),
                ('layer_knn_global', 'consistency/knn_global_mean'),
                ('layer_knn_local', 'consistency/knn_local_mean'),
                ('layer_trust_global', 'consistency/trust_global_mean'),
                ('layer_trust_local', 'consistency/trust_local_mean'),
            ]
            for src_key, summary_key in grand_keys:
                vals = [v for m in all_metrics for v in m.get(src_key, [])]
                mean_val = self._safe_mean(vals)
                if mean_val is not None:
                    run.summary[summary_key] = mean_val

        else:
            baseline_keys = [
                ('fidelity', 'consistency/fidelity'),
                ('correlation', 'consistency/correlation'),
                ('knn_recall_global', 'consistency/knn_global'),
                ('knn_recall_local', 'consistency/knn_local'),
                ('trust_global', 'consistency/trust_global'),
                ('trust_local', 'consistency/trust_local'),
            ]
            baseline_log_dict = {}
            for src_key, summary_key in baseline_keys:
                vals = [m[src_key] for m in all_metrics]
                mean_val = self._safe_mean(vals)
                if mean_val is not None:
                    run.summary[summary_key] = mean_val
                    baseline_log_dict[summary_key] = mean_val
                    
            if baseline_log_dict:
                run.log(baseline_log_dict)

    # =========================================================================
    # Data Extraction
    # =========================================================================

    def _get_embeddings(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Optional[Tuple]:
        """Extract embeddings from model."""
        adata = getattr(trainer.datamodule, 'adata', None)

        if (hasattr(pl_module, 'validation_step_outputs_vis') and
            pl_module.validation_step_outputs_vis is not None):
            return self._extract_baseline(pl_module, adata)

        return self._extract_difftree(trainer, pl_module, adata)

    def _extract_baseline(self, pl_module, adata) -> Tuple:
        """Extract from baseline model."""
        high_dim = pl_module.validation_step_outputs_high
        vis_2d = pl_module.validation_step_outputs_vis

        if hasattr(pl_module, 'hparams'):
            self._baseline_hparams = copy.deepcopy(dict(pl_module.hparams))
            self._baseline_method = self._baseline_hparams.get('method')

        if self.legacy_method == 'model':
            inferred_method = self._infer_baseline_local_method()
            if inferred_method is not None:
                self.local_method = inferred_method

        labels = self._get_labels(pl_module, adata, len(high_dim))

        if isinstance(high_dim, torch.Tensor):
            high_dim = high_dim.cpu().numpy()
        if isinstance(vis_2d, torch.Tensor):
            vis_2d = vis_2d.cpu().numpy()

        return high_dim, labels, [vis_2d], False

    def _infer_baseline_local_method(self) -> Optional[str]:
        """Infer baseline local method from the fitted baseline model."""
        if self._baseline_method in ('_tsne', 'tsne'):
            return 'tsne'
        if self._baseline_method in ('_umap', 'umap'):
            return 'umap'
        if self._baseline_method in ('_densmap', 'densmap'):
            return 'densmap'
        if self._baseline_method in ('_densne', 'densne'):
            return 'densne'
        if self._baseline_method in ('_phate', 'phate'):
            return 'phate'
        if self._baseline_method in ('_denssne', 'denssne'):
            return 'denssne'
        return None

    def _extract_difftree(self, trainer, pl_module, adata) -> Optional[Tuple]:
        """Extract multi-layer embeddings from DiffTree."""
        data_list, lat_vis_all, labels_list = [], [], []
        sample_count = 0

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                for batch in trainer.datamodule.val_dataloader():
                    if isinstance(batch, dict):
                        data_input = batch.get('data_input_item', batch.get('data'))
                        batch_labels = batch.get('label', batch.get('labels'))
                    elif isinstance(batch, (list, tuple)):
                        data_input = batch[0]
                        batch_labels = batch[1] if len(batch) > 1 else None
                    else:
                        data_input = batch
                        batch_labels = None

                    if data_input is None:
                        continue

                    data_input = data_input.to(pl_module.device).float()

                    try:
                        tau = getattr(pl_module.hparams, 'tau', 1.0) if hasattr(pl_module, 'hparams') else 1.0
                        output = pl_module(data_input, tau=tau)

                        if isinstance(output, tuple) and len(output) >= 4:
                            lat_vis_list = output[3]
                        else:
                            continue
                    except Exception as e:
                        print(f"[ConsistencyCallback] Forward error: {e}")
                        continue

                    data_list.append(data_input.cpu().numpy())
                    batch_vis = [lv.cpu().numpy() if isinstance(lv, torch.Tensor) else lv
                                for lv in lat_vis_list]
                    lat_vis_all.append(batch_vis)

                    if batch_labels is not None:
                        labels_list.append(batch_labels.cpu().numpy() if isinstance(batch_labels, torch.Tensor) else batch_labels)

                    sample_count += len(data_input)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if was_training:
                pl_module.train()

        if not data_list:
            return None

        data_high_dim = np.concatenate(data_list, axis=0)
        n_layers = len(lat_vis_all[0])
        layer_embeddings = [
            np.concatenate([b[i] for b in lat_vis_all], axis=0)
            for i in range(n_layers)
        ]

        if labels_list:
            labels = np.concatenate(labels_list, axis=0)
            if labels.ndim > 1:
                labels = labels[:, 0]
        else:
            labels = self._get_labels(None, None, len(data_high_dim))

        labels = self._normalize_labels(labels)

        print(f"[ConsistencyCallback] DiffTree: {data_high_dim.shape}, {n_layers} layers")
        return data_high_dim, labels, layer_embeddings, True

    # =========================================================================
    # Label Processing
    # =========================================================================

    def _get_labels(self, pl_module, adata, n_samples) -> np.ndarray:
        labels = None
        if pl_module is not None and hasattr(pl_module, '_labels') and pl_module._labels is not None:
            labels = pl_module._labels
        elif adata is not None and self.label_key in adata.obs.columns:
            labels = adata.obs[self.label_key].values[:n_samples]
        if labels is None:
            labels = np.zeros(n_samples, dtype=int)
        return self._normalize_labels(labels)

    def _normalize_labels(self, labels: np.ndarray) -> np.ndarray:
        if hasattr(labels, 'dtype') and labels.dtype.kind in ['U', 'S', 'O']:
            try:
                labels = np.array([int(l) for l in labels])
            except (ValueError, TypeError):
                unique = sorted(set(labels))
                label_map = {l: i for i, l in enumerate(unique)}
                labels = np.array([label_map[l] for l in labels])
        return labels.astype(int)

    def _get_label_sets(self, labels: np.ndarray) -> List[List]:
        if self.label_sets is not None:
            normalized = []
            for ls in self.label_sets:
                ns = [int(l) if isinstance(l, str) else l for l in ls]
                normalized.append(ns)
            return normalized

        unique = sorted(np.unique(labels).tolist())
        n = len(unique)
        sets = []
        if n >= 2:
            sets.append(unique[:2])
        if n >= 3:
            sets.append(unique[:3])
        sets.append(unique)
        return sets


    def train_local_model(
        self,
        subset_data: np.ndarray,
        subset_labels: np.ndarray,
        layer_indices: List[int],
        subset_mask: np.ndarray,
        label_set_name: str = None,
    ) -> List[np.ndarray]:
        """
        Train a fresh DiffTree model on subset data using the EXACT same
        pipeline as the main training: NNDescent k-NN, TraceMixup
        augmentation, DataSetBaseMultiLevel dataset, same hyperparameters.

        Args:
            subset_data: High-dim data for the subset
            subset_labels: Labels for the subset (used for augmentation dataset)
            layer_indices: Which layers to extract

        Returns:
            List of layer embeddings for the subset
        """
        if self._model_class is None or self._model_hparams is None:
            raise ValueError("Model class/hparams not available for micro-training")
        if self._datamodule_class is None or self._datamodule_init_args is None:
            raise ValueError("DataModule class/init args not available for micro-training")

        n_samples = len(subset_data)
        print(f"    [Micro-Training] Training fresh model on {n_samples} samples...")

        # Copy hyperparameters, override epochs and training data size
        hparams = copy.deepcopy(self._model_hparams)
        hparams['num_train_data'] = n_samples

        # Compute actual training budget based on strategy
        # ---------------------------------------------------------------
        # proportional: epoch 按子集占比缩放，保留 cosine LR
        #   epoch = global_max_epochs × (n_samples / full_dataset_size)
        #   例: 10K/60K × 2000 = 333 epochs
        #
        # converge: 常数 LR，不做 cosine 衰减
        #   micro_training_epochs 作为 max_epochs 上限
        #
        # fixed_epochs: 原始行为，所有子集相同 epoch
        # ---------------------------------------------------------------

        if self.micro_training_strategy == 'proportional':
            full_size = self._full_dataset_size or n_samples
            global_epochs = self._model_hparams.get('max_epochs', self.micro_training_epochs)
            actual_epochs = max(50, int(global_epochs * n_samples / full_size))
            print(f"    [Micro-Training] Strategy=proportional: "
                  f"{global_epochs} × ({n_samples}/{full_size}) = {actual_epochs} epochs")
        elif self.micro_training_strategy == 'converge':
            actual_epochs = self.micro_training_epochs
            print(f"    [Micro-Training] Strategy=converge: "
                  f"constant_lr, max_epochs={actual_epochs}")
        else:
            actual_epochs = self.micro_training_epochs
            print(f"    [Micro-Training] Strategy=fixed_epochs: {actual_epochs} epochs")

        hparams['max_epochs'] = actual_epochs

        try:
            pl.seed_everything(self._micro_seed, workers=True)

            # Create fresh model with same architecture and hyperparameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fresh_model = self._model_class(**hparams)

            device = self._device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            fresh_model = fresh_model.to(device)

            # For 'converge' strategy: replace cosine annealing with constant LR.
            # Cosine decay drops LR to ~0 by max_epochs, starving larger subsets.
            # For 'proportional' strategy: keep cosine annealing (epochs already scaled).
            if self.micro_training_strategy == 'converge':
                original_configure = fresh_model.configure_optimizers

                def _constant_lr_optimizers(self_model=fresh_model):
                    result = original_configure()
                    optimizer = result['optimizer']
                    result['lr_scheduler']['scheduler'] = torch.optim.lr_scheduler.ConstantLR(
                        optimizer, factor=1.0, total_iters=0
                    )
                    return result

                fresh_model.configure_optimizers = _constant_lr_optimizers

            print("    [Micro-Training] Replaying original datamodule pipeline on subset")
            subset_dm = create_subset_datamodule(
                datamodule_class=self._datamodule_class,
                datamodule_init_args=self._datamodule_init_args,
                subset_data=subset_data,
                subset_labels=subset_labels,
                subset_mask=subset_mask,
                original_adata=self._datamodule_adata,
                label_key=self.label_key,
            )

            # Train with same trainer configuration
            callbacks = []
            save_ckpt = self.save_micro_ckpt is not None
            if save_ckpt:
                tag = label_set_name or "unknown"
                ckpt_dir = os.path.join(self.save_micro_ckpt, f"micro_{tag}")
                os.makedirs(ckpt_dir, exist_ok=True)
                from lightning.pytorch.callbacks import ModelCheckpoint
                callbacks.append(ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=f"micro_{tag}",
                    save_last=True,
                ))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer = pl.Trainer(
                    max_epochs=actual_epochs,
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,
                    logger=False,
                    enable_checkpointing=save_ckpt,
                    callbacks=callbacks if callbacks else None,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    check_val_every_n_epoch=self._trainer_fit_params.get('check_val_every_n_epoch', 1),
                    num_sanity_val_steps=self._trainer_fit_params.get('num_sanity_val_steps', 0),
                )

            print(f"    [Micro-Training] Training for {actual_epochs} epochs...")
            trainer.fit(fresh_model, datamodule=subset_dm)

            stopped_epoch = trainer.current_epoch
            print(f"    [Micro-Training] Completed {stopped_epoch + 1} epochs")

            if save_ckpt:
                print(f"    [Micro-Training] Checkpoint saved to {ckpt_dir}")

            # Extract embeddings from fresh model
            fresh_model.eval()
            with torch.no_grad():
                data_tensor = torch.from_numpy(subset_data).float().to(device)
                # Double the data (model expects original + augmented pairs)
                data_doubled = torch.cat([data_tensor, data_tensor])

                tau = getattr(fresh_model.hparams, 'tau', 1.0) if hasattr(fresh_model, 'hparams') else 1.0
                output = fresh_model(data_doubled, tau=tau)

                if isinstance(output, tuple) and len(output) >= 4:
                    lat_vis_list = output[3]
                else:
                    raise ValueError(f"Unexpected output format: {type(output)}")

                # Take only first half, select layers
                all_layers = len(lat_vis_list)
                layer_embeddings = []
                for idx in layer_indices:
                    if idx < all_layers:
                        emb = lat_vis_list[idx][:n_samples]
                        layer_embeddings.append(emb.cpu().numpy())

            if hasattr(fresh_model, 'best_head_index'):
                print(f"    [Micro-Training] best_head_index={fresh_model.best_head_index}")
            print(f"    [Micro-Training] Extracted {len(layer_embeddings)} layers")

            # Save embeddings + data + labels for sub-cluster inspection
            if save_ckpt:
                npz_path = os.path.join(ckpt_dir, f"micro_{tag}_embeddings.npz")
                save_dict = {
                    'subset_data': subset_data,
                    'subset_labels': subset_labels,
                    'layer_indices': np.array(layer_indices),
                }
                for li, emb in zip(layer_indices, layer_embeddings):
                    save_dict[f'local_emb_layer_{li}'] = emb
                np.savez(npz_path, **save_dict)
                print(f"    [Micro-Training] Embeddings saved to {npz_path}")

        finally:
            if 'fresh_model' in locals():
                del fresh_model
            if 'subset_dm' in locals():
                del subset_dm
            if 'trainer' in locals():
                del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return layer_embeddings

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def _analyze_difftree(
        self,
        data_high_dim: np.ndarray,
        labels: np.ndarray,
        global_layer_embeddings: List[np.ndarray],
        label_sets: List[List],
        layer_indices: List[int]
    ) -> Tuple[Dict, List]:
        """Analyze DiffTree consistency: k-NN preservation, fidelity, correlation."""
        import wandb

        log_dict = {}
        all_metrics = []

        # Pre-compute min subset size for scaled training strategies
        for label_set in label_sets:
            label_str = '_'.join(map(str, label_set))

            mask = np.isin(labels, label_set)
            if mask.sum() < 10:
                print(f"  Skipping {label_set}: only {mask.sum()} samples")
                continue

            subset_data = data_high_dim[mask]
            subset_labels = labels[mask]

            print(f"  Analyzing labels {label_set}: {len(subset_data)} samples")

            # Get global embeddings for subset
            global_subset_embs = [emb[mask] for emb in global_layer_embeddings]

            # Train fresh local model on subset with exact same pipeline
            try:
                local_layer_embs = self.train_local_model(
                    subset_data, subset_labels, layer_indices,
                    subset_mask=mask,
                    label_set_name=label_str,
                )
            except Exception as e:
                import traceback
                print(f"    Micro-training failed: {e}")
                traceback.print_exc()
                continue

            if len(local_layer_embs) != len(layer_indices):
                print(f"    WARNING: expected {len(layer_indices)} layers, "
                      f"got {len(local_layer_embs)}. Skipping label set {label_set}.")
                continue

            # Per-layer metrics: fidelity, correlation, k-NN preservation
            layer_metrics = []
            for i, layer_idx in enumerate(layer_indices):
                global_emb = global_subset_embs[i]
                local_emb = local_layer_embs[i]

                # Fidelity + correlation (Procrustes-based)
                metrics = self.compute_layer_metrics(
                    subset_data, global_emb, local_emb, subset_labels
                )
                metrics['layer'] = layer_idx

                # k-NN preservation: HD -> embedding
                knn_global = knn_preservation_rate(subset_data, [global_emb], k=12)
                knn_local = knn_preservation_rate(subset_data, [local_emb], k=12)
                metrics['knn_recall_global'] = knn_global['per_scale_recall'][0]
                metrics['knn_recall_local'] = knn_local['per_scale_recall'][0]

                layer_metrics.append(metrics)

            # Choose the local best head for this label set by fidelity.
            best_layer = max(layer_metrics, key=lambda m: m['fidelity_score'])
            log_dict[f"consistency/labels_{label_str}/local_best_head"] = int(best_layer['layer'])

            for i, layer_idx in enumerate(layer_indices):
                global_emb = global_subset_embs[i]
                local_emb = local_layer_embs[i]
                layer_metric = layer_metrics[i]
                is_best_head = layer_idx == best_layer['layer']

                viz = LocalGlobalConsistencyVisualizer(
                    global_data=subset_data,
                    global_embedding=global_emb,
                    labels=subset_labels,
                    method=self.local_method,
                )
                aligned_global, aligned_local, _ = viz.align_embeddings(global_emb, local_emb)
                case_fig = self._plot_single_head_comparison(
                    aligned_global,
                    aligned_local,
                    subset_labels,
                    layer_metric,
                    layer_idx,
                    label_set,
                    is_best_head=is_best_head,
                    best_head_idx=best_layer['layer'],
                    full_global_emb=global_layer_embeddings[i],
                    full_labels=labels,
                    subset_mask=mask,
                )

                head_tag = "best" if is_best_head else ""
                log_dict[
                    f"consistency/labels_{label_str}/comparisons/head_{layer_idx}_{head_tag}"
                ] = wandb.Plotly(case_fig)
                self._save_plotly_figure(
                    case_fig,
                    os.path.join(
                        "difftree",
                        f"labels_{label_str}",
                        f"head_{layer_idx}_{head_tag}",
                    ),
                )

            best_head_fig_key = (
                f"consistency/labels_{label_str}/comparisons/head_{best_layer['layer']}_best"
            )
            log_dict[f"consistency/labels_{label_str}/local_best_comparison"] = log_dict[best_head_fig_key]

            case_metrics = {
                'labels': str(label_set),
                'label_slug': label_str,
                'n_samples': len(subset_data),
                'local_best_head': best_layer['layer'],
                'best_layer': best_layer['layer'],
                'best_fidelity': best_layer['fidelity_score'],
                'layer_indices': layer_indices,
                'layer_fidelities': [m['fidelity_score'] for m in layer_metrics],
                'layer_correlations': [m['global_local_correlation'] for m in layer_metrics],
                'layer_knn_global': [m['knn_recall_global'] for m in layer_metrics],
                'layer_knn_local': [m['knn_recall_local'] for m in layer_metrics],
                'layer_trust_global': [m['global_trustworthiness'] for m in layer_metrics],
                'layer_trust_local': [m['local_trustworthiness'] for m in layer_metrics],
            }
            all_metrics.append(case_metrics)

            # Console summary
            print(f"    Local best head: {best_layer['layer']} "
                  f"(fidelity={best_layer['fidelity_score']:.4f})")
            for m in layer_metrics:
                print(f"    Layer {m['layer']}: "
                      f"fidelity={m['fidelity_score']:.3f}  "
                      f"corr={m['global_local_correlation']:.3f}  "
                      f"kNN_g={m['knn_recall_global']:.3f}  "
                      f"kNN_l={m['knn_recall_local']:.3f}  "
                      f"trust_g={m['global_trustworthiness']:.3f}  "
                      f"trust_l={m['local_trustworthiness']:.3f}")

        if all_metrics:
            summary_fig = self._plot_summary(all_metrics, layer_indices)
            log_dict['consistency/summary'] = wandb.Plotly(summary_fig)
            self._save_plotly_figure(summary_fig, os.path.join("difftree", "summary"))

            # Per-head best fidelity line chart (wandb only)
            fidelity_fig = self._plot_per_head_fidelity(all_metrics, layer_indices)
            log_dict['consistency/per_head_fidelity'] = wandb.Plotly(fidelity_fig)

        return log_dict, all_metrics

    def _analyze_baseline(
        self,
        data_high_dim: np.ndarray,
        labels: np.ndarray,
        global_emb: np.ndarray,
        label_sets: List[List]
    ) -> Tuple[Dict, List]:
        """Analyze baseline (t-SNE/UMAP) consistency: k-NN preservation, fidelity, correlation."""
        import wandb

        log_dict = {}
        all_metrics = []

        for label_set in label_sets:
            mask = np.isin(labels, label_set)
            if mask.sum() < 10:
                continue

            subset_data = data_high_dim[mask]
            subset_global = global_emb[mask]
            subset_labels = labels[mask]

            print(f"  Analyzing labels {label_set}: {len(subset_data)} samples")

            local_emb = self.compute_local_embedding(subset_data)

            # Fidelity + correlation
            viz = LocalGlobalConsistencyVisualizer(
                global_data=subset_data,
                global_embedding=subset_global,
                labels=subset_labels,
                method=self.local_method
            )
            metrics = viz.compute_metrics(subset_data, subset_global, local_emb)

            # k-NN preservation: HD -> embedding
            knn_global = knn_preservation_rate(subset_data, [subset_global], k=12)
            knn_local = knn_preservation_rate(subset_data, [local_emb], k=12)
            metrics['knn_recall_global'] = knn_global['per_scale_recall'][0]
            metrics['knn_recall_local'] = knn_local['per_scale_recall'][0]

            viz = LocalGlobalConsistencyVisualizer(
                global_data=subset_data,
                global_embedding=subset_global,
                labels=subset_labels,
                method=self.local_method
            )
            aligned_global, aligned_local, _ = viz.align_embeddings(subset_global, local_emb)
            case_fig = self._plot_baseline_comparison(
                aligned_global,
                aligned_local,
                subset_labels,
                metrics,
                label_set,
            )
            label_str = '_'.join(map(str, label_set))
            log_dict[f"consistency/labels_{label_str}/comparison"] = wandb.Plotly(case_fig)
            
            # Log exact scalar metrics for this label set
            log_dict[f"consistency/labels_{label_str}/fidelity"] = float(metrics['fidelity_score'])
            log_dict[f"consistency/labels_{label_str}/correlation"] = float(metrics['global_local_correlation'])
            log_dict[f"consistency/labels_{label_str}/knn_global"] = float(metrics['knn_recall_global'])
            log_dict[f"consistency/labels_{label_str}/knn_local"] = float(metrics['knn_recall_local'])
            log_dict[f"consistency/labels_{label_str}/trust_global"] = float(metrics['global_trustworthiness'])
            log_dict[f"consistency/labels_{label_str}/trust_local"] = float(metrics['local_trustworthiness'])

            self._save_plotly_figure(
                case_fig,
                os.path.join(self.local_method, f"labels_{label_str}", "comparison"),
            )

            all_metrics.append({
                'labels': str(label_set),
                'n_samples': len(subset_data),
                'fidelity': metrics['fidelity_score'],
                'correlation': metrics['global_local_correlation'],
                'knn_recall_global': metrics['knn_recall_global'],
                'knn_recall_local': metrics['knn_recall_local'],
                'trust_global': metrics['global_trustworthiness'],
                'trust_local': metrics['local_trustworthiness'],
            })

            print(f"    fidelity={metrics['fidelity_score']:.4f}  "
                  f"corr={metrics['global_local_correlation']:.4f}  "
                  f"kNN_g={metrics['knn_recall_global']:.4f}  "
                  f"kNN_l={metrics['knn_recall_local']:.4f}  "
                  f"trust_g={metrics['global_trustworthiness']:.4f}  "
                  f"trust_l={metrics['local_trustworthiness']:.4f}")

        return log_dict, all_metrics



    def compute_local_embedding(self, subset_data: np.ndarray) -> np.ndarray:
        """Compute fresh local embedding for baselines."""
        n_samples = len(subset_data)

        if self._baseline_method in ('_tsne', '_umap', '_densmap', '_densne', '_phate', '_denssne'):
            return self._compute_baseline_compatible_embedding(subset_data)

        if self.local_method == 'tsne':
            perplexity = min(30, n_samples - 1)
            reducer = TSNE(n_components=2, perplexity=max(5, perplexity), random_state=42)
            return reducer.fit_transform(subset_data)

        if self.local_method == 'umap':
            n_neighbors = min(15, n_samples - 1)
            reducer = umap_lib.UMAP(n_components=2, n_neighbors=max(2, n_neighbors), random_state=42)
            return reducer.fit_transform(subset_data)

        if self.local_method == 'densmap':
            n_neighbors = min(15, n_samples - 1)
            reducer = umap_lib.UMAP(
                n_components=2,
                n_neighbors=max(2, n_neighbors),
                densmap=True,
                output_dens=False,
                random_state=42,
            )
            return reducer.fit_transform(subset_data)

        if self.local_method == 'densne':
            raise ImportError(
                "densNE local embedding requires a fitted densNE baseline or the `densne` package."
            )

        if self.local_method == 'phate':
            from model.baseline_tri import phate
            if phate is None:
                raise ImportError("PHATE local embedding requires the `phate` package.")
            reducer = phate.PHATE(
                n_components=2,
                knn=min(15, max(5, n_samples - 1)),
                random_state=42,
                verbose=0,
            )
            return reducer.fit_transform(subset_data)

        if self.local_method == 'denssne':
            raise ImportError(
                "densSNE local embedding requires a fitted densSNE baseline or the `densne` package."
            )

        raise ValueError(f"Unsupported local_method: {self.local_method}")

    def _compute_baseline_compatible_embedding(self, subset_data: np.ndarray) -> np.ndarray:
        """Refit the subset with the same baseline family and hyperparameters."""
        baseline_method = self._baseline_method
        random_state = self._baseline_hparams.get('random_state', 42)
        n_components = self._baseline_hparams.get('n_components', 2)
        from model.baseline_tri import (
            DMTEVT_model as BaselineModel,
            OpenTSNE,
            SklearnTSNE,
            phate,
        )

        p1 = self._baseline_hparams.get('p1', 2)
        p2 = self._baseline_hparams.get('p2', 2)

        if baseline_method == '_tsne':
            perplexity, early_exaggeration = BaselineModel._get_tsne_params(
                p1,
                p2,
                {
                    'perplexity': self._baseline_hparams.get('perplexity'),
                    'early_exaggeration': self._baseline_hparams.get('early_exaggeration'),
                },
            )
            perplexity = min(float(perplexity), max(5.0, float(len(subset_data) - 1)))

            if OpenTSNE is not None:
                reducer = OpenTSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    random_state=random_state,
                    n_jobs=-1,
                )
                embedding = reducer.fit(subset_data)
                return np.asarray(embedding)

            reducer = SklearnTSNE(
                n_components=n_components,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                random_state=random_state,
                init='pca',
                learning_rate='auto',
            )
            return reducer.fit_transform(subset_data)

        if baseline_method in ('_umap', '_densmap'):
            n_neighbors, min_dist = BaselineModel._get_umap_params(
                p1,
                p2,
                {
                    'n_neighbors': self._baseline_hparams.get('n_neighbors'),
                    'min_dist': self._baseline_hparams.get('min_dist'),
                },
            )
            n_neighbors = min(int(n_neighbors), max(2, len(subset_data) - 1))

            reducer = umap_lib.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                densmap=(baseline_method == '_densmap'),
                output_dens=False,
                random_state=random_state,
                n_jobs=-1,
            )
            return reducer.fit_transform(subset_data)

        if baseline_method == '_phate':
            available, message = BaselineModel.has_optional_dependency('_phate')
            if not available:
                raise ImportError(message)

            knn, decay = BaselineModel._get_phate_params(
                p1,
                p2,
                {
                    'knn': self._baseline_hparams.get('knn'),
                    'decay': self._baseline_hparams.get('decay'),
                },
            )
            knn = min(int(knn), max(5, len(subset_data) - 1))
            reducer = phate.PHATE(
                n_components=n_components,
                knn=knn,
                decay=decay,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )
            return reducer.fit_transform(subset_data)

        if baseline_method == '_densne':
            perplexity, early_exaggeration = BaselineModel._get_tsne_params(
                p1,
                p2,
                {
                    'perplexity': self._baseline_hparams.get('perplexity'),
                    'early_exaggeration': self._baseline_hparams.get('early_exaggeration'),
                },
            )
            perplexity = min(float(perplexity), max(5.0, float(len(subset_data) - 1)))
            runner = BaselineModel(
                method='_densne',
                data_name='subset',
                num_input_dim=subset_data.shape[1],
                n_components=n_components,
                p1=p1,
                p2=p2,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
            )
            return runner._run_densne(
                subset_data,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
            )

        if baseline_method == '_denssne':
            perplexity, early_exaggeration = BaselineModel._get_tsne_params(
                p1,
                p2,
                {
                    'perplexity': self._baseline_hparams.get('perplexity'),
                    'early_exaggeration': self._baseline_hparams.get('early_exaggeration'),
                },
            )
            perplexity = min(float(perplexity), max(5.0, float(len(subset_data) - 1)))
            runner = BaselineModel(
                method='_denssne',
                data_name='subset',
                num_input_dim=subset_data.shape[1],
                n_components=n_components,
                p1=p1,
                p2=p2,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
            )
            return runner._run_denssne(
                subset_data,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
            )

        raise ValueError(f"Unsupported baseline method: {baseline_method}")

    def compute_layer_metrics(
        self,
        subset_data: np.ndarray,
        global_embedding: np.ndarray,
        local_embedding: np.ndarray,
        subset_labels: np.ndarray
    ) -> Dict:
        """Compute consistency metrics for a layer."""
        viz = LocalGlobalConsistencyVisualizer(
            global_data=subset_data,
            global_embedding=global_embedding,
            labels=subset_labels,
            method=self.local_method
        )
        return viz.compute_metrics(subset_data, global_embedding, local_embedding)

    # =========================================================================
    # Visualization
    # =========================================================================

    def _plot_single_head_comparison(
        self,
        global_emb: np.ndarray,
        local_emb: np.ndarray,
        subset_labels: np.ndarray,
        metrics: Dict,
        layer_idx: int,
        selected_labels: List,
        is_best_head: bool = False,
        best_head_idx: Optional[int] = None,
        full_global_emb: Optional[np.ndarray] = None,
        full_labels: Optional[np.ndarray] = None,
        subset_mask: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """为单个投影头生成 Global vs Local 对比图，尺寸与 xc_plot_callback 一致。

        Global subplot shows ALL data: selected labels in color, others in gray.
        Local subplot shows only the subset data in color.
        """
        label_str = ', '.join(map(str, selected_labels))

        fid = metrics["fidelity_score"]
        corr = metrics["global_local_correlation"]
        knn_g = metrics["knn_recall_global"]
        knn_l = metrics["knn_recall_local"]
        trust_g = metrics["global_trustworthiness"]
        trust_l = metrics["local_trustworthiness"]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.55, 0.45],
            subplot_titles=[
                f'<b>Global</b> (Head {layer_idx})<br>'
                f'kNN: {knn_g:.3f} | Trust: {trust_g:.3f}',
                f'<b>Local</b> (Head {layer_idx})<br>'
                f'kNN: {knn_l:.3f} | Trust: {trust_l:.3f}',
            ],
            horizontal_spacing=0.06
        )

        # --- Global subplot: show all data with non-selected in gray ---
        if full_global_emb is not None and full_labels is not None and subset_mask is not None:
            # Plot non-selected points in gray first (background)
            non_selected_mask = ~subset_mask
            if non_selected_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=full_global_emb[non_selected_mask, 0],
                        y=full_global_emb[non_selected_mask, 1],
                        mode='markers',
                        marker=dict(size=2, color='lightgray', opacity=0.4),
                        name='Other',
                        legendgroup='other',
                        showlegend=True,
                        hoverinfo='skip',
                    ),
                    row=1, col=1
                )

            # Plot selected labels in color (foreground)
            subset_full_emb = full_global_emb[subset_mask]
            for label in sorted(np.unique(subset_labels)):
                lmask = subset_labels == label
                color = D3_COLORS[int(label) % len(D3_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=subset_full_emb[lmask, 0], y=subset_full_emb[lmask, 1],
                        mode='markers',
                        marker=dict(size=2, color=color, opacity=0.7),
                        name=f'Digit {label}',
                        legendgroup=str(label),
                        showlegend=True,
                    ),
                    row=1, col=1
                )
        else:
            # Fallback: only plot subset (old behavior)
            for label in sorted(np.unique(subset_labels)):
                lmask = subset_labels == label
                color = D3_COLORS[int(label) % len(D3_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=global_emb[lmask, 0], y=global_emb[lmask, 1],
                        mode='markers',
                        marker=dict(size=2, color=color, opacity=0.6),
                        name=f'Digit {label}',
                        legendgroup=str(label),
                        showlegend=True,
                    ),
                    row=1, col=1
                )

        # --- Local subplot: only subset data in color ---
        for label in sorted(np.unique(subset_labels)):
            lmask = subset_labels == label
            color = D3_COLORS[int(label) % len(D3_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=local_emb[lmask, 0], y=local_emb[lmask, 1],
                    mode='markers',
                    marker=dict(size=2, color=color, opacity=0.6),
                    name=f'Digit {label}',
                    legendgroup=str(label),
                    showlegend=False,
                ),
                row=1, col=2
            )

        best_head_suffix = " | BEST HEAD" if is_best_head else ""
        best_head_note = (
            f" | Best Head={best_head_idx}" if best_head_idx is not None and best_head_idx >= 0 else ""
        )

        fig.update_layout(
            title=dict(
                text=f'<b>Head {layer_idx}{best_head_suffix} | Labels [{label_str}]</b> | '
                     f'N={len(subset_labels)} | '
                     f'Fidelity={fid:.3f} | Corr={corr:.3f}{best_head_note}',
                x=0.5, xanchor='center',
            ),
            height=550,
            width=1200,
            showlegend=True,
            legend=dict(
                x=0.5, y=-0.1, xanchor='center', orientation='h',
            ),
            plot_bgcolor='white',
        )

        for col in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=1, col=col)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=1, col=col)

        return fig

    def _plot_baseline_comparison(
        self,
        aligned_global: np.ndarray,
        aligned_local: np.ndarray,
        subset_labels: np.ndarray,
        metrics: Dict,
        selected_labels: List
    ) -> go.Figure:
        """Baseline comparison plot with 3 core metrics."""
        label_str = ', '.join(map(str, selected_labels))
        method_name = self.local_method.upper()

        knn_g = metrics.get('knn_recall_global', 0)
        knn_l = metrics.get('knn_recall_local', 0)
        trust_g = metrics.get('global_trustworthiness', 0)
        trust_l = metrics.get('local_trustworthiness', 0)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'<b>Global {method_name}</b><br>kNN: {knn_g:.3f} | Trust: {trust_g:.3f}',
                f'<b>Local {method_name}</b> (fresh fit)<br>kNN: {knn_l:.3f} | Trust: {trust_l:.3f}',
            ]
        )

        for label in sorted(np.unique(subset_labels)):
            mask = subset_labels == label
            color = D3_COLORS[int(label) % len(D3_COLORS)]

            fig.add_trace(
                go.Scatter(
                    x=aligned_global[mask, 0], y=aligned_global[mask, 1],
                    mode='markers',
                    marker=dict(size=5, color=color, opacity=0.7),
                    name=f'Digit {label}',
                    legendgroup=str(label),
                    showlegend=True,
                    hovertemplate=f'Digit {label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra>Global</extra>',
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=aligned_local[mask, 0], y=aligned_local[mask, 1],
                    mode='markers',
                    marker=dict(size=5, color=color, opacity=0.7),
                    name=f'Digit {label}',
                    legendgroup=str(label),
                    showlegend=False,
                    hovertemplate=f'Digit {label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra>Local</extra>',
                ),
                row=1, col=2
            )

        fig.update_layout(
            title=dict(
                text=f'<b>{method_name} Local-Global Consistency Analysis</b><br>'
                     f'<span style="font-size:13px; color:#555;">'
                     f'Comparing: Global (fitted on ALL data) vs Local (fitted on SUBSET only)</span><br>'
                     f'<span style="font-size:12px;">'
                     f'Labels: [{label_str}] | N={len(subset_labels)} | '
                     f'Fidelity: {metrics["fidelity_score"]:.3f} | '
                     f'Correlation: {metrics["global_local_correlation"]:.3f} | '
                     f'kNN: {knn_g:.3f}/{knn_l:.3f} | '
                     f'Trust: {trust_g:.3f}/{trust_l:.3f}</span>',
                x=0.5, xanchor='center', y=0.95
            ),
            height=500,
            width=950,
            showlegend=True,
            legend=dict(
                x=0.5, y=-0.12, xanchor='center', orientation='h',
                title=dict(text='<b>Classes:</b> ', font=dict(size=11))
            ),
            plot_bgcolor='white',
            margin=dict(t=100, b=80)
        )

        fig.add_annotation(
            x=0.5, y=-0.18,
            text=f"<b>Metrics:</b> Fidelity={metrics['fidelity_score']:.3f} (Procrustes) | "
                 f"Correlation={metrics['global_local_correlation']:.3f} (Pearson) | "
                 f"kNN: Global={knn_g:.3f}, Local={knn_l:.3f} | "
                 f"Trust: Global={trust_g:.3f}, Local={trust_l:.3f}",
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=10, color='#444'),
            bgcolor='rgba(240,240,240,0.9)', borderpad=6
        )

        for i in range(2):
            fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=1, col=i+1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=1, col=i+1)

        return fig

    def _plot_per_head_fidelity(self, all_metrics: List[Dict], layer_indices: List[int]) -> go.Figure:
        """Line chart: best fidelity per head across projection heads."""
        fig = go.Figure()
        head_labels = [f"Head {idx}" for idx in layer_indices]

        # Each head's best fidelity = max across label sets
        n_heads = len(layer_indices)
        best_per_head = []
        for i in range(n_heads):
            vals = [m['layer_fidelities'][i] for m in all_metrics
                    if i < len(m.get('layer_fidelities', []))]
            best_per_head.append(max(vals) if vals else 0.0)

        fig.add_trace(go.Scatter(
            x=head_labels,
            y=best_per_head,
            mode='lines+markers',
            marker=dict(size=10),
            name='Best Fidelity',
        ))

        fig.update_layout(
            title=dict(text='<b>Per-Head Best Fidelity</b>', x=0.5, xanchor='center'),
            xaxis_title='Projection Head',
            yaxis_title='Fidelity',
            yaxis=dict(range=[0, 1.05]),
            height=400,
            width=600,
        )
        return fig

    def _plot_summary(self, all_metrics: List[Dict], layer_indices: List[int]) -> go.Figure:
        """Summary bar chart: Fidelity, Correlation, k-NN Recall by layer."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                '<b>Fidelity</b> (Global vs Local)',
                '<b>Correlation</b> (Global vs Local)',
                '<b>k-NN Recall</b> (HD preservation)',
            ],
            horizontal_spacing=0.08,
        )

        x_labels = [m['labels'] for m in all_metrics]

        for i, layer_idx in enumerate(layer_indices):
            fidelities = [m['layer_fidelities'][i] for m in all_metrics]
            fig.add_trace(
                go.Bar(x=x_labels, y=fidelities, name=f'Layer {layer_idx}'),
                row=1, col=1
            )

        for i, layer_idx in enumerate(layer_indices):
            corrs = [m['layer_correlations'][i] for m in all_metrics]
            fig.add_trace(
                go.Bar(x=x_labels, y=corrs, name=f'Layer {layer_idx}', showlegend=False),
                row=1, col=2
            )

        for i, layer_idx in enumerate(layer_indices):
            knn_g = [m['layer_knn_global'][i] for m in all_metrics]
            fig.add_trace(
                go.Bar(x=x_labels, y=knn_g, name=f'Layer {layer_idx}', showlegend=False),
                row=1, col=3
            )

        fig.update_layout(
            title=dict(text='<b>Layer Consistency Summary</b>', x=0.5, xanchor='center'),
            height=400,
            width=1200,
            barmode='group',
            legend=dict(x=0.5, y=-0.2, xanchor='center', orientation='h'),
        )

        fig.update_yaxes(title_text='Fidelity', row=1, col=1)
        fig.update_yaxes(title_text='Correlation', row=1, col=2)
        fig.update_yaxes(title_text='k-NN Recall', row=1, col=3)

        return fig


class DMTEVTConsistencyCallback(ConsistencyCallback):
    """Backward-compatible alias for older experiment configs."""


class QuickConsistencyCallback(pl.Callback):
    """
    Backward-compatible placeholder for legacy configs.

    The original quick callback implementation is not present in this
    repository snapshot. Keeping this class as a no-op preserves old
    configs without changing the current training pipeline.
    """

    def __init__(
        self,
        eval_every_n_epochs: int = None,
        max_samples: int = None,
        label_key: str = 'cell_type',
        use_model_embedding: bool = None,
        **kwargs,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.max_samples = max_samples
        self.label_key = label_key
        self.use_model_embedding = use_model_embedding
        self.extra_kwargs = kwargs
