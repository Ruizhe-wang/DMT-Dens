import json
import os

import torch
import torch.nn.functional as F
import lightning as pl
import wandb
import numpy as np

from eval.fidelity_eval import summarize_embedding_metrics, compute_asr, compute_frdc
from callbacks.encoder_benchmark import embedding_health_metrics
from callbacks.wandb_utils import safe_wandb_step

ALIAS_METRIC_NAMES = {"knn", "trust", "dist_corr", "den_corr", "ldc", "spir"}

PRIMARY_METRIC_LOG_NAMES = {
    "density_preservation_score": "val_density_score",
    "density_fidelity_score": "val_density_fidelity_score",
    "structure_score": "val_structure_score",
    "multi_scale_local_radius_correlation_fixed": "val_density_lrc_fixed",
    "multi_scale_local_radius_correlation_visual": "val_density_lrc_visual",
    "multi_scale_local_density_distortion_error": "val_density_ldd",
    "multi_scale_pairwise_density_ratio_error": "val_density_dre",
    "multi_scale_density_pair_order_accuracy": "val_density_dpa",
    "multi_scale_density_quantile_calibration_error": "val_density_dqce",
    "multi_scale_high_density_overlap_q10": "val_density_hdor_q10",
    "multi_scale_low_density_overlap_q10": "val_density_ldor_q10",
    "density_correlation": "val_visible_density_correlation",
    "local_density_correlation": "val_local_density_correlation",
    "svc_accuracy": "val_svc_acc",
    "trustworthiness": "val_trustworthiness",
    "knn_preservation": "val_knn_preservation",
    # Two different quantities: the first is a Spearman rank correlation over
    # sampled distance pairs (Shepard), the second is the dCor statistic. They
    # are named apart so the summary table cannot conflate them.
    "shepard_spearman_correlation": "val_shepard_spearman",
    "distance_correlation_dcor": "val_distance_correlation_dcor",
    "msdp_auc": "val_msdp_auc",
}


def build_metric_log_payload(metrics, include_primary_aliases=True):
    payload = {}

    if include_primary_aliases:
        for metric_name, log_name in PRIMARY_METRIC_LOG_NAMES.items():
            if metric_name in metrics:
                payload[log_name] = float(metrics[metric_name])

        if "density_correlation" in metrics:
            payload["val_legacy_density_correlation"] = float(metrics["density_correlation"])
        svc_value = metrics.get("svc_accuracy", metrics.get("svc_acc"))
        if svc_value is not None:
            payload["val_svc_acc"] = float(svc_value)
            payload["val_svc"] = float(svc_value)

        # Combined objective: equal-weighted mean of visible density correlation
        # and SVC accuracy. Logged so a single-metric wandb sweep can optimize
        # density preservation and label separability jointly (used by the
        # celegan wide bayes sweep). Additive: does not affect other metrics.
        den_corr_value = metrics.get("density_correlation")
        if den_corr_value is not None and svc_value is not None:
            payload["val_density_svc_combined"] = (
                0.5 * float(den_corr_value) + 0.5 * float(svc_value)
            )

    knn = metrics.get("knn")
    trust = metrics.get("trust")
    dist_corr = metrics.get("dist_corr")
    if knn is not None and trust is not None and dist_corr is not None:
        score = (float(knn) + float(trust) + float(dist_corr)) / 3.0
        payload["val_knn"] = float(knn)
        payload["val_trust"] = float(trust)
        payload["val_dist_corr"] = float(dist_corr)
        payload["val_score"] = float(score)

    return payload


class FidelityEvalCallback(pl.Callback):
    """
    Callback for evaluating model fidelity metrics:
    - kNN Preservation
    - Trustworthiness
    - Distance Correlation
    - Density Correlation
    - Continuity
    """
    def __init__(self, every_n_epochs=1, down_sample=3000, knn_k=12, density_k=15, seed=42,
                 use_mellon=False, mellon_pca_dim=20,
                 bn_train_eval_diagnostic=False, diagnostic_output_path=None):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.down_sample = down_sample
        self.knn_k = knn_k
        self.density_k = density_k
        self.seed = seed
        self.use_mellon = use_mellon
        self.mellon_pca_dim = mellon_pca_dim
        self.bn_train_eval_diagnostic = bool(bn_train_eval_diagnostic)
        self.diagnostic_output_path = diagnostic_output_path
        # Retain one verified full-HD array so ASR/FR-DC's id-based reference
        # caches survive later evaluation epochs. This changes evaluation cost,
        # not metric values or training state.
        self._independent_hd_reference = None

    def _stable_full_hd_reference(self, hd_full):
        reference = self._independent_hd_reference
        if reference is None or reference.shape != hd_full.shape:
            self._independent_hd_reference = hd_full
            return hd_full

        # Verify data order cheaply before reusing the reference object. A new
        # validation tensor is allocated every epoch even when its contents are
        # identical, which otherwise defeats the downstream id-based caches.
        indices = np.unique([0, len(hd_full) // 2, len(hd_full) - 1])
        if not np.array_equal(reference[indices], hd_full[indices], equal_nan=True):
            self._independent_hd_reference = hd_full
            return hd_full
        return reference

    def _get_wandb_run(self, trainer):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None
        experiment = getattr(logger, "experiment", None)
        return experiment if experiment is not None else None

    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _resolve_tau(self, pl_module):
        if hasattr(pl_module, "hparams"):
            return getattr(pl_module.hparams, "tau", 1.0)
        return 1.0

    def _get_cached_baseline_embeddings(self, pl_module):
        lat_vis = getattr(pl_module, "validation_step_outputs_vis", None)
        data_input = getattr(pl_module, "validation_step_outputs_high", None)
        if data_input is None:
            data_input = getattr(pl_module, "validation_origin_input", None)

        if lat_vis is None or data_input is None:
            return None

        lat_vis = torch.as_tensor(lat_vis, dtype=torch.float32)
        if lat_vis.ndim == 2:
            lat_vis = lat_vis.unsqueeze(0)

        data_input = torch.as_tensor(data_input, dtype=torch.float32)
        labels = getattr(pl_module, "_labels", None)
        labels = torch.as_tensor(labels).cpu() if labels is not None else None
        return lat_vis.cpu(), data_input.cpu(), labels

    @staticmethod
    def _classification_labels(labels_np):
        if labels_np is None:
            return None
        labels_np = np.asarray(labels_np)
        if labels_np.ndim > 1:
            return labels_np[:, 0]
        return labels_np

    def get_embeddings(self, trainer, pl_module):
        if self._is_baseline_model(pl_module):
            cached = self._get_cached_baseline_embeddings(pl_module)
            if cached is not None:
                return cached

        data_input = []
        data_list = []
        labels = []

        with torch.inference_mode():
            import inspect
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values())

            for batch in trainer.datamodule.val_dataloader():
                data_input_item = batch["data_input_item"].to(pl_module.device)
                
                if supports_tau:
                    model_output = pl_module(
                        data_input_item,
                        tau=self._resolve_tau(pl_module),
                    )
                else:
                    model_output = pl_module(data_input_item)
                if isinstance(model_output, tuple) and len(model_output) >= 4:
                    lat_vis_list = model_output[3]
                else:
                    lat_vis = torch.as_tensor(
                        model_output,
                        dtype=torch.float32,
                        device=data_input_item.device,
                    )
                    lat_vis_list = [lat_vis]

                lat_vis_list_stack = torch.stack(lat_vis_list, dim=0)
                data_list.append(lat_vis_list_stack.detach().float().cpu())
                data_input.append(data_input_item.detach().float().cpu())
                if "label" in batch:
                    labels.append(torch.as_tensor(batch["label"]).detach().cpu())

        # Concatenate on CPU
        data = torch.cat(data_list, dim=1) # (num_layers, total_batch, dim)
        data_input = torch.cat(data_input, dim=0) # (total_batch, data_dim)
        labels = torch.cat(labels, dim=0) if labels else None

        return data, data_input, labels

    @staticmethod
    def _extract_last_embedding(model_output):
        if isinstance(model_output, tuple) and len(model_output) >= 4:
            return model_output[3][-1]
        if isinstance(model_output, tuple) and len(model_output) >= 3:
            return model_output[2]
        return torch.as_tensor(model_output)

    def _projection_and_bn_diagnostics(self, trainer, pl_module):
        metrics = {}
        try:
            final_linear = pl_module.vis_list[-1][-1].block[0]
            weight = final_linear.weight.detach().float()
            if weight.ndim == 2 and weight.shape[0] >= 2:
                metrics["projection/weight_row_cosine"] = float(
                    F.cosine_similarity(weight[0], weight[1], dim=0).cpu()
                )
        except (AttributeError, IndexError, TypeError):
            pass

        if not self.bn_train_eval_diagnostic:
            return metrics

        try:
            batch = next(iter(trainer.datamodule.val_dataloader()))
            inputs = batch["data_input_item"].to(pl_module.device)
        except (AttributeError, KeyError, StopIteration, TypeError):
            return metrics

        was_training = pl_module.training
        buffer_snapshot = {
            name: value.detach().clone()
            for name, value in pl_module.named_buffers()
        }
        try:
            with torch.inference_mode():
                pl_module.eval()
                eval_output = self._extract_last_embedding(
                    pl_module(inputs, tau=self._resolve_tau(pl_module))
                ).detach().float()
                pl_module.train()
                train_output = self._extract_last_embedding(
                    pl_module(inputs, tau=self._resolve_tau(pl_module))
                ).detach().float()
        finally:
            for name, value in pl_module.named_buffers():
                if name in buffer_snapshot:
                    value.copy_(buffer_snapshot[name])
            pl_module.train(was_training)

        finite = torch.isfinite(eval_output).all(dim=1) & torch.isfinite(
            train_output
        ).all(dim=1)
        if int(finite.sum().item()) < 2:
            metrics["projection_bn/train_eval_nonfinite_fraction"] = 1.0
            return metrics

        eval_clean = eval_output[finite]
        train_clean = train_output[finite]
        reference_rms = eval_clean.square().mean().sqrt().clamp_min(1.0e-12)
        gap = (train_clean - eval_clean).square().mean().sqrt() / reference_rms
        metrics["projection_bn/train_eval_relative_rms_gap"] = float(gap.cpu())
        metrics["projection_bn/train_eval_nonfinite_fraction"] = float(
            1.0 - finite.float().mean().cpu()
        )
        return metrics

    def _append_diagnostic_record(self, epoch_num, metrics):
        if not self.diagnostic_output_path:
            return
        output_dir = os.path.dirname(self.diagnostic_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        record = {"epoch": int(epoch_num)}
        record.update(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        with open(self.diagnostic_output_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _independent_density_metrics(self, lat_vis, data_input, labels):
        """Compute ASR / FR-DC on the full embedding, averaged over layers.

        Returns a dict of ``{"fidelity/<name>": mean_over_layers}``. The HD
        reference quantities (S_c, n_HD) are cached per HD array inside the
        metric functions, so they are computed once per dataset and reused
        across layers (and, being deterministic, are identical across methods).
        """
        hd_full = data_input.numpy()
        labels_full = self._classification_labels(labels.numpy() if labels is not None else None)

        per_layer = {"fidelity/asr": [], "fidelity/asr_mpd": [], "fidelity/frdc": []}
        for i in range(lat_vis.shape[0]):
            emb_full = lat_vis[i].numpy()
            finite_mask = np.isfinite(emb_full).all(axis=1) & np.isfinite(hd_full).all(axis=1)
            emb_clean = emb_full[finite_mask]
            if finite_mask.all():
                hd_clean = self._stable_full_hd_reference(hd_full)
            else:
                hd_clean = hd_full[finite_mask]
            labels_clean = labels_full[finite_mask] if labels_full is not None else None
            if len(emb_clean) < 3:
                continue

            asr, asr_mpd = compute_asr(emb_clean, hd_clean, labels_clean)
            frdc = compute_frdc(hd_clean, emb_clean)
            if not np.isnan(asr):
                per_layer["fidelity/asr"].append(float(asr))
            if not np.isnan(asr_mpd):
                per_layer["fidelity/asr_mpd"].append(float(asr_mpd))
            per_layer["fidelity/frdc"].append(float(frdc))

        return {name: float(np.mean(vals)) for name, vals in per_layer.items() if vals}

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

        run = self._get_wandb_run(trainer)
        
        # Get embeddings and high-dim data
        lat_vis, data_input, labels = self.get_embeddings(trainer, pl_module)

        # Collapse health is a property of the complete coordinate set. Compute
        # it before the O(N^2) quality metrics are down-sampled so rare rays are
        # not silently missed by a random subset.
        full_health_values = {}
        for i in range(lat_vis.shape[0]):
            for health_name, health_value in embedding_health_metrics(
                lat_vis[i].numpy()
            ).items():
                full_health_values.setdefault(
                    f"embedding/{health_name}", []
                ).append(float(health_value))

        # Independent density-fidelity metrics (ASR, FR-DC). Computed on the
        # FULL embedding -- before the down_sample step below -- so they use
        # their own subsampling budgets and stay decoupled from the O(N^2)
        # legacy metrics. These break the train/test circularity of
        # density_correlation: ASR is class-level and uses withheld labels;
        # FR-DC fixes the radius and counts neighbours (dual of the kNN-radius
        # density the loss optimizes).
        independent_values = self._independent_density_metrics(lat_vis, data_input, labels)

        # Downsample if necessary for efficiency
        if self.down_sample and data_input.shape[0] > self.down_sample:
            generator = torch.Generator().manual_seed(self.seed)
            indices_t = torch.randperm(
                data_input.shape[0], generator=generator
            )[:self.down_sample]
            data_input = data_input[indices_t]
            lat_vis = lat_vis[:, indices_t, :]
            if labels is not None:
                labels = labels[indices_t]
            
        data_input_np = data_input.numpy()
        labels_np = self._classification_labels(labels.numpy() if labels is not None else None)

        up_dict = {}
        metric_values = dict(full_health_values)
        
        for i in range(lat_vis.shape[0]):
            emb_np = lat_vis[i].numpy()

            # Remove infinite/NaN values
            finite_mask = np.isfinite(emb_np).all(axis=1) & np.isfinite(data_input_np).all(axis=1)
            emb_np_clean = emb_np[finite_mask]
            data_input_clean = data_input_np[finite_mask]
            labels_clean = labels_np[finite_mask] if labels_np is not None else None
            
            if len(emb_np_clean) < 3:
                continue

            metrics = summarize_embedding_metrics(
                hd_data=data_input_clean,
                emb_data=emb_np_clean,
                labels=labels_clean,
                knn_k=self.knn_k,
                density_k=self.density_k,
                seed=self.seed,
                use_mellon=self.use_mellon,
                mellon_pca_dim=self.mellon_pca_dim,
            )

            score_payload = build_metric_log_payload(
                metrics,
                include_primary_aliases=True,
            )
            for log_name, value in score_payload.items():
                metric_values.setdefault(log_name, []).append(float(value))
            
            # Collect full metric logs with flat names.
            for metric_name, value in metrics.items():
                if metric_name not in ALIAS_METRIC_NAMES:
                    metric_values.setdefault(f"fidelity/{metric_name}", []).append(float(value))

        for log_name, values in metric_values.items():
            value = float(np.mean(values))
            up_dict[log_name] = value
            pl_module.log(log_name, value, sync_dist=True)

        if "embedding/collapsed" in up_dict:
            up_dict["val_embedding_collapsed"] = up_dict["embedding/collapsed"]
            pl_module.log(
                "val_embedding_collapsed",
                up_dict["val_embedding_collapsed"],
                sync_dist=True,
            )
        if "embedding/near_line" in up_dict:
            up_dict["val_embedding_near_line"] = up_dict["embedding/near_line"]
            pl_module.log(
                "val_embedding_near_line",
                up_dict["val_embedding_near_line"],
                sync_dist=True,
            )
        if "embedding/nonfinite_fraction" in up_dict:
            up_dict["val_embedding_nonfinite_fraction"] = up_dict[
                "embedding/nonfinite_fraction"
            ]
            pl_module.log(
                "val_embedding_nonfinite_fraction",
                up_dict["val_embedding_nonfinite_fraction"],
                sync_dist=True,
            )

        # Merge the independent density metrics (already averaged over layers).
        for log_name, value in independent_values.items():
            up_dict[log_name] = value
            pl_module.log(log_name, value, sync_dist=True)

        for log_name, value in self._projection_and_bn_diagnostics(
            trainer, pl_module
        ).items():
            up_dict[log_name] = value
            pl_module.log(log_name, value, sync_dist=True)

        self._append_diagnostic_record(epoch_num, up_dict)

        if run is not None and up_dict:
            # We also let PL module log it, but explicit wandb log keeps step unified
            run.log(up_dict, step=safe_wandb_step(trainer, run))
