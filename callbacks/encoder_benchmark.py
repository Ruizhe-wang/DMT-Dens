"""Training-health and capacity logging for the encoder benchmark.

By default this callback is observational. Diagnostic configs may enable
``fail_on_nonfinite`` to terminate a run immediately after recording the first
non-finite loss, unscaled gradient, or parameter.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

try:
    import lightning as pl
    from lightning.pytorch.loggers.logger import DummyLogger
except ModuleNotFoundError:  # Allows pure helper/unit tests without Lightning.

    class _Callback:
        pass

    class _LightningFallback:
        Callback = _Callback

    class DummyLogger:  # type: ignore[no-redef]
        pass

    pl = _LightningFallback()


LOSS_NAMES = ("manifold_loss", "density_loss", "total_loss")


def loss_stability_metrics(
    values,
    *,
    cv_threshold=0.25,
    direction_change_threshold=0.5,
    spike_ratio_threshold=3.0,
):
    """Return deterministic tail-stability metrics for an epoch-loss series."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "tail_cv": float("nan"),
            "direction_change_rate": float("nan"),
            "spike_ratio": float("nan"),
            "oscillation_detected": True,
        }

    magnitude = np.abs(array)
    mean_abs = max(float(np.mean(magnitude)), 1.0e-12)
    tail_cv = float(np.std(array) / mean_abs)
    median_abs = max(float(np.median(magnitude)), 1.0e-12)
    spike_ratio = float(np.max(magnitude) / median_abs)

    deltas = np.diff(array)
    signs = np.sign(deltas[np.abs(deltas) > 1.0e-12])
    if signs.size < 2:
        direction_change_rate = 0.0
    else:
        direction_change_rate = float(np.mean(signs[1:] != signs[:-1]))

    oscillation = bool(
        (
            tail_cv >= float(cv_threshold)
            and direction_change_rate >= float(direction_change_threshold)
        )
        or spike_ratio >= float(spike_ratio_threshold)
    )
    return {
        "tail_cv": tail_cv,
        "direction_change_rate": direction_change_rate,
        "spike_ratio": spike_ratio,
        "oscillation_detected": oscillation,
    }


def embedding_health_metrics(
    embedding,
    *,
    std_threshold=1.0e-6,
    axis_ratio_threshold=1.0e-3,
    near_line_axis_ratio_threshold=5.0e-2,
):
    """Measure non-finite output and point/line collapse in a 2-D embedding."""
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[1] < 2 or embedding.shape[0] < 2:
        return {
            "nonfinite_fraction": 1.0,
            "std_min": 0.0,
            "axis_ratio": 0.0,
            "axis_ratio_trim_top_0_1pct": 0.0,
            "norm_median": 0.0,
            "norm_max": float("nan"),
            "norm_max_median_ratio": float("nan"),
            "near_line": 1.0,
            "collapsed": 1.0,
        }

    finite_rows = np.isfinite(embedding).all(axis=1)
    nonfinite_fraction = float(1.0 - np.mean(finite_rows))
    finite = embedding[finite_rows]
    if finite.shape[0] < 2:
        return {
            "nonfinite_fraction": nonfinite_fraction,
            "std_min": 0.0,
            "axis_ratio": 0.0,
            "axis_ratio_trim_top_0_1pct": 0.0,
            "norm_median": 0.0,
            "norm_max": float("nan"),
            "norm_max_median_ratio": float("nan"),
            "near_line": 1.0,
            "collapsed": 1.0,
        }

    centered = finite - finite.mean(axis=0, keepdims=True)
    std_min = float(np.min(np.std(centered, axis=0)))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest = max(float(singular_values[0]), 1.0e-12)
    axis_ratio = float(singular_values[1] / largest)
    # Keep norm tails in the model's actual coordinate system. The collapse
    # report's ray amplification is an origin-relative phenomenon; centering
    # here would hide a shared offset and change max/median comparability.
    norms = np.linalg.norm(finite, axis=1)
    norm_median = float(np.median(norms))
    norm_max = float(np.max(norms))
    norm_max_median_ratio = float(norm_max / max(norm_median, 1.0e-12))

    trim_count = int(np.ceil(0.001 * finite.shape[0]))
    if finite.shape[0] - trim_count >= 2 and trim_count > 0:
        keep = np.argsort(norms)[: finite.shape[0] - trim_count]
        trimmed = finite[keep]
        trimmed = trimmed - trimmed.mean(axis=0, keepdims=True)
        trimmed_singular_values = np.linalg.svd(trimmed, compute_uv=False)
        trimmed_largest = max(float(trimmed_singular_values[0]), 1.0e-12)
        axis_ratio_trimmed = float(trimmed_singular_values[1] / trimmed_largest)
    else:
        axis_ratio_trimmed = axis_ratio
    collapsed = bool(
        nonfinite_fraction > 0.0
        or std_min < float(std_threshold)
        or axis_ratio < float(axis_ratio_threshold)
    )
    near_line = bool(
        nonfinite_fraction > 0.0
        or std_min < float(std_threshold)
        or axis_ratio < float(near_line_axis_ratio_threshold)
    )
    return {
        "nonfinite_fraction": nonfinite_fraction,
        "std_min": std_min,
        "axis_ratio": axis_ratio,
        "axis_ratio_trim_top_0_1pct": axis_ratio_trimmed,
        "norm_median": norm_median,
        "norm_max": norm_max,
        "norm_max_median_ratio": norm_max_median_ratio,
        "near_line": float(near_line),
        "collapsed": float(collapsed),
    }


class EncoderBenchmarkCallback(pl.Callback):
    """Log final losses, numerical health, oscillation and model capacity."""

    def __init__(
        self,
        tail_epochs=20,
        cv_threshold=0.25,
        direction_change_threshold=0.5,
        spike_ratio_threshold=3.0,
        check_grad_every_n_steps=50,
        fail_on_nonfinite=False,
    ):
        super().__init__()
        self.tail_epochs = int(tail_epochs)
        self.cv_threshold = float(cv_threshold)
        self.direction_change_threshold = float(direction_change_threshold)
        self.spike_ratio_threshold = float(spike_ratio_threshold)
        self.check_grad_every_n_steps = int(check_grad_every_n_steps)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        self._epoch_values = defaultdict(list)
        self._history = defaultdict(list)
        self._nonfinite_loss = False
        self._nonfinite_gradient = False
        self._nonfinite_gradient_param = None
        self._nonfinite_parameter = False
        self._first_nonfinite_kind = None
        self._first_nonfinite_name = None
        self._first_nonfinite_epoch = None
        self._first_nonfinite_step = None
        self._amp_overflow_count = 0
        self._first_amp_overflow_parameter = None
        self._first_amp_overflow_epoch = None
        self._first_amp_overflow_step = None

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().float().cpu().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _logger_summary(trainer):
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        # Lightning's dummy experiment can answer attributes with no-op
        # callables rather than a mapping.
        # Only something that actually supports item assignment is usable.
        if summary is None or not hasattr(summary, "__setitem__"):
            return None
        return summary

    @staticmethod
    def _log_metrics(trainer, metrics):
        """Deliver metrics to the run record, or print them if that is not possible.

        ``fast_dev_run`` deliberately swaps the real logger for ``DummyLogger``
        ("Logging and checkpointing is suppressed"), which is exactly the mode
        used for the required two-batch smoke test -- the mode where the
        capacity and batch-size numbers matter most. Rather than let them
        vanish, they are printed whenever the logger cannot take them.
        """
        summary = EncoderBenchmarkCallback._logger_summary(trainer)
        if summary is not None:
            try:
                for key, value in metrics.items():
                    summary[key] = value
                return
            except (TypeError, AttributeError):
                pass  # fall through to the metric stream

        logger = getattr(trainer, "logger", None)
        if (
            logger is not None
            and hasattr(logger, "log_metrics")
            and not isinstance(logger, DummyLogger)
        ):
            try:
                logger.log_metrics(
                    metrics, step=int(getattr(trainer, "global_step", 0))
                )
                return
            except Exception:  # noqa: BLE001 - logging must never break training
                pass

        for key, value in sorted(metrics.items()):
            print(f"[encoder-bench] {key}={value}")

    @staticmethod
    def _log_history_metrics(trainer, metrics):
        """Write numeric step metrics to the logger history, not just summary."""
        logger = getattr(trainer, "logger", None)
        if logger is not None and hasattr(logger, "log_metrics"):
            try:
                logger.log_metrics(
                    metrics, step=int(getattr(trainer, "global_step", 0))
                )
                return
            except Exception:  # noqa: BLE001 - diagnostics cannot hide training
                pass
        for key, value in sorted(metrics.items()):
            print(f"[encoder-bench] {key}={value}")

    def _record_nonfinite(self, trainer, kind, name):
        """Persist the first numerical failure and optionally abort the run."""
        if self._first_nonfinite_kind is None:
            self._first_nonfinite_kind = str(kind)
            self._first_nonfinite_name = str(name)
            self._first_nonfinite_epoch = int(getattr(trainer, "current_epoch", -1))
            self._first_nonfinite_step = int(getattr(trainer, "global_step", -1))
            self._log_metrics(
                trainer,
                {
                    "numerics/first_nonfinite_kind": self._first_nonfinite_kind,
                    "numerics/first_nonfinite_parameter": self._first_nonfinite_name,
                    "numerics/first_nonfinite_epoch": self._first_nonfinite_epoch,
                    "numerics/first_nonfinite_step": self._first_nonfinite_step,
                },
            )
            print(
                "[encoder-bench] first non-finite "
                f"{kind} at {name} (epoch={self._first_nonfinite_epoch}, "
                f"step={self._first_nonfinite_step})"
            )
        if self.fail_on_nonfinite:
            trainer.should_stop = True
            raise FloatingPointError(
                f"Non-finite {kind} detected at {name}; stopping diagnostic run"
            )

    def on_fit_start(self, trainer, pl_module):
        self._epoch_values.clear()
        self._history.clear()
        self._nonfinite_loss = False
        self._nonfinite_gradient = False
        self._nonfinite_gradient_param = None
        self._nonfinite_parameter = False
        self._first_nonfinite_kind = None
        self._first_nonfinite_name = None
        self._first_nonfinite_epoch = None
        self._first_nonfinite_step = None
        self._amp_overflow_count = 0
        self._first_amp_overflow_parameter = None
        self._first_amp_overflow_epoch = None
        self._first_amp_overflow_step = None

        capacity = getattr(pl_module, "encoder_capacity", {}) or {}
        datamodule = getattr(trainer, "datamodule", None)
        batch_size = getattr(datamodule, "batch_size", None)
        total_params = sum(
            parameter.numel()
            for parameter in pl_module.parameters()
            if parameter.requires_grad
        )
        metrics = {
            "engineering/total_trainable_params": int(total_params),
            "engineering/encoder_params": int(capacity.get("params", 0)),
            "engineering/baseline_encoder_params": int(
                capacity.get("baseline_params", 0)
            ),
            "engineering/encoder_param_ratio": float(
                capacity.get("param_ratio", float("nan"))
            ),
            "engineering/encoder_param_in_band": int(
                bool(capacity.get("param_in_band", False))
            ),
        }
        if batch_size is not None:
            metrics["engineering/actual_batch_size"] = int(batch_size)
        self._log_metrics(trainer, metrics)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_values.clear()

    def on_before_backward(self, trainer, pl_module, loss):
        """Stop before backward if the scalar objective is already invalid."""
        if not torch.isfinite(loss.detach()).all():
            self._nonfinite_loss = True
            self._record_nonfinite(trainer, "loss", "training_loss")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        latest = getattr(pl_module, "_latest_training_losses", None) or {}
        for name in LOSS_NAMES:
            value = self._as_float(latest.get(name))
            if value is None:
                continue
            if math.isfinite(value):
                self._epoch_values[name].append(value)
            else:
                self._nonfinite_loss = True
                self._record_nonfinite(trainer, "loss", name)

        # This hook runs after the automatic optimizer step. Check every batch
        # so a corrupted parameter cannot survive until the end of the epoch.
        for name, parameter in pl_module.named_parameters():
            if not torch.isfinite(parameter.detach()).all():
                self._nonfinite_parameter = True
                self._record_nonfinite(trainer, "parameter", name)
                break

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Checks gradients after AMP unscaling.

        Deliberately not ``on_after_backward``: under ``precision: 16-mixed``
        gradients are still multiplied by the GradScaler factor at that point,
        and overflowing to inf there is normal, benign scaler behaviour (the
        step is skipped and the scale is lowered). Checking there would report
        a non-finite gradient on virtually every fp16 run and make the flag
        useless. By this hook the gradients have been unscaled, so a non-finite
        value is a genuine problem.
        """
        interval = self.check_grad_every_n_steps
        if interval <= 0:
            return
        global_step = int(getattr(trainer, "global_step", 0))
        if global_step % interval != 0:
            return

        # The final projection BatchNorm scale is the suspected collapse path.
        # Gradients are unscaled at this hook, so these are physically useful
        # curves under mixed precision rather than GradScaler artefacts.
        try:
            final_bn = pl_module.vis_list[-1][-1].block[1]
            gamma_grad = final_bn.weight.grad
            if gamma_grad is not None and gamma_grad.numel() >= 2:
                gamma_grad = gamma_grad.detach().float()
                self._log_history_metrics(
                    trainer,
                    {
                        "projection_bn/gamma_grad_axis0": float(gamma_grad[0]),
                        "projection_bn/gamma_grad_axis1": float(gamma_grad[1]),
                        "projection_bn/gamma_grad_norm": float(gamma_grad.norm()),
                    },
                )
        except (AttributeError, IndexError, TypeError):
            pass

        nonfinite_gradient_names = [
            name
            for name, parameter in pl_module.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if not nonfinite_gradient_names:
            return

        first_name = nonfinite_gradient_names[0]
        scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
        if scaler is not None and hasattr(scaler, "get_scale"):
            # GradScaler intentionally leaves Inf/NaN gradients visible after
            # unscaling. scaler.step() consumes that signal, skips the unsafe
            # optimizer step, and lowers the scale. Raising from this hook runs
            # before scaler.step() and incorrectly turns a recoverable overflow
            # into a fatal training error.
            self._amp_overflow_count += 1
            try:
                amp_scale = float(scaler.get_scale())
            except (TypeError, ValueError):
                amp_scale = float("nan")
            if self._first_amp_overflow_parameter is None:
                self._first_amp_overflow_parameter = first_name
                self._first_amp_overflow_epoch = int(
                    getattr(trainer, "current_epoch", -1)
                )
                self._first_amp_overflow_step = global_step
                print(
                    "[encoder-bench] first recoverable AMP gradient overflow at "
                    f"{first_name} (epoch={self._first_amp_overflow_epoch}, "
                    f"step={global_step}, scale={amp_scale})"
                )
            self._log_history_metrics(
                trainer,
                {
                    "numerics/amp_overflow_count": self._amp_overflow_count,
                    "numerics/amp_scale_at_overflow": amp_scale,
                },
            )
            return

        self._nonfinite_gradient = True
        self._nonfinite_gradient_param = first_name
        self._record_nonfinite(trainer, "gradient", first_name)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_metrics = {}
        for name in LOSS_NAMES:
            values = self._epoch_values.get(name, [])
            if not values:
                continue
            mean_value = float(np.mean(values))
            self._history[name].append(mean_value)
            epoch_metrics[f"train_epoch/{name}"] = mean_value

        for parameter in pl_module.parameters():
            if not torch.isfinite(parameter.detach()).all():
                self._nonfinite_parameter = True
                break

        if epoch_metrics:
            logger = getattr(trainer, "logger", None)
            if logger is not None and hasattr(logger, "log_metrics"):
                logger.log_metrics(
                    epoch_metrics,
                    step=int(getattr(trainer, "global_step", 0)),
                )

    def _final_metrics(self):
        metrics = {}
        oscillation_any = False
        for name in LOSS_NAMES:
            history = self._history.get(name, [])
            if history:
                metrics[f"train_status/final_{name}"] = float(history[-1])
            tail = history[-max(1, self.tail_epochs) :]
            stability = loss_stability_metrics(
                tail,
                cv_threshold=self.cv_threshold,
                direction_change_threshold=self.direction_change_threshold,
                spike_ratio_threshold=self.spike_ratio_threshold,
            )
            for metric_name, value in stability.items():
                if metric_name == "oscillation_detected":
                    oscillation_any = oscillation_any or bool(value)
                    metrics[f"train_status/{name}_{metric_name}"] = int(bool(value))
                else:
                    metrics[f"train_status/{name}_{metric_name}"] = float(value)

        nonfinite_any = bool(
            self._nonfinite_loss
            or self._nonfinite_gradient
            or self._nonfinite_parameter
        )
        metrics.update(
            {
                "train_status/nonfinite_loss_detected": int(self._nonfinite_loss),
                "train_status/nonfinite_gradient_detected": int(
                    self._nonfinite_gradient
                ),
                "train_status/nonfinite_parameter_detected": int(
                    self._nonfinite_parameter
                ),
                "train_status/nonfinite_detected": int(nonfinite_any),
                "train_status/amp_overflow_count": int(self._amp_overflow_count),
                "train_status/oscillation_detected": int(oscillation_any),
            }
        )
        if self._first_amp_overflow_parameter is not None:
            metrics.update(
                {
                    "numerics/first_amp_overflow_parameter": self._first_amp_overflow_parameter,
                    "numerics/first_amp_overflow_epoch": self._first_amp_overflow_epoch,
                    "numerics/first_amp_overflow_step": self._first_amp_overflow_step,
                }
            )
        if self._first_nonfinite_kind is not None:
            metrics.update(
                {
                    "numerics/first_nonfinite_kind": self._first_nonfinite_kind,
                    "numerics/first_nonfinite_parameter": self._first_nonfinite_name,
                    "numerics/first_nonfinite_epoch": self._first_nonfinite_epoch,
                    "numerics/first_nonfinite_step": self._first_nonfinite_step,
                }
            )
        return metrics

    def on_fit_end(self, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        self._log_metrics(trainer, self._final_metrics())
