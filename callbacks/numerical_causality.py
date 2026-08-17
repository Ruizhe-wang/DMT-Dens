"""Opt-in backward tracing for short numerical-causality experiments.

The regular benchmark callback intentionally inspects gradients only after AMP
unscaling.  This callback complements it by recording both the scaled backward
pass and the unscaled optimizer-step state in a JSONL artifact.  It is meant for
short canary runs, not routine training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

try:
    import lightning as pl
except ModuleNotFoundError:  # Allows pure helper/unit tests without Lightning.

    class _Callback:
        pass

    class _LightningFallback:
        Callback = _Callback

    pl = _LightningFallback()


def _tensor_records(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_records(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_records(item)


def _nonfinite_tensor_summary(value):
    """Return a compact description of the first non-finite tensor, if any."""
    for tensor in _tensor_records(value):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        if finite.all():
            continue
        finite_fraction = float(finite.float().mean().cpu())
        finite_values = detached[finite]
        max_abs_finite = (
            float(finite_values.float().abs().max().cpu())
            if finite_values.numel()
            else None
        )
        return {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "finite_fraction": finite_fraction,
            "max_abs_finite": max_abs_finite,
        }
    return None


class NumericalCausalityCallback(pl.Callback):
    """Trace the first module and parameter touched by a non-finite backward."""

    def __init__(self, output_path, trace_leaf_modules=True):
        super().__init__()
        self.output_path = str(output_path)
        self.trace_leaf_modules = bool(trace_leaf_modules)
        self._handles = []
        self._trainer = None
        self._batch_idx = None
        self._loss = None
        self._amp_scale = None
        self._module_event = None

    @staticmethod
    def _scaler_scale(trainer):
        scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
        if scaler is None or not hasattr(scaler, "get_scale"):
            return None
        try:
            return float(scaler.get_scale())
        except (TypeError, ValueError):
            return None

    def _append(self, phase, **payload):
        trainer = self._trainer
        record = {
            "phase": phase,
            "epoch": int(getattr(trainer, "current_epoch", -1)),
            "step": int(getattr(trainer, "global_step", -1)),
            "batch_idx": self._batch_idx,
            "loss": self._loss,
            "amp_scale": self._amp_scale,
            **payload,
        }
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _module_hook(self, name):
        def hook(_module, grad_input, grad_output):
            if self._module_event is not None:
                return
            for kind, value in (("grad_output", grad_output), ("grad_input", grad_input)):
                summary = _nonfinite_tensor_summary(value)
                if summary is not None:
                    self._module_event = {"module": name, "kind": kind, **summary}
                    print(
                        "[numerics-causal] first scaled-backward non-finite "
                        f"at {name}.{kind} (epoch={getattr(self._trainer, 'current_epoch', -1)}, "
                        f"step={getattr(self._trainer, 'global_step', -1)}, "
                        f"amp_scale={self._amp_scale})"
                    )
                    return

        return hook

    def on_fit_start(self, trainer, pl_module):
        self._trainer = trainer
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        if not self.trace_leaf_modules:
            return
        for name, module in pl_module.named_modules():
            if not name or any(module.children()):
                continue
            self._handles.append(module.register_full_backward_hook(self._module_hook(name)))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module, batch
        self._trainer = trainer
        self._batch_idx = int(batch_idx)
        self._loss = None
        self._amp_scale = self._scaler_scale(trainer)
        self._module_event = None

    def on_before_backward(self, trainer, pl_module, loss):
        del pl_module
        self._trainer = trainer
        scalar = float(loss.detach().float().cpu())
        self._loss = scalar if math.isfinite(scalar) else str(scalar)
        self._amp_scale = self._scaler_scale(trainer)

    def on_after_backward(self, trainer, pl_module):
        self._trainer = trainer
        scaled_nonfinite = [
            name
            for name, parameter in pl_module.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if self._module_event is not None or scaled_nonfinite:
            self._append(
                "after_backward_scaled",
                first_module_event=self._module_event,
                nonfinite_parameters=scaled_nonfinite,
            )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del optimizer
        self._trainer = trainer
        nonfinite = []
        max_abs = 0.0
        max_abs_parameter = None
        for name, parameter in pl_module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            finite = torch.isfinite(gradient)
            if not finite.all():
                nonfinite.append(name)
            finite_values = gradient.detach()[finite]
            if finite_values.numel():
                candidate = float(finite_values.float().abs().max().cpu())
                if candidate > max_abs:
                    max_abs = candidate
                    max_abs_parameter = name
        self._append(
            "before_optimizer_step_unscaled",
            nonfinite_parameters=nonfinite,
            max_abs_finite_gradient=max_abs,
            max_abs_finite_gradient_parameter=max_abs_parameter,
            first_module_event=self._module_event,
        )

    def on_fit_end(self, trainer, pl_module):
        del trainer, pl_module
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

