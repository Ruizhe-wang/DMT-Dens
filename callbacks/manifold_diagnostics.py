"""Training-time diagnostics for the manifold affinity and hard-pair loss."""

from collections import defaultdict

import lightning as pl
import torch


class ManifoldDiagnosticsCallback(pl.Callback):
    """Aggregate scalar manifold diagnostics and write them to the logger.

    The model computes only scalar summaries from its internal pairwise tensors.
    This callback samples those summaries every ``every_n_steps`` training steps,
    aggregates them over an epoch, and writes ``diagnostics/manifold/*`` metrics
    to Lightning's configured logger (including W&B).
    """

    def __init__(
        self,
        every_n_steps: int = 1,
        log_every_n_epochs: int = 1,
        prefix: str = "diagnostics/manifold",
    ):
        super().__init__()
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        if log_every_n_epochs <= 0:
            raise ValueError("log_every_n_epochs must be positive")

        self.every_n_steps = int(every_n_steps)
        self.log_every_n_epochs = int(log_every_n_epochs)
        self.prefix = prefix.rstrip("/")
        self._reset()

    def _reset(self):
        self._sums = defaultdict(float)
        self._observations = defaultdict(int)
        self._mins = {}
        self._maxes = {}
        self._sampled_batches = 0

    def on_fit_start(self, trainer, pl_module):
        pl_module._manifold_diagnostics_enabled = True
        pl_module._manifold_diagnostics_every_n_steps = self.every_n_steps

    def on_fit_end(self, trainer, pl_module):
        pl_module._manifold_diagnostics_enabled = False
        pl_module._latest_manifold_diagnostics = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        diagnostics = getattr(pl_module, "_latest_manifold_diagnostics", None)
        if not diagnostics:
            return

        # Consume each sampled batch exactly once.
        pl_module._latest_manifold_diagnostics = None
        self._sampled_batches += 1

        for name, value in diagnostics.items():
            value = torch.as_tensor(value).detach().float().cpu().item()
            if name.endswith("_min"):
                self._mins[name] = min(self._mins.get(name, value), value)
            elif name.endswith("_max"):
                self._maxes[name] = max(self._maxes.get(name, value), value)
            elif name.endswith("_count"):
                self._sums[name] += value
            else:
                self._sums[name] += value
                self._observations[name] += 1

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_number = trainer.current_epoch + 1
        if self._sampled_batches == 0:
            return
        if epoch_number % self.log_every_n_epochs != 0:
            return
        if not trainer.is_global_zero or trainer.logger is None:
            return

        metrics = {
            f"{self.prefix}/{name}": value
            for name, value in self._mins.items()
        }
        metrics.update(
            {
                f"{self.prefix}/{name}": value
                for name, value in self._maxes.items()
            }
        )

        for name, total in self._sums.items():
            if name.endswith("_count"):
                metrics[f"{self.prefix}/{name}"] = total
            else:
                metrics[f"{self.prefix}/{name}"] = (
                    total / max(1, self._observations[name])
                )

        metrics[f"{self.prefix}/sampled_batches"] = self._sampled_batches
        metrics[f"{self.prefix}/epoch"] = epoch_number
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
