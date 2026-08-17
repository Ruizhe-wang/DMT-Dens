"""Minimal callback that prints peak GPU memory of the training steps.

Captures the maximum CUDA memory allocated during training batches (the
batch-bounded training peak), independent of any later validation/eval
callbacks, and prints it once at fit end as a single grep-able line:

    PEAKMEM_MB <value>

Append at run time without editing configs:
    --trainer.callbacks+=callbacks.peakmem_print.PeakMemPrint
"""
import torch
import lightning as pl


class PeakMemPrint(pl.Callback):
    def __init__(self):
        super().__init__()
        self._peak_bytes = 0

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if torch.cuda.is_available():
            self._peak_bytes = max(self._peak_bytes, torch.cuda.max_memory_allocated())

    def on_fit_end(self, trainer, pl_module):
        mb = self._peak_bytes / (1024.0 * 1024.0)
        print(f"PEAKMEM_MB {mb:.1f}")
        # Also surface to the wandb run summary when running under a sweep.
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary["gpu_peak_mem_mb"] = round(mb, 1)
        except Exception:
            pass
