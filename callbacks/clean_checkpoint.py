# callbacks/clean_checkpoint.py
import os
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def _to_cpu_state_dict(state_dict):
    # 避免保存时引用 GPU/storage，统一到 CPU
    return {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in state_dict.items()
    }


class CleanCheckpointCallback(Callback):
    """Save clean weights-only checkpoints (state_dict only), with best+last."""

    def __init__(
        self,
        dirpath: str = "zzl_checkpoints/clean",
        filename_last: str = "weights-last.ckpt",
        filename_best: str = "weights-best-{epoch:04d}-{monitor:.4f}.ckpt",
        monitor: str | None = None,  # e.g., "val_loss" / "loss_all"
        mode: str = "min",  # "min" or "max"
        save_last: bool = True,
        save_best: bool = True,
        every_n_epochs: int | None = None,  # 例如每 100 个 epoch 额外落盘
    ):
        self.dirpath = dirpath
        self.filename_last = filename_last
        self.filename_best = filename_best
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_best = save_best
        self.best_score = None
        self.best_path = None
        self.every_n_epochs = every_n_epochs

    def setup(self, trainer, pl_module, stage: str):
        os.makedirs(self.dirpath, exist_ok=True)

    @rank_zero_only
    def _save_weights(self, trainer, pl_module, path: str, monitor_val=None):
        state = _to_cpu_state_dict(pl_module.state_dict())
        payload = {"state_dict": state}
        if monitor_val is not None:
            payload["monitor"] = float(monitor_val)
        # 幂等保存
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        # 多进程同步
        trainer.strategy.barrier()

    def _is_better(self, current, best):
        if best is None:
            return True
        return (current < best) if self.mode == "min" else (current > best)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        # 每 N 个 epoch 额外落盘（可选）
        if self.every_n_epochs and epoch % self.every_n_epochs == 0:
            path = os.path.join(self.dirpath, f"weights-epoch{epoch:04d}.ckpt")
            self._save_weights(trainer, pl_module, path)

        # 维护 best
        if self.save_best and self.monitor is not None:
            metrics = trainer.callback_metrics
            if self.monitor in metrics and metrics[self.monitor] is not None:
                val = float(metrics[self.monitor])
                if self._is_better(val, self.best_score):
                    self.best_score = val
                    path = os.path.join(
                        self.dirpath,
                        self.filename_best.format(epoch=epoch, monitor=val),
                    )
                    self.best_path = path
                    self._save_weights(trainer, pl_module, path, monitor_val=val)

        # 保存 last
        if self.save_last:
            last_path = os.path.join(self.dirpath, self.filename_last)
            self._save_weights(trainer, pl_module, last_path)

    def on_train_end(self, trainer, pl_module):
        # 训练结束再稳妥落一次 last
        if self.save_last:
            last_path = os.path.join(self.dirpath, self.filename_last)
            self._save_weights(trainer, pl_module, last_path)
