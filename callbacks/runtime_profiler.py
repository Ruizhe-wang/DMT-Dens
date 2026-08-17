import csv
import os
import re
import statistics
import time
from datetime import datetime, timezone

try:
    import lightning as pl
except ModuleNotFoundError:
    class _Callback:
        pass

    class _LightningFallback:
        Callback = _Callback

    pl = _LightningFallback()


def _is_dummy_logger(logger):
    """True for Lightning's no-op logger, installed by ``fast_dev_run``."""
    try:
        from lightning.pytorch.loggers.logger import DummyLogger
    except ModuleNotFoundError:
        return False
    return isinstance(logger, DummyLogger)


class RuntimeProfilerCallback(pl.Callback):
    """Record runtime scalabilty timings to a CSV file."""

    FIELDNAMES = [
        "timestamp",
        "run_name",
        "method",
        "data_name",
        "seed",
        "training_seed",
        "data_seed",
        "sample_data_size",
        "actual_n_samples",
        "batch_size",
        "max_epochs",
        "current_epoch",
        "global_step",
        "fit_wall_time_sec",
        "train_time_sec",
        "validation_time_sec",
        "mean_epoch_time_sec",
        "median_epoch_time_sec",
        "epoch_time_std_sec",
        "steady_epoch_time_sec",
        "peak_cuda_memory_mb",
    ]

    def __init__(
        self,
        output_path="results/runtime/runtime_runs.csv",
        time_fn=None,
        timestamp_fn=None,
    ):
        super().__init__()
        self.output_path = output_path
        self.time_fn = time_fn or time.perf_counter
        self.timestamp_fn = timestamp_fn or self._default_timestamp
        self._fit_start = None
        self._train_epoch_start = None
        self._validation_start = None
        self._train_time_sec = 0.0
        self._validation_time_sec = 0.0
        self._epoch_times = []

    @staticmethod
    def _default_timestamp():
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _safe_getattr(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _reset(self):
        self._fit_start = self.time_fn()
        self._train_epoch_start = None
        self._validation_start = None
        self._train_time_sec = 0.0
        self._validation_time_sec = 0.0
        self._epoch_times = []

    def _reset_cuda_peak_memory(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    @staticmethod
    def _sync_cuda():
        """Make wall-clock boundaries include outstanding asynchronous work."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def _peak_cuda_memory_mb(self):
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return None
        return None

    def _actual_n_samples(self, trainer):
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "dataset", None)
        if dataset is None:
            return None
        try:
            return len(dataset)
        except TypeError:
            return None

    def _run_name(self, trainer):
        logger = getattr(trainer, "logger", None)
        # NOTE: Lightning's WandbLogger.name returns the *project* name, not the
        # per-run name, which collapses every run to one bucket. The real run
        # name (e.g. "runtime_difftree_mnist_n10000", set via logger init_args)
        # lives on the wandb run object at logger.experiment.name. Prefer it.
        experiment = getattr(logger, "experiment", None)
        run_name = getattr(experiment, "name", None)
        if run_name:
            return run_name
        return getattr(logger, "name", None) or "unknown"

    @staticmethod
    def _parse_runtime_run_name(run_name):
        if not isinstance(run_name, str) or not run_name.startswith("runtime_"):
            return None, None
        body = run_name[len("runtime_") :]
        if "_n" in body:
            body = body.rsplit("_n", 1)[0]
        if "_" not in body:
            return body, None
        method, data_name = body.split("_", 1)
        return method, data_name

    @staticmethod
    def _parse_seed_from_run_name(run_name):
        if not isinstance(run_name, str):
            return None
        match = re.search(r"_seed(\d+)(?:$|_)", run_name)
        return int(match.group(1)) if match else None

    def _training_seed(self, trainer, run_name):
        # Lightning's seed_everything exports PL_GLOBAL_SEED. Prefer the
        # explicit run-name suffix as a second source because runtime configs
        # deliberately keep the data-subsampling seed fixed at 42.
        global_seed = os.environ.get("PL_GLOBAL_SEED")
        if global_seed is not None:
            try:
                return int(global_seed)
            except ValueError:
                pass

        parsed_seed = self._parse_seed_from_run_name(run_name)
        if parsed_seed is not None:
            return parsed_seed

        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        config = getattr(experiment, "config", None)
        if config is not None:
            try:
                value = config.get("seed_everything")
                if value is not None:
                    return int(value)
            except (AttributeError, TypeError, ValueError):
                pass
        return None

    def _row(self, trainer, pl_module, fit_wall_time_sec):
        hparams = getattr(pl_module, "hparams", None)
        datamodule = getattr(trainer, "datamodule", None)
        mean_epoch = sum(self._epoch_times) / len(self._epoch_times) if self._epoch_times else None
        median_epoch = statistics.median(self._epoch_times) if self._epoch_times else None
        if len(self._epoch_times) > 1:
            epoch_std = statistics.stdev(self._epoch_times)
        elif self._epoch_times:
            epoch_std = 0.0
        else:
            epoch_std = None
        steady_times = self._epoch_times[5:]
        steady_epoch = sum(steady_times) / len(steady_times) if steady_times else mean_epoch
        peak_memory = self._peak_cuda_memory_mb()
        run_name = self._run_name(trainer)
        parsed_method, parsed_data_name = self._parse_runtime_run_name(run_name)
        training_seed = self._training_seed(trainer, run_name)
        data_seed = self._safe_getattr(datamodule, "seed", None)

        return {
            "timestamp": self.timestamp_fn(),
            "run_name": run_name,
            "method": parsed_method or getattr(pl_module, "method", pl_module.__class__.__name__),
            "data_name": parsed_data_name or self._safe_getattr(hparams, "data_name", None),
            # Keep the historical `seed` column, but make it mean the training
            # seed. Previous versions silently wrote the fixed data seed here,
            # causing seed 43/44 runs to be mislabeled as seed 42.
            "seed": training_seed,
            "training_seed": training_seed,
            "data_seed": data_seed,
            "sample_data_size": self._safe_getattr(datamodule, "sample_data_size", None),
            "actual_n_samples": self._actual_n_samples(trainer),
            "batch_size": self._safe_getattr(datamodule, "batch_size", None),
            "max_epochs": getattr(trainer, "max_epochs", None),
            "current_epoch": getattr(trainer, "current_epoch", None),
            "global_step": getattr(trainer, "global_step", None),
            "fit_wall_time_sec": f"{fit_wall_time_sec:.6f}",
            "train_time_sec": f"{self._train_time_sec:.6f}",
            "validation_time_sec": f"{self._validation_time_sec:.6f}",
            "mean_epoch_time_sec": f"{mean_epoch:.6f}" if mean_epoch is not None else "",
            "median_epoch_time_sec": f"{median_epoch:.6f}" if median_epoch is not None else "",
            "epoch_time_std_sec": f"{epoch_std:.6f}" if epoch_std is not None else "",
            "steady_epoch_time_sec": f"{steady_epoch:.6f}" if steady_epoch is not None else "",
            "peak_cuda_memory_mb": f"{peak_memory:.3f}" if peak_memory is not None else "",
        }

    def _write_row(self, row):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        file_exists = os.path.exists(self.output_path)
        with open(self.output_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _as_number(value):
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _runtime_metrics(self, row):
        metric_fields = [
            "seed",
            "training_seed",
            "data_seed",
            "sample_data_size",
            "actual_n_samples",
            "batch_size",
            "max_epochs",
            "current_epoch",
            "global_step",
            "fit_wall_time_sec",
            "train_time_sec",
            "validation_time_sec",
            "mean_epoch_time_sec",
            "median_epoch_time_sec",
            "epoch_time_std_sec",
            "steady_epoch_time_sec",
            "peak_cuda_memory_mb",
        ]
        metrics = {}
        for field in metric_fields:
            value = self._as_number(row.get(field))
            if value is not None:
                metrics[f"runtime_{field}"] = value
        return metrics

    def _log_metrics(self, trainer, row):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return

        metrics = self._runtime_metrics(row)
        if not metrics:
            return

        experiment = getattr(logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        # Not every logger exposes a mapping here: Lightning's dummy experiment
        # and some offline wandb runs answer any attribute with a no-op
        # callable, which raises TypeError on item assignment.
        if summary is not None and hasattr(summary, "__setitem__"):
            try:
                for key, value in metrics.items():
                    summary[key] = value
                return
            except (TypeError, AttributeError):
                pass

        if hasattr(logger, "log_metrics") and not _is_dummy_logger(logger):
            try:
                logger.log_metrics(
                    metrics, step=self._as_number(row.get("global_step"))
                )
                return
            except Exception:  # noqa: BLE001 - logging must never break training
                pass

        # fast_dev_run swaps in DummyLogger and suppresses logging, so print the
        # runtime numbers instead of losing them. The CSV row is written either
        # way by on_fit_end.
        for key, value in sorted(metrics.items()):
            print(f"[runtime] {key}={value}")

    def on_fit_start(self, trainer, pl_module):
        self._sync_cuda()
        self._reset()
        self._reset_cuda_peak_memory()

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync_cuda()
        self._train_epoch_start = self.time_fn()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._train_epoch_start is None:
            return
        self._sync_cuda()
        elapsed = self.time_fn() - self._train_epoch_start
        self._epoch_times.append(elapsed)
        self._train_time_sec += elapsed
        self._train_epoch_start = None

    def on_validation_start(self, trainer, pl_module):
        self._sync_cuda()
        self._validation_start = self.time_fn()

    def on_validation_end(self, trainer, pl_module):
        if self._validation_start is None:
            return
        self._sync_cuda()
        self._validation_time_sec += self.time_fn() - self._validation_start
        self._validation_start = None

    def on_fit_end(self, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        if self._fit_start is None:
            return
        self._sync_cuda()
        fit_wall_time_sec = self.time_fn() - self._fit_start
        row = self._row(trainer, pl_module, fit_wall_time_sec)
        self._write_row(row)
        self._log_metrics(trainer, row)
