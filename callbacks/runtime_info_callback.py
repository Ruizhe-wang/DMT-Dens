"""Runtime / environment info callback for appendix reporting.

Captures, per run, the information needed to write a complete experimental
appendix:
  * hardware: hostname, CPU model + core counts, total RAM, GPU model(s)
  * software: python / torch / CUDA versions, OS platform
  * peak memory: GPU peak (allocated + reserved) and CPU peak RSS
  * runtime: wall-clock fit duration
  * log paths: wandb run dir/id/path, trainer root + log dir, checkpoint dir

Outputs:
  1. A JSON file at ``<output_dir>/<run_name>_<timestamp>.json``.
  2. Flattened ``runtime/*`` fields on the wandb run summary, so they export
     to your metrics CSV together with the fidelity metrics.
  3. A concise human-readable block printed to stdout.

Usage (add to a config's trainer.callbacks list):
    - class_path: callbacks.runtime_info_callback.RuntimeInfoCallback
      init_args:
        output_dir: outputs/runtime_info

All third-party / platform-specific imports are guarded so the callback is a
no-op-safe addition on both CPU baseline runs and GPU DiffTree runs.
"""
import os
import json
import time
import socket
import platform
import subprocess
from datetime import datetime, timezone

import torch
import lightning as pl

try:  # Linux-only; best source of process peak RSS
    import resource
except Exception:  # pragma: no cover - Windows/dev hosts
    resource = None

try:
    import psutil
except Exception:
    psutil = None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _cpu_model() -> str:
    name = _safe(platform.processor, "") or ""
    if not name and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return name or "unknown"


def _total_ram_gb():
    if psutil is not None:
        return _safe(lambda: round(psutil.virtual_memory().total / 1e9, 2))
    return _safe(
        lambda: round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2)
    )


def _peak_cpu_rss_gb():
    # resource.ru_maxrss is in KiB on Linux, bytes on macOS.
    if resource is not None:
        ru = _safe(lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if ru:
            scale = 1024 if platform.system() == "Linux" else 1
            return round(ru * scale / 1e9, 3)
    if psutil is not None:  # current RSS, not peak — flagged in payload
        return _safe(lambda: round(psutil.Process().memory_info().rss / 1e9, 3))
    return None


def _nvidia_smi_names():
    out = _safe(
        lambda: subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    )
    if not out:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def _git_commit():
    value = _safe(
        lambda: subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    )
    return value or None


class RuntimeInfoCallback(pl.Callback):
    def __init__(self, output_dir: str = "outputs/runtime_info"):
        super().__init__()
        self.output_dir = output_dir
        self._t0 = None
        self._wall_start = None

    # ---- lifecycle -------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        self._wall_start = datetime.now(timezone.utc).isoformat()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                _safe(lambda i=i: torch.cuda.reset_peak_memory_stats(i))

    def on_fit_end(self, trainer, pl_module):
        runtime_s = None
        if self._t0 is not None:
            runtime_s = round(time.perf_counter() - self._t0, 2)

        payload = {
            "run": self._run_meta(trainer),
            "timing": {
                "wall_start_utc": self._wall_start,
                "wall_end_utc": datetime.now(timezone.utc).isoformat(),
                "fit_runtime_seconds": runtime_s,
                "fit_runtime_hms": _hms(runtime_s),
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": _safe(lambda: torch.backends.cudnn.version()),
                "platform": platform.platform(),
            },
            "hardware": self._hardware(),
            "peak_memory": self._peak_memory(trainer),
            "log_paths": self._log_paths(trainer),
        }

        self._write_json(payload)
        self._log_to_wandb(payload)
        self._print(payload)

    # ---- collectors ------------------------------------------------------
    def _run_meta(self, trainer):
        name = "run"
        logger = getattr(trainer, "logger", None)
        for attr in ("name", "experiment"):
            val = _safe(lambda a=attr: getattr(logger, a))
            if isinstance(val, str) and val:
                name = val
                break
        # Only accept a real string: Lightning's dummy experiment answers every
        # attribute with a bound no-op method, which would otherwise end up in
        # the output filename.
        wandb_name = _safe(lambda: trainer.logger.experiment.name)
        if not isinstance(wandb_name, str):
            wandb_name = None
        return {
            "name": wandb_name or name,
            "hostname": socket.gethostname(),
            "git_commit": _git_commit(),
        }

    def _hardware(self):
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = _safe(lambda i=i: torch.cuda.get_device_properties(i))
                gpus.append({
                    "index": i,
                    "name": _safe(lambda i=i: torch.cuda.get_device_name(i)),
                    "total_mem_gb": round(props.total_memory / 1e9, 2) if props else None,
                })
        return {
            "cpu_model": _cpu_model(),
            "cpu_physical_cores": _safe(lambda: psutil.cpu_count(logical=False)) if psutil else None,
            "cpu_logical_cores": os.cpu_count(),
            "total_ram_gb": _total_ram_gb(),
            "gpus": gpus,
            "gpus_visible_on_node": _nvidia_smi_names(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def _peak_memory(self, trainer):
        gpu_peak = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_peak.append({
                    "index": i,
                    "peak_allocated_gb": _safe(
                        lambda i=i: round(torch.cuda.max_memory_allocated(i) / 1e9, 3)
                    ),
                    "peak_reserved_gb": _safe(
                        lambda i=i: round(torch.cuda.max_memory_reserved(i) / 1e9, 3)
                    ),
                })
        return {
            "accelerator": _safe(lambda: type(trainer.accelerator).__name__),
            "cpu_peak_rss_gb": _peak_cpu_rss_gb(),
            "cpu_peak_is_current_rss_fallback": resource is None,
            "gpu": gpu_peak,
        }

    def _log_paths(self, trainer):
        paths = {
            "default_root_dir": _safe(lambda: str(trainer.default_root_dir)),
            "trainer_log_dir": _safe(lambda: str(trainer.log_dir)),
            "wandb_run_dir": _safe(lambda: trainer.logger.experiment.dir),
            "wandb_run_id": _safe(lambda: trainer.logger.experiment.id),
            "wandb_run_path": _safe(lambda: "/".join(trainer.logger.experiment.path)),
            "cwd": os.getcwd(),
        }
        ckpt = _safe(lambda: getattr(trainer.checkpoint_callback, "dirpath", None))
        if ckpt:
            paths["checkpoint_dir"] = str(ckpt)
        return paths

    # ---- sinks -----------------------------------------------------------
    def _write_json(self, payload):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = str(payload["run"]["name"]).replace("/", "_")[:80]
            path = os.path.join(self.output_dir, f"{safe_name}_{stamp}.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"[RuntimeInfo] wrote {path}")
        except Exception as e:
            print(f"[RuntimeInfo] failed to write json: {e}")

    def _log_to_wandb(self, payload):
        try:
            import wandb
            if wandb.run is None:
                return
            flat = _flatten(payload, prefix="runtime")
            for k, v in flat.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    wandb.run.summary[k] = v
        except Exception as e:
            print(f"[RuntimeInfo] failed to log wandb summary: {e}")

    def _print(self, payload):
        hw = payload["hardware"]
        pm = payload["peak_memory"]
        gpu_names = ", ".join(g.get("name", "?") for g in hw["gpus"]) or "none"
        gpu_peak = max([g.get("peak_allocated_gb") or 0 for g in pm["gpu"]], default=0)
        print(
            "[RuntimeInfo] "
            f"runtime={payload['timing']['fit_runtime_hms']} | "
            f"host={payload['run']['hostname']} | "
            f"cpu={hw['cpu_model']} ({hw['cpu_logical_cores']} threads) | "
            f"ram={hw['total_ram_gb']}GB | "
            f"gpu=[{gpu_names}] | "
            f"gpu_peak_alloc={gpu_peak}GB | "
            f"cpu_peak_rss={pm['cpu_peak_rss_gb']}GB"
        )


def _hms(seconds):
    if seconds is None:
        return None
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _flatten(d, prefix="", sep="/"):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out
