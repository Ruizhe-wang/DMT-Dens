#!/usr/bin/env python
"""Collect hardware / environment info for the paper appendix.

Run this ONCE on the machine where the runs were executed. It gathers:
  * 硬件配置 / hardware: hostname, OS, CPU model + cores, total RAM
  * GPU / CPU: GPU model(s) + VRAM, driver + CUDA version
  * peak memory: peak GPU memory used by the runs (summarized from the runtime
    CSV produced by callbacks.runtime_profiler.RuntimeProfilerCallback), plus
    GPU capacity for reference
  * 日志路径 / log paths: where wandb logs, runtime CSV, and outputs live

Outputs (both written + printed):
  * <out-dir>/hardware_info.json   -- structured, full detail
  * <out-dir>/hardware_info.md     -- ready-to-paste appendix block

Usage:
  python tools/collect_hardware_info.py
  python tools/collect_hardware_info.py --runtime-csv results/runtime/runtime_runs.csv --out-dir results/appendix

All third-party imports are optional; the script degrades gracefully.
"""
import os
import re
import csv
import json
import socket
import argparse
import platform
import subprocess
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _run(cmd, timeout=10):
    return _safe(
        lambda: subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    )


def cpu_model():
    # Linux: /proc/cpuinfo carries the real model name. platform.processor()
    # often returns just the arch ("x86_64"), so try the descriptive sources
    # first and only fall back to the arch token.
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    lscpu = _run(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    mac = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if mac:
        return mac
    return _safe(platform.processor, "") or "unknown"


def total_ram_gb():
    if psutil is not None:
        return _safe(lambda: round(psutil.virtual_memory().total / 1e9, 1))
    return _safe(
        lambda: round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    )


def collect_hardware():
    hw = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_physical_cores": _safe(lambda: psutil.cpu_count(logical=False)) if psutil else None,
        "cpu_logical_cores": os.cpu_count(),
        "total_ram_gb": total_ram_gb(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [],
        "gpu_driver_version": None,
        "cuda_version": None,
    }

    # Prefer torch for per-device VRAM, fall back to nvidia-smi for everything.
    if torch is not None and _safe(lambda: torch.cuda.is_available()):
        hw["cuda_version"] = _safe(lambda: torch.version.cuda)
        for i in range(_safe(lambda: torch.cuda.device_count(), 0) or 0):
            props = _safe(lambda i=i: torch.cuda.get_device_properties(i))
            hw["gpus"].append({
                "index": i,
                "name": _safe(lambda i=i: torch.cuda.get_device_name(i)),
                "total_mem_gb": round(props.total_memory / 1e9, 1) if props else None,
            })

    smi = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    if smi:
        rows = [r.strip() for r in smi.splitlines() if r.strip()]
        if not hw["gpus"]:
            for i, r in enumerate(rows):
                parts = [p.strip() for p in r.split(",")]
                hw["gpus"].append({
                    "index": i,
                    "name": parts[0] if parts else None,
                    "total_mem_gb": _safe(
                        lambda p=parts: round(float(p[1].split()[0]) / 1024, 1)
                    ) if len(parts) > 1 else None,
                })
        if rows and hw["gpu_driver_version"] is None:
            last = [p.strip() for p in rows[0].split(",")]
            if len(last) > 2:
                hw["gpu_driver_version"] = last[2]
    if hw["cuda_version"] is None:
        header = _run(["nvidia-smi"])
        if header and "CUDA Version" in header:
            hw["cuda_version"] = _safe(
                lambda: header.split("CUDA Version:")[1].split()[0]
            )
    return hw


def collect_software():
    return {
        "python": platform.python_version(),
        "torch": _safe(lambda: torch.__version__) if torch else None,
        "torch_cuda_build": _safe(lambda: torch.version.cuda) if torch else None,
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def summarize_runtime_csv(path):
    """Peak memory + runtime summary from RuntimeProfilerCallback output."""
    if not os.path.exists(path):
        return {"available": False, "csv_path": os.path.abspath(path)}
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return {"available": False, "csv_path": os.path.abspath(path), "error": str(e)}

    def fnum(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return None

    peaks = [p for p in (fnum(r, "peak_cuda_memory_mb") for r in rows) if p is not None]
    times = [t for t in (fnum(r, "fit_wall_time_sec") for r in rows) if t is not None]

    per_config = {}
    for r in rows:
        key = _config_key(r)
        pk = fnum(r, "peak_cuda_memory_mb")
        tm = fnum(r, "fit_wall_time_sec")
        agg = per_config.setdefault(key, {"peak_mb": [], "time_s": []})
        if pk is not None:
            agg["peak_mb"].append(pk)
        if tm is not None:
            agg["time_s"].append(tm)

    per_config_out = {}
    for key, v in sorted(per_config.items()):
        method, dataset, n = _split_run_config(key)
        per_config_out[key] = {
            "method": method,
            "dataset": dataset,
            "n": n,
            "max_peak_gpu_mem_mb": round(max(v["peak_mb"]), 1) if v["peak_mb"] else None,
            "max_runtime_sec": round(max(v["time_s"]), 1) if v["time_s"] else None,
            "n_runs": max(len(v["peak_mb"]), len(v["time_s"])),
        }

    return {
        "available": True,
        "csv_path": os.path.abspath(path),
        "n_runs": len(rows),
        "overall_peak_gpu_mem_mb": round(max(peaks), 1) if peaks else None,
        "overall_max_runtime_sec": round(max(times), 1) if times else None,
        "per_config": per_config_out,
    }


def _config_key(r):
    """Stable grouping key per run config; run_name is present on every row,
    unlike data_name which is empty for the DiffTree GPU runs."""
    name = (r.get("run_name") or "").strip()
    if name:
        name = re.sub(r"^runtime_", "", name)
        name = re.sub(r"_seed\d+$", "", name)
        return name or "unknown"
    return r.get("data_name") or "unknown"


def _split_run_config(key):
    """'difftree_mnist_n10000' -> ('difftree', 'mnist', 10000)."""
    m = re.match(r"^(.+?)_(.+)_n(\d+)$", key)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, key, None


def collect_log_paths(runtime_csv):
    candidates = {
        "wandb_dir": "wandb",
        "runtime_csv": runtime_csv,
        "runtime_dir": "results/runtime",
        "runtime_info_json_dir": "outputs/runtime_info",
        "outputs_dir": "outputs",
        "results_dir": "results",
    }
    return {
        k: {"path": os.path.abspath(v), "exists": os.path.exists(v)}
        for k, v in candidates.items()
    }


def to_markdown(payload):
    hw = payload["hardware"]
    sw = payload["software"]
    rt = payload["runtime_summary"]
    lp = payload["log_paths"]

    gpu_lines = "; ".join(
        f"{g.get('name','?')} ({g.get('total_mem_gb','?')} GB)" for g in hw["gpus"]
    ) or "none detected"

    lines = [
        "## Appendix: hardware & runtime",
        "",
        "| item | value |",
        "|------|-------|",
        f"| Host | {hw['hostname']} |",
        f"| OS | {hw['platform']} |",
        f"| CPU | {hw['cpu_model']} ({hw['cpu_physical_cores']} cores / {hw['cpu_logical_cores']} threads) |",
        f"| RAM | {hw['total_ram_gb']} GB |",
        f"| GPU | {gpu_lines} |",
        f"| GPU driver | {hw['gpu_driver_version']} |",
        f"| CUDA | {hw['cuda_version']} |",
        f"| Python / Torch | {sw['python']} / {sw['torch']} |",
    ]
    if rt.get("available"):
        lines += [
            f"| Peak GPU memory (runs) | {rt['overall_peak_gpu_mem_mb']} MB |",
            f"| Max wall-clock runtime | {rt['overall_max_runtime_sec']} s |",
            f"| Runtime runs | {rt['n_runs']} |",
        ]
    lines += ["", "### Log paths", ""]
    for k, v in lp.items():
        mark = "" if v["exists"] else "  (missing)"
        lines.append(f"- **{k}**: `{v['path']}`{mark}")

    if rt.get("available") and rt.get("per_config"):
        lines += ["", "### Peak GPU memory / runtime per run config", "",
                  "_Peak GPU memory is 0 for CPU-only baselines "
                  "(t-SNE/UMAP/PaCMAP/PHATE/densNE/densMAP); only DiffTree runs on GPU._",
                  "",
                  "| method | dataset | N | max peak GPU mem (MB) | max runtime (s) | seeds |",
                  "|--------|---------|---|----------------------|-----------------|-------|"]
        for key, v in rt["per_config"].items():
            n = v["n"] if v["n"] is not None else ""
            lines.append(
                f"| {v['method'] or ''} | {v['dataset'] or key} | {n} "
                f"| {v['max_peak_gpu_mem_mb']} | {v['max_runtime_sec']} | {v['n_runs']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Collect hardware/runtime info for the appendix.")
    ap.add_argument("--runtime-csv", default="results/runtime/runtime_runs.csv",
                    help="RuntimeProfilerCallback CSV to summarize peak memory / runtime.")
    ap.add_argument("--out-dir", default="results/appendix",
                    help="Directory for hardware_info.json / .md.")
    args = ap.parse_args()

    payload = {
        "hardware": collect_hardware(),
        "software": collect_software(),
        "runtime_summary": summarize_runtime_csv(args.runtime_csv),
        "log_paths": collect_log_paths(args.runtime_csv),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "hardware_info.json")
    md_path = os.path.join(args.out_dir, "hardware_info.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    md = to_markdown(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"[collect_hardware_info] wrote {json_path}")
    print(f"[collect_hardware_info] wrote {md_path}")


if __name__ == "__main__":
    main()
