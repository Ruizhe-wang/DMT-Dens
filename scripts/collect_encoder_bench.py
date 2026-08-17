"""Collect encoder-benchmark results from a W&B sweep into a markdown table.

Written to run *on the server inside tmux*, so a sweep's results are gathered
and written to disk even when no laptop is connected. Pair it with a wait:

    python scripts/collect_encoder_bench.py <entity/project/sweep_id> \\
        --wait --out results/encoder_tuning/E13_results.md

The metric keys are the ones the callbacks actually log. ``export_wandb_sweep.py``
uses ``fidelity/*`` names that the current callbacks do not emit, which yields
empty columns; this script uses the ``val_*`` names verified against live runs.

Credentials are read only from the environment, never printed.
"""

import argparse
import os
import sys
import time

import wandb

# Verified against live runs in TopoBranch_encoder_bench_*.
COLUMNS = [
    ("Density", "val_visible_density_correlation"),
    ("Local dens.", "val_local_density_correlation"),
    ("SVC", "val_svc_acc"),
    ("Trust", "val_trustworthiness"),
    ("kNN", "val_knn_preservation"),
    ("FR-DC", "fidelity/frdc"),
    ("ASR", "fidelity/asr"),
]
ENGINEERING = [
    ("Params", "encoder_params", "config"),
    ("Peak MB", "runtime_peak_cuda_memory_mb", "summary"),
    ("s/epoch", "runtime_mean_epoch_time_sec", "summary"),
    ("Batch", "runtime_batch_size", "summary"),
]
HEALTH = [
    ("collapse", "val_embedding_collapsed"),
    ("nonfinite", "train_status/nonfinite_detected"),
]


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def fmt(value, digits=4):
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def score(summary):
    """The predeclared density-first score S."""
    den = num(summary.get("val_visible_density_correlation"))
    ldc = num(summary.get("val_local_density_correlation"))
    if den is None or ldc is None:
        return None
    return 0.5 * den + 0.5 * ldc


def collect(sweep_path):
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    rows = []
    for run in sorted(sweep.runs, key=lambda r: r.name):
        s = run.summary
        row = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "S": score(s),
        }
        for label, key in COLUMNS:
            row[label] = num(s.get(key))
        for label, key, where in ENGINEERING:
            src = run.config if where == "config" else s
            row[label] = num(src.get(key))
        for label, key in HEALTH:
            row[label] = num(s.get(key))
        rows.append(row)
    return sweep, rows


def render(sweep, sweep_path, rows):
    headers = ["Run", "W&B", "State", "S"] + [c[0] for c in COLUMNS] \
        + [e[0] for e in ENGINEERING] + [h[0] for h in HEALTH]
    lines = [
        f"# Sweep {sweep_path}",
        "",
        f"State: {sweep.state}; runs: {len(rows)}; "
        f"collected {time.strftime('%Y-%m-%d %H:%M %z')}",
        "",
        "S = 0.5 * density_correlation + 0.5 * local_density_correlation.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [row["name"], f"`{row['id']}`", row["state"], fmt(row["S"])]
        cells += [fmt(row[c[0]]) for c in COLUMNS]
        cells += [fmt(row[e[0]], 1) for e in ENGINEERING]
        cells += [fmt(row[h[0]], 0) for h in HEALTH]
        lines.append("| " + " | ".join(cells) + " |")
    unfinished = [r["name"] for r in rows if r["state"] != "finished"]
    if unfinished:
        lines += ["", f"**Not finished ({len(unfinished)}):** " + ", ".join(unfinished)]
    return "\n".join(lines) + "\n"


def wait_for_sweep(sweep_path, poll_seconds, timeout_seconds):
    """Block until every run has left the running/pending states."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        api = wandb.Api()
        states = [r.state for r in api.sweep(sweep_path).runs]
        pending = [s for s in states if s in ("running", "pending")]
        print(
            f"[{time.strftime('%H:%M:%S')}] {len(states) - len(pending)}/{len(states)} "
            f"settled",
            flush=True,
        )
        if states and not pending:
            return True
        time.sleep(poll_seconds)
    print("timeout reached; collecting whatever has settled", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_path", help="entity/project/sweep_id")
    parser.add_argument("--out", required=True, help="markdown output path")
    parser.add_argument("--wait", action="store_true", help="poll until all runs settle")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()

    if not os.environ.get("WANDB_BASE_URL"):
        print("WANDB_BASE_URL is not set; refusing to guess the server", file=sys.stderr)
        return 2

    if args.wait:
        wait_for_sweep(args.sweep_path, args.poll_seconds, args.timeout_hours * 3600)

    sweep, rows = collect(args.sweep_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render(sweep, args.sweep_path, rows))
    print(f"wrote {args.out} ({len(rows)} runs, sweep state {sweep.state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
