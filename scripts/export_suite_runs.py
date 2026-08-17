"""Export every run of a sweep as a tidy CSV, one row per run.

Written for the E18 full-suite comparison: 9 datasets x 2 encoders x 5 seeds.
The output is machine readable so the join against the existing 5-seed MLP
baseline can happen wherever that baseline lives.

Duplicate run names are kept and flagged rather than silently dropped - the
sweep contains one duplicated name and which copy to trust is a decision for
the analysis, not for the exporter.

Usage:
    python scripts/export_suite_runs.py zelinzang/TopoBranch_encoder_suite/62y8lrrn \\
        --out results/encoder_tuning/E18_suite_runs.csv
"""

import argparse
import csv
import os
import sys

import wandb

METRICS = {
    "density": "val_visible_density_correlation",
    "local_density": "val_local_density_correlation",
    "svc": "val_svc_acc",
    "trust": "val_trustworthiness",
    "knn": "val_knn_preservation",
    "frdc": "fidelity/frdc",
    "asr": "fidelity/asr",
    "msdp_auc": "val_msdp_auc",
    "dist_corr": "val_dist_corr",
    "collapsed": "val_embedding_collapsed",
    "nonfinite": "train_status/nonfinite_detected",
    "peak_mb": "runtime_peak_cuda_memory_mb",
    "sec_per_epoch": "runtime_mean_epoch_time_sec",
    "batch_size": "runtime_batch_size",
    "epochs": "runtime_current_epoch",
}


def num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_path", help="entity/project/sweep_id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not os.environ.get("WANDB_BASE_URL"):
        print("WANDB_BASE_URL is not set", file=sys.stderr)
        return 2

    api = wandb.Api()
    sweep = api.sweep(args.sweep_path)
    rows = []
    seen = {}
    for run in sweep.runs:
        name = run.name
        seen[name] = seen.get(name, 0) + 1
        # name format: {encoder}_{dataset}_seed{n}
        parts = name.rsplit("_seed", 1)
        stem, seed = (parts[0], parts[1]) if len(parts) == 2 else (name, "")
        encoder, _, dataset = stem.partition("_")
        row = {
            "run_name": name,
            "run_id": run.id,
            "state": run.state,
            "encoder": encoder,
            "dataset": dataset,
            "seed": seed,
            "duplicate_index": seen[name],
            "params": num(run.config.get("encoder_params")),
        }
        for out_key, summary_key in METRICS.items():
            row[out_key] = num(run.summary.get(summary_key))
        rows.append(row)

    rows.sort(key=lambda r: (r["dataset"], r["encoder"], r["seed"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dupes = {k: v for k, v in seen.items() if v > 1}
    print(f"wrote {args.out} with {len(rows)} rows, sweep state {sweep.state}")
    if dupes:
        print(f"duplicate run names (kept, flagged by duplicate_index): {dupes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
