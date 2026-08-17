"""Export benchmark metrics from one W&B sweep as JSON or CSV.

The script intentionally reads credentials only through W&B's normal
environment/configuration mechanism and never prints them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import wandb


DEFAULT_KEYS = (
    "fidelity/density_correlation",
    "fidelity/local_density_correlation",
    "fidelity/svc_acc",
    "fidelity/trustworthiness",
    "fidelity/knn_preservation",
    "fidelity/distance_correlation",
    "engineering/encoder_params",
    "engineering/encoder_param_ratio",
    "engineering/actual_batch_size",
    "runtime/peak_memory/gpu",
    "runtime_mean_epoch_time_sec",
    "runtime_fit_wall_time_sec",
    "train_status/nonfinite_detected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep_path",
        help="entity/project/sweep_id, for example entity/project/abc123",
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_KEYS),
        help="W&B summary keys to export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = wandb.Api().sweep(args.sweep_path)
    records = []
    for run in sweep.runs:
        record = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "config_path": run.config.get("config"),
        }
        record.update({key: run.summary.get(key) for key in args.keys})
        records.append(record)

    if args.format == "json":
        json.dump(records, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    fields = ["run_id", "run_name", "state", "config_path", *args.keys]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)


if __name__ == "__main__":
    main()
