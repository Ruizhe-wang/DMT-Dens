"""Print selected metrics from local W&B summary files.

Usage:
    python scripts/summarize_wandb_runs.py <wandb-root> <run-id> [...]
"""

import argparse
import json
from pathlib import Path


FIELDS = (
    "runtime/run/name",
    "fidelity/density_correlation",
    "fidelity/local_density_correlation",
    "fidelity/svc_acc",
    "fidelity/trustworthiness",
    "fidelity/knn_preservation",
    "fidelity/frdc",
    "engineering/encoder_params",
    "runtime_peak_cuda_memory_mb",
    "runtime_mean_epoch_time_sec",
    "runtime_fit_wall_time_sec",
    "engineering/actual_batch_size",
    "embedding/collapsed",
    "train_status/nonfinite_detected",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_root", type=Path)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()

    summaries = {}
    for path in args.wandb_root.glob("run-*-*/files/wandb-summary.json"):
        run_id = path.parents[1].name.rsplit("-", 1)[-1]
        if run_id in args.run_ids:
            summaries[run_id] = json.loads(path.read_text(encoding="utf-8"))

    print("run_id\tS\t" + "\t".join(FIELDS))
    for run_id in args.run_ids:
        summary = summaries.get(run_id)
        if summary is None:
            raise FileNotFoundError(f"no local summary found for run {run_id}")
        density = summary["fidelity/density_correlation"]
        local_density = summary["fidelity/local_density_correlation"]
        score = 0.5 * (density + local_density)
        values = [run_id, f"{score:.6f}"]
        values.extend(str(summary.get(field, "")) for field in FIELDS)
        print("\t".join(values))


if __name__ == "__main__":
    main()
