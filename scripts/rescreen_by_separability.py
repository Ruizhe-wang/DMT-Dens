"""Re-screen finished encoder-benchmark runs under a separability-first rule.

Every selection so far used the density-first score

    S = 0.5 * density_correlation + 0.5 * local_density_correlation

and asked whether an encoder *beats* the MLP on it. A different question was
never asked of the same data: which encoder **matches** the MLP on density while
maximizing label separability and neighbourhood preservation? That is the
criterion under which a modern encoder could replace the MLP even without a
density win, and it costs no GPU time to evaluate.

Rule applied here, per dataset:
  * the MLP run with the highest S is the reference;
  * a candidate passes the density gate if its S is no more than `--tolerance`
    below the reference (default 0.024, the single-seed 2-sigma band);
  * passing candidates are ranked by SVC accuracy, with kNN preservation and
    trustworthiness reported alongside.

Runs are read from W&B; nothing is retrained. Credentials come from the
environment only and are never printed.
"""

import argparse
import os
import sys
from collections import defaultdict

import wandb

# W&B sweep agents override the `project` set in a run config with the sweep's
# own project, so runs are NOT grouped by dataset on the server: the E12 ACT runs
# live in the NG20 project and the E13 MNIST runs live in the HCL project. The
# dataset therefore has to be read from the run name, not from the project.
SOURCE_PROJECTS = [
    "TopoBranch_encoder_bench_NG20",
    "TopoBranch_encoder_bench_ACT",
    "TopoBranch_encoder_bench_MNIST",
    "TopoBranch_encoder_bench_HCL",
]
DATASET_TAGS = [("NG20", "_ng20"), ("ACT", "_act"), ("MNIST", "_mnist"), ("HCL", "_hcl")]

KEYS = {
    "den": "val_visible_density_correlation",
    "ldc": "val_local_density_correlation",
    "svc": "val_svc_acc",
    "trust": "val_trustworthiness",
    "knn": "val_knn_preservation",
    "frdc": "fidelity/frdc",
}


def num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def dataset_of(name):
    lowered = name.lower()
    for dataset, tag in DATASET_TAGS:
        if tag in lowered:
            return dataset
    return None


def load(entity, project):
    api = wandb.Api()
    rows = []
    try:
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as exc:  # noqa: BLE001 - a missing project is informative
        print(f"[{project}] unavailable: {type(exc).__name__}")
        return rows
    for run in runs:
        if run.state != "finished":
            continue
        vals = {k: num(run.summary.get(key)) for k, key in KEYS.items()}
        if vals["den"] is None or vals["ldc"] is None:
            continue
        vals["S"] = 0.5 * vals["den"] + 0.5 * vals["ldc"]
        vals["name"] = run.name
        vals["id"] = run.id
        vals["collapsed"] = num(run.summary.get("val_embedding_collapsed"))
        vals["dataset"] = dataset_of(run.name)
        rows.append(vals)
    return rows


def family(name):
    """Coarse encoder family from the run name, for grouping only."""
    lowered = name.lower()
    if lowered.startswith("mlp"):
        return "mlp"
    if "resmlp" in lowered:
        return "resmlp"
    if "latent" in lowered:
        return "latent"
    if lowered.startswith("ft") or "ft-transformer" in lowered:
        return "ft"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="zelinzang")
    parser.add_argument(
        "--tolerance", type=float, default=0.024,
        help="how far below the MLP reference S a candidate may sit (2 sigma)",
    )
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    if not os.environ.get("WANDB_BASE_URL"):
        print("WANDB_BASE_URL is not set; refusing to guess the server", file=sys.stderr)
        return 2

    print(
        f"Separability-first re-screen. Density gate: S >= MLP_best_S - "
        f"{args.tolerance}. Ranked by SVC.\n"
    )

    everything = []
    for project in SOURCE_PROJECTS:
        everything.extend(load(args.entity, project))
    by_dataset = defaultdict(list)
    for row in everything:
        if row["dataset"]:
            by_dataset[row["dataset"]].append(row)
    print(
        "runs loaded per dataset: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_dataset.items()))
        + "\n"
    )

    for dataset, _ in DATASET_TAGS:
        rows = by_dataset.get(dataset, [])
        if not rows:
            continue
        mlps = [r for r in rows if family(r["name"]) == "mlp"]
        if not mlps:
            print(f"[{dataset}] no finished MLP run to use as reference\n")
            continue
        ref = max(mlps, key=lambda r: r["S"])
        gate = ref["S"] - args.tolerance

        passing = [
            r for r in rows
            if r["S"] >= gate and family(r["name"]) != "mlp" and not r["collapsed"]
        ]
        passing.sort(key=lambda r: (r["svc"] is not None, r["svc"]), reverse=True)

        print(f"=== {dataset} ===")
        print(
            f"MLP reference  {ref['name'][:42]:42s} S={fmt(ref['S'])} "
            f"svc={fmt(ref['svc'])} knn={fmt(ref['knn'])} trust={fmt(ref['trust'])}"
        )
        print(f"density gate: S >= {fmt(gate)}   candidates passing: {len(passing)}")
        if not passing:
            print("  (none)\n")
            continue
        best_by_family = defaultdict(list)
        for row in passing:
            best_by_family[family(row["name"])].append(row)
        for row in passing[: args.top]:
            dsvc = None if row["svc"] is None or ref["svc"] is None else row["svc"] - ref["svc"]
            dknn = None if row["knn"] is None or ref["knn"] is None else row["knn"] - ref["knn"]
            print(
                f"  {row['name'][:42]:42s} S={fmt(row['S'])} "
                f"svc={fmt(row['svc'])} ({fmt(dsvc, 3):>7s}) "
                f"knn={fmt(row['knn'])} ({fmt(dknn, 3):>7s}) trust={fmt(row['trust'])}"
            )
        print(
            "  families passing: "
            + ", ".join(f"{k}x{len(v)}" for k, v in sorted(best_by_family.items()))
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
