#!/usr/bin/env python3
"""Summarize and plot root-cause generalization JSONL diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CONDITION_ORDER = [
    "distance_allpair",
    "distance_p_rownorm_allpair",
    "rank_allpair",
    "distance_hardpair",
]
CONDITION_LABELS = {
    "distance_allpair": "distance + all-pair",
    "distance_p_rownorm_allpair": "distance-P row norm + all-pair",
    "rank_allpair": "rank + all-pair",
    "distance_hardpair": "distance + hard-pair",
}
COLORS = {
    "distance_allpair": "#4D4D4D",
    "distance_p_rownorm_allpair": "#0072B2",
    "rank_allpair": "#CC79A7",
    "distance_hardpair": "#E69F00",
}
METRICS = [
    ("embedding/axis_ratio", "Full-coordinate axis ratio", False),
    ("embedding/norm_max_median_ratio", "Max / median norm", True),
    ("projection/weight_row_cosine", "Projection row cosine", False),
    ("projection_bn/train_eval_relative_rms_gap", "BN train/eval RMS gap", True),
    ("fidelity/density_correlation", "Density correlation", False),
    ("fidelity/trustworthiness", "Trustworthiness", False),
    ("fidelity/knn_preservation", "kNN preservation", False),
    ("fidelity/svc_accuracy", "SVC accuracy", False),
    ("embedding/nonfinite_fraction", "NaN/Inf fraction", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Directories containing copied rootcause_generalization outputs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def infer_metadata(path: Path) -> tuple[str, str]:
    parts = path.parts
    marker_name = next(
        part for part in parts if part.startswith("rootcause_generalization")
    )
    marker = parts.index(marker_name)
    dataset = parts[marker + 1]
    condition = parts[marker + 3]
    return dataset, condition


def load_records(inputs: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[Path] = set()
    for root in inputs:
        for path in sorted(root.rglob("full_geometry_and_quality.jsonl")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            dataset, condition = infer_metadata(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                record["dataset"] = dataset.upper()
                record["condition"] = condition
                rows.append(record)
    if not rows:
        raise SystemExit("No full_geometry_and_quality.jsonl files found")
    frame = pd.DataFrame(rows)
    frame["condition"] = pd.Categorical(
        frame["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return frame.sort_values(["dataset", "condition", "epoch"]).reset_index(drop=True)


def recompute_epoch1000_geometry(inputs: list[Path]) -> pd.DataFrame:
    rows = []
    for root in inputs:
        for path in sorted(root.rglob("*epoch1000_layer0.csv")):
            dataset, condition = infer_metadata(path)
            z = pd.read_csv(path, usecols=["x", "y"]).to_numpy(dtype=np.float64)
            finite = np.isfinite(z).all(axis=1)
            clean = z[finite]
            norms = np.linalg.norm(clean, axis=1)

            def axis_ratio(values: np.ndarray) -> float:
                singular = np.linalg.svd(
                    values - values.mean(axis=0, keepdims=True),
                    full_matrices=False,
                    compute_uv=False,
                )
                return float(singular[1] / max(singular[0], 1.0e-12))

            trimmed = clean[norms <= np.quantile(norms, 0.999)]
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "condition": condition,
                    "n_coordinates": len(z),
                    "nonfinite_fraction": 1.0 - float(finite.mean()),
                    "axis_ratio": axis_ratio(clean),
                    "axis_ratio_trim_top_0_1pct": axis_ratio(trimmed),
                    "norm_max": float(norms.max()),
                    "norm_median": float(np.median(norms)),
                    "norm_max_median_ratio": float(
                        norms.max() / max(np.median(norms), 1.0e-12)
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_dataset(frame: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = frame[frame["dataset"] == dataset]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (metric, title, log_scale) in zip(axes.flat, METRICS):
        for condition in CONDITION_ORDER:
            values = subset[subset["condition"] == condition]
            if values.empty or metric not in values:
                continue
            ax.plot(
                values["epoch"],
                values[metric],
                marker="o",
                linewidth=2.2 if condition == "distance_p_rownorm_allpair" else 1.7,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        if log_scale:
            ax.set_yscale("log")
        if metric == "embedding/axis_ratio":
            ax.axhline(0.05, color="#999999", linestyle=":", linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"{dataset}: row normalization prevents near-line geometry but quality remains metric-dependent",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        output_dir / f"{dataset.lower()}_rootcause_trajectory.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    frame = load_records(args.inputs)
    frame.to_csv(args.output_dir / "rootcause_metrics_long.csv", index=False)
    last = frame.loc[frame.groupby(["dataset", "condition"], observed=True)["epoch"].idxmax()]
    columns = ["dataset", "condition", "epoch"] + [metric for metric, _, _ in METRICS]
    last[columns].to_csv(args.output_dir / "rootcause_final_summary.csv", index=False)

    expected_epochs = list(range(50, 1001, 50))
    audit_rows = []
    for (dataset, condition), values in frame.groupby(
        ["dataset", "condition"], observed=True
    ):
        values = values.sort_values("epoch")
        final = values.iloc[-1]
        audit_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "records": len(values),
                "epoch_grid_complete": values["epoch"].tolist() == expected_epochs,
                "near_line_epochs": int(values["embedding/near_line"].sum()),
                "nonfinite_max": values["embedding/nonfinite_fraction"].max(),
                "axis_ratio_min": values["embedding/axis_ratio"].min(),
                "axis_ratio_final": final["embedding/axis_ratio"],
                "norm_ratio_max": values["embedding/norm_max_median_ratio"].max(),
                "norm_ratio_final": final["embedding/norm_max_median_ratio"],
                "projection_abs_cosine_max": values[
                    "projection/weight_row_cosine"
                ].abs().max(),
                "projection_cosine_final": final["projection/weight_row_cosine"],
                "bn_gap_max": values[
                    "projection_bn/train_eval_relative_rms_gap"
                ].max(),
                "bn_gap_final": final[
                    "projection_bn/train_eval_relative_rms_gap"
                ],
                "density_final": final["fidelity/density_correlation"],
                "trust_final": final["fidelity/trustworthiness"],
                "knn_final": final["fidelity/knn_preservation"],
                "svc_final": final["fidelity/svc_accuracy"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        args.output_dir / "rootcause_audit_summary.csv", index=False
    )

    paired_rows = []
    for dataset in frame["dataset"].unique():
        base = frame[
            (frame["dataset"] == dataset)
            & (frame["condition"] == "distance_allpair")
        ].set_index("epoch")
        rownorm = frame[
            (frame["dataset"] == dataset)
            & (frame["condition"] == "distance_p_rownorm_allpair")
        ].set_index("epoch")
        comparisons = {
            "axis_ratio": rownorm["embedding/axis_ratio"]
            > base["embedding/axis_ratio"],
            "norm_ratio_lower": rownorm["embedding/norm_max_median_ratio"]
            < base["embedding/norm_max_median_ratio"],
            "projection_abs_cosine_lower": rownorm[
                "projection/weight_row_cosine"
            ].abs()
            < base["projection/weight_row_cosine"].abs(),
            "bn_gap_lower": rownorm[
                "projection_bn/train_eval_relative_rms_gap"
            ]
            < base["projection_bn/train_eval_relative_rms_gap"],
            "density": rownorm["fidelity/density_correlation"]
            > base["fidelity/density_correlation"],
            "trust": rownorm["fidelity/trustworthiness"]
            > base["fidelity/trustworthiness"],
            "knn": rownorm["fidelity/knn_preservation"]
            > base["fidelity/knn_preservation"],
            "svc": rownorm["fidelity/svc_accuracy"]
            > base["fidelity/svc_accuracy"],
        }
        paired_rows.append(
            {"dataset": dataset, **{name: int(values.sum()) for name, values in comparisons.items()}}
        )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "rootcause_paired_win_counts.csv", index=False
    )
    recomputed = recompute_epoch1000_geometry(args.inputs)
    if not recomputed.empty:
        recomputed.to_csv(
            args.output_dir / "rootcause_epoch1000_geometry_recomputed.csv",
            index=False,
        )
    for dataset in frame["dataset"].unique():
        plot_dataset(frame, dataset, args.output_dir)


if __name__ == "__main__":
    main()
