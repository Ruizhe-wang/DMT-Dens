#!/usr/bin/env python
"""Select the best baseline hyperparameter (p1, p2) per (method, dataset) from one or more
wandb sweep exports, using a balanced density-vs-separability score.

The new metrics (ASR, FR-DC) are deliberately NOT used for selection: tuning baselines
on the metric we then compare them on would be circular. Selection uses only the
pre-existing metrics (density correlation + SVC accuracy).

Input
-----
One or more CSVs exported from wandb. Each must contain:
    method                              (e.g. "_umap"; leading underscore is stripped)
    Name                                (e.g. "baseline_umap_tree"; dataset parsed from it)
                                        -- OR an explicit data_name/dataset column
    model.init_args.p1 / p1
    model.init_args.p2 / p2
    fidelity/density_correlation        (or val_legacy_density_correlation)
    fidelity/svc_accuracy               (or fidelity/svc_acc / val_svc_acc)

Pass several CSVs to merge sweeps (e.g. the main export plus the tree/epi/act top-up):
    python select_best_hparams.py main_export.csv tree_epi_act_export.csv

Output
------
best_hparams.csv : method, data_name, best_p1, best_p2, score, density_corr, svc, n_configs
"""
import argparse
import re
import pandas as pd
import numpy as np

METHODS = ["densmap", "denssne", "densne", "pacmap", "phate", "tsne", "umap"]
PAPER_DATASETS = {"tree", "hcl", "gast10k", "mca", "epi", "emnist", "ng20", "act"}


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_dataset(name):
    s = str(name)
    if s.startswith("baseline_"):
        s = s[len("baseline_"):]
    for me in sorted(METHODS, key=len, reverse=True):
        if s.startswith(me + "_"):
            return re.sub(r"_seed\d+$", "", s[len(me) + 1:])
    return None


def norm01(s):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_csv", nargs="+", help="one or more wandb export CSVs to merge")
    ap.add_argument("--w_density", type=float, default=0.5)
    ap.add_argument("--out", default="best_hparams.csv")
    ap.add_argument("--all_datasets", action="store_true",
                    help="keep every dataset, not just the paper 8")
    args = ap.parse_args()

    frames = [pd.read_csv(f, low_memory=False) for f in args.sweep_csv]
    df = pd.concat(frames, ignore_index=True)

    m_col = first_col(df, ["method", "method_clean"])
    d_col = first_col(df, ["data_name", "dataset"])
    p1_col = first_col(df, ["model.init_args.p1", "p1"])
    p2_col = first_col(df, ["model.init_args.p2", "p2"])
    dc_col = first_col(df, ["fidelity/density_correlation", "val_legacy_density_correlation",
                            "fidelity/layer0/density_correlation"])
    svc_col = first_col(df, ["fidelity/svc_accuracy", "fidelity/svc_acc", "val_svc_acc"])
    for nm, c in [("method", m_col), ("p1", p1_col), ("p2", p2_col),
                  ("density_correlation", dc_col), ("svc", svc_col)]:
        if c is None:
            raise SystemExit(f"required column for '{nm}' not found in the export(s)")

    df["m"] = df[m_col].astype(str).str.lstrip("_").str.lower()
    df["d"] = df[d_col] if d_col else df["Name"].apply(parse_dataset)
    df["d"] = df["d"].astype(str).str.lower()
    df = df.dropna(subset=["m", "d", p1_col, p2_col])
    # drop blank/unknown method rows and rows with no usable metric
    df = df[df["m"].isin(METHODS)]
    df = df.dropna(subset=[dc_col, svc_col], how="all")

    rows = []
    for (m, d), g in df.groupby(["m", "d"]):
        if not args.all_datasets and d not in PAPER_DATASETS:
            continue
        g = g.copy()
        g["_s"] = args.w_density * norm01(g[dc_col]) + (1 - args.w_density) * norm01(g[svc_col])
        g = g.sort_values(["_s", dc_col], ascending=False)
        b = g.iloc[0]
        rows.append(dict(
            method=m, data_name=d,
            best_p1=int(b[p1_col]), best_p2=int(b[p2_col]),
            score=round(float(b["_s"]), 3),
            density_corr=round(float(pd.to_numeric(b[dc_col], errors="coerce")), 3),
            svc=round(float(pd.to_numeric(b[svc_col], errors="coerce")), 3),
            n_configs=g[[p1_col, p2_col]].drop_duplicates().shape[0]))
    out = pd.DataFrame(rows).sort_values(["method", "data_name"])
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nwrote {args.out}  ({len(out)} method-dataset cells)")
    thin = out[out["n_configs"] < 25]
    if len(thin):
        print("\n[warn] cells with < 25 swept configs (best chosen from a smaller grid):")
        print(thin[["method", "data_name", "n_configs"]].to_string(index=False))


if __name__ == "__main__":
    main()
