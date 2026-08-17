#!/usr/bin/env python
"""Patch the per-dataset baseline configs in this folder with the per-dataset BEST
hyperparameters selected by select_best_hparams.py.

WHY THIS EXISTS
---------------
The source configs (configs/dmtme_dataset_baselines/<method>/<dataset>.yaml) carry a single
DEFAULT (p1, p2) per method, identical across all datasets, and the original Table-1 runs
used those defaults — i.e. baselines were NOT tuned per dataset. This script writes the
real per-dataset winners into the configs here, so the new 5-seed runs are tuned per
dataset (a fair comparison).

INPUT
-----
best_hparams.csv  (output of select_best_hparams.py), columns:
    method, data_name, best_p1, best_p2[, best_p3]

Method names are matched case-insensitively with aliasing:
    densne == denssne == den-sne == densne   ->  folder 'denssne'
    tsne   == t-sne,  umap, pacmap, phate, densmap

USAGE
-----
    python apply_best_hparams.py best_hparams.csv
    python apply_best_hparams.py best_hparams.csv --dry-run     # show changes only
"""
import argparse, os, io, sys
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ALIAS = {
    "tsne": "tsne", "t-sne": "tsne",
    "umap": "umap", "pacmap": "pacmap", "phate": "phate", "densmap": "densmap",
    "densne": "denssne", "denssne": "denssne", "den-sne": "denssne",
}
DATASETS = {"tree", "hcl", "gast10k", "mca", "epi", "emnist", "ng20", "act"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best_csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bh = pd.read_csv(args.best_csv)
    bh.columns = [c.strip().lower() for c in bh.columns]
    need = {"method", "data_name", "best_p1", "best_p2"}
    if not need.issubset(bh.columns):
        sys.exit(f"best_hparams.csv must contain {need}; got {list(bh.columns)}")

    n_ok, n_miss = 0, []
    for _, r in bh.iterrows():
        folder = ALIAS.get(str(r["method"]).strip().lower())
        ds = str(r["data_name"]).strip().lower()
        if folder is None or ds not in DATASETS:
            n_miss.append((r["method"], r["data_name"], "unmapped method/dataset")); continue
        path = os.path.join(HERE, folder, ds + ".yaml")
        if not os.path.exists(path):
            n_miss.append((r["method"], r["data_name"], "config not found")); continue
        cfg = yaml.safe_load(io.open(path, encoding="utf-8"))
        ia = cfg["model"]["init_args"]
        old = (ia.get("p1"), ia.get("p2"), ia.get("p3"))
        ia["p1"] = int(r["best_p1"]); ia["p2"] = int(r["best_p2"])
        if "best_p3" in bh.columns and pd.notna(r.get("best_p3")):
            ia["p3"] = int(r["best_p3"])
        new = (ia.get("p1"), ia.get("p2"), ia.get("p3"))
        print(f"{folder:8}/{ds:8}  p1,p2,p3: {old} -> {new}")
        if not args.dry_run:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("# Per-dataset BEST hyperparameters applied by apply_best_hparams.py\n")
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        n_ok += 1
    print(f"\npatched {n_ok} configs" + (" (dry-run, nothing written)" if args.dry_run else ""))
    if n_miss:
        print("UNRESOLVED:")
        for x in n_miss:
            print("  ", x)


if __name__ == "__main__":
    main()
