"""
Generator for Experiment 1 (mechanism-decomposition ablation).

Reads each per-dataset template configs/ablation/component/<ds>_full.yaml
(which already carries the correct data settings: num_input_dim,
num_train_data, data class_path, etc.) and stamps out one YAML per variant,
toggling only the mechanism knobs and rewriting run names / output paths.

Factors (defaults in DiffTreeVQ_density.py reproduce the original behaviour):
  - manifold_affinity  : "rank" | "distance"        (M1-a)
  - manifold_symmetric : "bidirectional" | "unidirectional" (M1-b)
  - manifold_hardpair  : True | False               (M2)
  - density            : "multi" | "single" | "no"  (M3)

Design: remove-one-from-full (necessity) + add-one-to-base (sufficiency).

Run:  python configs/ablation/experiment1_mechanism/_generate.py
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE.parent / "component"

DATASETS = ["act", "aqc", "emnist", "epi", "gast10k", "hcl", "mca", "mnist", "ng20", "tree"]

# name -> (affinity, symmetric, hardpair, density_mode)
# density axis has three levels (no / single / multi) so the multi-scale claim
# (contribution 3) can be tested at both the full-manifold and base ends.
VARIANTS = {
    "full": ("rank", "bidirectional", True, "multi"),
    # remove-one-from-full (necessity)
    "r1_distance": ("distance", "bidirectional", True, "multi"),
    "r2_unidirectional": ("rank", "unidirectional", True, "multi"),
    "r3_allpair": ("rank", "bidirectional", False, "multi"),
    "r4_single_density": ("rank", "bidirectional", True, "single"),
    "r5_no_density": ("rank", "bidirectional", True, "no"),
    # add-one-to-base (sufficiency)
    "base": ("distance", "unidirectional", False, "no"),
    "b1_rank": ("rank", "unidirectional", False, "no"),
    "b2_bidirectional": ("distance", "bidirectional", False, "no"),
    "b3_hardpair": ("distance", "unidirectional", True, "no"),
    "b4_single_density": ("distance", "unidirectional", False, "single"),
    "b5_multi_density": ("distance", "unidirectional", False, "multi"),
}

HARDPAIR_K = 100


def apply_density(model_args, mode):
    if mode == "multi":
        model_args["density_scale_mode"] = "multi"
        # keep template density_weight (per-dataset tuned)
    elif mode == "single":
        model_args["density_scale_mode"] = "single"
    elif mode == "no":
        model_args["density_weight"] = 0.0
    else:
        raise ValueError(mode)


def rewrite_paths(cfg, ds, variant):
    trainer = cfg["trainer"]
    logger = trainer.get("logger")
    if isinstance(logger, dict):
        logger["init_args"]["name"] = f"exp1_{ds}_{variant}"
    for cb in trainer.get("callbacks", []):
        cp = cb.get("class_path", "")
        ia = cb.setdefault("init_args", {})
        if cp.endswith("ModelCheckpoint"):
            ia["dirpath"] = f"outputs/ablation/experiment1/checkpoints/{ds}/{variant}"
            ia["filename"] = f"exp1-{ds}-{variant}-{{epoch:04d}}"
        elif cp.endswith("VisualizationCallback"):
            ia["output_dir"] = f"outputs/ablation/experiment1/plots/{ds}/{variant}"
        elif cp.endswith("HeterogeneityPlotCallback"):
            ia["output_dir"] = (
                f"outputs/ablation/experiment1/plots/{ds}/{variant}/heterogeneity"
            )


def main():
    n = 0
    for ds in DATASETS:
        template_path = TEMPLATE_DIR / f"{ds}_full.yaml"
        # utf-8-sig strips any BOM present in the template
        with open(template_path, "r", encoding="utf-8-sig") as f:
            base_cfg = yaml.safe_load(f)
        for variant, (aff, sym, hard, dens) in VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)
            ma = cfg["model"]["init_args"]
            ma["manifold_affinity"] = aff
            ma["manifold_symmetric"] = sym
            ma["manifold_hardpair"] = hard
            ma["hardpair_k"] = HARDPAIR_K
            apply_density(ma, dens)
            rewrite_paths(cfg, ds, variant)
            out = HERE / f"{ds}_{variant}.yaml"
            with open(out, "w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            n += 1
    print(f"Generated {n} config(s) into {HERE}")


if __name__ == "__main__":
    main()
