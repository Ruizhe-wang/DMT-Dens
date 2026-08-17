"""Generate the legacy TopoBranch mechanism ablation for latent-bn.

The mechanism combinations intentionally match
``configs/ablation/experiment1_mechanism/_generate.py``.  Only the experiment
template is updated to the current seed-42 latent Transformer paper config.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
SWEEP_DIR = HERE / "sweep"

DATASETS = ("hcl", "epi", "mnist")
SEED = 42
HARDPAIR_K = 100

# name -> (affinity, symmetric, hard-pair, density mode)
# Kept byte-for-byte equivalent in meaning to the legacy mechanism design.
VARIANTS = {
    "full": ("rank", "bidirectional", True, "multi"),
    # Remove one mechanism from the full model (necessity).
    "r1_distance": ("distance", "bidirectional", True, "multi"),
    "r2_unidirectional": ("rank", "unidirectional", True, "multi"),
    "r3_allpair": ("rank", "bidirectional", False, "multi"),
    "r4_single_density": ("rank", "bidirectional", True, "single"),
    "r5_no_density": ("rank", "bidirectional", True, "no"),
    # Add one mechanism to the minimal base (sufficiency).
    "base": ("distance", "unidirectional", False, "no"),
    "b1_rank": ("rank", "unidirectional", False, "no"),
    "b2_bidirectional": ("distance", "bidirectional", False, "no"),
    "b3_hardpair": ("distance", "unidirectional", True, "no"),
    "b4_single_density": ("distance", "unidirectional", False, "single"),
    "b5_multi_density": ("distance", "unidirectional", False, "multi"),
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def one_callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix}, found {len(matches)}")
    return matches[0]


def apply_density(model_args: dict, mode: str) -> None:
    if mode == "multi":
        model_args["density_scale_mode"] = "multi"
        model_args["density_weight"] = 0.0018
    elif mode == "single":
        model_args["density_scale_mode"] = "single"
        model_args["density_weight"] = 0.0018
    elif mode == "no":
        # Leave scale mode explicit for reproducibility; it is inactive when
        # density_weight is zero.
        model_args["density_scale_mode"] = "multi"
        model_args["density_weight"] = 0.0
    else:
        raise ValueError(mode)


def build(dataset: str, variant: str, factors: tuple[str, str, bool, str]) -> dict:
    source = ROOT / (
        "configs/encoder_bench/paper_general_seed42/runs/"
        f"paper_latent-bn_{dataset}_seed42.yaml"
    )
    cfg = copy.deepcopy(load(source))
    affinity, symmetric, hardpair, density_mode = factors
    run_name = f"ablation_latent-bn_{dataset}_{variant}_seed{SEED}"
    out_root = f"outputs/ablation/latent_mechanism/{dataset}/{variant}/seed{SEED}"

    model_args = cfg["model"]["init_args"]
    model_args["manifold_affinity"] = affinity
    model_args["manifold_symmetric"] = symmetric
    model_args["manifold_hardpair"] = hardpair
    model_args["hardpair_k"] = HARDPAIR_K
    apply_density(model_args, density_mode)

    logger = cfg["trainer"]["logger"]["init_args"]
    logger.update(
        {
            "name": run_name,
            "project": "TopoBranch_latent_mechanism_ablation",
            "save_dir": "wandb/latent_mechanism_ablation",
        }
    )

    callbacks = cfg["trainer"]["callbacks"]
    one_callback(callbacks, "ModelCheckpoint")["init_args"].update(
        {
            "dirpath": f"{out_root}/checkpoints",
            "filename": f"{run_name}-{{epoch:04d}}",
            "save_last": True,
            "save_top_k": 0,
        }
    )
    one_callback(callbacks, "xc_plot_callback.VisualizationCallback")["init_args"].update(
        {
            "output_dir": f"{out_root}/diagnostics",
            "save_embeddings": True,
            "embedding_method_name": "latent-bn",
        }
    )
    one_callback(callbacks, "HeterogeneityPlotCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/heterogeneity"
    one_callback(callbacks, "RuntimeProfilerCallback")["init_args"][
        "output_path"
    ] = f"results/ablation/latent_mechanism/{run_name}/runtime.csv"
    one_callback(callbacks, "RuntimeInfoCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/runtime_info"
    one_callback(callbacks, "PaperEmbeddingCallback")["init_args"].update(
        {
            "output_dir": f"{out_root}/figure",
            "method_name": f"latent-bn-{variant}",
            "color_key": "final_annotation",
            "every_n_epochs": None,
            "point_size": 6.0,
            "alpha": 0.85,
            "cmap": "tab20",
            "overflow_cmap": "gist_ncar",
            "figsize": 4.0,
            "dpi": 300,
            "formats": ["png", "pdf", "svg"],
        }
    )
    one_callback(callbacks, "SaveConsolidatedEmbeddingsCallback")["init_args"].update(
        {
            "dataset_name": f"{dataset}_{variant}_seed{SEED}",
            "output_dir": f"{out_root}/final_embedding",
            "save_format": "both",
            "embedding_method_name": f"latent-bn-{variant}",
        }
    )
    return cfg


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in RUN_DIR.glob("*.yaml"):
        stale.unlink()

    run_paths: list[str] = []
    for dataset in DATASETS:
        for variant, factors in VARIANTS.items():
            cfg = build(dataset, variant, factors)
            run_name = f"ablation_latent-bn_{dataset}_{variant}_seed{SEED}"
            path = RUN_DIR / f"{run_name}.yaml"
            header = (
                "# Generated by configs/encoder_bench/ablation_mechanism_latent/"
                "generate_configs.py\n"
                f"# Legacy mechanism ablation: {dataset}, {variant}, seed {SEED}.\n"
                "# Encoder, data, projection head, optimizer, callbacks and all "
                "non-ablation settings match the current latent-bn paper config.\n"
            )
            with path.open("w", encoding="utf-8") as handle:
                handle.write(header)
                yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
            run_paths.append(path.relative_to(ROOT).as_posix())

    sweep = {
        "program": "main.py",
        "method": "grid",
        "project": "TopoBranch_latent_mechanism_ablation",
        "name": "latent_bn_old_mechanism_mnist_hcl_epi_seed42",
        "parameters": {"config": {"values": run_paths}},
        "command": ["${env}", "${interpreter}", "${program}", "fit", "${args}"],
    }
    sweep_path = SWEEP_DIR / "sweep_seed42.yaml"
    with sweep_path.open("w", encoding="utf-8") as handle:
        handle.write("# HCL, EPI and MNIST; 12 legacy mechanism variants; seed 42.\n")
        yaml.safe_dump(sweep, handle, sort_keys=False, allow_unicode=True)
    print(f"Wrote {len(run_paths)} configs and {sweep_path}")


if __name__ == "__main__":
    main()
