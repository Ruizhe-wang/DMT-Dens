"""Generate the targeted collapse-causality experiment matrix.

The matrix has two deliberately separate groups:

* ``current``: the historical detached-statistics standardization at seeds
  43/44, used only to measure collapse reproducibility.
* ``fixed``: seed 42 with differentiable float32 standardization and a 1e-4
  standard-deviation floor, used to test whether the known gradient pathology
  causes the collapse.

All other scientific settings are copied from the completed v2 seed-42
mechanism ablation.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
SWEEP_DIR = HERE / "sweep"
PROJECT = "TopoBranch_latent_collapse_diagnostic"

TARGETS = {
    "epi": (
        "full",
        "base",
        "b2_bidirectional",
        "b3_hardpair",
        "b4_single_density",
        "b5_multi_density",
    ),
    "hcl": ("full", "b4_single_density"),
    "mnist": ("full", "b5_multi_density"),
}
CURRENT_SEEDS = (43, 44)
FIXED_SEEDS = (42,)


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def one_callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix}, found {len(matches)}")
    return matches[0]


def source_config(dataset: str, variant: str) -> Path:
    return (
        ROOT
        / "configs/encoder_bench/ablation_mechanism_latent_v2/runs"
        / f"ablation_v2_latent-bn_{dataset}_{variant}_seed42.yaml"
    )


def build(dataset: str, variant: str, seed: int, mode: str) -> tuple[str, dict]:
    cfg = copy.deepcopy(load(source_config(dataset, variant)))
    fixed = mode == "fixed"
    if mode not in {"current", "fixed"}:
        raise ValueError(mode)

    run_name = f"collapse_diag_{mode}_latent-bn_{dataset}_{variant}_seed{seed}"
    out_root = f"outputs/ablation/collapse_diagnostic/{mode}/{dataset}/{variant}/seed{seed}"
    cfg["seed_everything"] = seed

    model_args = cfg["model"]["init_args"]
    model_args["stable_embedding_standardization"] = fixed
    model_args["embedding_std_floor"] = 1.0e-4

    logger = cfg["trainer"]["logger"]["init_args"]
    logger.update(
        {
            "name": run_name,
            "project": PROJECT,
            "save_dir": "wandb/collapse_diagnostic",
            "tags": [
                "collapse-causal-test",
                mode,
                dataset,
                variant,
                f"seed{seed}",
            ],
        }
    )

    callbacks = cfg["trainer"]["callbacks"]
    one_callback(callbacks, "ModelCheckpoint")["init_args"].update(
        {
            "dirpath": f"{out_root}/checkpoints",
            "filename": f"{run_name}-{{epoch:04d}}",
        }
    )
    one_callback(callbacks, "xc_plot_callback.VisualizationCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/diagnostics"
    one_callback(callbacks, "HeterogeneityPlotCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/heterogeneity"
    one_callback(callbacks, "RuntimeProfilerCallback")["init_args"][
        "output_path"
    ] = f"results/ablation/collapse_diagnostic/{run_name}/runtime.csv"
    one_callback(callbacks, "RuntimeInfoCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/runtime_info"

    method_name = f"latent-bn-{variant}-{mode}-collapse-diag"
    one_callback(callbacks, "PaperEmbeddingCallback")["init_args"].update(
        {"output_dir": f"{out_root}/figure", "method_name": method_name}
    )
    one_callback(callbacks, "SaveConsolidatedEmbeddingsCallback")["init_args"].update(
        {
            "dataset_name": f"{dataset}_{variant}_{mode}_collapse_diag_seed{seed}",
            "output_dir": f"{out_root}/final_embedding",
            "embedding_method_name": method_name,
        }
    )
    return run_name, cfg


def write_sweep(name: str, run_paths: list[Path]) -> None:
    sweep = {
        "program": "main.py",
        "method": "grid",
        "project": PROJECT,
        "name": name,
        "parameters": {
            "config": {
                "values": [path.relative_to(ROOT).as_posix() for path in run_paths]
            }
        },
        "command": ["${env}", "${interpreter}", "${program}", "fit", "${args}"],
    }
    path = SWEEP_DIR / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Targeted collapse diagnostic; generated file.\n")
        yaml.safe_dump(sweep, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in RUN_DIR.glob("*.yaml"):
        stale.unlink()
    for stale in SWEEP_DIR.glob("*.yaml"):
        stale.unlink()

    groups: dict[tuple[str, str], list[Path]] = {
        ("current", "epi"): [],
        ("fixed", "epi"): [],
        ("current", "hcl_mnist"): [],
        ("fixed", "hcl_mnist"): [],
    }

    for mode, seeds in (("current", CURRENT_SEEDS), ("fixed", FIXED_SEEDS)):
        for dataset, variants in TARGETS.items():
            machine_group = "epi" if dataset == "epi" else "hcl_mnist"
            for variant in variants:
                for seed in seeds:
                    run_name, cfg = build(dataset, variant, seed, mode)
                    path = RUN_DIR / f"{run_name}.yaml"
                    header = (
                        "# Generated by ablation_collapse_diagnostic/generate_configs.py\n"
                        f"# mode={mode}; dataset={dataset}; variant={variant}; seed={seed}\n"
                    )
                    with path.open("w", encoding="utf-8") as handle:
                        handle.write(header)
                        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
                    groups[(mode, machine_group)].append(path)

    write_sweep("collapse_diag_current_epi_seed43_44", groups[("current", "epi")])
    write_sweep("collapse_diag_fixed_epi_seed42", groups[("fixed", "epi")])
    write_sweep(
        "collapse_diag_current_hcl_mnist_seed43_44",
        groups[("current", "hcl_mnist")],
    )
    write_sweep(
        "collapse_diag_fixed_hcl_mnist_seed42",
        groups[("fixed", "hcl_mnist")],
    )
    print(
        f"Wrote {sum(len(v) for v in groups.values())} configs and "
        f"{len(groups)} machine-local sweeps"
    )


if __name__ == "__main__":
    main()
