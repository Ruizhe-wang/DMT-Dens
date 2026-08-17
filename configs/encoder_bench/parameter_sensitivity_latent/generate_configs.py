"""Generate five-seed latent-Transformer parameter-sensitivity configs.

The three axes share the paper operating point (A=512, k=12,
lambda_d=0.0018), so that point is generated only once per dataset/seed.

The original seed-42 sweeps are kept for provenance. Follow-up sweeps contain
only seeds 43--46, while retry sweeps contain only seed-42 configurations whose
first attempt failed because another process occupied the assigned GPU.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_DIR = HERE / "sweep"
PROJECT = "TopoBranch_latent_parameter_sensitivity"

DATASETS = ("gast10k", "mca", "hcl", "ng20")
SEEDS = (42, 43, 44, 45, 46)
FOLLOWUP_SEEDS = (43, 44, 45, 46)
ANCHORS = (128, 256, 512, 768, 1024, 1536)
NEIGHBORHOOD_K = (5, 10, 12, 15, 20, 25, 30, 40)
LAMBDA_D = (0.0001, 0.0005, 0.001, 0.0018, 0.003, 0.006, 0.01, 0.02)
BASELINE = {"density_num_anchors": 512, "density_k": 12, "density_weight": 0.0018}

# W&B-confirmed failed-only seed-42 settings from the first sweep attempt.
# MCA completed all settings.
SEED42_RETRIES = {
    "gast10k": {"k_40"},
    "hcl": {
        "baseline_a512_k12_l0p0018",
        "k_10",
        "k_20",
        "k_30",
        "k_40",
        "lambda_0p0001",
        "lambda_0p0005",
        "lambda_0p001",
        "lambda_0p003",
        "lambda_0p006",
        "lambda_0p01",
        "lambda_0p02",
    },
    "ng20": {
        "anchors_128",
        "anchors_256",
        "anchors_768",
        "anchors_1024",
        "anchors_1536",
        "baseline_a512_k12_l0p0018",
        "k_5",
        "k_10",
        "k_15",
        "k_20",
        "k_25",
        "k_30",
        "k_40",
        "lambda_0p0001",
        "lambda_0p001",
        "lambda_0p006",
        "lambda_0p01",
        "lambda_0p02",
    },
}

# W&B-audited missing full runs as of 2026-08-07.  A setting is considered
# complete only when a finished run reached epoch 999.  The completion sweeps
# are intentionally dataset-local so MCA stays on c90 and HCL stays on c82.
COMPLETION_RETRIES = {
    "mca": {
        44: {"baseline_a512_k12_l0p0018", "lambda_0p02"},
        45: {"anchors_128", "anchors_256", "anchors_768", "anchors_1024"},
        46: {
            "baseline_a512_k12_l0p0018",
            "k_10",
            "k_15",
            "k_20",
            "k_25",
            "k_30",
            "k_40",
            "lambda_0p0001",
            "lambda_0p0005",
            "lambda_0p001",
            "lambda_0p003",
            "lambda_0p006",
            "lambda_0p01",
            "lambda_0p02",
        },
    },
    "hcl": {
        44: {
            "anchors_1536",
            "baseline_a512_k12_l0p0018",
            "k_5",
            "lambda_0p006",
            "lambda_0p01",
            "lambda_0p02",
        },
        45: "*",
        46: "*",
    },
}


def value_slug(value: int | float) -> str:
    return str(value).replace(".", "p")


def settings() -> list[tuple[str, int | float | str, dict[str, int | float]]]:
    result: list[tuple[str, int | float | str, dict[str, int | float]]] = []
    for value in ANCHORS:
        if value != BASELINE["density_num_anchors"]:
            result.append(("anchors", value, {**BASELINE, "density_num_anchors": value}))
    for value in NEIGHBORHOOD_K:
        if value != BASELINE["density_k"]:
            result.append(("k", value, {**BASELINE, "density_k": value}))
    for value in LAMBDA_D:
        if value != BASELINE["density_weight"]:
            result.append(("lambda", value, {**BASELINE, "density_weight": value}))
    result.append(("baseline", "a512_k12_l0p0018", dict(BASELINE)))
    return result


def load_template(dataset: str) -> dict:
    path = ROOT / "configs" / "encoder_bench" / "sweep_e18" / "runs" / f"{dataset}_latent_bn_seed42.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_callbacks(config: dict, run_key: str, seed: int) -> None:
    output_root = f"outputs/encoder_tuning/parameter_sensitivity_latent/{run_key}"
    results_root = f"results/encoder_tuning/parameter_sensitivity_latent/{run_key}"
    config["trainer"]["callbacks"] = [
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": f"{output_root}/checkpoints",
                "filename": f"{run_key}-{{epoch:04d}}",
                "save_last": True,
                "every_n_epochs": 1000,
            },
        },
        {
            "class_path": "callbacks.Eval_density.FidelityEvalCallback",
            "init_args": {
                "every_n_epochs": 1000,
                "down_sample": 3000,
                "knn_k": 12,
                "density_k": 15,
                "seed": seed,
            },
        },
        {
            "class_path": "callbacks.runtime_profiler.RuntimeProfilerCallback",
            "init_args": {"output_path": f"{results_root}/runtime.csv"},
        },
        {
            "class_path": "callbacks.encoder_benchmark.EncoderBenchmarkCallback",
            "init_args": {
                "tail_epochs": 20,
                "cv_threshold": 0.25,
                "direction_change_threshold": 0.5,
                "spike_ratio_threshold": 3.0,
                "check_grad_every_n_steps": 50,
            },
        },
        {
            "class_path": "callbacks.runtime_info_callback.RuntimeInfoCallback",
            "init_args": {"output_dir": f"{output_root}/runtime_info"},
        },
    ]


def make_config(dataset: str, seed: int, axis: str, value: int | float | str, overrides: dict) -> tuple[str, dict]:
    config = copy.deepcopy(load_template(dataset))
    value_name = value if axis == "baseline" else value_slug(value)
    run_key = f"{dataset}_{axis}_{value_name}_seed{seed}"

    config["seed_everything"] = seed
    config["data"]["init_args"]["batch_size"] = 4096
    model = config["model"]["init_args"]
    model.update(overrides)
    model["density_scale_mode"] = "multi"
    model["stable_embedding_standardization"] = False
    model["embedding_std_floor"] = 1.0e-4
    model["max_epochs"] = 1000

    trainer = config["trainer"]
    trainer["max_epochs"] = 1000
    trainer["check_val_every_n_epoch"] = 1000
    trainer["logger"]["init_args"].update(
        {
            "name": run_key,
            "project": PROJECT,
            "save_dir": "wandb/parameter_sensitivity_latent",
        }
    )
    configure_callbacks(config, run_key, seed)
    return run_key, config


def write_sweep(dataset: str, paths: list[str], suffix: str) -> None:
    sweep = {
        "program": "main.py",
        "method": "grid",
        "project": PROJECT,
        "name": f"latent_parameter_sensitivity_{dataset}_{suffix}",
        "parameters": {"config": {"values": paths}},
        "command": ["${env}", "${interpreter}", "${program}", "fit", "${args}"],
    }
    with (SWEEP_DIR / f"latent_parameter_sensitivity_{dataset}_{suffix}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sweep, handle, sort_keys=False)


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    expected_names: set[str] = set()
    for dataset in DATASETS:
        paths_by_seed: dict[int, list[str]] = {seed: [] for seed in SEEDS}
        seed42_retry_paths: list[str] = []
        completion_retry_paths: list[str] = []
        for seed in SEEDS:
            for axis, value, overrides in settings():
                run_key, config = make_config(dataset, seed, axis, value, overrides)
                filename = f"{run_key}.yaml"
                expected_names.add(filename)
                path = RUNS_DIR / filename
                header = "# Generated by parameter_sensitivity_latent/generate_configs.py\n"
                path.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                relative_path = path.relative_to(ROOT).as_posix()
                paths_by_seed[seed].append(relative_path)
                setting_key = run_key.removeprefix(f"{dataset}_").removesuffix(f"_seed{seed}")
                if seed == 42 and setting_key in SEED42_RETRIES.get(dataset, set()):
                    seed42_retry_paths.append(relative_path)
                completion_keys = COMPLETION_RETRIES.get(dataset, {}).get(seed, set())
                if completion_keys == "*" or setting_key in completion_keys:
                    completion_retry_paths.append(relative_path)

        write_sweep(dataset, paths_by_seed[42], "seed42")
        followup_paths = [path for seed in FOLLOWUP_SEEDS for path in paths_by_seed[seed]]
        write_sweep(dataset, followup_paths, "seeds43_46")
        if seed42_retry_paths:
            write_sweep(dataset, seed42_retry_paths, "seed42_retry")
        if completion_retry_paths:
            write_sweep(dataset, completion_retry_paths, "completion_retry_20260807")

    for stale in RUNS_DIR.glob("*.yaml"):
        if stale.name not in expected_names:
            stale.unlink()
    sweep_count = len(list(SWEEP_DIR.glob("latent_parameter_sensitivity_*.yaml")))
    print(f"Generated {len(expected_names)} run configs and {sweep_count} sweeps")


if __name__ == "__main__":
    main()
