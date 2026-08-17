"""Validate the isolated latent parameter-sensitivity matrix."""

from __future__ import annotations

from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_DIR = HERE / "sweep"
PROJECT = "TopoBranch_latent_parameter_sensitivity"
DATASETS = ("gast10k", "mca", "hcl", "ng20")
SEEDS = (42, 43, 44, 45, 46)
EXPECTED_CALLBACKS = {
    "lightning.pytorch.callbacks.ModelCheckpoint",
    "callbacks.Eval_density.FidelityEvalCallback",
    "callbacks.runtime_profiler.RuntimeProfilerCallback",
    "callbacks.encoder_benchmark.EncoderBenchmarkCallback",
    "callbacks.runtime_info_callback.RuntimeInfoCallback",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    run_files = sorted(RUNS_DIR.glob("*.yaml"))
    if len(run_files) != 400:
        fail(f"expected 400 five-seed configs, found {len(run_files)}")

    output_paths: set[str] = set()
    for path in run_files:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = config["model"]["init_args"]
        trainer = config["trainer"]
        data = config["data"]["init_args"]

        seed = config["seed_everything"]
        assert seed in SEEDS, path
        assert data["batch_size"] == 4096, path
        assert trainer["max_epochs"] == 1000, path
        assert trainer["check_val_every_n_epoch"] == 1000, path
        assert trainer["logger"]["init_args"]["project"] == PROJECT, path
        assert model["encoder_type"] == "latent_transformer", path
        assert model["encoder_kwargs"]["num_latents"] == 32, path
        assert model["encoder_kwargs"]["d_token"] == 224, path
        assert model["encoder_kwargs"]["latent_rank"] == 16, path
        assert model["encoder_kwargs"]["num_layers"] == 2, path
        assert model["encoder_kwargs"]["num_heads"] == 4, path
        assert model["lr"] == 0.001, path
        assert model["density_scale_mode"] == "multi", path
        assert model["stable_embedding_standardization"] is False, path
        assert model["density_num_anchors"] in {128, 256, 512, 768, 1024, 1536}, path
        assert model["density_k"] in {5, 10, 12, 15, 20, 25, 30, 40}, path
        assert model["density_weight"] in {0.0001, 0.0005, 0.001, 0.0018, 0.003, 0.006, 0.01, 0.02}, path

        callbacks = {entry["class_path"]: entry.get("init_args", {}) for entry in trainer["callbacks"]}
        assert set(callbacks) == EXPECTED_CALLBACKS, path
        fidelity = callbacks["callbacks.Eval_density.FidelityEvalCallback"]
        assert fidelity["every_n_epochs"] == 1000, path
        assert fidelity["density_k"] == 15, path
        assert fidelity["knn_k"] == 12, path
        assert fidelity["seed"] == seed, path
        runtime_path = callbacks["callbacks.runtime_info_callback.RuntimeInfoCallback"]["output_dir"]
        if runtime_path in output_paths:
            fail(f"duplicate output path: {runtime_path}")
        output_paths.add(runtime_path)

    listed_seed42: list[str] = []
    listed_followup: list[str] = []
    for dataset in DATASETS:
        sweep_path = SWEEP_DIR / f"latent_parameter_sensitivity_{dataset}_seed42.yaml"
        sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
        assert sweep["project"] == PROJECT, sweep_path
        values = sweep["parameters"]["config"]["values"]
        assert len(values) == 20, sweep_path
        assert all(f"/{dataset}_" in value for value in values), sweep_path
        assert all("_seed42.yaml" in value for value in values), sweep_path
        listed_seed42.extend(values)

        followup_path = SWEEP_DIR / f"latent_parameter_sensitivity_{dataset}_seeds43_46.yaml"
        followup = yaml.safe_load(followup_path.read_text(encoding="utf-8"))
        assert followup["project"] == PROJECT, followup_path
        followup_values = followup["parameters"]["config"]["values"]
        assert len(followup_values) == 80, followup_path
        assert all(f"/{dataset}_" in value for value in followup_values), followup_path
        assert all(any(f"_seed{seed}.yaml" in value for seed in SEEDS[1:]) for value in followup_values), followup_path
        listed_followup.extend(followup_values)

    assert len(listed_seed42) == len(set(listed_seed42)) == 80
    assert len(listed_followup) == len(set(listed_followup)) == 320

    retry_expected = {"gast10k": 1, "hcl": 12, "ng20": 18}
    for dataset, expected in retry_expected.items():
        retry_path = SWEEP_DIR / f"latent_parameter_sensitivity_{dataset}_seed42_retry.yaml"
        retry = yaml.safe_load(retry_path.read_text(encoding="utf-8"))
        retry_values = retry["parameters"]["config"]["values"]
        assert len(retry_values) == expected, retry_path
        assert all(f"/{dataset}_" in value and "_seed42.yaml" in value for value in retry_values), retry_path

    print("PASS: 400 five-seed configs; 320 follow-up runs and 31 isolated seed-42 retries")


if __name__ == "__main__":
    main()
