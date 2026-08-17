"""Static integrity checks for both Latent-Transformer case-study profiles."""

from __future__ import annotations

from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
DATASETS = {"celegan", "dyngen"}
ENCODERS = {"latent-bn"}
PROFILES = {"native", "aligned"}
SEEDS = {41, 42, 43, 44, 45}

COMMON_CALLBACKS = {
    "lightning.pytorch.callbacks.ModelCheckpoint",
    "callbacks.xc_plot_callback.VisualizationCallback",
    "callbacks.case_study_callback.VisualizationCallback",
    "callbacks.xc_save_consolidated_embeddings.SaveConsolidatedEmbeddingsCallback",
    "callbacks.Eval_density.FidelityEvalCallback",
    "callbacks.runtime_profiler.RuntimeProfilerCallback",
    "callbacks.encoder_benchmark.EncoderBenchmarkCallback",
    "callbacks.runtime_info_callback.RuntimeInfoCallback",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_profile_strings(value):
    if isinstance(value, dict):
        return {key: normalize_profile_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_profile_strings(item) for item in value]
    if isinstance(value, str):
        return value.replace("_native_", "_aligned_").replace("/native/", "/aligned/")
    return value


def main() -> None:
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == 20, f"expected 20 run configs, found {len(paths)}"
    expected = {
        f"{encoder}_{profile}_{dataset}_seed{seed}"
        for dataset in DATASETS
        for encoder in ENCODERS
        for profile in PROFILES
        for seed in SEEDS
    }
    seen: set[str] = set()
    artifact_paths: set[str] = set()

    for path in paths:
        run_name = path.stem
        assert run_name in expected, f"unexpected config {run_name}"
        seen.add(run_name)
        encoder, profile, dataset_seed = run_name.split("_", 2)
        dataset, seed_text = dataset_seed.rsplit("_seed", 1)
        seed = int(seed_text)
        cfg = load(path)

        assert cfg["seed_everything"] == seed
        assert cfg["data"]["init_args"]["seed"] == 42
        assert cfg["data"]["init_args"]["batch_size"] == 1000
        assert cfg["trainer"]["max_epochs"] == 1000
        assert cfg["model"]["init_args"]["max_epochs"] == 1000
        assert cfg["model"]["init_args"]["lr"] == 0.001
        assert cfg["model"]["init_args"]["t_output_dim"] == 40
        assert cfg["trainer"]["check_val_every_n_epoch"] == 50
        assert cfg["trainer"]["precision"] == "16-mixed"
        assert cfg["trainer"]["logger"]["init_args"]["name"] == run_name
        assert cfg["trainer"]["logger"]["init_args"]["project"] == "TopoBranch_encoder_case_study"

        model_args = cfg["model"]["init_args"]
        assert encoder == "latent-bn"
        assert model_args["encoder_type"] == "latent_transformer"
        assert model_args["encoder_kwargs"]["num_latents"] == 32
        assert model_args["encoder_kwargs"]["d_token"] == 224
        assert model_args["encoder_kwargs"]["num_layers"] == 2
        assert model_args["encoder_kwargs"]["latent_rank"] == 16
        assert model_args["encoder_kwargs"]["input_norm"] == "batchnorm"

        if profile == "aligned":
            expected_nu_lat = 0.01
            expected_density = 0.0018
            expected_anchors = 512
        else:
            expected_nu_lat = 0.01 if dataset == "celegan" else 0.005
            expected_density = 0.01 if dataset == "celegan" else 0.018
            expected_anchors = 1000 if dataset == "celegan" else 256
        expected_k = 50 if dataset == "celegan" else 500
        assert model_args["nu_lat"] == expected_nu_lat
        assert model_args["density_weight"] == expected_density
        assert model_args["density_num_anchors"] == expected_anchors
        assert cfg["data"]["init_args"]["K"] == expected_k

        callbacks = cfg["trainer"]["callbacks"]
        classes = {item["class_path"] for item in callbacks}
        assert COMMON_CALLBACKS <= classes
        if dataset == "celegan":
            assert "callbacks.celegan_gt_metrics_callback.CeleganGroundTruthMetricsCallback" in classes
        else:
            assert "callbacks.dyngen_gt_metrics_callback.DyngenGroundTruthMetricsCallback" in classes
            assert "callbacks.case_study_fateprob_callback.FateProbabilityVisualizationCallback" in classes
            assert "callbacks.case_study_truetree_callback.TrueTreeOverlayVisualizationCallback" in classes

        for item in callbacks:
            args = item.get("init_args", {})
            for key in ("output_dir", "output_path", "dirpath"):
                value = args.get(key)
                if value and run_name.replace("-", "_") not in value.replace("/", "_"):
                    # Paths are hierarchical, so verify the exact dataset/seed/encoder tuple instead.
                    assert profile in value and dataset in value and f"seed{seed}" in value
                if value:
                    artifact_paths.add(value)

    assert seen == expected
    sweep_names: set[str] = set()
    for profile in PROFILES:
        sweep = load(HERE / f"sweep_{profile}.yaml")
        values = sweep["parameters"]["config"]["values"]
        profile_expected = {name for name in expected if f"_{profile}_" in name}
        assert len(values) == len(set(values)) == 10
        assert {Path(value).stem for value in values} == profile_expected
        sweep_names.update(Path(value).stem for value in values)
    assert sweep_names == expected

    # Apart from profile-qualified names/paths, the two batches may differ only
    # in the three density/manifold fields explicitly requested for alignment.
    for dataset in DATASETS:
        for seed in SEEDS:
            native = load(RUN_DIR / f"latent-bn_native_{dataset}_seed{seed}.yaml")
            aligned = load(RUN_DIR / f"latent-bn_aligned_{dataset}_seed{seed}.yaml")
            native_args = native["model"]["init_args"]
            native_args.update(
                {
                    "nu_lat": 0.01,
                    "density_weight": 0.0018,
                    "density_num_anchors": 512,
                }
            )
            assert normalize_profile_strings(native) == aligned
    print(
        "PASS: 20 unique latent-bn configs; native/aligned density settings, "
        "callbacks, seeds, and both sweep matrices are consistent"
    )


if __name__ == "__main__":
    main()
