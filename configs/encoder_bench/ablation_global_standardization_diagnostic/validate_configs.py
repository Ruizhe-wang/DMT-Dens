"""Validate isolation, callbacks, and sweep coverage for the diagnostic."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from generate_configs import HERE, PROJECT, ROOT, RUN_DIR, SEED, TARGETS, source_config


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalized(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    args = cfg["model"]["init_args"]
    for key in (
        "embedding_standardization",
        "embedding_std_floor",
        "density_distance_floor",
        "density_pearson_floor",
    ):
        args.pop(key, None)
    trainer = cfg["trainer"]
    trainer.pop("logger")
    trainer.pop("log_every_n_steps", None)
    for callback in trainer["callbacks"]:
        suffix = callback["class_path"].split(".")[-1]
        init = callback.get("init_args", {})
        for key in {
            "ModelCheckpoint": ("dirpath", "filename"),
            "VisualizationCallback": ("output_dir",),
            "HeterogeneityPlotCallback": ("output_dir",),
            "RuntimeProfilerCallback": ("output_path",),
            "RuntimeInfoCallback": ("output_dir",),
            "PaperEmbeddingCallback": ("output_dir", "method_name"),
            "SaveConsolidatedEmbeddingsCallback": (
                "dataset_name",
                "output_dir",
                "embedding_method_name",
            ),
            "EncoderBenchmarkCallback": (
                "check_grad_every_n_steps",
                "fail_on_nonfinite",
            ),
        }.get(suffix, ()):
            init.pop(key, None)
    return cfg


def main() -> None:
    expected = sum(len(variants) for variants in TARGETS.values())
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == expected == 7

    for path in paths:
        cfg = load(path)
        name = path.stem
        dataset = next(ds for ds in TARGETS if f"_{ds}_" in name)
        variant = next(v for v in TARGETS[dataset] if f"_{v}_seed" in name)
        args = cfg["model"]["init_args"]
        assert cfg["seed_everything"] == SEED
        assert args["embedding_standardization"] == "differentiable_global"
        assert args["embedding_std_floor"] == 1.0e-4
        assert args["density_distance_floor"] == 1.0e-4
        assert args["density_pearson_floor"] == 1.0e-4
        assert cfg["trainer"]["log_every_n_steps"] == 1
        assert cfg["trainer"]["logger"]["init_args"]["name"] == name
        assert cfg["trainer"]["logger"]["init_args"]["project"] == PROJECT
        callback = next(
            item
            for item in cfg["trainer"]["callbacks"]
            if item["class_path"].endswith("EncoderBenchmarkCallback")
        )
        assert callback["init_args"]["check_grad_every_n_steps"] == 1
        assert callback["init_args"]["fail_on_nonfinite"] is True
        assert normalized(cfg) == normalized(load(source_config(dataset, variant)))

    sweep_paths = sorted((HERE / "sweep").glob("*.yaml"))
    assert len(sweep_paths) == 1
    sweep = load(sweep_paths[0])
    assert sweep["project"] == PROJECT
    values = sweep["parameters"]["config"]["values"]
    assert len(values) == len(set(values)) == expected
    assert {Path(value).resolve() for value in values} == {p.resolve() for p in paths}
    print("PASS: 7 isolated global-standardization diagnostic configs")


if __name__ == "__main__":
    main()
