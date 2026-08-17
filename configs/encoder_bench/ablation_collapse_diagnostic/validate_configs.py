"""Validate isolation, counts, outputs and machine-local sweep grouping."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from generate_configs import (
    CURRENT_SEEDS,
    FIXED_SEEDS,
    HERE,
    PROJECT,
    ROOT,
    RUN_DIR,
    TARGETS,
)


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalized(cfg: dict) -> dict:
    """Drop fields that are expected to differ from the v2 source."""
    cfg = copy.deepcopy(cfg)
    cfg.pop("seed_everything")
    args = cfg["model"]["init_args"]
    args.pop("stable_embedding_standardization", None)
    args.pop("embedding_std_floor", None)
    trainer = cfg["trainer"]
    trainer.pop("logger")
    for callback in trainer["callbacks"]:
        suffix = callback["class_path"].split(".")[-1]
        init = callback.get("init_args", {})
        if suffix == "ModelCheckpoint":
            init.pop("dirpath")
            init.pop("filename")
        elif suffix == "VisualizationCallback":
            init.pop("output_dir")
        elif suffix == "HeterogeneityPlotCallback":
            init.pop("output_dir")
        elif suffix == "RuntimeProfilerCallback":
            init.pop("output_path")
        elif suffix == "RuntimeInfoCallback":
            init.pop("output_dir")
        elif suffix == "PaperEmbeddingCallback":
            init.pop("output_dir")
            init.pop("method_name")
        elif suffix == "SaveConsolidatedEmbeddingsCallback":
            init.pop("dataset_name")
            init.pop("output_dir")
            init.pop("embedding_method_name")
    return cfg


def main() -> None:
    expected = sum(len(v) for v in TARGETS.values())
    expected_total = expected * (len(CURRENT_SEEDS) + len(FIXED_SEEDS))
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == expected_total == 30

    outputs = set()
    for path in paths:
        cfg = load(path)
        name = path.stem
        mode = "fixed" if "_fixed_" in name else "current"
        dataset = next(ds for ds in TARGETS if f"_{ds}_" in name)
        variant = next(v for v in TARGETS[dataset] if f"_{v}_seed" in name)
        seed = int(name.rsplit("_seed", 1)[1])

        assert cfg["seed_everything"] == seed
        args = cfg["model"]["init_args"]
        assert args["stable_embedding_standardization"] is (mode == "fixed")
        assert args["embedding_std_floor"] == 1.0e-4
        logger = cfg["trainer"]["logger"]["init_args"]
        assert logger["name"] == name
        assert logger["project"] == PROJECT

        source = load(
            ROOT
            / "configs/encoder_bench/ablation_mechanism_latent_v2/runs"
            / f"ablation_v2_latent-bn_{dataset}_{variant}_seed42.yaml"
        )
        assert normalized(cfg) == normalized(source | {"seed_everything": seed})

        paper = next(
            c
            for c in cfg["trainer"]["callbacks"]
            if c["class_path"].endswith("PaperEmbeddingCallback")
        )
        outputs.add(paper["init_args"]["output_dir"])
    assert len(outputs) == expected_total

    sweep_paths = sorted((HERE / "sweep").glob("*.yaml"))
    assert len(sweep_paths) == 4
    all_values = []
    for path in sweep_paths:
        sweep = load(path)
        values = sweep["parameters"]["config"]["values"]
        assert len(values) == len(set(values))
        if "epi" in path.stem:
            assert all("_epi_" in value for value in values)
        else:
            assert all(("_hcl_" in value or "_mnist_" in value) for value in values)
        all_values.extend(values)
    assert len(all_values) == len(set(all_values)) == expected_total
    assert {Path(v).resolve() for v in all_values} == {p.resolve() for p in paths}
    print("PASS: 30 isolated collapse-diagnostic configs in four machine-local sweeps")


if __name__ == "__main__":
    main()
