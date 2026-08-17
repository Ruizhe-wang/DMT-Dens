"""Validate counts, pairing, isolation, callbacks and sweep coverage."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import yaml

from generate_configs import (
    HERE,
    LEARNING_RATES,
    PROJECT,
    ROOT,
    RUN_DIR,
    SEED,
    TARGETS,
    source_config,
)


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalized(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["model"]["init_args"].pop("lr")
    cfg["model"]["init_args"].pop("stable_embedding_standardization", None)
    cfg["model"]["init_args"].pop("embedding_std_floor", None)
    trainer = cfg["trainer"]
    trainer.pop("logger")
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
        }.get(suffix, ()):
            init.pop(key)
    return cfg


def main() -> None:
    condition_count = sum(len(variants) for variants in TARGETS.values())
    expected = condition_count * len(LEARNING_RATES)
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == expected == 22

    pairs = Counter()
    output_paths = set()
    configs = {}
    for path in paths:
        cfg = load(path)
        name = path.stem
        dataset = next(ds for ds in TARGETS if f"_{ds}_" in name)
        variant = next(v for v in TARGETS[dataset] if f"_{v}_lr" in name)
        lr_label = next(label for label in LEARNING_RATES if f"_{label}_" in name)
        pairs[(dataset, variant)] += 1
        configs[(dataset, variant, lr_label)] = cfg

        args = cfg["model"]["init_args"]
        assert cfg["seed_everything"] == SEED
        assert cfg["trainer"]["max_epochs"] == 1000
        assert args["lr"] == LEARNING_RATES[lr_label]
        assert args["stable_embedding_standardization"] is False
        assert args["embedding_std_floor"] == 1.0e-4
        logger = cfg["trainer"]["logger"]["init_args"]
        assert logger["name"] == name
        assert logger["project"] == PROJECT

        source = load(source_config(dataset, variant))
        assert normalized(cfg) == normalized(source)

        runtime = next(
            c for c in cfg["trainer"]["callbacks"]
            if c["class_path"].endswith("RuntimeProfilerCallback")
        )
        output_paths.add(runtime["init_args"]["output_path"])

    assert set(pairs.values()) == {len(LEARNING_RATES)}
    assert len(output_paths) == expected
    for dataset, variants in TARGETS.items():
        for variant in variants:
            left = normalized(configs[(dataset, variant, "lr1e3")])
            right = normalized(configs[(dataset, variant, "lr3e4")])
            assert left == right

    sweep_paths = sorted((HERE / "sweep").glob("*.yaml"))
    assert len(sweep_paths) == 1
    sweep = load(sweep_paths[0])
    assert sweep["project"] == PROJECT
    values = sweep["parameters"]["config"]["values"]
    assert len(values) == len(set(values)) == expected
    assert {Path(value).resolve() for value in values} == {p.resolve() for p in paths}
    print("PASS: 22 paired LR-diagnostic configs; only LR and run/output identity vary")


if __name__ == "__main__":
    main()
