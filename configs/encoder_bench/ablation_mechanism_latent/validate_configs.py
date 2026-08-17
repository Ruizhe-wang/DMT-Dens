"""Static validation for the latent-bn legacy mechanism ablation."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from generate_configs import DATASETS, HARDPAIR_K, ROOT, RUN_DIR, SEED, VARIANTS


HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def one_callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    assert len(matches) == 1, f"{suffix}: expected one, found {len(matches)}"
    return matches[0]


def main() -> None:
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == len(DATASETS) * len(VARIANTS) == 36
    figure_dirs: set[str] = set()
    embedding_dirs: set[str] = set()

    for dataset in DATASETS:
        source = load(
            ROOT
            / "configs/encoder_bench/paper_general_seed42/runs"
            / f"paper_latent-bn_{dataset}_seed42.yaml"
        )
        for variant, (affinity, symmetric, hardpair, density_mode) in VARIANTS.items():
            path = RUN_DIR / f"ablation_latent-bn_{dataset}_{variant}_seed{SEED}.yaml"
            cfg = load(path)
            expected = copy.deepcopy(source)

            assert cfg["seed_everything"] == SEED
            assert cfg["data"] == expected["data"]
            assert cfg["trainer"]["max_epochs"] == 1000
            assert cfg["data"]["init_args"]["batch_size"] == 4096

            actual_args = cfg["model"]["init_args"]
            source_args = expected["model"]["init_args"]
            for key in (
                "manifold_affinity",
                "manifold_symmetric",
                "manifold_hardpair",
                "hardpair_k",
                "density_scale_mode",
            ):
                source_args.pop(key, None)
            source_args["density_weight"] = actual_args["density_weight"]
            assert {
                key: value
                for key, value in actual_args.items()
                if key
                not in {
                    "manifold_affinity",
                    "manifold_symmetric",
                    "manifold_hardpair",
                    "hardpair_k",
                    "density_scale_mode",
                }
            } == source_args

            assert actual_args["encoder_type"] == "latent_transformer"
            assert actual_args["encoder_kwargs"] == source["model"]["init_args"][
                "encoder_kwargs"
            ]
            assert actual_args["manifold_affinity"] == affinity
            assert actual_args["manifold_symmetric"] == symmetric
            assert actual_args["manifold_hardpair"] is hardpair
            assert actual_args["hardpair_k"] == HARDPAIR_K
            assert actual_args["density_weight"] == (
                0.0 if density_mode == "no" else 0.0018
            )
            assert actual_args["density_scale_mode"] == (
                "single" if density_mode == "single" else "multi"
            )

            callbacks = cfg["trainer"]["callbacks"]
            for suffix in (
                "FidelityEvalCallback",
                "EncoderBenchmarkCallback",
                "RuntimeProfilerCallback",
                "RuntimeInfoCallback",
                "PaperEmbeddingCallback",
                "SaveConsolidatedEmbeddingsCallback",
            ):
                one_callback(callbacks, suffix)

            paper = one_callback(callbacks, "PaperEmbeddingCallback")["init_args"]
            assert paper["every_n_epochs"] is None
            assert paper["formats"] == ["png", "pdf", "svg"]
            assert paper["dpi"] == 300
            figure_dirs.add(paper["output_dir"])

            exporter = one_callback(callbacks, "SaveConsolidatedEmbeddingsCallback")[
                "init_args"
            ]
            assert exporter["save_format"] == "both"
            embedding_dirs.add(exporter["output_dir"])

            logger = cfg["trainer"]["logger"]["init_args"]
            assert logger["project"] == "TopoBranch_latent_mechanism_ablation"
            assert logger["name"] == path.stem

    assert len(figure_dirs) == len(embedding_dirs) == 36
    sweep = load(HERE / "sweep" / "sweep_seed42.yaml")
    values = sweep["parameters"]["config"]["values"]
    assert len(values) == len(set(values)) == 36
    assert {Path(value).resolve() for value in values} == {
        path.resolve() for path in paths
    }
    print(
        "PASS: 36 seed-42 latent-bn configs reproduce the 12 legacy mechanism "
        "variants on MNIST/HCL/EPI with identical non-ablation settings"
    )


if __name__ == "__main__":
    main()
