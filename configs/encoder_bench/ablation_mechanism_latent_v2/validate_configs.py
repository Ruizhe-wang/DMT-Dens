"""Validate that v2 changes identity only, not scientific configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from generate_configs import DATASETS, ROOT, RUN_DIR, SEED, VARIANTS


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
    assert len(paths) == 36
    output_roots = set()

    for dataset in DATASETS:
        for variant in VARIANTS:
            old = load(
                ROOT
                / "configs/encoder_bench/ablation_mechanism_latent/runs"
                / f"ablation_latent-bn_{dataset}_{variant}_seed{SEED}.yaml"
            )
            path = (
                RUN_DIR
                / f"ablation_v2_latent-bn_{dataset}_{variant}_seed{SEED}.yaml"
            )
            new = load(path)

            assert new["seed_everything"] == old["seed_everything"] == SEED
            assert new["data"] == old["data"]
            assert new["model"] == old["model"]
            for key in (
                "max_epochs",
                "precision",
                "benchmark",
                "check_val_every_n_epoch",
                "log_every_n_steps",
            ):
                assert new["trainer"][key] == old["trainer"][key]

            logger = new["trainer"]["logger"]["init_args"]
            assert logger["name"] == path.stem
            assert logger["project"] == "TopoBranch_latent_mechanism_ablation_v2"

            callbacks = new["trainer"]["callbacks"]
            for suffix in (
                "FidelityEvalCallback",
                "EncoderBenchmarkCallback",
                "RuntimeProfilerCallback",
                "RuntimeInfoCallback",
                "PaperEmbeddingCallback",
                "SaveConsolidatedEmbeddingsCallback",
            ):
                one_callback(callbacks, suffix)

            out = one_callback(callbacks, "PaperEmbeddingCallback")["init_args"]
            assert out["formats"] == ["png", "pdf", "svg"]
            assert out["dpi"] == 300
            output_roots.add(out["output_dir"])

    assert len(output_roots) == 36
    values = load(HERE / "sweep" / "sweep_seed42.yaml")["parameters"]["config"][
        "values"
    ]
    assert len(values) == len(set(values)) == 36
    assert {Path(value).resolve() for value in values} == {
        path.resolve() for path in paths
    }
    print("PASS: v2 preserves all 36 scientific configs and uses unique outputs")


if __name__ == "__main__":
    main()
