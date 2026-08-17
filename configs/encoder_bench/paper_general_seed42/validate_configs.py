"""Static checks for final general-dataset paper-embedding reruns."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs"
DATASETS = {
    "act",
    "emnist",
    "epi",
    "gast10k",
    "hcl",
    "mca",
    "mnist",
    "ng20",
    "tree",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def one_callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    assert len(matches) == 1, f"{suffix}: expected one, found {len(matches)}"
    return matches[0]


def main() -> None:
    paths = sorted(RUN_DIR.glob("*.yaml"))
    assert len(paths) == 9
    seen = set()
    figure_dirs = set()
    embedding_dirs = set()

    for path in paths:
        cfg = load(path)
        dataset = path.stem.removeprefix(
            "paper_exact_latent-bn_"
        ).removesuffix("_seed42")
        assert dataset in DATASETS
        seen.add(dataset)
        source = load(
            ROOT
            / f"configs/encoder_bench/sweep_e18/runs/{dataset}_latent_bn_seed42.yaml"
        )

        # Scientific configuration is unchanged from E18.
        assert cfg["seed_everything"] == source["seed_everything"] == 42
        expected_data = source["data"]
        if expected_data["init_args"].get("data_path") == "/zangzelin/data":
            expected_data["init_args"]["data_path"] = (
                "/usr/storage/ruizhe/zangzelin/data"
            )
        assert cfg["data"] == expected_data
        assert cfg["model"] == source["model"]
        assert cfg["trainer"]["max_epochs"] == source["trainer"]["max_epochs"] == 1000
        assert cfg["trainer"]["precision"] == source["trainer"]["precision"] == "16-mixed"

        logger = cfg["trainer"]["logger"]["init_args"]
        assert logger["name"] == f"paper_exact_latent-bn_{dataset}_seed42"
        assert logger["project"] == "TopoBranch_encoder_paper_figures"
        assert logger["group"] == "exact_e18_paper_callback_rerun_seed42_v3"

        callbacks = cfg["trainer"]["callbacks"]
        paper = one_callback(
            callbacks,
            "xc_paper_embedding_callback.PaperEmbeddingCallback",
        )["init_args"]
        assert paper["every_n_epochs"] is None
        assert paper["point_size"] == 1.5
        assert paper["alpha"] == 0.5
        assert paper["cmap"] == "tab20"
        assert paper["figsize"] == 4.0
        assert paper["dpi"] == 300
        assert paper["formats"] == ["png"]
        assert paper["log_to_wandb"] is True
        assert paper["wandb_key"] == "paper_embedding/figure"
        figure_dirs.add(paper["output_dir"])

        exporter = one_callback(callbacks, "SaveConsolidatedEmbeddingsCallback")[
            "init_args"
        ]
        assert exporter["save_format"] == "both"
        embedding_dirs.add(exporter["output_dir"])

        checkpoint = one_callback(callbacks, "ModelCheckpoint")["init_args"]
        assert checkpoint["save_last"] is True
        assert checkpoint["save_top_k"] == 0

    assert seen == DATASETS
    assert len(figure_dirs) == len(embedding_dirs) == 9
    values = load(HERE / "sweep" / "exact_e18_paper_callback_rerun_v3.yaml")[
        "parameters"
    ]["config"]["values"]
    assert len(values) == len(set(values)) == 9
    assert {Path(value).stem for value in values} == {
        f"paper_exact_latent-bn_{dataset}_seed42" for dataset in DATASETS
    }
    print(
        "PASS: 9 seed-42 configs preserve E18 scientific settings and export "
        "final exact-E18 PNG figures locally and in W&B plus CSV/NPZ "
        "embeddings"
    )


if __name__ == "__main__":
    main()
