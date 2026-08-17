"""Validate that the ACT batch-4096 reruns preserve the E18 configuration."""

from copy import deepcopy
from pathlib import Path

import yaml

from generate_configs import DATE_TAG, PROJECT, REPO_ROOT, RUNS_DIR, SEEDS


SOURCE_DIR = REPO_ROOT / "configs" / "encoder_bench" / "sweep_e18" / "runs"


def normalized(config: dict) -> dict:
    result = deepcopy(config)
    result["data"]["init_args"]["batch_size"] = "<batch-size>"
    logger = result["trainer"]["logger"]["init_args"]
    logger["name"] = "<run-name>"
    logger["project"] = "<project>"
    for callback in result["trainer"]["callbacks"]:
        init_args = callback.setdefault("init_args", {})
        path = callback["class_path"]
        if path.endswith("ModelCheckpoint"):
            init_args["dirpath"] = "<checkpoint-dir>"
            init_args["filename"] = "<checkpoint-name>"
        elif path.endswith(("VisualizationCallback", "HeterogeneityPlotCallback", "PaperEmbeddingCallback", "RuntimeInfoCallback")):
            init_args["output_dir"] = "<output-dir>"
            if path.endswith("PaperEmbeddingCallback"):
                init_args.pop("log_to_wandb", None)
                init_args.pop("wandb_key", None)
        elif path.endswith("RuntimeProfilerCallback"):
            init_args["output_path"] = "<output-path>"
    return result


def main() -> None:
    expected = set(SEEDS)
    observed = set()
    for seed in SEEDS:
        run_name = f"act_latent_bn_bs4096_seed{seed}_c90_{DATE_TAG}"
        target_path = RUNS_DIR / f"{run_name}.yaml"
        source_path = SOURCE_DIR / f"act_latent_bn_seed{seed}.yaml"
        target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        observed.add(target["seed_everything"])

        assert normalized(target) == normalized(source), f"scientific config drift: seed {seed}"
        assert target["data"]["init_args"]["batch_size"] == 4096
        logger = target["trainer"]["logger"]["init_args"]
        assert logger["name"] == run_name
        assert logger["project"] == PROJECT
        paper = [
            callback for callback in target["trainer"]["callbacks"]
            if callback["class_path"].endswith("PaperEmbeddingCallback")
        ]
        assert len(paper) == 1
        paper_args = paper[0]["init_args"]
        assert paper_args["every_n_epochs"] is None
        assert paper_args["color_key"] == "final_annotation"
        assert paper_args["dpi"] == 300
        assert paper_args["formats"] == ["png"]
        assert paper_args["log_to_wandb"] is True
        assert paper_args["wandb_key"] == "paper_embedding/final_annotation"

    assert observed == expected
    print("validated=5 seeds=42,43,44,45,46 batch_size=4096 paper_callback=enabled")


if __name__ == "__main__":
    main()
