"""Generate an identity-isolated rerun of the collapsed MNIST b5 condition."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / (
    "configs/encoder_bench/ablation_mechanism_latent_v2/runs/"
    "ablation_v2_latent-bn_mnist_b5_multi_density_seed42.yaml"
)
OUTPUT = ROOT / (
    "configs/encoder_bench/ablation_mechanism_latent_v2/reruns/"
    "ablation_v3_latent-bn_mnist_b5_multi_density_seed42_rerun1.yaml"
)


def one_callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix}, found {len(matches)}")
    return matches[0]


def main() -> None:
    with SOURCE.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    cfg = copy.deepcopy(source)
    run_name = "ablation_v3_latent-bn_mnist_b5_multi_density_seed42_rerun1"
    out_root = (
        "outputs/ablation/latent_mechanism_v3/mnist/"
        "b5_multi_density/seed42/rerun1"
    )

    logger = cfg["trainer"]["logger"]["init_args"]
    logger.update(
        {
            "name": run_name,
            "project": "TopoBranch_latent_mechanism_ablation_v2",
            "save_dir": "wandb/latent_mechanism_ablation_v3",
            "tags": [
                "collapse-reproduction",
                "scientific-config-unchanged",
                "mnist",
                "b5_multi_density",
                "seed42",
            ],
        }
    )

    callbacks = cfg["trainer"]["callbacks"]
    one_callback(callbacks, "ModelCheckpoint")["init_args"].update(
        {
            "dirpath": f"{out_root}/checkpoints",
            "filename": f"{run_name}-{{epoch:04d}}",
        }
    )
    one_callback(callbacks, "xc_plot_callback.VisualizationCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/diagnostics"
    one_callback(callbacks, "HeterogeneityPlotCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/heterogeneity"
    one_callback(callbacks, "RuntimeProfilerCallback")["init_args"][
        "output_path"
    ] = f"results/ablation/latent_mechanism_v3/{run_name}/runtime.csv"
    one_callback(callbacks, "RuntimeInfoCallback")["init_args"][
        "output_dir"
    ] = f"{out_root}/runtime_info"
    paper = one_callback(callbacks, "PaperEmbeddingCallback")["init_args"]
    paper.update(
        {
            "output_dir": f"{out_root}/figure",
            "method_name": "latent-bn-b5_multi_density-v3-rerun1",
            "log_to_wandb": True,
            "wandb_key": "paper_embedding/figure",
        }
    )
    one_callback(callbacks, "SaveConsolidatedEmbeddingsCallback")["init_args"].update(
        {
            "dataset_name": "mnist_b5_multi_density_v3_seed42_rerun1",
            "output_dir": f"{out_root}/final_embedding",
            "embedding_method_name": "latent-bn-b5_multi_density-v3-rerun1",
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Exact scientific-config rerun of the collapsed v2 MNIST b5 run.\n"
            "# Only W&B/run/output identity and optional paper-image upload differ.\n"
        )
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
