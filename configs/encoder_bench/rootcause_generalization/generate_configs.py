"""Generate strict root-cause generalization pairs for HCL and MNIST."""

from copy import deepcopy
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE_ROOT = REPO_ROOT / "configs" / "encoder_bench" / "ablation_mechanism_latent_v2" / "runs"
RUN_ROOT = HERE / "runs_epoch1000"
TOTAL_EPOCHS = 1000

EXPERIMENTS = {
    ("hcl", "b2_bidirectional"): SOURCE_ROOT
    / "ablation_v2_latent-bn_hcl_b2_bidirectional_seed42.yaml",
    ("mnist", "b5_multi_density"): SOURCE_ROOT
    / "ablation_v2_latent-bn_mnist_b5_multi_density_seed42.yaml",
}

CONDITIONS = {
    "distance_allpair": {},
    "distance_p_rownorm_allpair": {"distance_p_row_normalize": True},
    "rank_allpair": {"manifold_affinity": "rank"},
    "distance_hardpair": {"manifold_hardpair": True},
}


def _rewrite_callbacks(config, output_root, run_name, dataset, variant, condition):
    for callback in config["trainer"]["callbacks"]:
        class_path = callback["class_path"]
        args = callback.setdefault("init_args", {})
        if class_path.endswith("ModelCheckpoint"):
            args.update(
                dirpath=f"{output_root}/checkpoints",
                filename=f"{run_name}-{{epoch:04d}}",
                every_n_epochs=50,
            )
        elif class_path.endswith("VisualizationCallback"):
            args.update(
                output_dir=f"{output_root}/epoch_embeddings",
                every_n_epochs=50,
                save_embeddings=True,
                embedding_method_name=run_name,
            )
        elif class_path.endswith("HeterogeneityPlotCallback"):
            args.update(output_dir=f"{output_root}/heterogeneity", every_n_epochs=50)
        elif class_path.endswith("FidelityEvalCallback"):
            args.update(
                every_n_epochs=50,
                down_sample=3000,
                knn_k=12,
                density_k=15,
                seed=42,
                bn_train_eval_diagnostic=True,
                diagnostic_output_path=f"{output_root}/metrics/full_geometry_and_quality.jsonl",
            )
        elif class_path.endswith("RuntimeProfilerCallback"):
            args.update(output_path=f"{output_root}/runtime/runtime.csv")
        elif class_path.endswith("EncoderBenchmarkCallback"):
            args.update(check_grad_every_n_steps=1, fail_on_nonfinite=True)
        elif class_path.endswith("PaperEmbeddingCallback"):
            args.update(
                output_dir=f"{output_root}/figure",
                method_name=run_name,
            )
        elif class_path.endswith("RuntimeInfoCallback"):
            args.update(output_dir=f"{output_root}/runtime_info")
        elif class_path.endswith("SaveConsolidatedEmbeddingsCallback"):
            args.update(
                dataset_name=f"{dataset}_{variant}_{condition}_seed42",
                output_dir=f"{output_root}/final_embedding",
                embedding_method_name=run_name,
            )


def build_config(dataset, variant, source_path, condition, overrides):
    with source_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_args = config["model"]["init_args"]
    model_args.update(
        max_epochs=TOTAL_EPOCHS,
        embedding_standardization="differentiable_global",
        embedding_std_floor=1.0e-4,
        density_distance_floor=1.0e-4,
        density_pearson_floor=1.0e-4,
        distance_p_row_normalize=False,
    )
    model_args.update(overrides)

    trainer = config["trainer"]
    trainer.update(
        max_epochs=TOTAL_EPOCHS,
        precision="32-true",
        check_val_every_n_epoch=50,
        log_every_n_steps=1,
    )

    run_name = f"rootcause_gen1000_{dataset}_{variant}_{condition}_seed42"
    output_root = f"outputs/rootcause_generalization_1000/{dataset}/{variant}/{condition}/seed42"
    logger_args = trainer["logger"]["init_args"]
    logger_args.update(
        name=run_name,
        project="TopoBranch_rootcause_generalization",
        save_dir="wandb/rootcause_generalization",
        tags=[
            "rootcause-generalization",
            "distance-p-row-mass",
            "fp32",
            "epoch1000",
            dataset,
            variant,
            condition,
            "seed42",
        ],
    )
    _rewrite_callbacks(config, output_root, run_name, dataset, variant, condition)
    return config


def _scientific_signature(config):
    signature = {
        "seed_everything": config["seed_everything"],
        "data": deepcopy(config["data"]),
        "model": deepcopy(config["model"]),
        "trainer": {
            key: deepcopy(config["trainer"][key])
            for key in (
                "max_epochs",
                "accelerator",
                "devices",
                "precision",
                "benchmark",
                "check_val_every_n_epoch",
                "num_sanity_val_steps",
            )
        },
    }
    signature["model"]["init_args"].pop("distance_p_row_normalize")
    return signature


def validate_pair(baseline, rownorm):
    assert baseline["model"]["init_args"]["distance_p_row_normalize"] is False
    assert rownorm["model"]["init_args"]["distance_p_row_normalize"] is True
    assert _scientific_signature(baseline) == _scientific_signature(rownorm)


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for (dataset, variant), source_path in EXPERIMENTS.items():
        generated = {}
        for condition, overrides in CONDITIONS.items():
            config = build_config(
                dataset, variant, source_path, condition, overrides
            )
            generated[condition] = config
            output_path = RUN_ROOT / f"rootcause_gen1000_{dataset}_{variant}_{condition}_seed42.yaml"
            header = (
                "# Generated by configs/encoder_bench/rootcause_generalization/generate_configs.py\n"
                f"# dataset={dataset}; variant={variant}; condition={condition}; seed=42\n"
            )
            with output_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(header)
                yaml.safe_dump(config, stream, sort_keys=False)
        validate_pair(
            generated["distance_allpair"],
            generated["distance_p_rownorm_allpair"],
        )


if __name__ == "__main__":
    main()
