"""Generate the final 5-seed Latent-Transformer case-study configs.

The case-study-specific TopoBranch objective, projection head, data settings,
and ground-truth callbacks come from the existing paper configs.  This script
only installs the canonical E18 ``latent-bn`` encoder package and gives every
run an isolated artifact directory so parallel runs cannot overwrite figures
or embeddings.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RUN_DIR = OUT_DIR / "runs"
SEEDS = (41, 42, 43, 44, 45)

PROFILES = {
    # Preserve the case-study settings selected for the existing paper runs.
    "native": {},
    # Match the E18 full-suite density/manifold settings exactly.
    "aligned": {
        "nu_lat": 0.01,
        "density_weight": 0.0018,
        "density_num_anchors": 512,
    },
}

SPECS = {
    "celegan": ROOT / "configs/case_study/celegan/5seed_rep/celegan_topobranch_5seed.yaml",
    "dyngen": ROOT / "configs/case_study/dyngen/difftree_dyngen_best5seed.yaml",
}

ENCODERS = {
    "latent-bn": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "latent_rank": 16,
            "pooling": "mean",
            "final_norm": True,
            "input_norm": "batchnorm",
            "force_mem_efficient": True,
        },
    },
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def callback(callbacks: list[dict], suffix: str) -> dict:
    matches = [item for item in callbacks if item["class_path"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {suffix} callback, found {len(matches)}")
    return matches[0]


def make_run(
    dataset: str, encoder: str, profile: str, seed: int, base: dict
) -> tuple[str, dict]:
    cfg = copy.deepcopy(base)
    run_name = f"{encoder}_{profile}_{dataset}_seed{seed}"
    encoder_slug = encoder.replace("-", "_")
    artifact_root = (
        f"outputs/encoder_case_study/{encoder_slug}/{profile}/{dataset}/seed{seed}"
    )

    # Keep the case-study data split/sample fixed at seed 42, matching the
    # existing paper comparison.  Only model/training randomness varies.
    cfg["seed_everything"] = seed

    model_args = cfg["model"]["init_args"]
    model_args["lr"] = 0.001  # canonical encoder-suite LR selected before E18
    model_args.update(copy.deepcopy(ENCODERS[encoder]))
    model_args.update(PROFILES[profile])

    trainer = cfg["trainer"]
    trainer["enable_checkpointing"] = True
    trainer["logger"]["init_args"].update(
        {
            "name": run_name,
            "project": "TopoBranch_encoder_case_study",
            "save_dir": "wandb/encoder_case_study",
        }
    )

    callbacks = trainer["callbacks"]
    callback(callbacks, "xc_plot_callback.VisualizationCallback")["init_args"].update(
        {
            "output_dir": f"{artifact_root}/diagnostics",
            "embedding_method_name": encoder,
        }
    )
    case_plot = callback(callbacks, "case_study_callback.VisualizationCallback")
    case_plot["init_args"].update(
        {
            "output_dir": f"{artifact_root}/paper",
            "method_name": f"TopoBranch {encoder}",
        }
    )
    consolidated = callback(callbacks, "SaveConsolidatedEmbeddingsCallback")
    consolidated["init_args"].update(
        {
            "dataset_name": f"{dataset}_seed{seed}",
            "output_dir": f"{artifact_root}/embeddings",
            "embedding_method_name": encoder,
        }
    )

    if dataset == "dyngen":
        for suffix in (
            "FateProbabilityVisualizationCallback",
            "TrueTreeOverlayVisualizationCallback",
        ):
            item = callback(callbacks, suffix)
            item["init_args"].update(
                {
                    "output_dir": f"{artifact_root}/paper",
                    "method_name": f"TopoBranch {encoder}",
                }
            )

    callbacks.insert(
        0,
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": f"{artifact_root}/checkpoints",
                "filename": f"{run_name}-{{epoch:04d}}",
                "save_last": True,
                "every_n_epochs": 100,
                # Keep only last.ckpt; 20 runs do not need 10 periodic copies each.
                "save_top_k": 0,
            },
        },
    )
    callbacks.extend(
        [
            {
                "class_path": "callbacks.runtime_profiler.RuntimeProfilerCallback",
                "init_args": {
                    "output_path": f"results/encoder_case_study/{run_name}/runtime.csv"
                },
            },
            {
                "class_path": "callbacks.encoder_benchmark.EncoderBenchmarkCallback",
                "init_args": {
                    "tail_epochs": 20,
                    "cv_threshold": 0.25,
                    "direction_change_threshold": 0.5,
                    "spike_ratio_threshold": 3.0,
                    "check_grad_every_n_steps": 50,
                },
            },
            {
                "class_path": "callbacks.runtime_info_callback.RuntimeInfoCallback",
                "init_args": {"output_dir": f"{artifact_root}/runtime_info"},
            },
        ]
    )
    return run_name, cfg


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for stale_config in RUN_DIR.glob("*.yaml"):
        stale_config.unlink()
    run_paths: dict[str, list[str]] = {profile: [] for profile in PROFILES}
    for dataset, base_path in SPECS.items():
        base = load_yaml(base_path)
        for encoder in ENCODERS:
            for profile in PROFILES:
                for seed in SEEDS:
                    run_name, cfg = make_run(dataset, encoder, profile, seed, base)
                    path = RUN_DIR / f"{run_name}.yaml"
                    header = (
                        "# Generated by configs/encoder_bench/case_study_latent_5seed/generate_configs.py\n"
                        f"# Case-study run: {dataset}, {encoder}, profile={profile}, seed {seed}.\n"
                        "# Data sampling remains fixed at seed 42 to match the existing paper baseline.\n"
                    )
                    with path.open("w", encoding="utf-8") as handle:
                        handle.write(header)
                        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
                    run_paths[profile].append(path.relative_to(ROOT).as_posix())

    stale_sweep = OUT_DIR / "sweep.yaml"
    if stale_sweep.exists():
        stale_sweep.unlink()
    for profile, paths in run_paths.items():
        sweep = {
            "program": "main.py",
            "method": "grid",
            "project": "TopoBranch_encoder_case_study",
            "name": f"latent_bn_case_study_{profile}_2dataset_5seed",
            "parameters": {"config": {"values": paths}},
            "command": [
                "${env}",
                "${interpreter}",
                "${program}",
                "fit",
                "${args}",
            ],
        }
        sweep_path = OUT_DIR / f"sweep_{profile}.yaml"
        with sweep_path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# {profile}: 2 datasets x 1 encoder x 5 baked seeds = 10 runs.\n"
            )
            yaml.safe_dump(sweep, handle, sort_keys=False, allow_unicode=True)

    total = sum(len(paths) for paths in run_paths.values())
    print(f"Wrote {total} run configs and {len(PROFILES)} sweep files in {OUT_DIR}")


if __name__ == "__main__":
    main()
