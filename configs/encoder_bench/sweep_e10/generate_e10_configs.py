"""Generates the E10 normalization-hypothesis run configs and sweep YAML.

E10 asks whether the baseline MLP's density advantage comes from cross-sample
BatchNorm rather than from being an MLP. Every arm changes exactly one thing
against a control that already has a completed run; nothing else in the data,
augmentation, losses, projection head, schedule, batch or callbacks moves.

Controls:
  A1, A2 -> published NG20 ResMLP (3 seeds, sweeps v92rl72v / 8e0a7n28)
  A3     -> E03 latent M32/r16/d224 mean pooling, W&B run d9830su3
  A4     -> A3
  A5     -> A4

Run:
    python configs/encoder_bench/sweep_e10/generate_e10_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = (
    HERE.parent / "sweep" / "encoder_bench_ng20_e10_normalization_seed42.yaml"
)
TEMPLATE = (
    HERE.parent
    / "sweep_e08"
    / "runs"
    / "ng20_latent_transformer_m32_d192_r16_readout512_256_lr0p001.yaml"
)

PROJECT = "TopoBranch_encoder_bench_NG20"
SEED = 42
LEARNING_RATES = [("0p005", 0.005), ("0p001", 0.001)]

# The E03 latent winner (W&B d9830su3), reproduced exactly. A3/A4/A5 each add
# one key to this block and change nothing else.
LATENT_CONTROL = {
    "num_latents": 32,
    "d_token": 224,
    "num_layers": 2,
    "num_heads": 4,
    "ffn_ratio": 4.0,
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "latent_rank": 16,
    "pooling": "mean",
    "final_norm": True,
    "force_mem_efficient": True,
}


def latent(**overrides):
    kwargs = copy.deepcopy(LATENT_CONTROL)
    kwargs.update(overrides)
    return kwargs


# slug -> (arm, encoder_type, encoder_kwargs, wandb method label, change)
ARMS = {
    "ng20_resmlp_bn": (
        "A1",
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"},
        "resmlp-bn",
        "ResMLP LayerNorm -> BatchNorm1d",
    ),
    "ng20_resmlp_lrelu": (
        "A2",
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "activation": "leaky_relu"},
        "resmlp-lrelu",
        "ResMLP GELU -> LeakyReLU(0.1)",
    ),
    "ng20_latent_m32_r16_nodrop": (
        "A3",
        "latent_transformer",
        latent(dropout=0.0, attn_dropout=0.0),
        "latent-m32-r16-nodrop",
        "latent dropout 0.1 -> 0.0 (matches the MLP, which has none)",
    ),
    "ng20_latent_m32_r16_nodrop_bn": (
        "A4",
        "latent_transformer",
        latent(dropout=0.0, attn_dropout=0.0, block_norm="batchnorm"),
        "latent-m32-r16-nodrop-bn",
        "A3 + token LayerNorm -> BatchNorm over tokens",
    ),
    "ng20_latent_m32_r16_nodrop_bn_input": (
        "A5",
        "latent_transformer",
        latent(
            dropout=0.0,
            attn_dropout=0.0,
            block_norm="batchnorm",
            input_norm="batchnorm",
        ),
        "latent-m32-r16-nodrop-bn-input",
        "A4 + input BatchNorm1d(D) stem before compression",
    ),
}


def build_run_config(template, slug, arm, encoder_type, encoder_kwargs, label, lr_tag, lr):
    cfg = copy.deepcopy(template)
    name = f"{slug}_lr{lr_tag}"
    out_root = f"outputs/encoder_tuning/E10/{name}"

    cfg["seed_everything"] = SEED
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = lr
    model_args["encoder_type"] = encoder_type
    model_args["encoder_kwargs"] = copy.deepcopy(encoder_kwargs)

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_ng20_lr{lr}_seed{SEED}"
    logger_args["project"] = PROJECT

    for callback in cfg["trainer"]["callbacks"]:
        init_args = callback.get("init_args", {})
        path = callback["class_path"]
        if path.endswith("ModelCheckpoint"):
            init_args["dirpath"] = f"{out_root}/checkpoints"
            init_args["filename"] = name + "-{epoch:04d}"
        elif path.endswith("VisualizationCallback"):
            init_args["output_dir"] = f"{out_root}/plots"
            init_args["embedding_method_name"] = label
        elif path.endswith("HeterogeneityPlotCallback"):
            init_args["output_dir"] = f"{out_root}/heterogeneity"
        elif path.endswith("RuntimeProfilerCallback"):
            init_args["output_path"] = f"results/encoder_tuning/E10/{name}/runtime.csv"
        elif path.endswith("PaperEmbeddingCallback"):
            init_args["output_dir"] = f"{out_root}/paper"
            init_args["method_name"] = label
        elif path.endswith("RuntimeInfoCallback"):
            init_args["output_dir"] = f"{out_root}/runtime_info"

    return name, cfg


def main():
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for slug, (arm, encoder_type, encoder_kwargs, label, change) in ARMS.items():
        for lr_tag, lr in LEARNING_RATES:
            name, cfg = build_run_config(
                template, slug, arm, encoder_type, encoder_kwargs, label, lr_tag, lr
            )
            header = (
                "# Generated by configs/encoder_bench/sweep_e10/generate_e10_configs.py\n"
                f"# E10 arm {arm}: {change}.\n"
                f"# Learning rate: {lr}; seed: {SEED}; batch 4096 (8192 encoder rows).\n"
                "# Data, augmentation, losses, projection head, schedule and callbacks\n"
                "# are unchanged from the E01-E09 NG20 protocol.\n"
            )
            path = RUNS_DIR / f"{name}.yaml"
            path.write_text(
                header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            written.append(f"configs/encoder_bench/sweep_e10/runs/{name}.yaml")
            print(f"wrote {path.relative_to(HERE.parents[2])}")

    sweep = (
        "# E10: the normalization hypothesis, NG20 seed 42.\n"
        "# 5 arms x 2 learning rates = 10 runs. Each run config has its own\n"
        "# checkpoint, plot, runtime-CSV and runtime-info paths so concurrent\n"
        "# agents cannot share artifact directories (the E01 failure mode).\n"
        "program: main.py\n"
        "method: grid\n"
        f"project: {PROJECT}\n"
        "name: encoder_bench_ng20_e10_normalization_seed42\n"
        "command:\n"
        "  - ${env}\n"
        "  - ${interpreter}\n"
        "  - ${program}\n"
        "  - fit\n"
        "  - ${args}\n"
        "parameters:\n"
        "  config:\n"
        "    values:\n"
        + "".join(f"      - {p}\n" for p in written)
    )
    SWEEP_PATH.write_text(sweep, encoding="utf-8")
    print(f"wrote {SWEEP_PATH.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
