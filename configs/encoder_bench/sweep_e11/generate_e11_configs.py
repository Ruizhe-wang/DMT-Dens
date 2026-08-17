"""Generates the E11 run configs and sweep YAML.

E11 does two things and nothing else.

1. **Isolate the E10 mechanism.** E10-A5 confounded input BatchNorm with
   token-axis BatchNorm. Since A4 (token BatchNorm alone) was null, the gain is
   almost certainly the input norm, but that minimal configuration has not been
   run. A6 adds `input_norm` to A3 with the block LayerNorms retained; A7 makes
   the same change on the non-attention side, ResMLP with LayerNorm blocks plus
   an input BatchNorm.

2. **Confirm the two E10 winners on seeds 43 and 44.** Every E10 conclusion
   rests on seed 42, which is exactly what made the E03-E06 chain unreliable.

Run:
    python configs/encoder_bench/sweep_e11/generate_e11_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = (
    HERE.parent / "sweep" / "encoder_bench_ng20_e11_input_norm_and_seeds.yaml"
)
TEMPLATE = (
    HERE.parent
    / "sweep_e08"
    / "runs"
    / "ng20_latent_transformer_m32_d192_r16_readout512_256_lr0p001.yaml"
)

PROJECT = "TopoBranch_encoder_bench_NG20"

LATENT_CONTROL = {
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
    "force_mem_efficient": True,
}

RESMLP_CONTROL = {"width": 512, "num_blocks": 3, "dropout": 0.0}


def latent(**overrides):
    kwargs = copy.deepcopy(LATENT_CONTROL)
    kwargs.update(overrides)
    return kwargs


def resmlp(**overrides):
    kwargs = copy.deepcopy(RESMLP_CONTROL)
    kwargs.update(overrides)
    return kwargs


# slug -> (encoder_type, encoder_kwargs, wandb label, purpose, seeds, lr tags)
SPECS = {
    # 1. mechanism isolation, seed 42 only
    "ng20_latent_m32_r16_nodrop_inputbn": (
        "latent_transformer",
        latent(input_norm="batchnorm"),
        "latent-m32-r16-nodrop-inputbn",
        "A6: A3 + input BatchNorm only, block LayerNorm retained",
        [42],
        ["0p005", "0p001"],
    ),
    "ng20_resmlp_inputbn": (
        "resmlp",
        resmlp(input_norm="batchnorm"),
        "resmlp-inputbn",
        "A7: ResMLP LayerNorm blocks + input BatchNorm",
        [42],
        ["0p005", "0p001"],
    ),
    # 2. seed confirmation of the E10 winners
    "ng20_resmlp_bn": (
        "resmlp",
        resmlp(norm="batchnorm"),
        "resmlp-bn",
        "E10-A1 seed confirmation",
        [43, 44],
        ["0p005", "0p001"],
    ),
    "ng20_latent_m32_r16_nodrop_bn_input": (
        "latent_transformer",
        latent(block_norm="batchnorm", input_norm="batchnorm"),
        "latent-m32-r16-nodrop-bn-input",
        "E10-A5 seed confirmation",
        [43, 44],
        ["0p001"],
    ),
}

LR_VALUES = {"0p005": 0.005, "0p001": 0.001}


def build(template, slug, encoder_type, encoder_kwargs, label, lr_tag, lr, seed):
    cfg = copy.deepcopy(template)
    name = f"{slug}_lr{lr_tag}_seed{seed}"
    out_root = f"outputs/encoder_tuning/E11/{name}"

    cfg["seed_everything"] = seed
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = lr
    model_args["encoder_type"] = encoder_type
    model_args["encoder_kwargs"] = copy.deepcopy(encoder_kwargs)

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_ng20_lr{lr}_seed{seed}"
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
            init_args["output_path"] = f"results/encoder_tuning/E11/{name}/runtime.csv"
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
    for slug, (enc_type, kwargs, label, purpose, seeds, lr_tags) in SPECS.items():
        for seed in seeds:
            for lr_tag in lr_tags:
                name, cfg = build(
                    template, slug, enc_type, kwargs, label,
                    lr_tag, LR_VALUES[lr_tag], seed,
                )
                header = (
                    "# Generated by configs/encoder_bench/sweep_e11/generate_e11_configs.py\n"
                    f"# {purpose}.\n"
                    f"# Learning rate: {LR_VALUES[lr_tag]}; seed: {seed}; batch 4096\n"
                    "# (8192 encoder rows). Data, augmentation, losses, projection head,\n"
                    "# schedule and callbacks unchanged from the E01-E10 NG20 protocol.\n"
                )
                path = RUNS_DIR / f"{name}.yaml"
                path.write_text(
                    header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                written.append(f"configs/encoder_bench/sweep_e11/runs/{name}.yaml")
                print(f"wrote {name}.yaml")

    sweep = (
        "# E11: isolate the E10 normalization mechanism and confirm its winners\n"
        "# on seeds 43/44. A6/A7 are seed 42; the rest are seed confirmations.\n"
        "# Each run config owns its checkpoint, plot, runtime-CSV and\n"
        "# runtime-info paths so concurrent agents cannot share directories.\n"
        "program: main.py\n"
        "method: grid\n"
        f"project: {PROJECT}\n"
        "name: encoder_bench_ng20_e11_input_norm_and_seeds\n"
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
    print(f"wrote {SWEEP_PATH.name} with {len(written)} runs")


if __name__ == "__main__":
    main()
