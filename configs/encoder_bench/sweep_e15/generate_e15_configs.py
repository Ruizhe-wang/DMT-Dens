"""Generates the E15 run configs and sweep YAML: post-norm on both branches, HCL.

Direction. E13 showed the latent Transformer's deficit against the MLP shrinks
monotonically with dimension (NG20 -0.074, MNIST -0.028, HCL -0.023) and that on
HCL it already beats the MLP on SVC, kNN preservation and speed. The Transformer
branch is therefore developed on HCL, the target domain, not on NG20 - E14
independently concluded NG20-only architecture search should stop.

Hypothesis. E10-E12 established that cross-sample normalization helps density,
but the Transformer blocks are pre-norm, so the residual stream itself - the
quantity that is pooled and projected - is never normalized. A4's null result
for token BatchNorm was measured in the *pre-norm slot*. The evidence predicts
the untested cell is **post-norm**, and the same prediction applies to the
ResMLP branch, so both are tested in one wave against the same E13 HCL controls.

Comparison chain (all against E13 HCL, same machine and protocol):
    latent-bn   0.8532   control for T1/T3
    T1 = latent-bn + post-norm              isolates norm *position*
    T2 = T1 + token BatchNorm               isolates norm *family* in the post slot
    T3 = latent-bn + rank 32                the M*rank bottleneck, 5.9x -> 3.0x
    resmlp-bn   0.8616   control for C1
    C1 = resmlp-bn + post-norm              the same hypothesis without attention
    mlp         0.8758   the bar to beat

Capacity, HCL baseline 2,632,312 and a 2x protocol ceiling of 5,264,624:
T1/T2 approximately 2.90M (1.10x), T3 approximately 4.6M (1.75x), C1
approximately 3.16M (1.20x). Rank 64 is excluded: its compression matrix alone
is 3038 x 2048 = 6.2M parameters, 2.36x, outside the band. M is not raised
because memory scales with M and 6.1 GB is already the heaviest run.

Run:
    python configs/encoder_bench/sweep_e15/generate_e15_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = HERE.parent / "sweep" / "encoder_bench_e15_postnorm_hcl_seed42.yaml"
TEMPLATE = HERE.parent / "hcl_mlp.yaml"
PROJECT = "TopoBranch_encoder_bench_HCL"
SEED = 42
LEARNING_RATES = [("0p001", 0.001), ("0p0003", 0.0003)]

# The E13 HCL latent-bn recipe, reproduced exactly.
LATENT_BN = {
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
    "input_norm": "batchnorm",
}
# The E13 HCL resmlp-bn recipe, reproduced exactly.
RESMLP_BN = {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"}


def latent(**overrides):
    kwargs = copy.deepcopy(LATENT_BN)
    kwargs.update(overrides)
    return kwargs


def resmlp(**overrides):
    kwargs = copy.deepcopy(RESMLP_BN)
    kwargs.update(overrides)
    return kwargs


ARMS = {
    "hcl_latent_postnorm_ln": (
        "latent_transformer",
        latent(block_norm_position="post"),
        "latent-postnorm-ln",
        "T1: latent-bn + post-norm, LayerNorm retained (isolates norm position)",
    ),
    "hcl_latent_postnorm_bn": (
        "latent_transformer",
        latent(block_norm_position="post", block_norm="batchnorm"),
        "latent-postnorm-bn",
        "T2: T1 + token BatchNorm (isolates norm family in the post slot)",
    ),
    "hcl_latent_rank32": (
        "latent_transformer",
        latent(latent_rank=32),
        "latent-rank32",
        "T3: latent-bn + rank 32, M*rank 512 -> 1024, compression 5.9x -> 3.0x",
    ),
    "hcl_resmlp_postnorm_bn": (
        "resmlp",
        resmlp(norm_position="post"),
        "resmlp-postnorm-bn",
        "C1: resmlp-bn + post-norm, the same hypothesis without attention",
    ),
}


def build(template, slug, enc_type, enc_kwargs, label, purpose, lr_tag, lr):
    cfg = copy.deepcopy(template)
    name = f"{slug}_lr{lr_tag}_seed{SEED}"
    out_root = f"outputs/encoder_tuning/E15/{name}"

    cfg["seed_everything"] = SEED
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = lr
    model_args["encoder_type"] = enc_type
    model_args["encoder_kwargs"] = copy.deepcopy(enc_kwargs)

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_hcl_lr{lr}_seed{SEED}"
    logger_args["project"] = PROJECT

    for callback in cfg["trainer"]["callbacks"]:
        init_args = callback.setdefault("init_args", {})
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
            init_args["output_path"] = f"results/encoder_tuning/E15/{name}/runtime.csv"
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
    for slug, (enc_type, enc_kwargs, label, purpose) in ARMS.items():
        for lr_tag, lr in LEARNING_RATES:
            name, cfg = build(
                template, slug, enc_type, enc_kwargs, label, purpose, lr_tag, lr
            )
            header = (
                "# Generated by configs/encoder_bench/sweep_e15/generate_e15_configs.py\n"
                f"# {purpose}.\n"
                f"# HCL D=3038; learning rate {lr}; seed {SEED}; batch 4096.\n"
                "# Data, augmentation, losses, projection head, schedule and callbacks\n"
                "# are the HCL dataset's own, identical to the E13 controls.\n"
            )
            path = RUNS_DIR / f"{name}.yaml"
            path.write_text(
                header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            written.append(f"configs/encoder_bench/sweep_e15/runs/{name}.yaml")
            print(f"wrote {name}.yaml")

    sweep = (
        "# E15: post-norm on both encoder branches, plus the M*rank bottleneck,\n"
        "# on HCL (D=3038) at seed 42. Controls are the E13 HCL runs on the same\n"
        "# machine and protocol: mlp 0.8758, resmlp-bn 0.8616, latent-bn 0.8532.\n"
        "# The bar is the MLP, not the previous Transformer best.\n"
        "program: main.py\n"
        "method: grid\n"
        f"project: {PROJECT}\n"
        "name: encoder_bench_e15_postnorm_hcl_seed42\n"
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
