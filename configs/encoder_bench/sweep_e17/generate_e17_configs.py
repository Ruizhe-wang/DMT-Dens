"""Generates E17: the three untested Transformer combinations, 3 seeds each.

Goal: parity or better against the MLP on NG20 / ACT / MNIST / HCL, with a
Transformer-family encoder. HCL is already at parity (latent-bn matches the
5-seed MLP baseline on density and is +0.012 on SVC). This round attacks the two
datasets where the gap is a *missing run* rather than a tested failure.

A. ACT latent-bn - a pure gap. ACT's only latent numbers are the pre-BatchNorm
   version. Input BatchNorm was worth +0.05 on NG20 and brought the latent
   encoder to parity on HCL; it was simply never run on ACT.

B. NG20 FT + the E10-E12 recipe - never combined. FT's last formal run was
   E08/E09, *before* input BatchNorm was found. Its best recorded S of 0.5635
   was measured with dropout 0.1, LayerNorm and no input norm. D=100 is also the
   only dimension where FT is feasible at all (101 tokens), and it is the
   architecture whose premise actually fits a low-dimensional input.

C. NG20 latent as a real compression - M*rank = 16*4 = 64 < D = 100. The current
   M32/r16 gives M*rank = 512, a 5x *expansion* on NG20, so the architecture's
   premise is inverted there; that is the likely reason it trails by 0.059 on
   NG20 while matching the MLP on HCL. M16/r4 was run in the E01 era but without
   any of the three elements later shown to matter (input BatchNorm, dropout 0,
   mean pooling).

Every arm runs 3 seeds from the start. Single-seed screening produced two wrong
verdicts in this project (the E03-E06 chain, and the rank32 correction), and at
an expected effect size near 0.01 against a single-seed 2-sigma band of 0.024 it
has no discriminating power.

Learning rate 1e-3 for all arms: it was the selected rate for every BatchNorm
variant on NG20 and ACT, and for FT's best recorded run.

Run:
    python configs/encoder_bench/sweep_e17/generate_e17_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = HERE.parent / "sweep" / "encoder_bench_e17_transformer_gaps.yaml"
BENCH = HERE.parent

NG20_TEMPLATE = (
    BENCH / "sweep_e08" / "runs"
    / "ng20_latent_transformer_m32_d192_r16_readout512_256_lr0p001.yaml"
)
ACT_TEMPLATE = BENCH / "act_resmlp.yaml"

SEEDS = [42, 43, 44]
LR = 0.001

ARMS = {
    # A. the ACT gap
    "act_latent_bn": {
        "template": "act",
        "project": "TopoBranch_encoder_bench_ACT",
        "dataset": "act",
        "encoder_type": "latent_transformer",
        "label": "latent-bn",
        "purpose": "A: ACT latent-bn, the input-BatchNorm recipe never run on ACT",
        "kwargs": {
            "num_latents": 32, "d_token": 224, "num_layers": 2, "num_heads": 4,
            "ffn_ratio": 4.0, "dropout": 0.0, "attn_dropout": 0.0,
            "latent_rank": 16, "pooling": "mean", "final_norm": True,
            "input_norm": "batchnorm", "force_mem_efficient": True,
        },
    },
    # B. FT with everything the round has since validated
    "ng20_ft_bn": {
        "template": "ng20",
        "project": "TopoBranch_encoder_bench_NG20",
        "dataset": "ng20",
        "encoder_type": "ft_transformer",
        "label": "ft-bn",
        "purpose": "B: NG20 FT + input BatchNorm + dropout 0 + mean pooling, never combined",
        "kwargs": {
            "d_token": 96, "num_layers": 2, "num_heads": 4, "ffn_ratio": 4.0,
            "dropout": 0.0, "attn_dropout": 0.0, "pooling": "mean",
            "final_norm": False, "input_norm": "batchnorm",
            "force_mem_efficient": True,
        },
    },
    # C. latent as an actual compression at D=100
    "ng20_latent_m16_r4_bn": {
        "template": "ng20",
        "project": "TopoBranch_encoder_bench_NG20",
        "dataset": "ng20",
        "encoder_type": "latent_transformer",
        "label": "latent-m16-r4-bn",
        "purpose": "C: NG20 latent with M*rank=64 < D=100, a real compression",
        "kwargs": {
            "num_latents": 16, "d_token": 224, "num_layers": 2, "num_heads": 4,
            "ffn_ratio": 4.0, "dropout": 0.0, "attn_dropout": 0.0,
            "latent_rank": 4, "pooling": "mean", "final_norm": True,
            "input_norm": "batchnorm", "force_mem_efficient": True,
        },
    },
}


def build(template, slug, arm, seed):
    cfg = copy.deepcopy(template)
    name = f"{slug}_lr0p001_seed{seed}"
    out_root = f"outputs/encoder_tuning/E17/{name}"
    label = arm["label"]

    cfg["seed_everything"] = seed
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = LR
    model_args["encoder_type"] = arm["encoder_type"]
    model_args["encoder_kwargs"] = copy.deepcopy(arm["kwargs"])

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_{arm['dataset']}_lr{LR}_seed{seed}"
    logger_args["project"] = arm["project"]

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
            init_args["output_path"] = f"results/encoder_tuning/E17/{name}/runtime.csv"
        elif path.endswith("PaperEmbeddingCallback"):
            init_args["output_dir"] = f"{out_root}/paper"
            init_args["method_name"] = label
        elif path.endswith("RuntimeInfoCallback"):
            init_args["output_dir"] = f"{out_root}/runtime_info"

    return name, cfg


def main():
    templates = {
        "ng20": yaml.safe_load(NG20_TEMPLATE.read_text(encoding="utf-8")),
        "act": yaml.safe_load(ACT_TEMPLATE.read_text(encoding="utf-8")),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for slug, arm in ARMS.items():
        for seed in SEEDS:
            name, cfg = build(templates[arm["template"]], slug, arm, seed)
            header = (
                "# Generated by configs/encoder_bench/sweep_e17/generate_e17_configs.py\n"
                f"# {arm['purpose']}.\n"
                f"# Learning rate {LR}; seed {seed}. Data, augmentation, losses,\n"
                "# projection head, batch, schedule and callbacks are the dataset's own.\n"
            )
            path = RUNS_DIR / f"{name}.yaml"
            path.write_text(
                header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            written.append(f"configs/encoder_bench/sweep_e17/runs/{name}.yaml")
            print(f"wrote {name}.yaml")

    sweep = (
        "# E17: the three untested Transformer combinations, 3 seeds each.\n"
        "# A ACT latent-bn (a gap, not a tested failure); B NG20 FT + the E10-E12\n"
        "# recipe (never combined; FT's last run predates the normalization finding);\n"
        "# C NG20 latent with M*rank < D (the current config is a 5x expansion there).\n"
        "# Sweep agents override a run config's project, so ACT runs will land in\n"
        "# whichever project this sweep declares - read results by run name.\n"
        "program: main.py\n"
        "method: grid\n"
        "project: TopoBranch_encoder_bench_NG20\n"
        "name: encoder_bench_e17_transformer_gaps\n"
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
