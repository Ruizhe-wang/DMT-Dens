"""Generates the E13 run configs and sweep YAML: MNIST and HCL.

Neither dataset has any existing encoder-benchmark run - the W&B projects
`TopoBranch_encoder_bench_MNIST` and `TopoBranch_encoder_bench_HCL` do not
exist, so stages C and D of the original brief were never executed. E13
therefore builds the controls and the treatment together rather than adding a
treatment to an existing control.

Grid: 4 encoders x 2 datasets x 2 learning rates, seed 42.

  mlp          the dataset's own baseline, the reference every claim is against
  resmlp       LayerNorm control, the encoder the mechanism was found on
  resmlp-bn    the treatment: LayerNorm -> BatchNorm1d, nothing else
  latent-bn    latent M32/r16/d224 + input BatchNorm, dropout 0 (the E11 A6
               recipe), which is also the first high-dimensional datapoint for
               the attention route

Learning rates 1e-3 and 3e-4 bracket the two candidates: 3e-4 is these
datasets' mainline value, 1e-3 was selected for the BatchNorm variants on both
NG20 and ACT. Running a single rate risks reading a rate mismatch as a null
result, which is the misreading this round is specifically guarding against.
5e-3 is omitted; it never won for a BatchNorm variant on either dataset.

On HCL the latent encoder's `M * rank = 512` against D=3038 is a 5.9x
compression - the bottleneck the earlier report identified as untestable on
NG20. E13 only establishes whether that configuration works at all; the
`M * rank` sweep itself is a separate round.

FT-Transformer is excluded by architecture: one token per input dimension is
3039 tokens on HCL, measured at roughly 51 GB against an 11 GB card.

Run:
    python configs/encoder_bench/sweep_e13/generate_e13_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = HERE.parent / "sweep" / "encoder_bench_e13_mnist_hcl_seed42.yaml"
BENCH = HERE.parent

DATASETS = {
    "mnist": {"template": BENCH / "mnist_mlp.yaml", "project": "TopoBranch_encoder_bench_MNIST"},
    "hcl": {"template": BENCH / "hcl_mlp.yaml", "project": "TopoBranch_encoder_bench_HCL"},
}

LEARNING_RATES = [("0p001", 0.001), ("0p0003", 0.0003)]

ENCODERS = {
    "mlp": ("mlp", None, "mlp", "dataset baseline"),
    "resmlp": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0},
        "resmlp",
        "LayerNorm control",
    ),
    "resmlp_bn": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"},
        "resmlp-bn",
        "treatment: LayerNorm -> BatchNorm1d",
    ),
    "latent_bn": (
        "latent_transformer",
        {
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
        },
        "latent-bn",
        "E11 A6 recipe, first high-dimensional attention datapoint",
    ),
}

SEED = 42


def build(template, dataset, project, enc_slug, enc_type, enc_kwargs, label, lr_tag, lr):
    cfg = copy.deepcopy(template)
    name = f"{dataset}_{enc_slug}_lr{lr_tag}_seed{SEED}"
    out_root = f"outputs/encoder_tuning/E13/{name}"

    cfg["seed_everything"] = SEED
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = lr
    model_args["encoder_type"] = enc_type
    model_args["encoder_kwargs"] = copy.deepcopy(enc_kwargs) if enc_kwargs else None

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_{dataset}_lr{lr}_seed{SEED}"
    logger_args["project"] = project

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
            init_args["output_path"] = f"results/encoder_tuning/E13/{name}/runtime.csv"
        elif path.endswith("PaperEmbeddingCallback"):
            init_args["output_dir"] = f"{out_root}/paper"
            init_args["method_name"] = label
        elif path.endswith("RuntimeInfoCallback"):
            init_args["output_dir"] = f"{out_root}/runtime_info"

    return name, cfg


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for dataset, spec in DATASETS.items():
        template = yaml.safe_load(spec["template"].read_text(encoding="utf-8"))
        for enc_slug, (enc_type, enc_kwargs, label, purpose) in ENCODERS.items():
            for lr_tag, lr in LEARNING_RATES:
                name, cfg = build(
                    template, dataset, spec["project"], enc_slug,
                    enc_type, enc_kwargs, label, lr_tag, lr,
                )
                header = (
                    "# Generated by configs/encoder_bench/sweep_e13/generate_e13_configs.py\n"
                    f"# E13 {dataset.upper()}: {purpose}.\n"
                    f"# Learning rate: {lr}; seed: {SEED}. Data, augmentation, losses,\n"
                    "# projection head, batch, schedule and callbacks are the dataset's own,\n"
                    "# identical across all four encoders.\n"
                )
                path = RUNS_DIR / f"{name}.yaml"
                path.write_text(
                    header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                written.append(f"configs/encoder_bench/sweep_e13/runs/{name}.yaml")
                print(f"wrote {name}.yaml")

    sweep = (
        "# E13: MNIST (D=784) and HCL (D=3038), seed 42.\n"
        "# Neither dataset had any prior encoder-benchmark run, so the controls\n"
        "# (mlp, resmlp) are built here alongside the treatment (resmlp-bn) and\n"
        "# the first high-dimensional attention datapoint (latent-bn).\n"
        "# Each run config carries its own project and its own artifact paths.\n"
        "program: main.py\n"
        "method: grid\n"
        "project: TopoBranch_encoder_bench_HCL\n"
        "name: encoder_bench_e13_mnist_hcl_seed42\n"
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
