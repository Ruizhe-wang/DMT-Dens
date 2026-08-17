"""Generates the full-suite encoder comparison: 10 datasets x 2 encoders x N seeds.

Compared against the existing 5-seed MLP baseline
(`D:/ruizhe/DMT_result/table1_supp_mean_std_best_hp_by_dataset_method.csv` and
`difftree_baseline_density_localdensity_svc_mean_std_by_dataset_method.csv`),
which used one unified hyperparameter setting rather than per-dataset tuning.

Two encoders, both at the configuration that won their family:

  resmlp-bn   width 512, 3 blocks, dropout 0, LayerNorm -> BatchNorm1d
  latent-bn   M=32, rank=16, d_token=224, 2 blocks, 4 heads, dropout 0,
              mean pooling, input BatchNorm

Every dataset's data block, input dimension and training-set size are taken from
that dataset's own existing config, while the model block, callbacks and
evaluation are taken from the encoder-benchmark template, so the metric set is
identical across all ten datasets.

`--smoke-only` writes just one seed per dataset/encoder, for the two-batch
feasibility pass (MCA at D=9120 and EMNIST at N=697,932 are the two unknowns).

Run:
    python configs/encoder_bench/sweep_e18/generate_e18_configs.py --seeds 42
    python configs/encoder_bench/sweep_e18/generate_e18_configs.py --seeds 42,43,44,45,46
"""

import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_DIR = HERE.parent / "sweep"
BENCH = HERE.parent
# Template supplies the model block, trainer and the full callback set.
TEMPLATE = BENCH / "hcl_mlp.yaml"

# dataset -> (source config for the data block, D, N, batch, measured minutes
# per 1000-epoch run under 8-way concurrency).
#
# The per-run minutes come from a measured 5-epoch pass, scaled by 1.53. That
# factor is calibrated, not guessed: HCL takes 6.42 s/epoch alone and E13
# measured 9.8 s/epoch for the same config under 8-way concurrency, and the E13
# figure already amortizes the validation callbacks.
#
# `aqc` is deliberately absent. It is not one of the paper's Table 1 datasets,
# and it runs 447 steps per epoch - 37x HCL - because its data module loads far
# more than the `num_train_data` its config declares. Including it would add
# 450 GPU-hours, 61% of the total, for a dataset the comparison does not need.
DATASETS = {
    "emnist": (BENCH.parent / "ablation/component/emnist_full.yaml", 784, 697932, 4096, 900),
    "epi": (BENCH.parent / "ablation/component/epi_full.yaml", 500, 100000, 4096, 166),
    "hcl": (BENCH / "hcl_mlp.yaml", 3038, 60000, 4096, 164),
    "tree": (BENCH.parent / "ablation/component/tree_full.yaml", 1000, 80000, 4096, 134),
    "mca": (BENCH / "mca_mlp.yaml", 9120, 23341, 4096, 131),
    "mnist": (BENCH / "mnist_mlp.yaml", 784, 60000, 4096, 110),
    "ng20": (BENCH / "ng20_mlp.yaml", 100, 18846, 4096, 50),
    "gast10k": (BENCH.parent / "ablation/component/gast10k_full.yaml", 1458, 10638, 4096, 45),
    "act": (BENCH / "act_mlp.yaml", 561, 10299, 5000, 44),
}

ENCODERS = {
    "resmlp_bn": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"},
        "resmlp-bn",
    ),
    "latent_bn": (
        "latent_transformer",
        {
            "num_latents": 32, "d_token": 224, "num_layers": 2, "num_heads": 4,
            "ffn_ratio": 4.0, "dropout": 0.0, "attn_dropout": 0.0,
            "latent_rank": 16, "pooling": "mean", "final_norm": True,
            "input_norm": "batchnorm", "force_mem_efficient": True,
        },
        "latent-bn",
    ),
}

# One unified learning rate, matching how the MLP baseline was produced. Per
# dataset selection would give the candidates an advantage the baseline never
# had.
LR = 0.001


def build(template, dataset, source_cfg, num_input_dim, num_train_data,
          batch_size, enc_slug, enc_type, enc_kwargs, label, seed):
    cfg = copy.deepcopy(template)
    name = f"{dataset}_{enc_slug}_seed{seed}"
    out_root = f"outputs/encoder_tuning/E18/{name}"

    cfg["seed_everything"] = seed
    # Data block comes wholesale from the dataset's own config.
    cfg["data"] = copy.deepcopy(source_cfg["data"])
    cfg["data"]["init_args"]["batch_size"] = batch_size

    model_args = cfg["model"]["init_args"]
    model_args["num_input_dim"] = num_input_dim
    model_args["num_train_data"] = num_train_data
    model_args["lr"] = LR
    model_args["encoder_type"] = enc_type
    model_args["encoder_kwargs"] = copy.deepcopy(enc_kwargs)

    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_{dataset}_seed{seed}"
    logger_args["project"] = "TopoBranch_encoder_suite"

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
            init_args["output_path"] = f"results/encoder_tuning/E18/{name}/runtime.csv"
        elif path.endswith("PaperEmbeddingCallback"):
            init_args["output_dir"] = f"{out_root}/paper"
            init_args["method_name"] = label
        elif path.endswith("RuntimeInfoCallback"):
            init_args["output_dir"] = f"{out_root}/runtime_info"

    return name, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42", help="comma separated seed list")
    parser.add_argument("--sweep-name", default="encoder_bench_e18_suite")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    # Longest-first (LPT). With 90 runs spanning 44 minutes to 15 hours, grid
    # order would leave the ten 15-hour EMNIST runs as a tail and roughly double
    # the makespan.
    ordered = sorted(DATASETS.items(), key=lambda kv: -kv[1][4])
    for dataset, (source_path, D, N, batch, _minutes) in ordered:
        source_cfg = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        for enc_slug, (enc_type, enc_kwargs, label) in ENCODERS.items():
            for seed in seeds:
                name, cfg = build(
                    template, dataset, source_cfg, D, N, batch,
                    enc_slug, enc_type, enc_kwargs, label, seed,
                )
                header = (
                    "# Generated by configs/encoder_bench/sweep_e18/generate_e18_configs.py\n"
                    f"# Full-suite comparison: {dataset} (D={D}, N={N}), {label}, seed {seed}.\n"
                    f"# Unified lr {LR} for every dataset and encoder, matching how the\n"
                    "# 5-seed MLP baseline was produced. Evaluation callbacks are identical\n"
                    "# across all datasets; the data block comes from the dataset's own config.\n"
                )
                path = RUNS_DIR / f"{name}.yaml"
                path.write_text(
                    header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                    encoding="utf-8",
                )
                written.append(f"configs/encoder_bench/sweep_e18/runs/{name}.yaml")

    total = sum(v[4] for v in DATASETS.values()) * len(ENCODERS) * len(seeds)
    print(f"wrote {len(written)} run configs for seeds {seeds}")
    print(f"estimated {total/60:.0f} GPU-hours, about {total/60/8:.0f} h wall clock on 8 GPUs")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    sweep_path = SWEEP_DIR / f"{args.sweep_name}.yaml"
    sweep = (
        "# E18: full-suite encoder comparison against the existing 5-seed MLP\n"
        "# baseline. Two encoders, one unified learning rate, identical evaluation.\n"
        "# Sweep agents override a run config's project, so read results by run name.\n"
        "program: main.py\n"
        "method: grid\n"
        "project: TopoBranch_encoder_suite\n"
        f"name: {args.sweep_name}\n"
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
    sweep_path.write_text(sweep, encoding="utf-8")
    print(f"wrote {sweep_path.name}")


if __name__ == "__main__":
    main()
