"""Generates the E12 run configs and sweep YAML.

E12 asks three questions and deliberately includes the one that can overturn
the E10/E11 conclusions.

Q1  Complete the 2x2. A1 (BatchNorm blocks) and A7 (input BatchNorm) are each
    positive on ResMLP; their combination has never been run. A8 = both.

Q2  Replicate the mechanism on a second dataset. ACT (D=561) already has
    3-seed MLP and ResMLP controls from sweep r1awjeoh, run from
    configs/encoder_bench/act_resmlp.yaml at lr 1e-3, so the only change is
    norm=batchnorm. This is the falsification test: the NG20 mechanism claim
    rests on one dataset.

Q3  Settle the open regression. A6 (latent, input BatchNorm only) has one seed;
    its SVC behaviour decides whether the input norm itself costs label
    separability.

No encoder code changes are required -- every switch already exists.

Run:
    python configs/encoder_bench/sweep_e12/generate_e12_configs.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
SWEEP_PATH = HERE.parent / "sweep" / "encoder_bench_e12_combination_and_act.yaml"
BENCH = HERE.parent

NG20_TEMPLATE = (
    BENCH / "sweep_e08" / "runs"
    / "ng20_latent_transformer_m32_d192_r16_readout512_256_lr0p001.yaml"
)
ACT_TEMPLATE = BENCH / "act_resmlp.yaml"

NG20_PROJECT = "TopoBranch_encoder_bench_NG20"
ACT_PROJECT = "TopoBranch_encoder_bench_ACT"

LATENT_A6 = {
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

# slug -> dict describing one run group
GROUPS = {
    # Q1: the missing 2x2 cell, NG20
    "ng20_resmlp_bn_inputbn": {
        "template": "ng20",
        "project": NG20_PROJECT,
        "encoder_type": "resmlp",
        "encoder_kwargs": {
            "width": 512,
            "num_blocks": 3,
            "dropout": 0.0,
            "norm": "batchnorm",
            "input_norm": "batchnorm",
        },
        "label": "resmlp-bn-inputbn",
        "purpose": "Q1/A8: BatchNorm blocks + input BatchNorm",
        "runs": [(42, "0p001", 0.001), (43, "0p001", 0.001), (44, "0p001", 0.001),
                 (42, "0p005", 0.005)],
    },
    # Q2: cross-dataset replication of the mechanism
    "act_resmlp_bn": {
        "template": "act",
        "project": ACT_PROJECT,
        "encoder_type": "resmlp",
        "encoder_kwargs": {
            "width": 512,
            "num_blocks": 3,
            "dropout": 0.0,
            "norm": "batchnorm",
        },
        "label": "resmlp-bn",
        "purpose": "Q2: ACT replication, only norm changes vs the ACT ResMLP control",
        "runs": [(42, "0p001", 0.001), (43, "0p001", 0.001), (44, "0p001", 0.001)],
    },
    # Q3: seeds for the simplified latent recipe
    "ng20_latent_m32_r16_nodrop_inputbn": {
        "template": "ng20",
        "project": NG20_PROJECT,
        "encoder_type": "latent_transformer",
        "encoder_kwargs": LATENT_A6,
        "label": "latent-m32-r16-nodrop-inputbn",
        "purpose": "Q3/A6 seed confirmation",
        "runs": [(43, "0p001", 0.001), (44, "0p001", 0.001)],
    },
}


def build(template, group, slug, seed, lr_tag, lr):
    cfg = copy.deepcopy(template)
    name = f"{slug}_lr{lr_tag}_seed{seed}"
    out_root = f"outputs/encoder_tuning/E12/{name}"
    label = group["label"]

    cfg["seed_everything"] = seed
    model_args = cfg["model"]["init_args"]
    model_args["lr"] = lr
    model_args["encoder_type"] = group["encoder_type"]
    model_args["encoder_kwargs"] = copy.deepcopy(group["encoder_kwargs"])

    dataset = "act" if group["template"] == "act" else "ng20"
    logger_args = cfg["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{label}_{dataset}_lr{lr}_seed{seed}"
    logger_args["project"] = group["project"]

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
            init_args["output_path"] = f"results/encoder_tuning/E12/{name}/runtime.csv"
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
    for slug, group in GROUPS.items():
        for seed, lr_tag, lr in group["runs"]:
            name, cfg = build(templates[group["template"]], group, slug, seed, lr_tag, lr)
            header = (
                "# Generated by configs/encoder_bench/sweep_e12/generate_e12_configs.py\n"
                f"# {group['purpose']}.\n"
                f"# Learning rate: {lr}; seed: {seed}. Data, augmentation, losses,\n"
                "# projection head, batch, schedule and callbacks are unchanged from the\n"
                "# corresponding dataset control.\n"
            )
            path = RUNS_DIR / f"{name}.yaml"
            path.write_text(
                header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            written.append(f"configs/encoder_bench/sweep_e12/runs/{name}.yaml")
            print(f"wrote {name}.yaml  [{group['template']}]")

    # The grid spans two W&B projects. The sweep's own project only decides
    # where the sweep object lives; each run config carries its own project, so
    # ACT runs land beside the ACT controls.
    sweep = (
        "# E12: complete the ResMLP normalization 2x2 (Q1), replicate the\n"
        "# mechanism on ACT against its existing 3-seed controls (Q2), and give\n"
        "# the simplified latent recipe its remaining seeds (Q3).\n"
        "# Run configs carry their own project, so ACT runs log to the ACT project.\n"
        "program: main.py\n"
        "method: grid\n"
        f"project: {NG20_PROJECT}\n"
        "name: encoder_bench_e12_combination_and_act\n"
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
