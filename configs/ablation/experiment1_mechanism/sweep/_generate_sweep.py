"""Generate the wandb sweep YAML for Experiment 1 (mechanism ablation).

Grid over all 12 variants x 9 datasets (108 configs) x 5 seeds = 540 runs.
Per-variant hyperparameters (manifold_*, density_*) live in each config, so the
sweep only varies `config` and `seed_everything`.

Run:  python configs/ablation/experiment1_mechanism/sweep/_generate_sweep.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT_DIR = HERE.parent
REL = "configs/ablation/experiment1_mechanism"

DATASETS = ["act", "aqc", "emnist", "epi", "gast10k", "hcl", "mca", "mnist", "ng20", "tree"]
VARIANTS = [
    "full",
    "r1_distance",
    "r2_unidirectional",
    "r3_allpair",
    "r4_single_density",
    "r5_no_density",
    "base",
    "b1_rank",
    "b2_bidirectional",
    "b3_hardpair",
    "b4_single_density",
    "b5_multi_density",
]

lines = [
    "program: main.py",
    "method: grid",
    "project: DiffTree_rz",
    "name: experiment1_mechanism_12variant_10dataset_5seed",
    "",
    "parameters:",
    "  config:",
    "    values:",
]
for ds in DATASETS:
    lines.append(f"    # --- {ds} ---")
    for v in VARIANTS:
        cfg = VARIANT_DIR / f"{ds}_{v}.yaml"
        assert cfg.exists(), f"missing {cfg}"
        lines.append(f"      - {REL}/{ds}_{v}.yaml")
lines += [
    "  seed_everything:",
    "    values: [42, 43, 44, 45, 46]",
    "  model.init_args.max_epochs:",
    "    values: [1000]",
    "  trainer.max_epochs:",
    "    values: [1000]",
    "  # num_workers=0 loads data in the main process (no os.fork), which fully",
    "  # avoids the fork OOM (Errno 12) seen with parallel sweep agents.",
    "  data.init_args.num_workers:",
    "    values: [0]",
    "",
    "command:",
    "  - ${env}",
    "  - ${interpreter}",
    "  - ${program}",
    "  - fit",
    "  - ${args}",
    "",
]

out = HERE / "experiment1_mechanism_sweep.yaml"
out.write_text("\n".join(lines), encoding="utf-8")
n = len(DATASETS) * len(VARIANTS)
print(f"Wrote {out} with {n} configs x 5 seeds = {n * 5} runs")
