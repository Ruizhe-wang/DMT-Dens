"""Validate the three-seed strict leave-one-out ablation configs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
GENERATOR_PATH = HERE / "generate_configs.py"
SPEC = importlib.util.spec_from_file_location("single_factor_3seed_generator", GENERATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {GENERATOR_PATH}")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)

EXPECTED = {
    "full": ("rank", "bidirectional", True, 0.0018, "multi"),
    "r1_distance": ("distance", "bidirectional", True, 0.0018, "multi"),
    "r2_unidirectional": ("rank", "unidirectional", True, 0.0018, "multi"),
    "r3_allpair": ("rank", "bidirectional", False, 0.0018, "multi"),
    "r4_single_density": ("rank", "bidirectional", True, 0.0018, "single"),
    "r5_no_density": ("rank", "bidirectional", True, 0.0, "multi"),
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    paths = sorted(GENERATOR.RUN_DIR.glob("*.yaml"))
    expected_count = (
        len(GENERATOR.DATASETS) * len(GENERATOR.SEEDS) * len(GENERATOR.VARIANTS)
    )
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} configs, found {len(paths)}")

    for dataset in GENERATOR.DATASETS:
        for seed in GENERATOR.SEEDS:
            for variant in GENERATOR.VARIANTS:
                run_name, expected_cfg = GENERATOR.build(dataset, variant, seed)
                path = GENERATOR.RUN_DIR / f"{run_name}.yaml"
                actual_cfg = load(path)
                if actual_cfg != expected_cfg:
                    raise ValueError(f"Config differs from builder output: {path}")

                args = actual_cfg["model"]["init_args"]
                signature = (
                    args["manifold_affinity"],
                    args["manifold_symmetric"],
                    args["manifold_hardpair"],
                    args["density_weight"],
                    args["density_scale_mode"],
                )
                if signature != EXPECTED[variant]:
                    raise ValueError(f"Unexpected mechanism signature in {path}")
                if args.get("distance_p_row_normalize", False):
                    raise ValueError(f"P row normalization must be disabled in {path}")
                if actual_cfg["seed_everything"] != seed:
                    raise ValueError(f"Unexpected seed in {path}")
                if actual_cfg["trainer"]["max_epochs"] != 1000:
                    raise ValueError(f"Unexpected trainer epochs in {path}")

    print(f"Validated {expected_count} matched full/single-factor configs")


if __name__ == "__main__":
    main()
