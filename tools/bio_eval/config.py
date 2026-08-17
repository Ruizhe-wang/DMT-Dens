from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import yaml


@dataclass(frozen=True)
class BioEvalConfig:
    dataset: str
    adata_path: Path
    label_key: str
    fine_label_key: str | None = None
    pseudotime_key: str | None = None
    diffusion_potential_key: str | None = None
    branch_probs_key: str | None = None
    branch_names_key: str | None = None
    transition_labels: tuple[str, ...] = ()
    transition_key: str | None = None
    rare_labels: tuple[str, ...] = ()
    rare_key: str | None = None
    rare_frequency_threshold: float | None = None
    var_symbol_keys: tuple[str, ...] = ()
    marker_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    panels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigValidationReport:
    supported: dict[str, bool]
    warnings: list[str]


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _normalise_marker_sets(value: Any) -> dict[str, tuple[str, ...]]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise TypeError("marker_sets must be a mapping of marker set name to genes")
    return {str(name): _as_tuple(genes) for name, genes in value.items()}


def load_bio_eval_config(path: str | Path) -> BioEvalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "dataset" not in raw:
        raise ValueError(f"{config_path} is missing required key: dataset")
    if "adata_path" not in raw:
        raise ValueError(f"{config_path} is missing required key: adata_path")
    if "label_key" not in raw:
        raise ValueError(f"{config_path} is missing required key: label_key")

    adata_path = Path(str(raw["adata_path"]))
    if not adata_path.is_absolute():
        adata_path = (config_path.parent / adata_path).resolve()

    return BioEvalConfig(
        dataset=str(raw["dataset"]),
        adata_path=adata_path,
        label_key=str(raw["label_key"]),
        fine_label_key=raw.get("fine_label_key"),
        pseudotime_key=raw.get("pseudotime_key"),
        diffusion_potential_key=raw.get("diffusion_potential_key"),
        branch_probs_key=raw.get("branch_probs_key"),
        branch_names_key=raw.get("branch_names_key"),
        transition_labels=_as_tuple(raw.get("transition_labels")),
        transition_key=raw.get("transition_key"),
        rare_labels=_as_tuple(raw.get("rare_labels")),
        rare_key=raw.get("rare_key"),
        rare_frequency_threshold=(
            float(raw["rare_frequency_threshold"])
            if raw.get("rare_frequency_threshold") is not None
            else None
        ),
        var_symbol_keys=_as_tuple(raw.get("var_symbol_keys")),
        marker_sets=_normalise_marker_sets(raw.get("marker_sets")),
        panels=_as_tuple(raw.get("panels")),
    )


def validate_config_against_adata(config: BioEvalConfig) -> ConfigValidationReport:
    warnings: list[str] = []
    supported: dict[str, bool] = {}

    if not config.adata_path.exists():
        raise FileNotFoundError(config.adata_path)

    adata = ad.read_h5ad(config.adata_path, backed="r")
    try:
        obs_keys = set(adata.obs.columns)
        obsm_keys = set(adata.obsm.keys())
        uns_keys = set(adata.uns.keys())
        var_keys = set(adata.var.columns)

        supported["label"] = config.label_key in obs_keys
        if not supported["label"]:
            warnings.append(f"label_key {config.label_key!r} is missing from adata.obs")

        supported["fine_label"] = bool(config.fine_label_key and config.fine_label_key in obs_keys)
        if config.fine_label_key and not supported["fine_label"]:
            warnings.append(f"fine_label_key {config.fine_label_key!r} is missing from adata.obs")

        supported["pseudotime"] = bool(config.pseudotime_key and config.pseudotime_key in obs_keys)
        if config.pseudotime_key and not supported["pseudotime"]:
            warnings.append(f"pseudotime_key {config.pseudotime_key!r} is missing from adata.obs")

        supported["diffusion_potential"] = bool(
            config.diffusion_potential_key and config.diffusion_potential_key in obs_keys
        )
        if config.diffusion_potential_key and not supported["diffusion_potential"]:
            warnings.append(
                f"diffusion_potential_key {config.diffusion_potential_key!r} is missing from adata.obs"
            )

        supported["branch_probabilities"] = bool(
            config.branch_probs_key and config.branch_probs_key in obsm_keys
        )
        if config.branch_probs_key and not supported["branch_probabilities"]:
            warnings.append(f"branch_probs_key {config.branch_probs_key!r} is missing from adata.obsm")

        supported["branch_names"] = bool(config.branch_names_key and config.branch_names_key in uns_keys)
        if config.branch_names_key and not supported["branch_names"]:
            warnings.append(f"branch_names_key {config.branch_names_key!r} is missing from adata.uns")

        missing_symbol_keys = [key for key in config.var_symbol_keys if key not in var_keys]
        supported["var_symbol_keys"] = not missing_symbol_keys
        for key in missing_symbol_keys:
            warnings.append(f"var_symbol_key {key!r} is missing from adata.var")

        supported["marker_sets"] = bool(config.marker_sets)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

    return ConfigValidationReport(supported=supported, warnings=warnings)
