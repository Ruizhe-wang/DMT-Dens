from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np

from .config import BioEvalConfig
from .markers import score_marker_set


@dataclass
class BioSignals:
    categorical: dict[str, np.ndarray] = field(default_factory=dict)
    continuous: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cell_ids: np.ndarray | None = None


def build_bio_signals(config: BioEvalConfig) -> BioSignals:
    adata = ad.read_h5ad(config.adata_path)
    signals = BioSignals(cell_ids=adata.obs_names.astype(str).to_numpy())

    if config.label_key in adata.obs:
        signals.categorical["label"] = adata.obs[config.label_key].astype(str).to_numpy()
    else:
        signals.warnings.append(f"label_key {config.label_key!r} missing")

    if config.fine_label_key and config.fine_label_key in adata.obs:
        signals.categorical["fine_label"] = adata.obs[config.fine_label_key].astype(str).to_numpy()
    elif config.fine_label_key:
        signals.warnings.append(f"fine_label_key {config.fine_label_key!r} missing")

    if config.pseudotime_key and config.pseudotime_key in adata.obs:
        signals.continuous["pseudotime"] = _numeric_obs(adata, config.pseudotime_key)
    elif config.pseudotime_key:
        signals.warnings.append(f"pseudotime_key {config.pseudotime_key!r} missing")

    if config.diffusion_potential_key and config.diffusion_potential_key in adata.obs:
        signals.continuous["diffusion_potential"] = _numeric_obs(adata, config.diffusion_potential_key)
    elif config.diffusion_potential_key:
        signals.warnings.append(f"diffusion_potential_key {config.diffusion_potential_key!r} missing")

    _add_branch_probabilities(adata, config, signals)
    _add_transition_mask(adata, config, signals)
    _add_rare_mask(adata, config, signals)
    _add_marker_scores(adata, config, signals)
    return signals


def _numeric_obs(adata, key: str) -> np.ndarray:
    return np.asarray(adata.obs[key], dtype=np.float32)


def _add_branch_probabilities(adata, config: BioEvalConfig, signals: BioSignals) -> None:
    if not config.branch_probs_key:
        return
    if config.branch_probs_key not in adata.obsm:
        signals.warnings.append(f"branch_probs_key {config.branch_probs_key!r} missing")
        return

    probs = np.asarray(adata.obsm[config.branch_probs_key], dtype=np.float32)
    names = _branch_names(adata, config, probs.shape[1])
    for index, name in enumerate(names):
        signals.continuous[f"branch_prob_{name}"] = probs[:, index]
    signals.continuous["max_branch_prob"] = np.max(probs, axis=1)
    entropy = -np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)
    if probs.shape[1] > 1:
        entropy = entropy / np.log(float(probs.shape[1]))
    signals.continuous["branch_entropy"] = entropy.astype(np.float32)
    signals.categorical["terminal_state"] = np.asarray(names, dtype=object)[np.argmax(probs, axis=1)].astype(str)


def _branch_names(adata, config: BioEvalConfig, n_branches: int) -> list[str]:
    if config.branch_names_key and config.branch_names_key in adata.uns:
        names = [str(item) for item in np.asarray(adata.uns[config.branch_names_key]).tolist()]
        if len(names) == n_branches:
            return names
    return [f"branch_{index}" for index in range(n_branches)]


def _add_transition_mask(adata, config: BioEvalConfig, signals: BioSignals) -> None:
    key = config.transition_key or config.fine_label_key or config.label_key
    if config.transition_labels and key in adata.obs:
        values = adata.obs[key].astype(str)
        signals.masks["transition"] = values.isin(config.transition_labels).to_numpy(dtype=bool)
        return

    if "max_branch_prob" in signals.continuous:
        max_prob = signals.continuous["max_branch_prob"]
        entropy = signals.continuous.get("branch_entropy")
        mask = max_prob < 0.8
        if entropy is not None:
            mask = mask & (entropy >= np.quantile(entropy, 0.5))
        signals.masks["transition"] = mask.astype(bool)
        return

    if "pseudotime" in signals.continuous:
        pt = signals.continuous["pseudotime"]
        valid = np.isfinite(pt)
        if valid.any():
            lo, hi = np.quantile(pt[valid], [0.25, 0.75])
            signals.masks["transition"] = ((pt >= lo) & (pt <= hi) & valid).astype(bool)


def _add_rare_mask(adata, config: BioEvalConfig, signals: BioSignals) -> None:
    key = config.rare_key or config.fine_label_key or config.label_key
    if config.rare_labels and key in adata.obs:
        values = adata.obs[key].astype(str)
        signals.masks["rare"] = values.isin(config.rare_labels).to_numpy(dtype=bool)
        return

    threshold = config.rare_frequency_threshold
    if threshold is None or key not in adata.obs:
        return
    values = adata.obs[key].astype(str)
    frequencies = values.value_counts(normalize=True)
    rare_labels = set(frequencies[frequencies < threshold].index.astype(str))
    signals.masks["rare"] = values.isin(rare_labels).to_numpy(dtype=bool)


def _add_marker_scores(adata, config: BioEvalConfig, signals: BioSignals) -> None:
    for name, genes in config.marker_sets.items():
        score = score_marker_set(adata, genes, symbol_keys=config.var_symbol_keys)
        if score is None:
            signals.warnings.append(f"marker set {name!r} has no genes present in AnnData")
            continue
        signals.continuous[f"marker_{name}"] = score
