from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .embeddings import MethodEmbedding
from .signals import BioSignals


def compute_metrics(embedding: MethodEmbedding, signals: BioSignals, n_neighbors: int = 15) -> dict[str, float | str]:
    n_cells = embedding.coords.shape[0]
    k = max(1, min(n_neighbors, n_cells - 1))
    result: dict[str, float | str] = {
        "method": embedding.method,
        "n_cells": float(n_cells),
        "layer": float(embedding.layer) if embedding.layer is not None else np.nan,
    }
    if n_cells < 2:
        return result

    indices = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(embedding.coords)
        .kneighbors(embedding.coords, return_distance=False)[:, 1:]
    )

    if "transition" in signals.masks:
        result["transition_neighbor_fraction"] = _mask_neighbor_fraction(signals.masks["transition"], indices)
        result["transition_cell_count"] = float(np.asarray(signals.masks["transition"], dtype=bool).sum())
    if "rare" in signals.masks:
        result["rare_neighbor_fraction"] = _mask_neighbor_fraction(signals.masks["rare"], indices)
        result["rare_cell_count"] = float(np.asarray(signals.masks["rare"], dtype=bool).sum())

    branch_keys = [key for key in signals.continuous if key.startswith("branch_prob_")]
    if branch_keys:
        branch_matrix = np.column_stack([signals.continuous[key] for key in branch_keys])
        result["fate_neighbor_std"] = float(np.mean([np.mean(np.std(branch_matrix[row], axis=0)) for row in indices]))

    marker_keys = [key for key in signals.continuous if key.startswith("marker_")]
    if marker_keys:
        marker_matrix = np.column_stack([signals.continuous[key] for key in marker_keys])
        result["marker_neighbor_std"] = float(np.mean([np.mean(np.std(marker_matrix[row], axis=0)) for row in indices]))

    return result


def write_metrics(rows: list[dict[str, float | str]], output_path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def _mask_neighbor_fraction(mask: np.ndarray, indices: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    selected_rows = np.where(selected)[0]
    return float(np.mean(selected[indices[selected_rows]]))
