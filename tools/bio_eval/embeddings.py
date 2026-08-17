from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodEmbedding:
    method: str
    cell_ids: np.ndarray
    coords: np.ndarray
    layer: int | None = None
    labels: np.ndarray | None = None
    source_path: Path | None = None


def load_embedding(path: str | Path, method: str | None = None, layer: int | None = None) -> MethodEmbedding:
    embedding_path = Path(path)
    suffix = embedding_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_embedding(embedding_path, method=method, layer=layer)
    if suffix == ".npz":
        return _load_npz_embedding(embedding_path, method=method, layer=layer)
    raise ValueError(f"Unsupported embedding format for {embedding_path}; expected .csv or .npz")


def _load_csv_embedding(path: Path, method: str | None, layer: int | None) -> MethodEmbedding:
    frame = pd.read_csv(path)
    required = {"cell_id", "x", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    work = frame.copy()
    if method is not None and "method" in work.columns:
        work = work[work["method"].astype(str) == str(method)].copy()
    resolved_method = _resolve_method(method, work, path)

    if "layer" in work.columns:
        resolved_layer = int(layer) if layer is not None else int(work["layer"].max())
        work = work[work["layer"].astype(int) == resolved_layer].copy()
    else:
        resolved_layer = layer

    if work.empty:
        raise ValueError(f"No embedding rows remain after filtering {path}")
    _check_duplicate_cell_ids(work["cell_id"].astype(str).to_numpy(), path)

    labels = work["label"].astype(str).to_numpy() if "label" in work.columns else None
    return MethodEmbedding(
        method=resolved_method,
        cell_ids=work["cell_id"].astype(str).to_numpy(),
        coords=work[["x", "y"]].to_numpy(dtype=np.float32),
        layer=resolved_layer,
        labels=labels,
        source_path=path,
    )


def _load_npz_embedding(path: Path, method: str | None, layer: int | None) -> MethodEmbedding:
    data = np.load(path, allow_pickle=True)
    if "cell_ids" not in data:
        raise ValueError(f"{path} is missing cell_ids")
    cell_ids = np.asarray(data["cell_ids"], dtype=str)
    _check_duplicate_cell_ids(cell_ids, path)

    layer_keys = sorted(
        (key for key in data.files if key.startswith("layer_")),
        key=lambda item: int(item.split("_", 1)[1]),
    )
    if not layer_keys:
        raise ValueError(f"{path} has no layer_N arrays")
    resolved_layer = int(layer) if layer is not None else int(layer_keys[-1].split("_", 1)[1])
    layer_key = f"layer_{resolved_layer}"
    if layer_key not in data:
        raise ValueError(f"{path} does not contain {layer_key}")

    if method is not None:
        resolved_method = str(method)
    elif "method" in data:
        resolved_method = str(np.asarray(data["method"]).item())
    else:
        resolved_method = path.stem.replace("_embeddings", "")

    labels = np.asarray(data["labels"], dtype=str) if "labels" in data else None
    return MethodEmbedding(
        method=resolved_method,
        cell_ids=cell_ids,
        coords=np.asarray(data[layer_key], dtype=np.float32),
        layer=resolved_layer,
        labels=labels,
        source_path=path,
    )


def align_embedding_to_adata(
    adata: ad.AnnData,
    embedding: MethodEmbedding,
    *,
    require_all: bool = True,
) -> MethodEmbedding:
    adata_ids = [str(item) for item in adata.obs_names]
    order_map = {cell_id: index for index, cell_id in enumerate(embedding.cell_ids.astype(str))}
    missing = [cell_id for cell_id in adata_ids if cell_id not in order_map]
    if missing and require_all:
        raise ValueError(
            f"Embedding {embedding.method!r} is missing {len(missing)} AnnData cells; "
            f"examples: {missing[:5]}"
        )

    aligned_ids: list[str] = []
    aligned_coords: list[np.ndarray] = []
    aligned_labels: list[str] = []
    for cell_id in adata_ids:
        if cell_id not in order_map:
            continue
        idx = order_map[cell_id]
        aligned_ids.append(cell_id)
        aligned_coords.append(embedding.coords[idx])
        if embedding.labels is not None:
            aligned_labels.append(str(embedding.labels[idx]))

    if not aligned_ids:
        raise ValueError(f"Embedding {embedding.method!r} has no cell ids in common with AnnData")

    labels = np.asarray(aligned_labels, dtype=str) if embedding.labels is not None else None
    return MethodEmbedding(
        method=embedding.method,
        cell_ids=np.asarray(aligned_ids, dtype=str),
        coords=np.asarray(aligned_coords, dtype=np.float32),
        layer=embedding.layer,
        labels=labels,
        source_path=embedding.source_path,
    )


def _resolve_method(method: str | None, frame: pd.DataFrame, path: Path) -> str:
    if method is not None:
        return str(method)
    if "method" in frame.columns and not frame["method"].dropna().empty:
        methods = frame["method"].astype(str).unique()
        if len(methods) == 1:
            return methods[0]
    return path.stem.replace("_embeddings", "")


def _check_duplicate_cell_ids(cell_ids: np.ndarray, path: Path) -> None:
    series = pd.Series(cell_ids.astype(str))
    if series.duplicated().any():
        examples = series[series.duplicated()].head().tolist()
        raise ValueError(f"{path} contains duplicate cell ids: {examples}")
