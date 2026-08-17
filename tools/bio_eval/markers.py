from __future__ import annotations

import numpy as np
import pandas as pd


def find_gene_indices(adata, genes: list[str] | tuple[str, ...], symbol_keys: tuple[str, ...] = ()) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, name in enumerate(adata.var_names.astype(str)):
        lookup.setdefault(name.lower(), index)

    for key in symbol_keys:
        if key not in adata.var:
            continue
        values = pd.Series(adata.var[key]).astype(str)
        for index, name in enumerate(values):
            if name and name.lower() != "nan":
                lookup.setdefault(name.lower(), index)

    hits: dict[str, int] = {}
    for gene in genes:
        idx = lookup.get(str(gene).lower())
        if idx is not None:
            hits[str(gene)] = idx
    return hits


def score_marker_set(adata, genes: list[str] | tuple[str, ...], symbol_keys: tuple[str, ...] = ()) -> np.ndarray | None:
    hits = find_gene_indices(adata, genes, symbol_keys=symbol_keys)
    if not hits:
        return None
    indices = list(hits.values())
    values = adata.X[:, indices]
    if hasattr(values, "toarray"):
        values = values.toarray()
    matrix = np.asarray(values, dtype=np.float32)
    return np.mean(np.log1p(np.maximum(matrix, 0.0)), axis=1).astype(np.float32)
