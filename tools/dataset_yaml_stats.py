"""Summarize dataset sizes from experiment YAML files.

The script reads YAML configs, resolves each ``data.init_args.data_path``, and
uses the lightweight file formats referenced by the local DataModule code to
report sample count, feature count, and class count without building dataloaders.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


@dataclass
class DatasetSummary:
    dataset: str
    yaml_path: Path
    class_path: str | None
    configured_data_path: str | None
    resolved_data_path: Path | None
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    label_source: str | None = None
    preprocessing: str | None = None
    data_files: str | None = None
    status: str = "ok"
    error: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_path_maps(values: Iterable[str]) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-map must be FROM=TO, got {value!r}")
        src, dst = value.split("=", 1)
        maps.append((src.rstrip("/\\"), dst.rstrip("/\\")))
    return maps


def resolve_data_path(configured: str | None, path_maps: list[tuple[str, str]] | None = None) -> Path | None:
    if not configured:
        return None
    candidate = configured
    for src, dst in path_maps or []:
        src_norm = src.replace("\\", "/").rstrip("/")
        cand_norm = candidate.replace("\\", "/")
        if cand_norm == src_norm or cand_norm.startswith(src_norm + "/"):
            candidate = dst + cand_norm[len(src_norm) :]
            break
    return Path(candidate).expanduser()


def count_classes(labels: np.ndarray | pd.Series | list[Any]) -> int:
    arr = np.asarray(labels)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return 0
    return int(pd.Series(arr).dropna().astype(str).nunique())


def flattened_feature_count(shape: tuple[int, ...], pad_to_even: bool = False) -> int:
    n_features = int(math.prod(shape[1:])) if len(shape) > 1 else 1
    if pad_to_even and n_features % 2:
        n_features += 1
    return n_features


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def summarize_npy_pair(
    root: Path,
    data_rel: str,
    label_rel: str,
    *,
    pad_to_even: bool = False,
    mca_filter: bool = False,
) -> tuple[int, int, int, str, str]:
    data_path = require_file(root / data_rel)
    label_path = require_file(root / label_rel)
    data = np.load(data_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")

    if mca_filter:
        selected_genes = np.asarray(data.max(axis=0) > 4)
        label_arr = np.asarray(labels)
        counts = pd.Series(label_arr).value_counts()
        keep_labels = set(counts[counts >= 500].index.tolist())
        mask = np.array([label in keep_labels for label in label_arr])
        n_samples = int(mask.sum())
        n_features = int(selected_genes.sum())
        if n_features % 2:
            n_features += 1
        n_classes = len(keep_labels)
        preprocessing = "genes with max expression > 4; remove classes with <500 cells; pad one zero feature if odd"
    else:
        n_samples = int(data.shape[0])
        n_features = flattened_feature_count(tuple(data.shape), pad_to_even=pad_to_even)
        n_classes = count_classes(labels)
        preprocessing = "flatten to 2D"
        if pad_to_even:
            preprocessing += "; pad one zero feature if feature count is odd"

    files = f"{data_rel}; {label_rel}"
    return n_samples, n_features, n_classes, str(label_path), files, preprocessing


def summarize_activity(root: Path) -> tuple[int, int, int, str, str, str]:
    train_path = require_file(root / "feature_select" / "Activity_train.csv")
    test_path = require_file(root / "feature_select" / "Activity_test.csv")
    all_data = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    feature_cols = [col for col in all_data.columns if col not in {"subject", "Activity"}]
    preprocessing = "concatenate train/test CSV; drop subject and Activity; min-max scale features"
    return (
        int(len(all_data)),
        int(len(feature_cols)),
        count_classes(all_data["Activity"]),
        "Activity column",
        f"{train_path.name}; {test_path.name}",
        preprocessing,
    )


def h5ad_shape(path: Path) -> tuple[int, int]:
    import h5py

    with h5py.File(path, "r") as h5:
        x = h5["X"]
        if "shape" in x.attrs:
            shape = tuple(int(v) for v in x.attrs["shape"])
        else:
            shape = tuple(int(v) for v in x.shape)
    if len(shape) != 2:
        raise ValueError(f"Unsupported h5ad X shape in {path}: {shape}")
    return shape


def decode_h5_values(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def read_h5ad_obs(path: Path, key: str) -> list[str]:
    import h5py

    with h5py.File(path, "r") as h5:
        obs = h5["obs"]
        if key not in obs:
            raise KeyError(key)
        node = obs[key]
        if isinstance(node, h5py.Dataset):
            return decode_h5_values(node[()])
        if "codes" in node and "categories" in node:
            codes = np.asarray(node["codes"][()])
            categories = decode_h5_values(node["categories"][()])
            return [categories[int(code)] if int(code) >= 0 else "" for code in codes]
    raise KeyError(key)


def read_h5ad_obs_first(path: Path, keys: Iterable[str]) -> tuple[str, list[str]]:
    for key in keys:
        try:
            return key, read_h5ad_obs(path, key)
        except KeyError:
            continue
    raise KeyError(f"none of {list(keys)} found in {path}")


def summarize_hcl(root: Path) -> tuple[int, int, int, str, str, str]:
    path = require_file(root / "HCL60kafter-elis-all.h5ad")
    n_obs, n_vars = h5ad_shape(path)
    labels = np.asarray(read_h5ad_obs(path, "louvain"))
    counts = pd.Series(labels).value_counts()
    keep = set(counts[counts >= 500].index.tolist())
    n_samples = int(sum(label in keep for label in labels))
    n_features = n_vars + (n_vars % 2)
    preprocessing = "MinMax scale; remove louvain classes with <500 cells; pad one zero feature if odd"
    return n_samples, n_features, len(keep), "obs['louvain']", path.name, preprocessing


def summarize_gast(root: Path) -> tuple[int, int, int, str, str, str]:
    path = require_file(root / "gast10kwithcelltype.h5ad")
    n_obs, n_vars = h5ad_shape(path)
    key, labels = read_h5ad_obs_first(path, ["celltype", "cell_type"])
    n_features = n_vars + (n_vars % 2)
    preprocessing = "MinMax scale; encode celltype labels; pad one zero feature if odd"
    return n_obs, n_features, count_classes(labels), f"obs[{key!r}]", path.name, preprocessing


def summarize_mouse_prenatal(root: Path, init_args: dict[str, Any]) -> tuple[int, int, int | None, str, str, str]:
    pattern = str(init_args.get("h5ad_pattern") or "adata_JAX_dataset_*.h5ad")
    files = sorted(Path(p) for p in glob.glob(str(root / pattern)))
    if not files:
        files = sorted(root.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad files match {root / pattern}")

    shapes = [h5ad_shape(path) for path in files]
    total_obs = sum(shape[0] for shape in shapes)
    sample_size = init_args.get("sample_data_size")
    n_samples = min(total_obs, int(sample_size)) if sample_size is not None else total_obs
    top_genes = init_args.get("top_genes")
    n_features = min(int(top_genes), shapes[0][1]) if top_genes is not None else shapes[0][1]

    metadata_file = init_args.get("metadata_file", "df_cell.csv")
    label_key = init_args.get("label_key", "celltype_update")
    meta_path = root / metadata_file
    n_classes: int | None = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path, usecols=lambda col: col == label_key)
        if label_key in meta:
            n_classes = count_classes(meta[label_key])

    preprocessing = "sample cells; select top variable genes; normalize_total; log1p"
    return n_samples, n_features, n_classes, f"{metadata_file}:{label_key}", "; ".join(p.name for p in files), preprocessing


def summarize_torchvision(name: str, root: Path, class_path: str) -> tuple[int, int, int, str, str, str]:
    lower = class_path.lower()
    if "emnist" in lower:
        return 697932, 784, 62, "EMNIST byclass train labels", str(root), "torchvision EMNIST byclass train split; flatten 28x28 images"
    if "mnist" in lower:
        return 60000, 784, 10, "MNIST train labels", str(root), "torchvision MNIST train split; flatten 28x28 images"
    if "cifar100" in name.lower() or "cifar100" in lower:
        return 50000, 3072, 100, "CIFAR-100 train labels", str(root), "torchvision CIFAR-100 train split; flatten 32x32x3 images"
    if "cifar" in lower:
        return 50000, 3072, 10, "CIFAR-10 train labels", str(root), "torchvision CIFAR-10 train split; flatten 32x32x3 images"
    raise ValueError(f"Unsupported torchvision dataset for {class_path}")


def summarize_by_class_path(
    dataset: str,
    class_path: str | None,
    root: Path | None,
    init_args: dict[str, Any],
) -> tuple[int, int, int | None, str, str, str]:
    if root is None:
        raise ValueError("data.init_args.data_path is missing")
    key = (class_path or "").lower()
    if "m1datamodel_activity" in key:
        return summarize_activity(root)
    if "m1datamodel_aqc" in key:
        return summarize_npy_pair(root, "dmthi/aqc_all_data_3000.npy", "dmthi/aqc_all_label.npy", pad_to_even=True)
    if "m1datamodel_epi" in key:
        return summarize_npy_pair(root, "difftreedata/data/EpitheliaCell_data_n.npy", "difftreedata/data/EpitheliaCell_label.npy", pad_to_even=True)
    if "m1datamodel_mca" in key:
        return summarize_npy_pair(root, "mca_data/mca_data_dim_34947.npy", "mca_data/mca_label_dim_34947.npy", mca_filter=True)
    if "m1datamodel_20ng" in key:
        return summarize_npy_pair(root, "20NG.npy", "20NG_labels.npy")
    if "m1datamodel_hcl" in key:
        return summarize_hcl(root)
    if "m1datamodel_gast" in key:
        return summarize_gast(root)
    if "m1datamodel_mouse_prenatal" in key:
        return summarize_mouse_prenatal(root, init_args)
    if any(token in key for token in ("mnist", "emnist", "cifar")):
        return summarize_torchvision(dataset, root, class_path or "")
    raise ValueError(f"No lightweight reader registered for class_path={class_path!r}")


def summarize_yaml_file(
    yaml_path: str | Path,
    path_maps: list[tuple[str, str]] | None = None,
) -> DatasetSummary:
    yaml_path = Path(yaml_path)
    cfg = load_yaml(yaml_path)
    data_cfg = cfg.get("data") or {}
    model_cfg = (cfg.get("model") or {}).get("init_args") or {}
    init_args = data_cfg.get("init_args") or {}
    class_path = data_cfg.get("class_path")
    configured_path = init_args.get("data_path")
    resolved_path = resolve_data_path(configured_path, path_maps)
    summary = DatasetSummary(
        dataset=yaml_path.stem,
        yaml_path=yaml_path,
        class_path=class_path,
        configured_data_path=configured_path,
        resolved_data_path=resolved_path,
    )

    try:
        n_samples, n_features, n_classes, label_source, files, preprocessing = summarize_by_class_path(
            yaml_path.stem,
            class_path,
            resolved_path,
            init_args,
        )
        summary.n_samples = n_samples
        summary.n_features = n_features
        summary.n_classes = n_classes
        summary.label_source = label_source
        summary.preprocessing = preprocessing
        summary.data_files = files
    except Exception as exc:
        summary.status = "error"
        summary.error = str(exc)
        if "num_train_data" in model_cfg:
            summary.n_samples = int(model_cfg["num_train_data"])
        if "num_input_dim" in model_cfg:
            summary.n_features = int(model_cfg["num_input_dim"])
    return summary


def iter_yaml_files(config_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.yaml" if recursive else "*.yaml"
    return sorted(path for path in config_dir.glob(pattern) if path.is_file())


def format_markdown(rows: list[DatasetSummary]) -> str:
    headers = ["dataset", "samples", "features", "classes", "data_path", "label_source", "preprocessing", "status"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row.dataset,
            "" if row.n_samples is None else str(row.n_samples),
            "" if row.n_features is None else str(row.n_features),
            "" if row.n_classes is None else str(row.n_classes),
            "" if row.resolved_data_path is None else str(row.resolved_data_path),
            row.label_source or "",
            row.preprocessing or "",
            row.status if row.status == "ok" else f"error: {row.error}",
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", type=Path, help="Directory containing YAML config files.")
    parser.add_argument("--path-map", action="append", default=[], help="Map YAML paths, e.g. /zangzelin/data=/mnt/data.")
    parser.add_argument("--recursive", action="store_true", help="Also read YAML files in child directories.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a Markdown table.")
    args = parser.parse_args()

    path_maps = parse_path_maps(args.path_map)
    rows = [summarize_yaml_file(path, path_maps) for path in iter_yaml_files(args.config_dir, args.recursive)]
    if args.json:
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2, default=str))
    else:
        print(format_markdown(rows))
    return 0 if all(row.status == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
