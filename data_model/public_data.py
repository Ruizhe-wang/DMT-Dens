"""Path-neutral data module for public DMT-Dens examples and user datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata
import joblib
import lightning as pl
import numpy as np
import scipy.sparse as sp
import torch
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


class PublicDataModule(pl.LightningDataModule):
    """Load a processed observation matrix from NPZ or H5AD.

    NPZ inputs must contain an observations-by-features array named ``X`` by
    default and may contain labels named ``y``. H5AD labels are read from the
    requested ``label_key`` in ``adata.obs``. Labels are retained for plotting
    and evaluation but are not used to construct neighbors unless ``uselabel``
    is explicitly enabled.
    """

    def __init__(
        self,
        data_path: str,
        data_name: str = "public_data",
        x_key: str = "X",
        label_key: str = "y",
        batch_size: int = 256,
        num_workers: int = 0,
        K: int = 20,
        pca_dim: int = 64,
        n_hop: int = 1,
        seed: int = 42,
        sample_data_size: int | None = None,
        num_positive_samples: int = 1,
        standardize: bool = True,
        uselabel: bool = False,
        use_cache: bool = True,
        cache_dir: str = ".cache/dmt_dens",
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.data_name = data_name
        self.x_key = x_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.K = K
        self.pca_dim = pca_dim
        self.n_hop = n_hop
        self.seed = seed
        self.sample_data_size = sample_data_size
        self.num_positive_samples = num_positive_samples
        self.standardize = standardize
        self.uselabel = uselabel
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.dataset_kwargs = kwargs

    def _load_npz(self, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            if self.x_key not in archive:
                raise KeyError(f"{path} does not contain the feature key {self.x_key!r}")
            data = archive[self.x_key]
            label = archive[self.label_key] if self.label_key in archive else None
        return data, label, None

    def _load_h5ad(self, path: Path):
        adata = anndata.read_h5ad(path)
        data = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        label = adata.obs[self.label_key].to_numpy() if self.label_key in adata.obs else None
        return data, label, adata

    def load_data(self):
        path = Path(self.data_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Generate the toy data first or "
                "override --data.init_args.data_path."
            )

        if path.suffix.lower() == ".npz":
            data, label, adata = self._load_npz(path)
        elif path.suffix.lower() == ".h5ad":
            data, label, adata = self._load_h5ad(path)
        else:
            raise ValueError(f"Unsupported dataset format {path.suffix!r}; use .npz or .h5ad")

        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, received shape {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("Input matrix contains NaN or infinite values")

        if label is None:
            label = np.zeros(data.shape[0], dtype=np.int64)
        label = np.asarray(label).reshape(-1)
        if label.shape[0] != data.shape[0]:
            raise ValueError(
                f"Feature/label row mismatch: X has {data.shape[0]} rows, "
                f"labels have {label.shape[0]}"
            )

        if self.sample_data_size is not None and data.shape[0] > self.sample_data_size:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(data.shape[0], self.sample_data_size, replace=False))
            data = data[keep]
            label = label[keep]
            adata = adata[keep].copy() if adata is not None else None

        if self.standardize:
            mean = data.mean(axis=0, keepdims=True)
            scale = data.std(axis=0, keepdims=True)
            data = (data - mean) / np.maximum(scale, 1e-8)

        if adata is None:
            adata = anndata.AnnData(X=data)
        display_label = label.astype(str)
        adata.obs["final_annotation"] = display_label
        adata.obs["cell_type"] = display_label
        adata.obs["batch"] = display_label
        self.info_list = ["final_annotation", "cell_type", "batch"]
        return adata, data, label

    def _cache_path(self, data_shape):
        signature = hashlib.sha256(
            f"{Path(self.data_path).resolve()}|{data_shape}|{self.K}|{self.pca_dim}|{self.seed}".encode()
        ).hexdigest()[:16]
        return Path(self.cache_dir) / f"neighbors_{signature}.pkl"

    def _neighbors(self, data, label):
        if data.shape[0] < 2:
            raise ValueError("DMT-Dens requires at least two observations")
        k = min(self.K, data.shape[0] - 1)
        cache_path = self._cache_path(data.shape)
        if self.use_cache and cache_path.is_file():
            return joblib.load(cache_path)

        search_data = data
        pca_dim = min(self.pca_dim, data.shape[0] - 1, data.shape[1])
        if 0 < pca_dim < data.shape[1]:
            search_data = PCA(n_components=pca_dim, random_state=self.seed).fit_transform(data)

        if self.uselabel:
            raise ValueError(
                "uselabel=True is disabled in the public loader because labels "
                "must not influence representation training."
            )

        index = NNDescent(search_data, n_jobs=-1, random_state=self.seed)
        neighbors, _ = index.query(search_data, k=k + 1)
        neighbors = neighbors[:, 1:]
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(neighbors, cache_path)
        return neighbors

    def setup(self, stage=None):
        adata, data, label = self.load_data()
        self.adata = adata
        augmenter = Augmenter(
            [{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}]
        )
        self.dataset = DataSetBaseMultiLevel(
            data=data,
            label=label,
            neighbors_index=self._neighbors(data, label),
            transform=augmenter,
        )

    def _loader(self, shuffle, drop_last):
        return DataLoader(
            self.dataset,
            batch_size=min(self.batch_size, len(self.dataset)),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._loader(shuffle=False, drop_last=False)

    def predict_dataloader(self):
        return self._loader(shuffle=False, drop_last=False)
