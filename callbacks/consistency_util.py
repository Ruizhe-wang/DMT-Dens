"""
Consistency Utility Classes

Contains dataset utilities for training fresh models on subset data,
replicating the exact same pipeline as the main training.
"""

import copy
import inspect
from types import MethodType

import anndata
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from pynndescent import NNDescent

from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


class MicroTrainingDataset(Dataset):
    """
    Legacy dataset for micro-training (noise-based augmentation).
    Kept for backward compatibility. Prefer create_subset_dataloader() instead.
    """
    def __init__(self, data: np.ndarray, add_noise: bool = True, noise_scale: float = 0.01):
        self.data = torch.from_numpy(data).float()
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.global_std = self.data.std()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        if self.add_noise:
            noise = torch.randn_like(data_item) * self.noise_scale * self.global_std
            data_aug = data_item + noise
        else:
            data_aug = data_item.clone()

        return {
            'data_input_item': data_item,
            'data_input_aug': data_aug,
            'index': idx,
        }


def compute_subset_neighbors(data: np.ndarray, k: int = 200, pca_dim: int = 64) -> np.ndarray:
    """
    Compute k-NN graph for a data subset, replicating the exact same
    pipeline as cal_near_index in M1datamodel_mnist_v2.py.

    Args:
        data: Subset data array, shape (n_samples, n_features)
        k: Number of nearest neighbors
        pca_dim: PCA dimensionality for neighbor computation

    Returns:
        neighbors_index: Array of shape (n_samples, k) with neighbor indices
    """
    x_reshaped = data.reshape((data.shape[0], -1))

    if pca_dim < x_reshaped.shape[1]:
        x_reshaped = PCA(n_components=pca_dim).fit_transform(x_reshaped)

    k_safe = min(k, x_reshaped.shape[0] - 1)
    index = NNDescent(x_reshaped, n_jobs=-1)
    neighbors_index, _ = index.query(x_reshaped, k=k_safe + 1)
    neighbors_index = neighbors_index[:, 1:]  # Remove self-loop

    return neighbors_index


def create_subset_dataloader(
    subset_data: np.ndarray,
    subset_labels: np.ndarray,
    k: int = 200,
    pca_dim: int = 64,
    n_hop: int = 1,
    batch_size: int = 5000,
) -> DataLoader:
    """
    Create a DataLoader for subset data that exactly replicates the main
    training pipeline: NNDescent k-NN, TraceMixup augmentation,
    DataSetBaseMultiLevel dataset.

    This mirrors the setup() method in M1datamodel_mnist_v2.DMTBaseDataModule.

    Args:
        subset_data: High-dimensional data for the subset
        subset_labels: Labels for the subset
        k: Number of nearest neighbors (matches DataModule.K)
        pca_dim: PCA dim for neighbor computation (matches DataModule.pca_dim)
        n_hop: Number of hops for TraceMixup (matches DataModule.n_hop)
        batch_size: Batch size for training

    Returns:
        DataLoader with identical structure to the main training pipeline
    """
    neighbors_index = compute_subset_neighbors(subset_data, k=k, pca_dim=pca_dim)

    k_actual = neighbors_index.shape[1]
    augmenter = Augmenter([{
        "name": "TraceMixup",
        "alpha_max": 1.0,
        "p": 1.0,
        "n_hop": n_hop,
        "knn": k_actual,
    }])

    dataset = DataSetBaseMultiLevel(
        data=subset_data,
        label=subset_labels,
        neighbors_index=neighbors_index,
        transform=augmenter,
    )

    dataloader = DataLoader(
        dataset,
        drop_last=True,
        shuffle=True,
        batch_size=min(batch_size, len(subset_data)),
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return dataloader


def extract_datamodule_init_args(datamodule) -> dict:
    """
    Recover constructor arguments from an instantiated datamodule.

    This lets micro-training reuse the original datamodule class instead of
    reimplementing its pipeline in callback code.
    """
    init_args = {}
    signature = inspect.signature(type(datamodule).__init__)
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if hasattr(datamodule, name):
            init_args[name] = copy.deepcopy(getattr(datamodule, name))
    return init_args


def _build_subset_adata(
    subset_data: np.ndarray,
    subset_labels: np.ndarray,
    original_adata=None,
    subset_mask: np.ndarray = None,
    label_key: str = "cell_type",
):
    """Create a subset AnnData while preserving original metadata when possible."""
    if original_adata is not None and subset_mask is not None and len(original_adata) == len(subset_mask):
        return original_adata[subset_mask].copy()

    adata = anndata.AnnData(X=subset_data.copy())
    label_str = subset_labels.astype(str)
    adata.obs["batch"] = label_str
    adata.obs["final_annotation"] = label_str
    adata.obs[label_key] = label_str
    return adata


def create_subset_datamodule(
    datamodule_class,
    datamodule_init_args: dict,
    subset_data: np.ndarray,
    subset_labels: np.ndarray,
    subset_mask: np.ndarray = None,
    original_adata=None,
    label_key: str = "cell_type",
):
    """
    Instantiate the original datamodule class and override only its data source.

    This keeps `setup()`, `train_dataloader()`, `val_dataloader()`, neighbor
    graph construction, augmentation, and dataloader settings aligned with the
    original training code path.
    """
    subset_data = np.asarray(subset_data)
    subset_labels = np.asarray(subset_labels)
    init_args = copy.deepcopy(datamodule_init_args or {})
    subset_adata = _build_subset_adata(
        subset_data=subset_data,
        subset_labels=subset_labels,
        original_adata=original_adata,
        subset_mask=subset_mask,
        label_key=label_key,
    )

    subset_dm = datamodule_class(**init_args)

    def _load_subset_data(self, data_path):
        self.adata = subset_adata.copy()
        if not hasattr(self, "info_list"):
            self.info_list = ["batch", "final_annotation"]
        return self.adata, subset_data.copy(), subset_labels.copy()

    subset_dm.load_data = MethodType(_load_subset_data, subset_dm)
    return subset_dm
