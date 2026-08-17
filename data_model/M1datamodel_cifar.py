import lightning as pl
from torch.utils.data import DataLoader
import joblib
import os
import torch
import numpy as np
from sklearn.decomposition import PCA

import scipy.sparse as sp

from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import anndata

from data_model.aug import Augmenter
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


def extract_cifar_data(dataset, flatten=False):
    """
    Extract CIFAR dataset images and labels.

    Args:
        dataset: CIFAR dataset
        flatten: If True, flatten images to vectors (3072 dims).
                 If False, keep original shape (3, 32, 32) for CNN/ResNet.

    Returns:
        data: numpy array of shape (N, 3072) if flatten else (N, 3, 32, 32)
        labels: numpy array of labels
    """
    data_list, label_list = [], []
    for img, label in dataset:
        if flatten:
            # Flatten to vector: (C, H, W) -> (H, W, C) -> flatten
            data_list.append(img.numpy().transpose(1, 2, 0).reshape(-1))
        else:
            # Keep original shape (C, H, W) for ResNet/CNN
            data_list.append(img.numpy())
        label_list.append(label)
    return np.stack(data_list).astype(np.float32), np.array(label_list)


class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/usr/storage/ruizhe/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = True,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        sample_data_size=600000,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        num_positive_samples=1,
        top_genes: int = None,
        use_cache: bool = True,
        n_hop=1,
        data_name: str = "cifar10",  # "cifar10" or "cifar100"
        flatten: bool = False,  # False for ResNet/CNN, True for MLP
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.num_positive_samples = num_positive_samples
        self.top_genes = top_genes
        self.use_cache = use_cache
        self.dataset_kwargs = kwargs
        self.sample_data_size = sample_data_size
        self.n_hop = n_hop
        self.data_name = data_name.lower()
        self.flatten = flatten

    def load_data(self, data_path):
        torch.set_float32_matmul_precision('medium')
        transform = transforms.Compose([transforms.ToTensor()])

        # Load CIFAR dataset based on data_name
        if self.data_name == "cifar100":
            train_data = CIFAR100(root=data_path, train=True, download=True, transform=transform)
            num_classes = 100
            print("Loading CIFAR-100 dataset...")
        else:
            # Default to CIFAR-10
            train_data = CIFAR10(root=data_path, train=True, download=True, transform=transform)
            num_classes = 10
            print("Loading CIFAR-10 dataset...")

        # Extract images (flatten=False keeps (N, 3, 32, 32) for ResNet)
        DATA, LABEL = extract_cifar_data(train_data, flatten=self.flatten)

        # Create anndata object for visualization callbacks
        adata = anndata.AnnData(X=DATA.reshape(DATA.shape[0], -1))  # Always store flattened for adata
        adata.obs['batch'] = LABEL.astype(str)
        adata.obs['final_annotation'] = LABEL.astype(str)
        adata.obs['cell_type'] = LABEL.astype(str)

        self.info_list = ['batch', 'final_annotation']
        self.num_classes = num_classes

        if self.flatten:
            print(f'CIFAR data shape: {DATA.shape} (flattened), num_classes: {num_classes}')
        else:
            print(f'CIFAR data shape: {DATA.shape} (C, H, W for ResNet), num_classes: {num_classes}')

        return adata, DATA, LABEL

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        # Use flattened shape for filename to ensure consistency
        flat_shape = (data.shape[0], np.prod(data.shape[1:]))
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__ + "_" + self.data_name + str(flat_shape), k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)

        if not os.path.exists(filename):
            print("Computing nearest neighbors, will save to:", filename)
            # Always flatten for nearest neighbor computation
            X_rshaped = data.reshape((data.shape[0], -1))
            if pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)
        else:
            print("Loading nearest neighbors from:", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path)
        self.adata = adata

        augmenter_list = [
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": 1, 'knn': self.K}]),
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, 'knn': self.K}])
        ]

        neighbors_index = self.cal_near_index(
            data=data,
            k=self.K,
            device="cuda",
            uselabel=self.uselabel,
            pca_dim=self.pca_dim,
        )

        self.dataset = DataSetBaseMultiLevel(
            data=data,
            label=label,
            neighbors_index=neighbors_index,
            transform=augmenter_list[-1],
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
