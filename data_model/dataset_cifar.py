import lightning as pl
from torch.utils.data import DataLoader
import joblib
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import anndata
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


def create_adata(data, label):
    adata = anndata.AnnData(X=data)
    if isinstance(label, np.ndarray) and len(label.shape) > 1 and label.shape[1] > 1:
        adata.obs['label'] = label[:, 0].astype(str)
    elif isinstance(label, np.ndarray):
        adata.obs['label'] = label.astype(str)
    else:
        adata.obs['label'] = label.astype(str)
    return adata


class Cifar10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
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
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.K = K
        self.n_hop = n_hop
        self.dataset_kwargs = kwargs

    def load_data(self, data_path):
        transform = transforms.Compose([transforms.ToTensor()])

        # Load CIFAR-10 train and test data
        train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

        # Extract and combine train and test data
        train_data = []
        train_labels = []
        for img, label in train_dataset:
            train_data.append(img.numpy().transpose(1, 2, 0).reshape(-1))
            train_labels.append(label)

        test_data = []
        test_labels = []
        for img, label in test_dataset:
            test_data.append(img.numpy().transpose(1, 2, 0).reshape(-1))
            test_labels.append(label)

        data = np.concatenate([np.stack(train_data), np.stack(test_data)]).astype(np.float32)
        label = np.concatenate([np.array(train_labels), np.array(test_labels)]).astype(np.int32)

        print('CIFAR-10 data.shape', data.shape)
        print('CIFAR-10 num_classes', 10)

        adata = create_adata(data, label)
        adata.obs['batch'] = label.astype(str)
        adata.obs['final_annotation'] = label.astype(str)
        adata.obs['cell_type'] = label.astype(str)
        self.info_list = ['batch', 'final_annotation']
        return adata, data, label

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path)
        self.adata = adata

        augmenter = Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}])
        neighbors_index = self.cal_near_index(
            data=data,
            label=label,
            k=self.K,
            uselabel=self.uselabel,
            pca_dim=self.pca_dim,
        )

        self.dataset = DataSetBaseMultiLevel(
            data=data,
            label=label,
            neighbors_index=neighbors_index,
            transform=augmenter,
        )

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__+str(data.shape), k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)

        if not os.path.exists(filename):
            print("save data to ", filename)
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
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


class Cifar100DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
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
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.K = K
        self.n_hop = n_hop
        self.dataset_kwargs = kwargs

    def load_data(self, data_path):
        transform = transforms.Compose([transforms.ToTensor()])

        # Load CIFAR-100 train and test data
        train_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=transform)

        # Extract and combine train and test data
        train_data = []
        train_labels = []
        for img, label in train_dataset:
            train_data.append(img.numpy().transpose(1, 2, 0).reshape(-1))
            train_labels.append(label)

        test_data = []
        test_labels = []
        for img, label in test_dataset:
            test_data.append(img.numpy().transpose(1, 2, 0).reshape(-1))
            test_labels.append(label)

        data = np.concatenate([np.stack(train_data), np.stack(test_data)]).astype(np.float32)
        label = np.concatenate([np.array(train_labels), np.array(test_labels)]).astype(np.int32)

        print('CIFAR-100 data.shape', data.shape)
        print('CIFAR-100 num_classes', 100)

        adata = create_adata(data, label)
        adata.obs['batch'] = label.astype(str)
        adata.obs['final_annotation'] = label.astype(str)
        adata.obs['cell_type'] = label.astype(str)
        self.info_list = ['batch', 'final_annotation']
        return adata, data, label

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path)
        self.adata = adata

        augmenter = Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}])
        neighbors_index = self.cal_near_index(
            data=data,
            label=label,
            k=self.K,
            uselabel=self.uselabel,
            pca_dim=self.pca_dim,
        )

        self.dataset = DataSetBaseMultiLevel(
            data=data,
            label=label,
            neighbors_index=neighbors_index,
            transform=augmenter,
        )

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__+str(data.shape), k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)

        if not os.path.exists(filename):
            print("save data to ", filename)
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
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
