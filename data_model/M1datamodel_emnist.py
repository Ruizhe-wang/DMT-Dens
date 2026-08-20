import lightning as pl
from torch.utils.data import DataLoader
import scanpy as sc
import joblib
import os
import torch
import numpy as np
from torch.utils import data
from sklearn.decomposition import PCA

from pynndescent import NNDescent

from sklearn.metrics import pairwise_distances
import anndata
import matplotlib.pyplot as plt
from data_model.aug import Augmenter
from torchvision import transforms

from torchvision.datasets import EMNIST

from data_model.dataset_base import DataSetBase
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel

def flatten_mnist_data(dataset):
    """
    Flatten EMNIST dataset images to vectors and extract labels.
    """
    data_list, label_list = [], []
    for img, label in dataset:
        data_list.append(img.numpy().squeeze())
        label_list.append(label)
    return np.stack(data_list).reshape((-1, 784)), np.array(label_list)

class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data",
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

    def load_data(self, data_path):
        torch.set_float32_matmul_precision('medium')
        transform = transforms.Compose([transforms.ToTensor()])

        # Load EMNIST dataset
        train_data = EMNIST(root=data_path, split='byclass', train=True, download=True, transform=transform)

        # Flatten image
        DATA, LABEL = flatten_mnist_data(train_data)
        
        adata = anndata.AnnData(X=DATA)
        adata.obs['batch'] = LABEL.astype(str)
        adata.obs['final_annotation'] = LABEL.astype(str)
        self.info_list = ['batch', 'final_annotation']
        return adata, DATA, LABEL

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__+str(data.shape), k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)
        
        if not os.path.exists(filename):
            print("save data to ", filename)
            X_rshaped = data.reshape((data.shape[0], -1))
            actual_pca_dim = min(pca_dim, X_rshaped.shape[0], X_rshaped.shape[1])
            if actual_pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=actual_pca_dim).fit_transform(X_rshaped)
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

    def sample_multi_level_data(self, data, label, sample_size_list = [2000, 60000]):
        data_list = []
        label_list = []
        neighbors_list = []
        for i, sample_size in enumerate(sample_size_list):
            if data.shape[0] < sample_size:
                sample_size = data.shape[0]
            
            if i == len(sample_size_list) - 1:
                sampled_data = data
                sampled_label = label
            else:
                index = np.random.choice(data.shape[0], sample_size, replace=False)
                sampled_data = data[index]
                sampled_label = label[index]
            
            neighbors_index = self.cal_near_index(
                data=sampled_data,
            label=label,
                k=self.K,
                device="cuda",
                uselabel=self.uselabel,
                pca_dim=self.pca_dim,
            )
            
            data_list.append(sampled_data)
            label_list.append(sampled_label)
            neighbors_list.append(neighbors_index)
        return data_list, label_list, neighbors_list

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path)
        
        augmenter_list = [
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": 1}]), 
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop}])
        ]
        
        self.adata = adata
        
        neighbors_index = self.cal_near_index(
            data=data,
            label=label,
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

    def _build_dataloader(self, drop_last: bool, shuffle: bool = False):
        dataloader_kwargs = dict(
            dataset=self.dataset,
            drop_last=drop_last,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=max(int(self.num_workers), 0),
            pin_memory=True,
        )
        if shuffle:
            dataloader_kwargs["shuffle"] = True
        if dataloader_kwargs["num_workers"] > 0:
            dataloader_kwargs["persistent_workers"] = True
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self):
        return self._build_dataloader(drop_last=True, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(drop_last=False)

    def test_dataloader(self):
        return self._build_dataloader(drop_last=True)
