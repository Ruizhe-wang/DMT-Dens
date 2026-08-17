import lightning as pl
from torch.utils.data import DataLoader
import joblib
import os
import torch
import numpy as np
import scanpy as sc
import sklearn.preprocessing
from sklearn.decomposition import PCA
from pynndescent import NNDescent
import anndata
from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


class DMTBaseDataModule(pl.LightningDataModule):
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
        sadata = sc.read(os.path.join(data_path, "gast10kwithcelltype.h5ad"))
        data = np.array(sadata.X.toarray() if hasattr(sadata.X, "toarray") else sadata.X).astype(np.float32)
        
        if 'celltype' in sadata.obs:
            label_col = sadata.obs['celltype']
        elif 'cell_type' in sadata.obs:
            label_col = sadata.obs['cell_type']
        else:
            label_col = np.zeros(data.shape[0])
            
        le = sklearn.preprocessing.LabelEncoder()
        label = le.fit_transform(label_col)

        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        if data.shape[1] % 2 != 0:
            zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)
            data = np.hstack((data, zeros_column))

        adata = anndata.AnnData(X=data)
        adata.obs['batch'] = label.astype(str)
        adata.obs['final_annotation'] = label.astype(str)
        self.info_list = ['batch', 'final_annotation']
        
        print(f"Gastrulation loaded: {data.shape}")
        return adata, data, label

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__ + str(data.shape), k, uselabel, pca_dim
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
                from sklearn.metrics import pairwise_distances
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)
        else:
            print("load data from ", filename)
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
