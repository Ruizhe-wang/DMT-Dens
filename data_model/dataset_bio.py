import lightning as pl
from torch.utils.data import DataLoader
import scanpy as sc
import joblib
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pynndescent import NNDescent
from sklearn.metrics import pairwise_distances
import anndata
import scipy.sparse as sp
import pandas as pd
from data_model.aug import Augmenter
from data_model.dataset_base import DataSetBase
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


class MCAD9119DataModule(pl.LightningDataModule):
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
        data = np.load(os.path.join(data_path, 'mca_data/mca_data_dim_34947.npy'))
        data = data[:, data.max(axis=0) > 4].astype(np.float32)
        label = np.load(os.path.join(data_path, 'mca_data/mca_label_dim_34947.npy'))

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1

        for i in list(label_count.keys()):
            if label_count[i] < 500:
                label[label == i] = -1

        data = data[label != -1]
        label = label[label != -1]
        label = label.astype(np.int32)

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


class HCLDataModule(pl.LightningDataModule):
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
        sadata = sc.read(os.path.join(data_path, "HCL60kafter-elis-all.h5ad"))
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)])).astype(np.int32)

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(bool)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]

        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)
        data = np.hstack((data, zeros_column))

        print(data.shape)
        adata = create_adata(data, label)
        adata.obs['batch'] = label.astype(str)
        adata.obs['final_annotation'] = label.astype(str)
        adata.obs['cell_type'] = label.astype(str)
        self.info_list = ['batch', 'final_annotation']
        return adata, data, label

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path)
        self.adata = adata

        # Updated augmenter with 'knn' key and n_hop parameter (matching M1datamodel_mnist_v2.py)
        augmenter = Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}])

        neighbors_index = self.cal_near_index(
            data=data,
            label=label,
            k=self.K,
            uselabel=self.uselabel,
            pca_dim=self.pca_dim,
        )

        # Use DataSetBaseMultiLevel instead of DataSetBase
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


class Gast10kDataModule(pl.LightningDataModule):
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
        sadata = sc.read(os.path.join(data_path, "gast10kwithcelltype.h5ad"))
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)
        data = np.hstack((data, zeros_column))

        label_train_str = list(sadata.obs['celltype'])
        label_train_str_set = sorted(list(set(label_train_str)))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str])
        label = np.array(label_train).astype(np.int32)
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


class SAMUSIKDataModule(pl.LightningDataModule):
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
        data = pd.read_csv(os.path.join(data_path, 'samusik_01.csv'))
        label_df = pd.read_csv(os.path.join(data_path, 'samusik01_labelnr.csv'))
        data.fillna(data.min(), inplace=True)
        label = np.array(label_df)[:,-1]
        data = np.array(data)[:,1:]
        data = (data-data.min())/(data.max()-data.min())
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label).astype(np.int32)
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


class ActivityDataModule(pl.LightningDataModule):
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
        train_data = pd.read_csv(os.path.join(data_path, 'feature_select/Activity_train.csv'))
        test_data = pd.read_csv(os.path.join(data_path, 'feature_select/Activity_test.csv'))
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(['subject', 'Activity'], axis=1).to_numpy()
        label_str = all_data['Activity'].tolist()
        label_str_set = sorted(list(set(label_str)))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data-data.min())/(data.max()-data.min())

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label).astype(np.int32)
        print('data.shape', data.shape)
        print(label)
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


class CeleganDataModule(pl.LightningDataModule):
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
        adata = sc.read(os.path.join(data_path, "difftreedata/data/celegan.h5ad"))
        data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = pd.read_csv(os.path.join(data_path, 'difftreedata/data/celegan_celltype_2.tsv'), sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype.values))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print(data)
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


class DARLINDataModule(pl.LightningDataModule):
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
        adata = sc.read(os.path.join(data_path, "DARLIN/RAW/GSM4185642_stateFate_inVitro.h5ad"))

        sc.pp.log1p(adata)
        adata.obs['celltype']=adata.obs['Cell type annotation']
        adata = adata[~adata.obs['celltype'].isna()]
        sc.pp.highly_variable_genes(adata, n_top_genes=500)
        adata = adata[:, adata.var['highly_variable']]

        if isinstance(adata.X, (sp.csr_matrix, sp.csc_matrix)):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.array(data).astype(np.float32)

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / std

        label_celltype = adata.obs['celltype']
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print('data.shape', data.shape, 'num_classes', len(label_train_str_set))
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


class EpitheliaCellDataModule(pl.LightningDataModule):
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
        adata = sc.read(os.path.join(data_path, "MouseDevelopment/EpitheliaCell.h5ad"))
        adata.obs['celltype']=adata.obs['cell_type']
        adata = adata[~adata.obs['celltype'].isna()]
        sc.pp.highly_variable_genes(adata, n_top_genes=500)
        adata = adata[:, adata.var['highly_variable']]
        data = adata.X

        if isinstance(adata.X, (sp.csr_matrix, sp.csc_matrix)):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = adata.obs['celltype']
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print('data.shape', data.shape)
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


class ImmuneDataModule(pl.LightningDataModule):
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
        adata = sc.read(os.path.join(data_path, "Immune_ALL_human.h5ad"))

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=5000,
            flavor='seurat_v3',
            batch_key='batch',
            subset=False
        )

        adata = adata[:, adata.var.highly_variable].copy()

        data = adata.X.toarray().astype(np.float32)

        annotation = list(adata.obs['final_annotation'])

        set_list = sorted(list(set(annotation)))
        label = np.array([set_list.index(i) for i in annotation]).astype(np.int32)

        adata.obs['batch'] = label.astype(str)
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


class PancreasDataModule(pl.LightningDataModule):
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
        adata = sc.read(os.path.join(data_path, "pancreas_verified.h5ad"))

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=50,
            flavor='seurat_v3',
        )

        adata = adata[:, adata.var.highly_variable].copy()

        data = adata.X

        annotation = list(adata.obs['celltype'])
        adata.obs['final_annotation'] = adata.obs['celltype']

        set_list = sorted(list(set(annotation)))
        label = np.array([set_list.index(i) for i in annotation]).astype(np.int32)

        adata.obs['batch'] = label.astype(str)
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


class AqcallDataModule(pl.LightningDataModule):
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
        data = np.load(os.path.join(data_path, "dmthi/aqc_all_data_3000.npy"))
        label = np.load(os.path.join(data_path, "dmthi/aqc_all_label.npy"))
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        print(data)
        label = np.array(label).astype(np.int32)
        print(label)
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


class EpiDataModule(pl.LightningDataModule):
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
        data = np.load(os.path.join(data_path, "difftreedata/data/EpitheliaCell_new_data_n.npy"))
        label = np.load(os.path.join(data_path, "difftreedata/data/epi_all_label_100_Mar.npy"))
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        print(data)
        label = np.array(label).astype(np.int32)
        print(label)
        print(np.max(label))
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


class LimbFilterDataModule(pl.LightningDataModule):
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
        data = np.load(os.path.join(data_path, "difftreedata/data/LimbFilter_data_n.npy"))
        label = np.load(os.path.join(data_path, "difftreedata/data/LimbFilter_label.npy"))
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        print('data.shape', data.shape)
        label = np.array(label).astype(np.int32)
        print('label.shape', label.shape)
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
