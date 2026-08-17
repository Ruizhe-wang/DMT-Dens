import os

import anndata
import joblib
import lightning as pl
import numpy as np
import scipy.io as sio
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel


class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data",
        dataset_file: str = "treedata.mat",
        data_name: str = "tree",
        data_key: str = "M",
        label_key: str = "C",
        batch_size: int = 2048,
        num_workers: int = 4,
        K: int = 100,
        uselabel: bool = False,
        pca_dim: int = 64,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 42,
        sample_data_size=None,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        num_positive_samples=1,
        top_genes: int = None,
        use_cache: bool = True,
        n_hop: int = 1,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_file = dataset_file
        self.data_name = data_name
        self.data_key = data_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.sample_data_size = sample_data_size
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.num_positive_samples = num_positive_samples
        self.top_genes = top_genes
        self.use_cache = use_cache
        self.n_hop = n_hop
        self.dataset_kwargs = kwargs

    def _resolve_mat_path(self):
        if os.path.isdir(self.data_path):
            return os.path.join(self.data_path, self.dataset_file)
        return self.data_path

    def load_data(self, data_path, dataset_file=None):
        file_path = self._resolve_mat_path()
        mat = sio.loadmat(file_path)

        if self.data_key in mat:
            data = mat[self.data_key]
        elif "X" in mat:
            data = mat["X"]
        else:
            raise KeyError(f"Cannot find data key '{self.data_key}' or fallback key 'X' in {file_path}")

        if self.label_key in mat:
            raw_label = mat[self.label_key]
        elif "Y" in mat:
            raw_label = mat["Y"]
        else:
            raise KeyError(f"Cannot find label key '{self.label_key}' or fallback key 'Y' in {file_path}")

        if hasattr(data, "toarray"):
            data = data.toarray()

        data = np.asarray(data, dtype=np.float32)
        raw_label = np.asarray(raw_label).reshape(-1)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {data.shape} from {file_path}")
        if data.shape[0] != raw_label.shape[0]:
            raise ValueError(
                f"Data/label row mismatch in {file_path}: data has {data.shape[0]} rows, labels have {raw_label.shape[0]} rows"
            )

        if self.sample_data_size is not None and data.shape[0] > self.sample_data_size:
            rng = np.random.default_rng(self.seed)
            selected = rng.choice(data.shape[0], self.sample_data_size, replace=False)
            data = data[selected]
            raw_label = raw_label[selected]

        unique_labels, label = np.unique(raw_label, return_inverse=True)
        label = label.astype(np.int64)

        adata = anndata.AnnData(X=data)
        adata.obs["batch"] = unique_labels[label].astype(str)
        adata.obs["final_annotation"] = unique_labels[label].astype(str)
        adata.obs["branch_id"] = unique_labels[label].astype(str)
        self.info_list = ["batch", "final_annotation", "branch_id"]

        print(f"Loaded {file_path}: data shape {data.shape}, label shape {label.shape}, branches {len(unique_labels)}")
        return adata, data, label

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        file_stem = os.path.splitext(os.path.basename(self.dataset_file))[0]
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__ + "_" + self.data_name + "_" + file_stem + "_" + str(data.shape),
            k,
            uselabel,
            pca_dim,
        )
        os.makedirs("save_near_index", exist_ok=True)

        if self.use_cache and os.path.exists(filename):
            print("load data from ", filename)
            return joblib.load(filename)

        print("save data to ", filename)
        reshaped = data.reshape((data.shape[0], -1))
        actual_pca_dim = min(pca_dim, reshaped.shape[0], reshaped.shape[1])
        if actual_pca_dim < reshaped.shape[1]:
            reshaped = PCA(n_components=actual_pca_dim).fit_transform(reshaped)

        if not uselabel:
            index = NNDescent(reshaped, n_jobs=-1)
            neighbors_index, _ = index.query(reshaped, k=k + 1)
            neighbors_index = neighbors_index[:, 1:]
        else:
            dis = pairwise_distances(reshaped)
            label_matrix = np.repeat(label.reshape(1, -1), reshaped.shape[0], axis=0)
            dis[(label_matrix - label_matrix.T) != 0] = dis.max() + 1
            neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]

        if self.use_cache:
            joblib.dump(value=neighbors_index, filename=filename)
        return neighbors_index

    def setup(self, stage: str):
        adata, data, label = self.load_data(self.data_path, self.dataset_file)
        self.adata = adata

        augmenter = Augmenter(
            [{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}]
        )

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
            transform=augmenter,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )