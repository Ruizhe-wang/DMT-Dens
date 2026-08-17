import os

import anndata
import joblib
import lightning as pl
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

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
        include_time_label: bool = True,
        standardize: bool = True,
        exclude_unannotated: bool = True,
        use_cache: bool = True,
        n_hop=1,
        **kwargs,
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
        self.include_time_label = include_time_label
        self.standardize = standardize
        self.exclude_unannotated = exclude_unannotated
        self.use_cache = use_cache
        self.dataset_kwargs = kwargs
        self.sample_data_size = sample_data_size
        self.n_hop = n_hop

    def _resolve_data_dir(self, data_path):
        candidates = [
            data_path,
            os.path.join(data_path, "celegan"),
            os.path.join(data_path, "difftreedata", "data"),
            os.path.join(data_path, "difftreedata", "data", "celegan"),
        ]
        required = [
            "celegan.h5ad",
            "celegan_celltype_2.tsv",
            "celegan_embryo_time.tsv",
        ]
        for candidate in candidates:
            if all(os.path.exists(os.path.join(candidate, name)) for name in required):
                return candidate
        raise FileNotFoundError(
            "Could not find celegan files under supported layouts: "
            + ", ".join(candidates)
        )

    @staticmethod
    def _embryo_time_to_float(value):
        text = str(value).strip()
        if "-" in text:
            return float(text.split("-")[-1].strip())
        if text.startswith("<"):
            return float(text[1:].strip()) - 50.0
        if text.startswith(">"):
            return float(text[1:].strip()) + 100.0
        return float(text)

    def _build_celltype_time_label(self, label_celltype, label_embryo_time):
        label_names = sorted(label_celltype.unique().tolist())
        label_lookup = {name: index for index, name in enumerate(label_names)}
        celltype_label = np.array([label_lookup[name] for name in label_celltype], dtype=np.int32)
        embryo_time_numeric = np.array(
            [self._embryo_time_to_float(value) for value in label_embryo_time],
            dtype=np.float32,
        )

        if self.include_time_label:
            label = np.column_stack([celltype_label.astype(np.float32), embryo_time_numeric])
            label_columns = ["celltype_id", "embryo_time_numeric"]
        else:
            label = celltype_label
            label_columns = ["celltype_id"]

        return label, celltype_label, embryo_time_numeric, label_names, label_columns

    def load_data(self, data_path):
        data_dir = self._resolve_data_dir(data_path)
        adata_raw = sc.read(os.path.join(data_dir, "celegan.h5ad"))
        label_celltype = pd.read_csv(
            os.path.join(data_dir, "celegan_celltype_2.tsv"),
            sep="\t",
            header=None,
        ).iloc[:, 0].astype(str)
        label_embryo_time = pd.read_csv(
            os.path.join(data_dir, "celegan_embryo_time.tsv"),
            sep="\t",
            header=None,
        ).iloc[:, 0].astype(str)

        if adata_raw.n_obs != len(label_celltype) or adata_raw.n_obs != len(label_embryo_time):
            raise ValueError(
                "celegan label lengths do not match h5ad observations: "
                f"adata={adata_raw.n_obs}, celltype={len(label_celltype)}, "
                f"embryo_time={len(label_embryo_time)}"
            )

        data = adata_raw.X.toarray() if sp.issparse(adata_raw.X) else np.asarray(adata_raw.X)
        data = data.astype(np.float32)
        var = adata_raw.var.copy()

        if self.standardize:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1.0
            data = (data - mean) / std

        obs = adata_raw.obs.copy()
        obs["celltype"] = label_celltype.to_numpy()
        obs["embryo_time"] = label_embryo_time.to_numpy()

        if self.exclude_unannotated:
            mask = label_celltype.to_numpy() != "unannotated"
            data = data[mask]
            obs = obs.iloc[mask].copy()
            label_celltype = label_celltype[mask].reset_index(drop=True)
            label_embryo_time = label_embryo_time[mask].reset_index(drop=True)

        selected_celltypes = sorted(label_celltype.unique().tolist())
        label, celltype_label, embryo_time_numeric, label_names, label_columns = self._build_celltype_time_label(
            label_celltype=label_celltype,
            label_embryo_time=label_embryo_time,
        )

        obs["celltype"] = label_celltype.to_numpy()
        obs["cell_type"] = label_celltype.to_numpy()
        obs["final_annotation"] = label_celltype.to_numpy()
        obs["celltype_id"] = celltype_label
        obs["embryo_time"] = label_embryo_time.to_numpy()
        obs["embryo_time_numeric"] = embryo_time_numeric
        obs["batch"] = label_embryo_time.to_numpy()
        obs["label_id"] = celltype_label.astype(str)

        if var.index.empty:
            var.index = pd.Index([f"gene_{i}" for i in range(data.shape[1])])
        var.index = var.index.astype(str)

        adata = anndata.AnnData(X=data, obs=obs, var=var)
        adata.var_names_make_unique()

        self.selected_celltypes = selected_celltypes
        self.label_names = label_names
        self.label_columns = label_columns
        self.info_list = ["final_annotation", "embryo_time_numeric"]
        print(
            "C. elegans loaded: "
            f"data={data.shape}, label={label.shape}, n_celltypes={len(selected_celltypes)}"
        )
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
                neighbors_index, _ = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                neighbor_label = label[:, 0] if label is not None and np.ndim(label) > 1 else label
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(neighbor_label.reshape(1, -1), X_rshaped.shape[0], axis=0)
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
