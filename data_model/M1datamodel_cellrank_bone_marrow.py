import os

import anndata
import joblib
import numpy as np
import pandas as pd
import sklearn.preprocessing
from pynndescent import NNDescent
from sklearn.decomposition import PCA

from data_model.runtime_sampling import subsample_adata_arrays

try:
    import lightning as pl
except ModuleNotFoundError:
    from types import SimpleNamespace

    class _LightningDataModule:
        pass

    pl = SimpleNamespace(LightningDataModule=_LightningDataModule)


class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/zangzelin/data/bone_marrow",
        h5ad_file: str = "bone_marrow.h5ad",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 42,
        sample_data_size=None,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        num_positive_samples=1,
        top_genes: int = 2000,
        use_cache: bool = True,
        n_hop=1,
        label_key: str = "clusters",
        pseudotime_key: str | None = "palantir_pseudotime",
        diffusion_potential_key: str | None = "palantir_diff_potential",
        branch_probs_key: str | None = "palantir_branch_probs",
        branch_names_key: str | None = "palantir_branch_probs_cell_types",
        use_obsm: str = None,
        minmax_scale: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.h5ad_file = h5ad_file
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
        self.label_key = label_key
        self.pseudotime_key = pseudotime_key
        self.diffusion_potential_key = diffusion_potential_key
        self.branch_probs_key = branch_probs_key
        self.branch_names_key = branch_names_key
        self.use_obsm = use_obsm
        self.minmax_scale = minmax_scale

    def _resolve_label_key(self, adata):
        candidates = [
            self.label_key,
            "celltype",
            "cell_type",
            "clusters",
            "leiden",
        ]
        for key in candidates:
            if key and key in adata.obs:
                return key
        raise KeyError(
            "No cluster or cell-type annotation found in CellRank bone marrow AnnData. "
            f"Tried: {', '.join(str(k) for k in candidates if k)}"
        )

    def _read_feature_matrix(self, adata):
        if self.use_obsm:
            if self.use_obsm not in adata.obsm:
                raise KeyError(f"AnnData.obsm does not contain {self.use_obsm!r}")
            data = np.asarray(adata.obsm[self.use_obsm], dtype=np.float32)
            var = pd.DataFrame(index=[f"{self.use_obsm}_{i}" for i in range(data.shape[1])])
            return data, var

        if self.top_genes is not None and int(self.top_genes) > 0 and int(self.top_genes) < adata.n_vars:
            try:
                import scanpy as sc

                sc.pp.highly_variable_genes(adata, n_top_genes=int(self.top_genes), flavor="seurat")
                adata = adata[:, adata.var["highly_variable"]].copy()
            except (ModuleNotFoundError, ValueError, FloatingPointError, OverflowError) as exc:
                print(
                    "[CellRankBoneMarrow] scanpy highly_variable_genes failed; "
                    f"falling back to variance selection: {exc}"
                )
                adata = self._select_top_variable_genes(adata, int(self.top_genes))

        data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        data = np.asarray(data, dtype=np.float32)
        var = adata.var.copy()
        if var.index.empty:
            var.index = pd.Index([f"gene_{i}" for i in range(data.shape[1])])
        var.index = var.index.astype(str)
        return data, var

    @staticmethod
    def _select_top_variable_genes(adata, top_genes):
        top_k = min(int(top_genes), adata.n_vars)
        x = adata.X

        if hasattr(x, "power"):
            mean = np.asarray(x.mean(axis=0)).ravel()
            mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
            variances = mean_sq - np.square(mean)
        else:
            dense = np.asarray(x, dtype=np.float32)
            variances = np.var(dense, axis=0)

        variances = np.nan_to_num(variances, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
        selected = np.argpartition(variances, -top_k)[-top_k:]
        return adata[:, np.sort(selected)].copy()

    @staticmethod
    def _pad_odd_feature(data, var):
        if data.shape[1] % 2 == 0:
            return data, var

        data = np.hstack((data, np.zeros((data.shape[0], 1), dtype=np.float32)))
        padding_name = "__padding_gene__"
        while padding_name in var.index:
            padding_name = f"_{padding_name}"
        var = pd.concat([var, pd.DataFrame(index=[padding_name])], axis=0)
        return data, var

    def _branch_names(self, sadata, n_branches):
        if self.branch_names_key and self.branch_names_key in sadata.uns:
            names = [str(name) for name in np.asarray(sadata.uns[self.branch_names_key]).tolist()]
            if len(names) == n_branches:
                return names
        return [f"branch_{i}" for i in range(n_branches)]

    def _add_palantir_metadata(self, sadata, obs, adata):
        if self.pseudotime_key and self.pseudotime_key in obs:
            obs["pseudotime"] = obs[self.pseudotime_key]

        if self.diffusion_potential_key and self.diffusion_potential_key in obs:
            obs["diffusion_potential"] = obs[self.diffusion_potential_key]

        if self.branch_probs_key and self.branch_probs_key in sadata.obsm:
            probs = np.asarray(sadata.obsm[self.branch_probs_key], dtype=np.float32)
            names = self._branch_names(sadata, probs.shape[1])
            adata.obsm[self.branch_probs_key] = probs
            for index, name in enumerate(names):
                obs[f"branch_prob_{name}"] = probs[:, index]
            obs["terminal_state"] = np.asarray(names, dtype=object)[np.argmax(probs, axis=1)]

    def _resolve_h5ad_path(self, data_path):
        file_path = os.path.join(data_path, self.h5ad_file)
        if os.path.exists(file_path):
            return file_path

        nested_path = os.path.join(data_path, "bone_marrow", self.h5ad_file)
        if os.path.exists(nested_path):
            return nested_path

        return file_path

    def load_data(self, data_path):
        file_path = self._resolve_h5ad_path(data_path)
        sadata = anndata.read_h5ad(file_path)

        label_key = self._resolve_label_key(sadata)
        label_col = sadata.obs[label_key].astype(str)

        data, var = self._read_feature_matrix(sadata.copy() if self.top_genes else sadata)

        if self.minmax_scale:
            scaler = sklearn.preprocessing.MinMaxScaler()
            data = scaler.fit_transform(data).astype(np.float32)

        data, var = self._pad_odd_feature(data, var)

        obs = sadata.obs.copy()
        obs["celltype"] = label_col.to_numpy(dtype=str)
        obs["cell_type"] = label_col.to_numpy(dtype=str)
        obs["final_annotation"] = label_col.to_numpy(dtype=str)

        adata = anndata.AnnData(X=data, obs=obs, var=var)
        for key in sadata.obsm.keys():
            if sadata.obsm[key].shape[0] == sadata.n_obs:
                adata.obsm[key] = np.asarray(sadata.obsm[key])
        self._add_palantir_metadata(sadata, adata.obs, adata)
        adata.var_names_make_unique()

        le = sklearn.preprocessing.LabelEncoder()
        label = le.fit_transform(label_col).astype(np.int32)

        adata, data, label, self.sampled_indices = subsample_adata_arrays(
            adata, data, label, self.sample_data_size, seed=self.seed
        )

        adata.obs["batch"] = label.astype(str)
        adata.obs["label_id"] = label.astype(str)

        self.label_encoder = le
        self.info_list = ["batch", "final_annotation"]
        if "pseudotime" in adata.obs:
            self.info_list.append("pseudotime")
        if "diffusion_potential" in adata.obs:
            self.info_list.append("diffusion_potential")
        if "terminal_state" in adata.obs:
            self.info_list.append("terminal_state")

        print(f"CellRank bone marrow loaded: {data.shape} from {file_path}")
        return adata, data, label

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}seed{}.pkl".format(
            self.__class__.__name__ + str(data.shape),
            k,
            uselabel,
            pca_dim,
            self.seed,
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
        from data_model.aug import Augmenter
        from data_model.dataset_base_multi_level import DataSetBaseMultiLevel

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
        from torch.utils.data import DataLoader

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
        from torch.utils.data import DataLoader

        return DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
