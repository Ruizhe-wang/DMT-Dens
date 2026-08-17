import os

import anndata
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing
from sklearn.decomposition import PCA

from data_model.runtime_sampling import subsample_adata_arrays

try:
    import lightning as pl
except Exception:
    from types import SimpleNamespace

    class _LightningDataModule:
        pass

    pl = SimpleNamespace(LightningDataModule=_LightningDataModule)


class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "/usr/storage/zelin181data/sclineage/single-cell-lineage1/data/figure3_fate/dyngen",
        h5ad_file: str = "fig3_fate_dyngen_shared_dynGenZL10k_hyperbranch_v2_seed42.h5ad",
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
        top_genes: int = None,
        use_cache: bool = True,
        n_hop=1,
        label_key: str = "fig3_true_terminal",
        state_key: str | None = "fig3_true_state",
        time_key: str | None = "fig3_true_time",
        branch_key: str | None = "true_branch",
        use_obsm: str = None,
        minmax_scale: bool = True,
        var_name_key: str | None = "gene_id",
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
        self.state_key = state_key
        self.time_key = time_key
        self.branch_key = branch_key
        self.use_obsm = use_obsm
        self.minmax_scale = minmax_scale
        self.var_name_key = var_name_key

    def _resolve_h5ad_path(self, data_path):
        if os.path.isfile(data_path):
            return data_path

        file_path = os.path.join(data_path, self.h5ad_file)
        if os.path.exists(file_path):
            return file_path

        nested_path = os.path.join(data_path, "dyngen", self.h5ad_file)
        if os.path.exists(nested_path):
            return nested_path

        return file_path

    def _resolve_label_key(self, adata):
        candidates = [
            self.label_key,
            "fig3_true_terminal",
            "fig3_true_state",
            "true_branch",
            "to",
            "from",
        ]
        for key in candidates:
            if key and key in adata.obs:
                return key
        raise KeyError(
            "No fate or lineage annotation found in dynGen fate AnnData. "
            f"Tried: {', '.join(str(k) for k in candidates if k)}"
        )

    @staticmethod
    def _select_top_variable_genes(adata, top_genes):
        top_k = min(int(top_genes), adata.n_vars)
        x = adata.X
        if sp.issparse(x):
            mean = np.asarray(x.mean(axis=0)).ravel()
            mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
            variances = mean_sq - np.square(mean)
        else:
            variances = np.var(np.asarray(x, dtype=np.float32), axis=0)

        variances = np.nan_to_num(variances, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
        selected = np.argpartition(variances, -top_k)[-top_k:]
        return adata[:, np.sort(selected)].copy()

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
                    "[DyngenFate] scanpy highly_variable_genes failed; "
                    f"falling back to variance selection: {exc}"
                )
                adata = self._select_top_variable_genes(adata, int(self.top_genes))

        data = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        data = np.asarray(data, dtype=np.float32)
        var = adata.var.copy()
        if self.var_name_key and self.var_name_key in var.columns:
            candidate_index = var[self.var_name_key].astype(str)
            if not candidate_index.isna().all() and candidate_index.str.len().gt(0).any():
                var.index = candidate_index
        if var.index.empty:
            var.index = pd.Index([f"gene_{i}" for i in range(data.shape[1])])
        var.index = var.index.astype(str)
        return data, var

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

    @staticmethod
    def _add_if_present(obs, source_key, target_key):
        if source_key and source_key in obs:
            obs[target_key] = obs[source_key]

    @staticmethod
    def _add_fate_probability_columns(obs):
        fate_keys = [key for key in obs.columns if str(key).startswith("fig3_fate_prob_")]
        for key in fate_keys:
            target = str(key).replace("fig3_", "", 1)
            obs[target] = obs[key].astype(np.float32)
        return fate_keys

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
        self._add_if_present(obs, self.time_key, "pseudotime")
        self._add_if_present(obs, self.branch_key, "branch")
        self._add_if_present(obs, self.state_key, "state")
        self._add_if_present(obs, "fig3_is_root", "is_root")
        self._add_if_present(obs, "fig3_is_branchpoint", "is_branchpoint")
        self._add_if_present(obs, "fig3_is_terminal", "is_terminal")
        fate_keys = self._add_fate_probability_columns(obs)

        le = sklearn.preprocessing.LabelEncoder()
        label = le.fit_transform(label_col).astype(np.int32)

        adata = anndata.AnnData(X=data, obs=obs, var=var)
        for key in sadata.obsm.keys():
            if sadata.obsm[key].shape[0] == sadata.n_obs:
                adata.obsm[key] = np.asarray(sadata.obsm[key])
        adata.var_names_make_unique()

        adata, data, label, self.sampled_indices = subsample_adata_arrays(
            adata, data, label, self.sample_data_size, seed=self.seed
        )

        adata.obs["batch"] = label.astype(str)
        adata.obs["label_id"] = label.astype(str)

        self.label_encoder = le
        self.fate_probability_keys = fate_keys
        self.info_list = ["batch", "final_annotation"]
        for key in ["pseudotime", "branch", "state", "is_root", "is_branchpoint", "is_terminal"]:
            if key in adata.obs:
                self.info_list.append(key)

        print(f"dynGen fate loaded: {data.shape} from {file_path}")
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
                from pynndescent import NNDescent

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
