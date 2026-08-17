import glob
import os
from types import SimpleNamespace

import anndata
import h5py
import joblib
import numpy as np
import pandas as pd
import sklearn.preprocessing
from pynndescent import NNDescent
from sklearn.decomposition import PCA

try:
    import lightning as pl
except ModuleNotFoundError:
    class _LightningDataModule:
        pass

    pl = SimpleNamespace(LightningDataModule=_LightningDataModule)

class DMTBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "A single-cell time-lapse of mouse prenatal development from gastrula to birth",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        sample_data_size=60000,
        rrc_rate: float = 0.8,
        trans_range: int = 6,
        num_positive_samples=1,
        top_genes: int = 1500,
        use_cache: bool = True,
        n_hop=1,
        label_key: str = "celltype_update",
        h5ad_pattern: str = "adata_JAX_dataset_*.h5ad",
        metadata_file: str = "df_cell.csv",
        gene_file: str = "df_gene.csv",
        metadata_chunksize: int = 200000,
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
        self.label_key = label_key
        self.h5ad_pattern = h5ad_pattern
        self.metadata_file = metadata_file
        self.gene_file = gene_file
        self.metadata_chunksize = metadata_chunksize

    def _discover_h5ad_files(self, data_path):
        files = sorted(glob.glob(os.path.join(data_path, self.h5ad_pattern)))
        if not files:
            files = sorted(glob.glob(os.path.join(data_path, "*.h5ad")))
        if not files:
            raise FileNotFoundError(f"No h5ad files found in {data_path}")
        return files

    @staticmethod
    def _h5ad_shape(path):
        with h5py.File(path, "r") as handle:
            x = handle["X"]
            if isinstance(x, h5py.Dataset):
                return x.shape
            shape = x.attrs.get("shape")
            if shape is None:
                raise ValueError(f"Cannot infer X shape from {path}")
            return tuple(int(value) for value in shape)

    def _sample_indices(self, total_size):
        if self.sample_data_size is None or self.sample_data_size >= total_size:
            return np.arange(total_size, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        sample_size = int(self.sample_data_size)
        return np.sort(rng.choice(total_size, size=sample_size, replace=False)).astype(np.int64)

    @staticmethod
    def _split_indices_by_file(global_indices, file_sizes):
        file_offsets = np.cumsum([0] + file_sizes)
        splits = []
        for file_idx, size in enumerate(file_sizes):
            start = file_offsets[file_idx]
            end = start + size
            mask = (global_indices >= start) & (global_indices < end)
            splits.append(global_indices[mask] - start)
        return splits

    def _read_metadata(self, data_path, global_indices):
        metadata_path = os.path.join(data_path, self.metadata_file)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        selected = []
        cursor = 0
        sorted_indices = np.asarray(global_indices, dtype=np.int64)
        for chunk in pd.read_csv(metadata_path, index_col=0, chunksize=self.metadata_chunksize):
            chunk_end = cursor + len(chunk)
            left = np.searchsorted(sorted_indices, cursor, side="left")
            right = np.searchsorted(sorted_indices, chunk_end, side="left")
            if right > left:
                local_positions = sorted_indices[left:right] - cursor
                selected.append(chunk.iloc[local_positions].copy())
            cursor = chunk_end
            if cursor > sorted_indices[-1]:
                break

        if not selected:
            raise ValueError("No metadata rows matched sampled cell indices")
        metadata = pd.concat(selected, axis=0)
        metadata.index = metadata.index.astype(str)
        return metadata

    def _read_var(self, data_path, n_vars):
        gene_path = os.path.join(data_path, self.gene_file)
        if os.path.exists(gene_path):
            var = pd.read_csv(gene_path, index_col=0)
            var.index = var.index.astype(str)
        else:
            var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
        return var.iloc[:n_vars].copy()

    def _feature_indices(self, n_vars):
        if self.top_genes is None or self.top_genes >= n_vars:
            return np.arange(n_vars, dtype=np.int64)
        return np.arange(int(self.top_genes), dtype=np.int64)

    def _read_expression(self, h5ad_files, local_indices_by_file, feature_indices):
        chunks = []
        for path, local_indices in zip(h5ad_files, local_indices_by_file):
            if len(local_indices) == 0:
                continue
            chunks.append(self._read_h5ad_expression_rows(path, local_indices, feature_indices))
        if not chunks:
            raise ValueError("No expression rows were loaded")
        return np.vstack(chunks)

    @staticmethod
    def _read_h5ad_expression_rows(path, row_indices, feature_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        with h5py.File(path, "r") as handle:
            x = handle["X"]
            if isinstance(x, h5py.Dataset):
                rows = []
                for row in row_indices:
                    rows.append(np.asarray(x[row, feature_indices], dtype=np.float32))
                return np.vstack(rows)

            encoding_type = x.attrs.get("encoding-type", "")
            if isinstance(encoding_type, bytes):
                encoding_type = encoding_type.decode("utf-8")
            if encoding_type != "csr_matrix":
                raise ValueError(f"Unsupported h5ad X encoding: {encoding_type}")

            indptr = x["indptr"]
            indices_ds = x["indices"]
            data_ds = x["data"]
            output = np.zeros((len(row_indices), len(feature_indices)), dtype=np.float32)

            contiguous_prefix = np.array_equal(feature_indices, np.arange(len(feature_indices)))
            if contiguous_prefix:
                max_col = len(feature_indices)
                for output_row, source_row in enumerate(row_indices):
                    start = int(indptr[source_row])
                    end = int(indptr[source_row + 1])
                    cols = indices_ds[start:end]
                    values = data_ds[start:end]
                    mask = cols < max_col
                    output[output_row, cols[mask]] = values[mask]
            else:
                feature_to_position = {int(col): pos for pos, col in enumerate(feature_indices)}
                for output_row, source_row in enumerate(row_indices):
                    start = int(indptr[source_row])
                    end = int(indptr[source_row + 1])
                    cols = indices_ds[start:end]
                    values = data_ds[start:end]
                    positions = [feature_to_position.get(int(col), -1) for col in cols]
                    positions = np.asarray(positions, dtype=np.int64)
                    mask = positions >= 0
                    output[output_row, positions[mask]] = values[mask]

            return output

    def _choose_label_values(self, metadata):
        for key in (self.label_key, "celltype_update", "major_trajectory", "day"):
            if key in metadata:
                return metadata[key].astype(str).fillna("unknown").to_numpy()
        return np.zeros(len(metadata), dtype=str)

    @staticmethod
    def _pad_odd_feature(data, var):
        if data.shape[1] % 2 == 0:
            return data, var
        data = np.hstack([data, np.zeros((data.shape[0], 1), dtype=np.float32)])
        padding_name = "__padding_gene__"
        while padding_name in var.index:
            padding_name = f"_{padding_name}"
        var = pd.concat([var, pd.DataFrame(index=[padding_name])], axis=0)
        return data, var

    def load_data(self, data_path):
        h5ad_files = self._discover_h5ad_files(data_path)
        shapes = [self._h5ad_shape(path) for path in h5ad_files]
        file_sizes = [shape[0] for shape in shapes]
        n_vars = shapes[0][1]
        if any(shape[1] != n_vars for shape in shapes):
            raise ValueError("All h5ad shards must have the same number of genes")

        global_indices = self._sample_indices(sum(file_sizes))
        local_indices_by_file = self._split_indices_by_file(global_indices, file_sizes)
        feature_indices = self._feature_indices(n_vars)

        metadata = self._read_metadata(data_path, global_indices)
        var = self._read_var(data_path, n_vars).iloc[feature_indices].copy()
        data = self._read_expression(h5ad_files, local_indices_by_file, feature_indices)

        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data).astype(np.float32)
        data, var = self._pad_odd_feature(data, var)

        label_values = self._choose_label_values(metadata)
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label = label_encoder.fit_transform(label_values).astype(np.int32)

        obs = metadata.copy()
        obs["batch"] = obs["day"].astype(str) if "day" in obs else label.astype(str)
        obs["final_annotation"] = label_values
        obs["cell_type"] = label_values
        obs["celltype"] = label_values
        obs["label_id"] = label.astype(str)

        adata = anndata.AnnData(X=data, obs=obs, var=var)
        adata.var_names_make_unique()
        self.adata = adata
        self.label_encoder = label_encoder
        self.sampled_global_indices = global_indices
        self.info_list = ["batch", "final_annotation", "major_trajectory"]

        print(f"Mouse prenatal development loaded: {data.shape}")
        return adata, data, label

    def cal_near_index(self, data, label=None, k=10, device="cuda", uselabel=False, pca_dim=100):
        actual_k = min(k, max(data.shape[0] - 1, 1))
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.__class__.__name__ + str(data.shape), actual_k, uselabel, pca_dim
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
                neighbors_index, _ = index.query(X_rshaped, k=actual_k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                from sklearn.metrics import pairwise_distances

                dis = pairwise_distances(X_rshaped)
                M = np.repeat(label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : actual_k + 1]
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
        augmenter_list = [
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": 1, "knn": self.K}]),
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, "knn": self.K}]),
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
