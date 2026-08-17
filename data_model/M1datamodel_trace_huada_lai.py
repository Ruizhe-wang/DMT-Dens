import os

import joblib
import lightning as pl
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from pynndescent import NNDescent
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from data_model.aug import Augmenter
from data_model.dataset_base_multi_level import DataSetBaseMultiLevel
from data_model.sampler import MultiClusterBatchSampler
from sklearn.cluster import KMeans



def flatten_mnist_data(dataset, ):
    """
    将MNIST数据集中的图像展平为向量，并提取标签。
    """
    data_list, label_list = [], []
    for img, label in dataset:
        data_list.append(img.numpy().squeeze())
        label_list.append(label)
    return np.stack(data_list).reshape((-1, 784)), np.array(label_list)

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
        top_genes: int = None,  # 添加top_genes参数
        use_cache: bool = True,  # 添加缓存控制参数
        n_hop=1,  # 添加n_hop参数
        # unique_cell_types = None,  # 添加unique_cell_types参数
        **kwargs  # 添加kwargs来接收其他数据集特定参数
    ):
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
        self.top_genes = top_genes  # 保存top_genes参数
        self.use_cache = use_cache  # 保存缓存控制参数
        self.dataset_kwargs = kwargs  # 保存额外的数据集参数
        self.sample_data_size = sample_data_size  # 默认采样数据大小
        self.n_hop = n_hop
        # self.unique_cell_types = unique_cell_types
        super().__init__()
        


    # def load_data_from_raw(self, sample_data_size=60000):

    #     adata_raw = sc.read_h5ad("/zangzelin/data/qced/cellbin_ed.h5ad")
    #     adata = anndata.AnnData(
    #         X=adata_raw.layers['counts'].copy(),
    #         obs=adata_raw.obs.copy(),var=adata_raw.var.copy())
    #     # del adata.obsm['X_pca']
    #     # import pdb; pdb.set_trace()
        
    #     conts = adata.X.sum(axis=1)
    #     mask_counts = (conts < 1500) & (conts > 800)
    #     adata = adata[mask_counts, :].copy()
    #     conts = adata.X.sum(axis=1)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(conts, bins=60, edgecolor='black')
    #     plt.title("Histogram of conts")
    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.savefig("conts_histogram.png")
        
    #     sc.pp.highly_variable_genes(adata, n_top_genes=60, flavor='seurat_v3')
    #     adata = adata[:, adata.var.highly_variable]
    #     sc.pp.normalize_total(adata, target_sum=1e4)
    #     sc.pp.log1p(adata)
    #     sc.pp.scale(adata, max_value=10)
    #     sc.tl.pca(adata, svd_solver='arpack')
        
    #     # leiden 
    #     sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    #     sc.tl.leiden(adata, resolution=10)
        
    #     cell_types = adata.obs['leiden']
    #     unique_cell_types = sorted(cell_types.unique())
    #     cell_type_to_num = {cell_type: i for i, cell_type in enumerate(unique_cell_types)}
        
    #     rng = np.random.default_rng(seed=42)
    #     down_sample_index_data = rng.choice(adata.shape[0], sample_data_size, replace=False)
    #     cell_types_sampled = adata[down_sample_index_data].obs['leiden']
    #     LABEL = np.array([cell_type_to_num[ct] for ct in cell_types_sampled], dtype=np.int32)
    #     # LABEL = conts[down_sample_index_data] # np.log1p(1+)
    #     # import pdb; pdb.set_trace()

    #     X_top_genes = np.asarray(adata.obsm['X_pca'])[down_sample_index_data]
        
    #     # 保存adata用于可视化
    #     self.adata_M2 = adata[down_sample_index_data, :].copy()
    #     self.down_sample_index_data = down_sample_index_data
        
    #     return X_top_genes, LABEL

    def load_data(self, data_path, sample_data_size=60000):
        
        adata = sc.read_h5ad("/root/hanning/mulleveldr/m1data/huada_lai.h5ad")

        # import pdb; pdb.set_trace()
        np.random.seed(42)
        # down_sample_index = np.random.choice(
        #     adata.shape[0], min(adata.shape[0], 600000), replace=False)
        # adata_tmp = adata[down_sample_index, :].copy()

        # # 选高变基因
        # X_matrix = adata_tmp.X
        # if issparse(X_matrix):
        #     X_dense = X_matrix.todense()
        # else:
        #     X_dense = X_matrix
        # vars = np.var(np.array(X_dense), axis=0)
        # mask_gene = np.argsort(vars).reshape(-1)[-500:]
        X_top500 = adata.obsm['X_pca']

        # cluster with kmeans
        # kmeans = KMeans(n_clusters=10, random_state=0).fit(X_top500)
        # adata.obs['kmeans_10'] = kmeans.labels_.astype(str)
        
        # kmeans = KMeans(n_clusters=25, random_state=0).fit(X_top500)
        # adata.obs['kmeans_25'] = kmeans.labels_.astype(str)
        
        adata.obs['batch'] = adata.obs['batch1'].astype(str)
        adata.obs['final_annotation'] = adata.obs['Cell type'].astype(str)
        adata.obs['cell_type'] = adata.obs['Cell type'].astype(str)
        self.info_list = ['batch', 'cell_type']

        # import pdb; pdb.set_trace()
        
        # X_top500_mean = X_top500.mean()
        # X_top500_std = X_top500.std()
        
        # X_top500 = (X_top500 - X_top500_mean) / X_top500_std
        
        # str_label = adata_tmp.obs['sample_id'].to_list()

        LABEL_str = adata.obs['final_annotation'].to_list()
        set_annotations = sorted(set(LABEL_str))
        annotation_to_num = {name: i for i, name in enumerate(set_annotations)}
        LABEL = np.array([annotation_to_num[name] for name in LABEL_str], dtype=np.int32)
        
        
        return adata, X_top500, LABEL

    def symmetrize_union(self, neighbors_index, n_samples):
        """
        邻居索引对称化 (并集方式, 稀疏表示)
        """
        rows = np.repeat(np.arange(n_samples), neighbors_index.shape[1])
        cols = neighbors_index.ravel()
        data = np.ones_like(cols, dtype=bool)

        # 稀疏邻接矩阵 (有向)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

        # 并集对称化：A->B 或 B->A 存在就置为 True
        adj = adj.maximum(adj.T).tocsr()
        return adj

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
                # adj = self.symmetrize_union(neighbors_index, X_rshaped.shape[0])
                # import pdb; pdb.set_trace()
                
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

    def adj_to_neighbors_index(self, adj):
        """
        将邻接矩阵转为邻居索引数组

        参数
        ----
        adj : scipy.sparse.csr_matrix 或 ndarray
            对称化后的邻接矩阵
        k : int 或 None
            每个点取多少个邻居；None 表示保留所有邻居

        返回
        ----
        neighbors_index : ndarray, shape (n_samples, k)
        """
        n_samples = adj.shape[0]
        neighbors_index = []

        for i in range(n_samples):
            neigh = adj[i].indices if hasattr(adj[i], "indices") else np.where(adj[i])[0]
            # if k is not None:
            #     neigh = neigh[:k]  # 如果有需要限制前 k 个
            neighbors_index.append(neigh)
        
        # pad 成矩阵（不等长时补 -1）
        max_len = max(len(n) for n in neighbors_index)
        mat = -np.ones((n_samples, max_len), dtype=int)
        for i, neigh in enumerate(neighbors_index):
            mat[i, :len(neigh)] = neigh
        
        return mat


    def sample_multi_level_data(self, data, label, sample_size_list = [2000, 60000]):
        
        data_list = []
        label_list = []
        neighbors_index_list = []
        for i, sample_size in enumerate(sample_size_list):
            if data.shape[0] < sample_size:
                raise ValueError(f"Sample size {sample_size} is larger than the dataset size {data.shape[0]}.")
            
            if sample_size == -1:
                sample_size = data.shape[0]

            if i == len(sample_size_list) - 1:
                # 最后一层使用全部数据
                sampled_data = data
                sampled_label = label
            else:
                index = np.random.choice(data.shape[0], sample_size, replace=False)
                sampled_data = data[index]
                sampled_label = label[index]
            
            neighbors_index = self.cal_near_index(
                data=sampled_data,
                k=self.K,
                device="cuda",
                uselabel=self.uselabel,
                pca_dim=self.pca_dim,
            )
            data_list.append(sampled_data)
            label_list.append(sampled_label)
            neighbors_index_list.append(neighbors_index)
        return data_list, label_list, neighbors_index_list


    def setup(self, stage: str):
        
        adata, data, label = self.load_data(self.data_path)
        self.adata = adata
        
        augmenter_list = [
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": 1, 'knn': self.K}]), 
            Augmenter([{"name": "TraceMixup", "alpha_max": 1.0, "p": 1.0, "n_hop": self.n_hop, 'knn': self.K}])
        ]
        
        self.adata = adata
        # self.data = data
        # self.label = label

        # data_list, label_list, neighbors_list = self.sample_multi_level_data(
        #     data=data,
        #     label=label,
        #     sample_size_list=[1000, -1]
        # )
        
        neighbors_index = self.cal_near_index(
            data=data,
            k=self.K,
            device="cuda",
            uselabel=self.uselabel,
            pca_dim=self.pca_dim,
        )
        
        self.sampler = MultiClusterBatchSampler(
            neighbors_index=neighbors_index,
            clusters_per_batch=100,
            cluster_size=10,
            num_batches=100
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
            # batch_sampler=self.sampler,
        )

    def val_dataloader(self):
        val1 = DataLoader(
            self.dataset,
            drop_last=False,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        return val1

    def test_dataloader(self):
        
        return DataLoader(
            self.dataset,
            drop_last=True,
            batch_size=min(self.batch_size, self.dataset.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        
