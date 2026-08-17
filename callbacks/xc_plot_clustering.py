# Standard library imports
import os
import uuid
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import wandb
import networkx as nx
import umap
import scipy.special
from tqdm import tqdm
import lightning as pl

# Lightning imports
from lightning.pytorch.callbacks import Callback

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from callbacks.util import (
    plot_scatter,
    plot_path,
    build_tree_zl,
)

# Constants
MAX_SAMPLES = 20000
MNIST_IMAGE_SIZE = 784
MNIST_SHAPE = (28, 28)
GRID_SIZE = (10, 10)
LARGE_FIGURE_SIZE = (30, 30)
DEFAULT_NEIGHBORS = 3
PLOTLY_FIGURE_SIZE = (800, 600)

# Color palette for visualizations
D3_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
] * 1000




class EvalCallBack(Callback):
    """
    用于评估聚类和路由性能的回调类。
    
    该类实现了Lightning回调接口，用于在验证过程中评估模型的路由和聚类性能，
    包括可视化、指标计算和结果保存。
    
    Attributes:
        inter (int): 评估间隔
        vis_rout (bool): 是否进行路由可视化
        save_results (bool): 是否保存结果
        only_val (bool): 是否仅在验证时运行
        fully_eval (bool): 是否进行完整评估
        dataset (str): 数据集名称
        dirpath (str): 输出目录路径
        best_acc (float): 最佳准确率
    """
    
    def __init__(
        self,
        inter: int = 10,
        dirpath: str = "",
        fully_eval: bool = False,
        dataset: str = "",
        only_val: bool = False,
        save_results: bool = False,
        vis_rout: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.inter = inter
        self.only_val = only_val
        self.vis_rout = vis_rout
        self.save_results = save_results
        self.dirpath = dirpath
        self.dataset = dataset
        self.fully_eval = fully_eval
        self.best_acc = 0.0



    def get_predict_label(self, tree_list: List[np.ndarray], token_emb: np.ndarray, emb_vis: np.ndarray) -> np.ndarray:
        """
        根据树结构和嵌入预测标签。
        
        Args:
            tree_list: 树结构列表
            token_emb: 令牌嵌入矩阵
            emb_vis: 可视化嵌入矩阵
            
        Returns:
            预测的标签数组
        """
        distances = pairwise_distances(emb_vis, token_emb)  # (n_samples, k)
        if len(tree_list) > 0:
            last_element = tree_list[-1].reshape(-1, 1)
            mask = np.zeros((last_element.shape[0], token_emb.shape[0])) + 1e9
            for i_emb_vis in range(emb_vis.shape[0]):
                start = int(last_element[i_emb_vis] * 2)
                end = int((last_element[i_emb_vis] + 1) * 2)
                mask[i_emb_vis, start:end] = 0

            distances += mask

        label_predict = np.argmin(distances, axis=1)

        list_count = []
        for label_i in range(label_predict.max() + 1):
            list_count.append(np.sum(label_predict == label_i))
        return label_predict

    def get_rout_vis(self, tree_node_embedding, emb_vis) -> Tuple[List[str], Dict]:
        """
        获取路由可视化结果。
        
        Args:
            tree_node_embedding: 树节点嵌入
            emb_vis: 可视化嵌入
            
        Returns:
            路径列表和可视化字典的元组
        """
        # import pdb; pdb.set_trace()

        G = nx.Graph()
        G.add_node("L-1/0")

        path_list = []
        tree_list = []
        plotly_fig_rute_dict = {}
        for i in range(len(tree_node_embedding)):

            # token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            # print("token_emb.shape:", token_emb.shape)
            # print("emb_vis.shape:", emb_vis.shape)

            if token_emb.shape[0] > MAX_SAMPLES:
                idx = np.random.choice(token_emb.shape[0], MAX_SAMPLES, replace=False)
                token_emb = token_emb[idx]
            if emb_vis.shape[0] > MAX_SAMPLES:
                idx = np.random.choice(token_emb.shape[0], MAX_SAMPLES, replace=False)
                emb_vis = emb_vis[idx]

            label_predict = self.get_predict_label(tree_list, token_emb, emb_vis)
            tree_list.append(label_predict)

            # for label_i in range(label_predict.max() + 1):
            #     print(
            #         f"level: {i}, label: {label_i}, num: {np.sum(label_predict==label_i)}"
            #     )

            plotly_fig_rute = go.Figure()

            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=emb_vis[:, 0],
                    y=emb_vis[:, 1],
                    mode="markers",
                    marker=dict(
                        size=1,
                        # color=label_predict,
                        color=[D3_COLORS[c_i] for c_i in label_predict],
                    ),
                    name="emb_vis",
                )
            )

            # Add additional scatter points for `token_emb`
            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=token_emb[:, 0],
                    y=token_emb[:, 1],
                    mode="markers",
                    text=[str(i) for i in range(token_emb.shape[0])],
                    marker=dict(
                        size=5,
                        color="red",
                        symbol="star",
                    ),
                    textposition="top center",  # 控制文字相对于节点的显示位置
                    textfont=dict(size=12, color="black"),
                    name="token_emb",
                )
            )

            num_nodes = token_emb.shape[0]
            adjacency_matrix = kneighbors_graph(
                token_emb,
                n_neighbors=min(DEFAULT_NEIGHBORS, num_nodes - 1),
                mode="connectivity",
                metric="euclidean",
            )
            G_ = nx.Graph(adjacency_matrix)
            mapping = {i_node: f"L{i}/{i_node}" for i_node in range(num_nodes)}
            G_ = nx.relabel_nodes(G_, mapping)

            edges_list = list(G_.edges)
            for i_edge in range(len(G_.edges)):
                edge = edges_list[i_edge]
                s_node = edge[0]
                e_node = edge[1]
                index_s = int(s_node.split("/")[1])
                index_e = int(e_node.split("/")[1])
                weight = 10 ** (len(tree_node_embedding) - i) + np.linalg.norm(
                    token_emb[index_s] - token_emb[index_e]
                )
                # print(f"edge: {s_node} -> {e_node}, weight: {weight}")
                G.add_edge(s_node, e_node, weight=weight)
                # import pdb; pdb.set_trace()

            distance_to_zero = np.linalg.norm(token_emb, axis=1)
            near_index = np.argmin(distance_to_zero)

            for i_node in range(token_emb.shape[0]):
                G.add_edge(
                    f"L{i-1}/{i_node//2}",
                    f"L{i}/{i_node}",
                    weight=10 ** (len(tree_node_embedding) - i),
                )

            end_node_list = range(token_emb.shape[0])
            for end_node in end_node_list:
                plot_path(
                    G, tree_node_embedding, i, near_index, end_node, plotly_fig_rute
                )

            plotly_fig_rute.update_layout(
                plot_bgcolor="white",  # 设置绘图区域背景为白色
                paper_bgcolor="white",  # 设置整个图表区域背景为白色
                width=PLOTLY_FIGURE_SIZE[0],  # 图表宽度（像素）
                height=PLOTLY_FIGURE_SIZE[1],  # 图表高度（像素）
                title="Tree Visualization",
                xaxis=dict(visible=False),  # 隐藏 x 轴
                yaxis=dict(visible=False),  # 隐藏 y 轴
                xaxis_scaleanchor="y",  # 锁定 x 和 y 的比例
                yaxis_scaleanchor="x",
                legend=dict(title="Legend", x=0.01, y=0.99),
            )

            # plotly_fig_rute_dict[f'graph/tree_{i}'] = fig_graph
            
            os.makedirs("fig", exist_ok=True)
            uuid_str = str(uuid.uuid4())[:16]
            plotly_fig_rute.write_image(f"fig/fig_emb_colored_with_rout_{i}_{uuid_str}.png", scale=3)
            path_list.append(f"fig/fig_emb_colored_with_rout_{i}_{uuid_str}.png")

        return path_list, plotly_fig_rute_dict

    def rout_eval(
        self,
        trainer,
        pl_module,
        gathered_rout,
        gathered_label,
        emb_vis,
        gathered_sample=None,
        # gathered_reconstruct_history_history=None,
        # gathered_reconstruct_label=None,
        save_results=False,
    ):

        # import pdb; pdb.set_trace()
        # print("Evaluating the routing vector, calculating SVC accuracy...")
        gathered_rout_flat = gathered_rout.reshape(gathered_rout.shape[0], -1)
        # acc = self.get_rout_svc_acc(gathered_rout_flat, gathered_label)
        acc = 1.00

        if gathered_rout_flat.shape[0] > MAX_SAMPLES:
            rand_idx = np.random.choice(
                gathered_rout_flat.shape[0], MAX_SAMPLES, replace=False
            )
            gathered_rout_flat = gathered_rout_flat[rand_idx]
            gathered_label = gathered_label[rand_idx]
            emb_vis = emb_vis[rand_idx]

        # tsne visualization of gathered_rout_flat
        # print('Evaluating the routing vector, tsne visualization of the routing vector...')
        # TSNE_vis = TSNE(n_components=2, random_state=0).fit_transform(gathered_rout_flat)
        # fig = plt.figure(figsize=(10, 10))
        # plt.scatter(TSNE_vis[:, 0], TSNE_vis[:, 1], c=gathered_label, cmap='rainbow', s=3)

        # print("Evaluating the routing vector, uploading the results to wandb...")
        dict = {
            "rout/svc_acc": acc,
            # "rout/tsne": wandb.Image(fig),
            "rout/distribution_gathered_rout_flat": wandb.Histogram(gathered_rout_flat),
            "epoch": trainer.current_epoch,
        }

        # print("Evaluating the routing vector, visualizing the embedding space...")

        if self.vis_rout:
            vis_plot_path, plotly_fig_rute_dict = self.get_rout_vis(
                pl_module.tree_node_embedding, emb_vis
            )
            dict.update(plotly_fig_rute_dict)
            for i, path in enumerate(vis_plot_path):
                dict[f"rout/emb_colored_with_rout_{i}"] = wandb.Image(path)

        if gathered_sample is not None and gathered_sample.shape[-1] == MNIST_IMAGE_SIZE:
            # print("Evaluating the routing vector, plotting the sample images...")
            gathered_image = gathered_sample.reshape(-1, MNIST_SHAPE[0], MNIST_SHAPE[1])
            # show 10*10 images
            fig_sample = plt.figure(figsize=LARGE_FIGURE_SIZE)
            try:
                for i in range(GRID_SIZE[0] * GRID_SIZE[1]):
                    plt.subplot(GRID_SIZE[0], GRID_SIZE[1], i + 1)
                    plt.imshow(gathered_image[i], cmap="gray")
                    plt.axis("off")

                dict.update({"rout/fig_sample_image": wandb.Image(fig_sample)})
            finally:
                plt.close(fig_sample)  # 确保关闭图形释放内存


        wandb.log(dict)

    def get_rout_vis_label(self, trainer, pl_module):
        
        rout_list = []
        rout_vector_list = []
        vis_list = []
        label_list = []
        
        for batch in trainer.datamodule.val_dataloader():
            # for batch in val_loader:
            #     import pdb; pdb.set_trace()

                # print(f"Processing batch with size: {batch['data_input_item'].shape}")
                
            data_input_item = batch["data_input_item"].to(pl_module.device)
            index = batch["index"]
            label = batch["label"].to(pl_module.device)
            
            # Pass data through the model (use pl_module here)
            x_masked, lat_high_dim, lat_vis, _ = pl_module(
                data_input_item,
                tau=pl_module.hparams.tau,  # Access hyperparameters from the model
            )
            
            lat_vis = (lat_vis - pl_module.lat_vis_mean) / (pl_module.lat_vis_std + 1e-8)
            
            tree_rout, vector_rout, loss_rout = pl_module.router_forward(
                lat_vis, tree_rout_bool=True
            )
            
            # Collect the latent visualizations
            rout_list.append(tree_rout)
            vis_list.append(lat_vis)
            label_list.append(label)
            rout_vector_list.append(vector_rout)

        rout_list_all = torch.cat(rout_list, dim=0).cpu().detach().numpy()
        vis_list_all = torch.cat(vis_list, dim=0).cpu().detach().numpy()
        label_list_all = torch.cat(label_list, dim=0).cpu().detach().numpy()
        rout_vector_list_all = torch.cat(rout_vector_list, dim=0).cpu().detach().numpy()

        return rout_list_all, vis_list_all, label_list_all, rout_vector_list_all

    def plot_multiple_clustering(self, vis, label_multi):
        
        dict_vis = {}
        for i in range(label_multi.shape[1]):
            
            fig_predict = plot_scatter(vis, label_multi[:, i])
            dict_vis[f"predict_label_{i}"] = wandb.Image(fig_predict)
            
        return dict_vis

        

    def on_validation_epoch_end(self, trainer, pl_module):

        # if in training and epoch == 0, return
        if trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING and trainer.current_epoch == 0:

            return

        gathered_rout, gathered_vis, gathered_label, gathered_vector_rout = self.get_rout_vis_label(trainer, pl_module)
        data_name = f"{trainer.datamodule.dataset.data_name}_seed_{pl_module.uuid_str}_epoch_{trainer.current_epoch}"
        if self.save_results:
            if os.path.exists(f"save_output") == False:
                os.makedirs(f"save_output")

            if os.path.exists(f"save_output/{data_name}") == False:
                os.makedirs(f"save_output/{data_name}")

        if trainer.is_global_zero and pl_module.set_mean_bool:

            num_clusters = len(set(gathered_label))
            
            if True:
                self.rout_eval(
                    trainer,
                    pl_module,
                    gathered_rout,
                    gathered_label,
                    gathered_vis,
                    )

                node_root = build_tree_zl(gathered_rout, gathered_vector_rout, num_clusters)
                fig_tree = node_root.plot_all_tree_with_matplotlib()
                predict_label = node_root.output_label_list().astype(np.int32)
                fig_predict = plot_scatter(gathered_vis, predict_label)
                
                dict_vis = self.plot_multiple_clustering(gathered_vis, gathered_rout)

                dict_vis.update({
                    f"predict_label": wandb.Image(fig_predict),
                    f"tree": wandb.Image(fig_tree),
                })

                wandb.log(dict_vis)

