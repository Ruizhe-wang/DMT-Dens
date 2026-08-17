import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import lightning as pl
import wandb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import uuid


def plot_arrow(fig, x0, y0, x1, y1, arrow_size=0.1, color="rgba(0, 0, 255, 0.5)"):
    """
    在图中绘制一个箭头，使用 Scatter 实现。

    :param fig: Plotly 图对象
    :param x0: 箭头起点的 x 坐标
    :param y0: 箭头起点的 y 坐标
    :param x1: 箭头终点的 x 坐标
    :param y1: 箭头终点的 y 坐标
    :param arrow_size: 箭头大小
    """
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=1),
            showlegend=False,
        )
    )

    dx, dy = x1 - x0, y1 - y0
    arrow_norm = (dx**2 + dy**2) ** 0.5
    dx /= arrow_norm
    dy /= arrow_norm

    mid_x1, mid_y1 = (x0 + x1) / 2, (y0 + y1) / 2

    arrow_x = [mid_x1, mid_x1 - arrow_size * (dx + dy), mid_x1 - arrow_size * (dx - dy)]
    arrow_y = [mid_y1, mid_y1 - arrow_size * (dy - dx), mid_y1 - arrow_size * (dy + dx)]

    # arrow_x = [x1, x1 - arrow_size * (dx + dy), x1 - arrow_size * (dx - dy)]
    # arrow_y = [y1, y1 - arrow_size * (dy - dx), y1 - arrow_size * (dy + dx)]

    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            fill="toself",
            fillcolor="blue",
            line=dict(color="blue", width=0),
            mode="lines",
            showlegend=False,
        )
    )
    return fig


def plot_path(G, tree_node_embedding, i, index_near_cluster, end_node, plotly_fig_rute):
    shortest_path = nx.shortest_path(
        G,
        source=f"L{i}/{index_near_cluster}",
        target=f"L{i}/{end_node}",
        weight="weight",
    )
    
    print(f"shortest_path: {shortest_path}")

    for index_str_path in range(len(shortest_path) - 1):
        s_node_str = shortest_path[index_str_path][1:]
        e_node_str = shortest_path[index_str_path + 1][1:]
        s_node_level, s_node_index = s_node_str.split("/")
        e_node_level, e_node_index = e_node_str.split("/")

        s_node_index = int(s_node_index)
        s_node_level = int(s_node_level)
        e_node_index = int(e_node_index)
        e_node_level = int(e_node_level)

        if e_node_level < 0:
            shortest_path[index_str_path + 1] = shortest_path[index_str_path]
        else:
            star_emb = (
                tree_node_embedding[s_node_level].detach().cpu().numpy()[s_node_index]
            )
            end_emb = (
                tree_node_embedding[e_node_level].detach().cpu().numpy()[e_node_index]
            )

            plotly_fig_rute = plot_arrow(
                plotly_fig_rute,
                star_emb[0],
                star_emb[1],
                end_emb[0],
                end_emb[1],
                color="rgba(0, 0, 255, 0.5)",
                arrow_size=0.2,
            )


def get_predict_label(tree_list, cluster_center_high_emb, data_vis):

    distances = pairwise_distances(data_vis, cluster_center_high_emb)  # (n_samples, k)
    if len(tree_list) > 0:
        last_element = tree_list[-1].reshape(-1, 1)
        mask = np.zeros((last_element.shape[0], cluster_center_high_emb.shape[0])) + 1e9
        for i_emb_vis in range(data_vis.shape[0]):
            start = int(last_element[i_emb_vis] * 2)
            end = int((last_element[i_emb_vis] + 1) * 2)
            mask[i_emb_vis, start:end] = 0

        distances += mask

    label_predict = np.argmin(distances, axis=1)

    list_count = []
    for label_i in range(label_predict.max() + 1):
        list_count.append(np.sum(label_predict == label_i))
    return label_predict


def plot_rout_single_level(
    plotly_fig_rute_dict, 
    data_vis, 
    cluster_center_high_emb, 
    cluster_center_low_emb, 
    G, 
    tree_node_embedding, 
    level, 
    index_near_cluster,
):

    plotly_fig_rute = go.Figure()

    plotly_fig_rute.add_trace(
        go.Scatter(
            x=data_vis[:, 0],
            y=data_vis[:, 1],
            mode="markers",
            marker=dict(size=1, color="grey"),
            name="data_vis",
        )
    )

    plotly_fig_rute.add_trace(
        go.Scatter(
            x=cluster_center_low_emb[:, 0],
            y=cluster_center_low_emb[:, 1],
            mode="markers",
            text=[str(i) for i in range(cluster_center_low_emb.shape[0])],
            marker=dict(
                size=5,
                color="red",
                symbol="star",
            ),
            textposition="top center",  # 控制文字相对于节点的显示位置
            textfont=dict(size=12, color="black"),
            name="cluster_center_low_emb",
        )
    )

    end_node_list = range(cluster_center_low_emb.shape[0])
    for end_node in end_node_list:
        plot_path(
            G, 
            tree_node_embedding, 
            level, 
            index_near_cluster, 
            end_node, 
            plotly_fig_rute)

    plotly_fig_rute.update_layout(
        plot_bgcolor="white",  # 设置绘图区域背景为白色
        paper_bgcolor="white",  # 设置整个图表区域背景为白色
        width=800,  # 图表宽度（像素）
        height=600,  # 图表高度（像素）
        title="Tree Visualization",
        xaxis=dict(visible=False),  # 隐藏 x 轴
        yaxis=dict(visible=False),  # 隐藏 y 轴
        xaxis_scaleanchor="y",  # 锁定 x 和 y 的比例
        yaxis_scaleanchor="x",
        legend=dict(title="Legend", x=0.01, y=0.99),
    )

    plotly_fig_rute_dict[f"tree/tree_{level}"] = plotly_fig_rute

    return plotly_fig_rute_dict

def update_graph(cluster_center_high_emb, K, level, num_all_level, G, label=None):

    num_cluster_center = cluster_center_high_emb.shape[0]
    adjacency_matrix = kneighbors_graph(
        cluster_center_high_emb,
        n_neighbors=min(K, num_cluster_center - 1),
        mode="connectivity",
        metric="euclidean",
    )
    G_ = nx.Graph(adjacency_matrix)
    mapping = {i_node: f"L{level}/{i_node}" for i_node in range(num_cluster_center)}
    G_ = nx.relabel_nodes(G_, mapping)

    edges_list = list(G_.edges)
    for i_edge in range(len(G_.edges)):
        edge = edges_list[i_edge]
        s_node = edge[0]
        e_node = edge[1]
        index_s = int(s_node.split("/")[1])
        index_e = int(e_node.split("/")[1])
        weight = 10 ** (num_all_level - level) + np.linalg.norm(
            cluster_center_high_emb[index_s] - cluster_center_high_emb[index_e]
        )
        G.add_edge(s_node, e_node, weight=weight)


    for i_node in range(cluster_center_high_emb.shape[0]):
        G.add_edge(
            f"L{level-1}/{i_node//2}",
            f"L{level}/{i_node}",
            weight=(10 ** (num_all_level - level))*3,
        )
    return G


def plot_multi_level_rout_with_plotly(
    cluster_center_high_multi_level, 
    cluster_center_low_multi_level, 
    data_high, 
    data_vis, 
    K=3, 
    label=None,
    centre_node=np.array([-2.5, -1.5]),
):
    G = nx.Graph()
    G.add_node("L-1/0")

    plotly_fig_rute_dict = {}
    
    num_all_level = len(cluster_center_high_multi_level)
    for level in range(8):
        cluster_center_high_emb = cluster_center_high_multi_level[level].detach().cpu().numpy()
        cluster_center_low_emb = cluster_center_low_multi_level[level].detach().cpu().numpy()

        G = update_graph(cluster_center_high_emb, K, level, num_all_level, G, label=label)

        if level > 3:
            distance_to_zero = np.linalg.norm(cluster_center_high_emb - centre_node, axis=1)
            index_near_cluster = np.argmin(distance_to_zero)
            plotly_fig_rute_dict = plot_rout_single_level(
                plotly_fig_rute_dict,
                data_vis,
                cluster_center_high_emb,
                cluster_center_low_emb,
                G,
                cluster_center_low_multi_level,
                level,
                index_near_cluster,
            )

    return plotly_fig_rute_dict


class VisualizationTrace(pl.Callback):
    def __init__(self, output_dir="output",centre_node=[-20, 10]):
        # adata = adata
        self.output_dir = output_dir
        self.centre_node = centre_node
        os.makedirs(output_dir, exist_ok=True)

    def get_tree_and_emb(self, trainer, pl_module, down_sample=10000):

        data_list = []
        data_high = []
        tree_rout_list = []
        vector_rout_list = []

        for batch in trainer.datamodule.val_dataloader():

            data_input_item = batch["data_input_item"].to(pl_module.device)
            index = batch["index"]

            x_masked, lat_high_dim, lat_vis, lat_vis_list = pl_module(
                data_input_item,
                tau=pl_module.hparams.tau,  # Access hyperparameters from the model
            )

            tree_rout, vector_rout, loss_rout = pl_module.router_forward(
                lat_high_dim.float().detach(),
                tree_rout_bool=True,
                ec_ce_weight=pl_module.hparams.ec_ce_weight,
            )

            data_list.append(lat_vis_list[-1])
            tree_rout_list.append(tree_rout)
            vector_rout_list.append(vector_rout)
            data_high.append(lat_high_dim)

        data = torch.cat(data_list, dim=0)
        tree_rout = torch.cat(tree_rout_list, dim=0)
        vector_rout = torch.cat(vector_rout_list, dim=0)
        data_high = torch.cat(data_high, dim=0)
        


        return data, data_high, tree_rout, vector_rout

    def on_validation_epoch_end(self, trainer, pl_module, down_sample=5000):

        data_vis, data_high, tree_rout, vector_rout = self.get_tree_and_emb(trainer, pl_module)
        
        if data_vis.shape[0] > down_sample:
            indices = torch.randperm(data_vis.shape[0])[:down_sample]
            data_vis = data_vis[indices].detach().cpu().numpy()
            data_high = data_high[indices].detach().cpu().numpy()
            tree_rout = tree_rout[indices].detach().cpu().numpy()
            vector_rout = vector_rout[indices].detach().cpu().numpy()

            adata = trainer.datamodule.adata
            adata = adata[indices.detach().cpu().numpy(), :].copy()
            label = adata.obs["cell_type"].to_numpy()

        cluster_center_high_multi_level = []
        cluster_center_low_multi_level = []
        for i in range(len(pl_module.tree_node_embedding)):
            node_high = pl_module.tree_node_embedding[i].weight
            node_low = pl_module.vis_list[3](node_high)
            cluster_center_high_multi_level.append(node_high)
            cluster_center_low_multi_level.append(node_low)

        centre_node = data_high[label == 'undiff'].mean(axis=0)
        plotly_fig_rute_dict = plot_multi_level_rout_with_plotly(
            cluster_center_high_multi_level,
            cluster_center_low_multi_level,
            data_high,
            data_vis,
            centre_node=centre_node,
            label=label,
            K=5,
        )

        trainer.logger.experiment.log(plotly_fig_rute_dict)
