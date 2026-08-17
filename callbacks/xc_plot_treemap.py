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
from anytree import Walker

import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import uuid

from anytree import Node, RenderTree
from anytree.search import find_by_attr, find, findall_by_attr
import anytree

import numpy as np
import torch
from anytree import PostOrderIter, LevelOrderIter

# from anytree.util import commonancestor
import plotly.express as px
import random

from scipy.interpolate import make_interp_spline, interp1d
import numpy as np
from scipy.ndimage import gaussian_filter1d


color_list = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#ad494a",
    "#8c6d31",
    "#843c39",
    "#636363",
]

def smooth_path_spline(x_coords, y_coords, sigma=2):
    """高斯平滑但保留起点和终点"""

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    if len(x_coords) <= 2:
        return x_coords, y_coords

    # 保存端点
    start_x, start_y = x_coords[0], y_coords[0]
    end_x, end_y = x_coords[-1], y_coords[-1]

    # 对中间部分进行平滑
    x_smooth = gaussian_filter1d(x_coords, sigma=sigma, mode="nearest")
    y_smooth = gaussian_filter1d(y_coords, sigma=sigma, mode="nearest")

    # 强制恢复端点
    x_smooth[0] = start_x
    y_smooth[0] = start_y
    x_smooth[-1] = end_x
    y_smooth[-1] = end_y

    return x_smooth, y_smooth


def get_effective_vis_emb(Graph_g, node, vis_emb_c, end_point):

    p_vis_emb = np.array(Graph_g.nodes[node]["p_vis_emb"])

    p_vis_emb[np.isnan(p_vis_emb)] = 100000

    dists_a = pairwise_distances(p_vis_emb, vis_emb_c.reshape(1, -1))
    dists_b = pairwise_distances(p_vis_emb, end_point.reshape(1, -1))
    dists = dists_a + dists_b
    dists[dists_a < 1e-6] = dists.max() + 1
    index_min = np.argmin(dists)
    # import ipdb; ipdb.set_trace()

    return p_vis_emb[index_min], dists[index_min]


def pick_random_deepest_node(root, target_label, seed=42):
    # 找到所有匹配 major_class 的节点
    cands = findall_by_attr(root, name="major_class", value=target_label)
    if not cands:
        return None

    # 找到最大的深度
    max_depth = max(len(n.ancestors) for n in cands)

    # 过滤出所有最深的节点
    deepest_nodes = [n for n in cands if len(n.ancestors) == max_depth]

    random.seed(seed)
    # 在这些节点里随机选一个
    return random.choice(deepest_nodes)


def anytree_to_networkx(root, n_neighbors=5, build_graph_knn_key="high_emb"):
    Graph_g = nx.Graph()  # 使用有向图
    # node_names = list(Graph_g.nodes())
    num_level = 10

    # 遍历所有节点
    for node in root.descendants:
        Graph_g.add_node(
            node.name,
            vis_emb=node.vis_emb if hasattr(node, "vis_emb") else None,
            high_emb=node.high_emb if hasattr(node, "high_emb") else None,
            label_distribution=(
                node.label_distribution if hasattr(node, "label_distribution") else None
            ),
            major_class=node.major_class if hasattr(node, "major_class") else None,
        )
        level = int(node.name.split(":")[0][1:])
        if node.parent and level > 5:
            weight = 40 ** (num_level - level + 1)
            Graph_g.add_edge(node.parent.name, node.name, weight=weight)
            # print(
            #     f"Add edge between {node.parent.name} and {node.name}, weight={weight}"
            # )

    # 添加根节点
    # Graph_g.add_node(root.name)
    node_names = list(Graph_g.nodes())
    for level in range(num_level + 1):
        nodes_c_level = [n for n in node_names if n.startswith(f"L{level}:")]
        list_emb = [
            (
                Graph_g.nodes[node_name][build_graph_knn_key]
                if build_graph_knn_key in Graph_g.nodes[node_name]
                else np.array([1e9, 1e9])
            )
            for node_name in nodes_c_level
        ]
        numpy_emb = np.array(list_emb)

        # cal knn in list list_emb
        weight_knn = 20 ** (num_level - level)
        if len(numpy_emb) > 41:
            knn_graph = kneighbors_graph(
                numpy_emb,
                n_neighbors=n_neighbors,
                include_self=True,
                metric="euclidean",
            )
            # knn_graph = radius_neighbors_graph(
            #     numpy_emb,
            #     radius=2,
            #     mode='connectivity',  # 'connectivity' 或 'distance'
            #     metric='euclidean',
            #     include_self=True
            # )

            mask_over_large = numpy_emb.sum(axis=1) > 1e9
            if mask_over_large.sum() > 0:
                # import ipdb; ipdb.set_trace()
                knn_graph[mask_over_large, :] = 0
                knn_graph[:, mask_over_large] = 0

            knn_graph = knn_graph.toarray()
            # import ipdb; ipdb.set_trace()

            num_node = numpy_emb.shape[0]
            for i in range(num_node):
                for j in range(num_node):
                    if i > j and knn_graph[i, j]:
                        Graph_g.add_edge(
                            nodes_c_level[i], nodes_c_level[j], weight=weight_knn
                        )
                        # print(
                        #     f"Add knn edge between {nodes_c_level[i]} and {nodes_c_level[j]}, weight={weight_knn}"
                        # )
    # import ipdb; ipdb.set_trace()
    return Graph_g


def _sum_value_from_label_dist(ld):
    """根据 label_distribution 计算节点 value（面积）"""
    if ld is None:
        return None
    # dict: {label: count}
    if isinstance(ld, dict):
        return int(sum(int(v) for v in ld.values()))
    # numpy
    if isinstance(ld, np.ndarray):
        return int(np.sum(ld))
    # torch
    if torch.is_tensor(ld):
        return int(ld.detach().cpu().sum().item())
    # list/tuple: 当作计数字符串/值相加；若为标签列表，则以长度为计数
    if isinstance(ld, (list, tuple)):
        # 如果是计数字典被转成 (k, v) 列表请自行改造，这里默认当作样本集合
        try:
            return int(sum(int(v) for v in ld))  # 若是数值列表
        except Exception:
            return int(len(ld))  # 若是标签列表，取长度
    # 其他类型：尝试转 int
    try:
        return int(ld)
    except Exception:
        return None


def _label_dist_to_str(ld, top_k=5):
    """把 label_distribution 转成 hover 友好的字符串"""
    if ld is None:
        return "—"
    if isinstance(ld, dict):
        # 取前 top_k 个最大的项
        items = sorted(ld.items(), key=lambda x: x[1], reverse=True)[:top_k]
        tot = sum(ld.values()) if ld else 0
        parts = [f"{k}: {v} ({v/tot:.1%})" if tot else f"{k}: {v}" for k, v in items]
        return "; ".join(parts)
    if isinstance(ld, np.ndarray):
        tot = int(np.sum(ld))
        # 显示前 top_k 个非零索引
        nz = np.where(ld > 0)[0]
        top = sorted([(i, int(ld[i])) for i in nz], key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        parts = [f"{i}: {v} ({v/tot:.1%})" if tot else f"{i}: {v}" for i, v in top]
        return "; ".join(parts) if parts else f"sum={tot}"
    if torch.is_tensor(ld):
        ld = ld.detach().cpu().numpy()
        return _label_dist_to_str(ld, top_k=top_k)
    if isinstance(ld, (list, tuple)):
        return f"len={len(ld)}"
    return str(ld)


def build_treemap_figure_from_anytree(
    root, title="Tree Treemap", color_by_major_label=False, max_depth=None
):
    """
    把 anytree 的 root 转成 Plotly Treemap 并返回 fig。
    - 面积 values 来自 label_distribution 的总数；若父节点缺失则自动用子节点之和回填。
    - hover 展示节点名和标签分布摘要。
    - color_by_major_label=True 时，用主导标签（分布中最大计数的键）给节点着色。
    - max_depth 控制绘制的最大层数（root 深度=0）；None 表示不限制。
    """
    # 先后序遍历计算每个节点的 value
    node_value, node_major = {}, {}
    for node in PostOrderIter(root):
        v = _sum_value_from_label_dist(getattr(node, "label_distribution", None))
        if v is None:
            v = sum(node_value.get(child, 0) for child in node.children)
        node_value[node] = int(v) if v is not None else 0

        ld = getattr(node, "label_distribution", None)
        major = None
        if isinstance(ld, dict) and len(ld) > 0:
            major = max(ld.items(), key=lambda x: x[1])[0]
        node_major[node] = major

    labels, parents, values, hover_texts, colors = [], [], [], [], []

    for node in LevelOrderIter(root):
        depth = len(node.ancestors)  # root 的深度=0
        if max_depth is not None and depth > max_depth:
            continue  # 超过层级就不画

        labels.append(node.name)
        parents.append("" if node.is_root else node.parent.name)
        values.append(node_value[node])
        hover_texts.append(
            _label_dist_to_str(getattr(node, "label_distribution", None))
        )
        colors.append(node_major[node])

    data = {
        "labels": labels,
        "parents": parents,
        "values": values,
        "hover": hover_texts,
        "major": colors,
    }

    if color_by_major_label:
        fig = px.treemap(
            data,
            names="labels",
            parents="parents",
            values="values",
            color="major",
            hover_data={"hover": True, "major": True, "values": True, "parents": False},
            title=title,
        )
    else:
        fig = px.treemap(
            data,
            names="labels",
            parents="parents",
            values="values",
            hover_data={"hover": True, "values": True, "parents": False},
            title=title,
        )

    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>"
        + "Value: %{value}<br>"
        + "%{customdata[0]}<extra></extra>",
        customdata=np.array(list(zip(data["hover"]))),
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


class PlotTreeMap(pl.Callback):
    def __init__(
        self,
        # vis_index=0,
        output_dir="output",
        # centre_node=[-20, 10],
        # lineage_start=[
        #     0, 0, 0, 0, 0, 0
        # ],
        # lineage_end=[1, 1, 1, 1, 1, 1],
    ):
        # adata = adata
        self.output_dir = output_dir
        # self.centre_node = centre_node
        # self.vis_index = vis_index

        # self.lineage_start = lineage_start
        # self.lineage_end = lineage_end
        os.makedirs(output_dir, exist_ok=True)

    def get_tree_and_emb(self, trainer, pl_module, vis_index=0, down_sample=10000):

        data_list = []
        data_high = []
        tree_rout_list = []
        vector_rout_list = []

        for batch in trainer.datamodule.val_dataloader():

            data_input_item = batch["data_input_item"].to(pl_module.device)
            index = batch["index"]

            x_masked, lat_high_dim, lat_vis, lat_vis_list = pl_module(
                data_input_item,
                tau=pl_module.hparams.tau,
            )
            # import pdb; pdb.set_trace()

            tree_rout, vector_rout, loss_rout = pl_module.router_forward(
                lat_high_dim.float().detach(),
                tree_rout_bool=True,
                ec_ce_weight=pl_module.hparams.ec_ce_weight,
            )

            data_list.append(lat_vis_list[vis_index])
            tree_rout_list.append(tree_rout)
            vector_rout_list.append(vector_rout)
            data_high.append(lat_high_dim)

        data = torch.cat(data_list, dim=0)
        tree_rout = torch.cat(tree_rout_list, dim=0)
        vector_rout = torch.cat(vector_rout_list, dim=0)
        data_high = torch.cat(data_high, dim=0)
        return data, data_high, tree_rout, vector_rout

    # def align_tree_node_labels(self,):

    def cal_label_distribution_in_node(self, label):

        num_all_data = label.shape[0]
        unique_labels, counts = np.unique(label, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts / num_all_data))

        # sort label_distribution
        label_distribution = dict(
            sorted(label_distribution.items(), key=lambda item: item[1], reverse=True)
        )

        # remove item less than 10%
        label_distribution = {k: v for k, v in label_distribution.items() if v >= 0.1}

        # round only 2 digits
        label_distribution = {k: round(v, 2) for k, v in label_distribution.items()}

        return label_distribution

    def build_tree_with_label(self, pl_module, tree_rout, label, data_vis, data_high):
        num_tree_depth = len(pl_module.tree_node_embedding)

        root = Node(
            "L0:N0",
            label_distribution=None,
            major_class="None",
            vis_emb=data_vis.mean(axis=0),
            high_emb=data_high.mean(axis=0),
        )
        for level in range(num_tree_depth):
            num_nodes_c_level = pl_module.tree_node_embedding[level].weight.shape[0]
            for node_index in range(num_nodes_c_level):
                mask = tree_rout[:, level] == node_index
                cal_label_distribution_in_node = self.cal_label_distribution_in_node(
                    label[mask]
                )
                node_name = f"L{level+1}:N{node_index}"
                node_father_name = f"L{level}:N{node_index//2}"
                node_father = find_by_attr(root, name="name", value=node_father_name)
                # import pdb; pdb.set_trace()
                if len(list(cal_label_distribution_in_node.keys())) > 0:
                    major_class = list(cal_label_distribution_in_node.keys())[0]
                else:
                    major_class = "None"
                    
                # print('node name:', node_name, 'major class:', major_class, 'label distribution:', cal_label_distribution_in_node)

                if mask.sum() > 0:
                    vis_emb = data_vis[mask].mean(axis=0)
                    high_emb = data_high[mask].mean(axis=0)
                else:
                    vis_emb = np.array([1e10] * data_vis.shape[1])
                    high_emb = np.array([1e10] * data_high.shape[1])

                Node(
                    node_name,
                    parent=node_father,
                    label_distribution=cal_label_distribution_in_node,
                    major_class=major_class,
                    vis_emb=vis_emb,
                    high_emb=high_emb,
                )
                # import pdb; pdb.set_trace()

        for pre, fill, node in RenderTree(root, maxlevel=6):
            print("%s %s %s" % (pre, node.name, node.label_distribution))
        return root

    def update_tree_with_possible_vis_emb(self, root):

        for node in PostOrderIter(root):
            # 如果是叶子节点，p_vis_emb 就是自己的 vis_emb
            if node.is_leaf:
                if hasattr(node, "vis_emb") and node.vis_emb is not None:
                    node.p_vis_emb = [node.vis_emb]
                else:
                    node.p_vis_emb = []
            else:
                # 收集所有子节点的 vis_emb
                all_child_vis_embs = []

                for child in node.children:
                    # 如果子节点有 p_vis_emb（已经收集了其所有后代的 vis_emb）
                    if hasattr(child, "p_vis_emb"):
                        all_child_vis_embs.extend(child.p_vis_emb)
                    # 如果子节点直接有 vis_emb
                    elif hasattr(child, "vis_emb") and child.vis_emb is not None:
                        all_child_vis_embs.append(child.vis_emb)

                # 如果当前节点自己也有 vis_emb，也加入列表
                if hasattr(node, "vis_emb") and node.vis_emb is not None:
                    all_child_vis_embs.append(node.vis_emb)

                node.p_vis_emb = all_child_vis_embs

        return root

    def get_depth_node_with_label_random_selection(
        self, Graph_g, label, select_method="farest"
    ):

        list_nodel = []
        level_list = []
        for node in Graph_g.nodes(data=True):
            # print('node name:', node, 'major class:', node[1].get("major_class"), 'label:', label)
            
            if node[1].get("major_class") == label:
                list_nodel.append(node[0])
                level_list.append(int(node[0].split(":")[0][1:]))

        max_level = max(level_list)

        list_nodel_only_with_max_level = [
            n for n, l in zip(list_nodel, level_list) if l == max_level
        ]

        if select_method == "random":
            random.shuffle(list_nodel_only_with_max_level)
            out = list_nodel_only_with_max_level[0]
        if select_method == "farest":
            out = max(
                list_nodel_only_with_max_level,
                key=lambda n: np.linalg.norm(Graph_g.nodes[n]["vis_emb"]),
            )

            # import ipdb; ipdb.set_trace()
        return out

    def gen_path_lineage_graph(self, Graph_g, lineage_start, lineage_end, seed=42):


        start_node = self.get_depth_node_with_label_random_selection(
            Graph_g, lineage_start, select_method="random"
        )
        end_node = self.get_depth_node_with_label_random_selection(
            Graph_g, lineage_end, select_method="farest"
        )

        path = nx.shortest_path(
            Graph_g, source=start_node, 
            target=end_node, weight="weight"
        )
        # path = nx.shortest_path(Graph_g, source='L1:N0', target='L1:N1', weight='weight')

        return path

    def gen_path_lineage(self, G, lineage_start, lineage_end, seed=42):

        # import ipdb; ipdb.set_trace()

        start_node = pick_random_deepest_node(root, lineage_start, seed=seed)
        end_node = pick_random_deepest_node(root, lineage_end, seed=seed)

        w = Walker()
        upwards, common, downwards = w.walk(start_node, end_node)
        root = self.update_tree_with_possible_vis_emb(root)
        path = list(upwards) + [common] + list(downwards)
        return path

    def plot_path(self, Graph_g, path, fig, color):

        x_coords = []
        y_coords = []
        # end_point = Graph_g.nodes[path[-1]]['vis_emb']
        for i, node in enumerate(path):

            if "vis_emb" in Graph_g.nodes[node]:
                vis_emb_c = Graph_g.nodes[node]["vis_emb"]

                # if not vis_emb_c[0] > 90000 and min_dist > 0  :
                x_coords.append(vis_emb_c[0])
                y_coords.append(vis_emb_c[1])

        # import ipdb; ipdb.set_trace()

        # x_smooth, y_smooth = smooth_path_spline(x_coords, y_coords)
        x_smooth, y_smooth = x_coords, y_coords
        # 创建 Plotly 图形


        # 路径用三角
        fig.add_trace(
            go.Scatter(
                x=x_coords[0:],
                y=y_coords[0:],
                mode="markers",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="triangle-up",
                    line=dict(color="darkred", width=1),
                ),
                name="Start Point",
                text=[f"{node}" for node in path],
            )
        )

        # 只添加一条折线
        fig.add_trace(
            go.Scatter(
                x=x_smooth, y=y_smooth, mode="lines", line=dict(color=color, width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[x_coords[0]],  # 起点的 x 坐标
                y=[y_coords[0]],  # 起点的 y 坐标
                mode="markers",
                marker=dict(
                    color="red",
                    size=15,  # 星星大小
                    symbol="star",  # 五角星符号
                    line=dict(color="darkred", width=1),  # 添加边框使其更明显
                ),
                name="Start Point",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_coords[-1]],  # 终点的 x 坐标
                y=[y_coords[-1]],  # 终点的 y 坐标
                mode="markers",
                marker=dict(
                    color="green",
                    size=15,  # 星星大小
                    symbol="circle",  # 圆形符号
                    line=dict(color="darkgreen", width=1),  # 添加边框使其更明显
                ),
                name="End Point",
            )
        )

        return fig

    def plot_single_lineage_with_label(
        self, root, lineage_start, lineage_end, data_vis, seed=42
    ):

        Graph_g = anytree_to_networkx(root=root)
        path_list = []
        for i in range(len(lineage_start)):
            path = self.gen_path_lineage_graph(
                Graph_g, lineage_start[i], lineage_end[i], seed=seed + i
            )
            # print(f"Generated path for seed {seed+i}: {path}")
            path_list.append(path)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data_vis[:, 0],
                y=data_vis[:, 1],
                mode="markers",
                marker=dict(color="lightgray", size=2),
                name="Data Points",
            )
        )

        for i, path in enumerate(path_list):
            color = color_list[i]
            fig = self.plot_path(Graph_g, path, fig, color=color)

        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            showlegend=False,
            template="plotly_white",
        )

        return fig

    def on_validation_epoch_end(self, trainer, pl_module, down_sample=20000):

        # if in training and epoch == 0, return
        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
        ):
            return

        data_vis, data_high, tree_rout, vector_rout = self.get_tree_and_emb(
            trainer, pl_module, vis_index=3
        )

        adata = trainer.datamodule.adata

        data_vis = data_vis.detach().cpu().numpy()
        data_high = data_high.detach().cpu().numpy()
        tree_rout = tree_rout.detach().cpu().numpy()
        vector_rout = vector_rout.detach().cpu().numpy()

        if data_vis.shape[0] > down_sample:
            indices = torch.randperm(data_vis.shape[0])[:down_sample]
            data_vis = data_vis[indices]
            data_high = data_high[indices]
            tree_rout = tree_rout[indices]
            vector_rout = vector_rout[indices]

            adata = adata[indices.detach().cpu().numpy(), :].copy()

        label = adata.obs["cell_type"].to_numpy()

        root = self.build_tree_with_label(
            pl_module, tree_rout, label, data_vis, data_high
        )

        log_dict = {}
        fig_treemap = build_treemap_figure_from_anytree(
            root,
            title="Cell-type Distribution Treemap",
            color_by_major_label=True,
            max_depth=8,
        )
        log_dict["treemap/treemap"] = fig_treemap


        trainer.logger.experiment.log(log_dict)
