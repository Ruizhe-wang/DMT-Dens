import os
import random

import numpy as np
import torch
import lightning as pl
import plotly.graph_objects as go

import networkx as nx
from sklearn.neighbors import kneighbors_graph

from anytree import Node
from anytree.search import find_by_attr, findall_by_attr
from anytree import PostOrderIter, LevelOrderIter
import math
import numpy as np
import plotly.graph_objects as go
from anytree import PostOrderIter, LevelOrderIter

from anytree import Node, PreOrderIter, RenderTree
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc


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


# def prune_tree_by_depth(root, max_depth):
#     """
#     返回一棵新的子树，只保留 depth <= max_depth 的节点。
#     """
#     # 用字典记录新节点，方便重新建立 parent
#     new_nodes = {}
#     for n in PreOrderIter(root):
#         if n.depth <= max_depth:
#             parent_new = new_nodes.get(n.parent, None)
#             new_node = Node(
#                 n.name,
#                 parent=parent_new,
#                 **{k: v for k, v in n.__dict__.items() if k not in ["name", "parent", "children"]}
#             )
#             new_nodes[n] = new_node
#     return new_nodes[root]  # 返回新树的根节点

def plot_radial_tree_major_class(root):
    # 1) 收集所有 major_class
    nodes = list(PreOrderIter(root))
    classes = sorted(set([n.major_class for n in nodes if n.major_class is not None]))
    
    # 2) 为每个类别分配颜色
    palette = pc.qualitative.Set3  # 12种离散颜色
    color_map = {}
    for i, c in enumerate(classes):
        color_map[c] = palette[i % len(palette)]

    # 3) 调用之前封装好的函数，指定 node_color 为 callable
    fig = plot_radial_tree_anytree(
        root,
        node_color=lambda n: color_map.get(n.major_class, "#d3d3d3"),  # None 类别用灰色
        node_size=lambda n: 8 if n.is_leaf else 10,
        hover_text=lambda n: f"{n.name}<br>Class={n.major_class}<br>{n.label_distribution}",
        show_scale=False  # 离散颜色不显示色带
    )

    # 可选：添加图例
    for c, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=str(c)
        ))

    return fig


def plot_radial_tree_anytree(
    root,
    *,
    angle_start=-np.pi/2,
    angle_end=3*np.pi/2,
    radius_step=1.0,
    node_size=4,
    node_color=None,
    colorscale="Viridis",
    showlegend=False,
    show_scale=True,
    edge_color="#9aa0a6",
    edge_width=1.2,
    edge_opacity=0.9,              # 边线透明度
    node_opacity=1.0,              # 节点透明度
    hover_text=None,
    figure_size=700,

    # --- 叶子外圈文字参数（用 annotations 实现旋转） ---
    show_leaf_labels=True,         # 是否显示叶子文字环
    leaf_label_fn=lambda n: f"{n.major_class}({n.name})",  # 叶子文本内容
    leaf_radius_offset=1.8,        # 文本相对最外层半径外扩
    leaf_font_size=10,
    leaf_font_color="#444",
    leaf_label_every=1,            # 每隔几个叶子标一次
    leaf_label_orientation="tangent",  # "tangent" 或 "radial"
):
    import numpy as np
    import plotly.graph_objects as go
    from anytree import PreOrderIter

    # -------- 1) 叶子均匀分配角度 --------
    leaves = [n for n in PreOrderIter(root) if n.is_leaf]
    if len(leaves) == 0:
        raise ValueError("树中没有叶子节点。")

    leaf_angles = {}
    L = len(leaves)
    for i, leaf in enumerate(leaves):
        theta = angle_start + (angle_end - angle_start) * i / max(L, 1)
        leaf_angles[leaf] = theta

    # -------- 2) 内部节点角度 = 子树叶角度的向量平均 --------
    def node_angle(n):
        if n.is_leaf:
            return leaf_angles[n]
        child_leaf_angles = [leaf_angles[d] for d in PreOrderIter(n) if d.is_leaf]
        if len(child_leaf_angles) == 0:
            return angle_start
        s = np.sum([np.exp(1j * a) for a in child_leaf_angles])
        return float(np.angle(s))

    angles = {n: node_angle(n) for n in PreOrderIter(root)}

    # -------- 3) 半径由深度给定，转为坐标 --------
    max_depth = max(n.depth for n in PreOrderIter(root))

    def node_radius(n):
        # 先归一化到 0~1，再平方放大
        t = n.depth / max_depth
        return (t ** 2) * (max_depth * radius_step)

    xy = {n: (node_radius(n) * np.cos(angles[n]),
              node_radius(n) * np.sin(angles[n]))
          for n in PreOrderIter(root)}

    # -------- 4) 画边 --------
    edge_traces = []
    for n in PreOrderIter(root):
        for c in n.children:
            x0, y0 = xy[n]
            x1, y1 = xy[c]
            edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hoverinfo="skip",
                showlegend=False,
                opacity=edge_opacity,
            ))

    # -------- 5) 节点样式 --------
    def eval_callable_or_const(spec, node, default=None):
        if spec is None:
            return default
        if callable(spec):
            return spec(node)
        if isinstance(spec, (int, float, str)):
            return spec
        if isinstance(spec, str):
            return getattr(node, spec, default)
        return default

    sizes = [eval_callable_or_const(node_size, n, node_size) for n in PreOrderIter(root)]

    if node_color is None:
        color_vals = [n.depth for n in PreOrderIter(root)]
        color_kwargs = dict(color=color_vals, colorscale=colorscale, showscale=show_scale)
    else:
        vals = [eval_callable_or_const(node_color, n, None) for n in PreOrderIter(root)]
        if all(isinstance(v, (int, float, np.floating)) for v in vals if v is not None):
            color_kwargs = dict(color=vals, colorscale=colorscale, showscale=show_scale)
        else:
            color_kwargs = dict(color=vals, showscale=False)

    def default_hover(n): 
        return str(getattr(n, "name", f"depth={n.depth}"))
    hovers = [eval_callable_or_const(hover_text, n, default_hover(n)) for n in PreOrderIter(root)]

    xs = [xy[n][0] for n in PreOrderIter(root)]
    ys = [xy[n][1] for n in PreOrderIter(root)]

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=sizes,
            line=dict(width=0.4, color="white"),
            opacity=node_opacity,
            **color_kwargs
        ),
        text=hovers,
        hoverinfo="text",
        showlegend=showlegend,
        opacity=1.0,  # 轨迹整体不降透明度
    )

    # -------- 5.5) 叶子文字环（annotations 以支持旋转） --------
    annotations = []
    if show_leaf_labels:
        max_depth = max(n.depth for n in PreOrderIter(root))
        r_text = (max_depth * radius_step) * 1.2 + float(leaf_radius_offset)

        leaves_ordered = [n for n in PreOrderIter(root) if n.is_leaf]
        for i, leaf in enumerate(leaves_ordered):
            if (i % max(int(leaf_label_every), 1)) != 0:
                continue

            text = str(leaf_label_fn(leaf))
            if not text:
                continue

            a = angles[leaf]
            x = r_text * np.cos(a)
            y = r_text * np.sin(a)

            # 角度（度）
            if leaf_label_orientation == "tangent":
                ang = -1*np.degrees(a) 
                # if 90 < (ang % 360) < 270:
                #     ang += 180
            else:  # radial
                ang = -1*np.degrees(a)
                # if -90 <= ang <= 90:
                #     ang += 180

            annotations.append(dict(
                x=x, y=y,
                xref="x", yref="y",
                text=text,
                showarrow=False,
                font=dict(size=leaf_font_size, color=leaf_font_color),
                xanchor="center", 
                yanchor="middle",
                textangle=ang,
                align="left",
                borderpad=0,
                opacity=1.0,
                captureevents=False,
            ))

    # -------- 6) 布局 --------
    fig = go.Figure(edge_traces + [node_trace])
    fig.update_layout(
        width=figure_size,
        height=figure_size,
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=36, r=36, t=36, b=36),  # 给外圈文字留空间
        annotations=annotations,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig




def _entropy_from_label_dist(ld):
    """基于 label_distribution(dict: label->count 或 ->比例) 计算熵，返回自然对数底。"""
    if not isinstance(ld, dict) or len(ld) == 0:
        return None
    vals = np.array(list(ld.values()), dtype=float)
    if vals.sum() <= 0:
        return None
    p = vals / vals.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _major_and_ratio(ld):
    """返回 (major_label, ratio[0..1])；若无则(None, None)"""
    if not isinstance(ld, dict) or len(ld) == 0:
        return None, None
    items = sorted(ld.items(), key=lambda x: x[1], reverse=True)
    total = float(sum(v for _, v in items)) or 1.0
    maj, cnt = items[0]
    return maj, float(cnt) / total

def _fmt_path(node):
    return " / ".join(n.name for n in node.path)





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
    Graph_g = nx.Graph()
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
            
    # 添加根节点（节点已在上面添加，这里进入分层连边/近邻构图）
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

        # KNN 构图
        weight_knn = 20 ** (num_level - level)
        if len(numpy_emb) > 41:
            knn_graph = kneighbors_graph(
                numpy_emb,
                n_neighbors=n_neighbors,
                include_self=True,
                metric="euclidean",
            )
            mask_over_large = numpy_emb.sum(axis=1) > 1e9
            if mask_over_large.sum() > 0:
                knn_graph[mask_over_large, :] = 0
                knn_graph[:, mask_over_large] = 0

            knn_graph = knn_graph.toarray()
            num_node = numpy_emb.shape[0]
            for i in range(num_node):
                for j in range(num_node):
                    if i > j and knn_graph[i, j]:
                        Graph_g.add_edge(
                            nodes_c_level[i], nodes_c_level[j], weight=weight_knn
                        )
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

    def build_tree_with_label(
        self, 
        pl_module, 
        tree_rout, 
        label, 
        data_vis, 
        data_high,
        num_tree_depth = 10,
        ):
        

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
                if len(list(cal_label_distribution_in_node.keys())) > 0:
                    major_class = list(cal_label_distribution_in_node.keys())[0]
                else:
                    major_class = "None"

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
        # 可选：如需调试可取消注释以下打印
        # for pre, fill, node in RenderTree(root, maxlevel=6):
        #     print(f"{pre} {node.name} {node.label_distribution}")
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

        return out




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
        # vector_rout is not used below; no need to convert

        if data_vis.shape[0] > down_sample:
            indices = torch.randperm(data_vis.shape[0])[:down_sample]
            idx_np = indices.detach().cpu().numpy()
            data_vis = data_vis[idx_np]
            data_high = data_high[idx_np]
            tree_rout = tree_rout[idx_np]
            adata = adata[idx_np, :].copy()

        label = adata.obs["cell_type"].to_numpy()

        root = self.build_tree_with_label(
            pl_module, tree_rout, label, data_vis, data_high, num_tree_depth=8
        )

        # root_with_pruned = prune_tree_by_depth(root, max_depth=8)
        log_dict = {}
        fig_radialtree = plot_radial_tree_major_class(
            root,
        )
        log_dict["radialtree/figure"] = fig_radialtree


        trainer.logger.experiment.log(log_dict)
