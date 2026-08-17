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

def build_sankey_detailed_from_anytree(
    root,
    title="Tree Sankey (Detailed)",
    max_depth=None,              # None: 全部层
    color_by_major_label=True,
    epsilon=1e-6,                # 0 值边/节点的极小替代
    flow_mode="subtree",         # "subtree" | "local"
    topk_hover=6,                # hover 中 label 分布显示前 k 项
    border_width=0.5,            # 节点描边
):
    """
    生成详细版 Sankey：节点与边 hover 信息极为丰富；从 L0:N0 全层绘制。
    依赖：每个节点最好已有 node.sample_count（本地样本数）。
    """

    # ——1) 收集本地样本数（local_count）、主导类别（major_label）与分布
    local_count, major_label, label_dist = {}, {}, {}
    for node in PostOrderIter(root):
        cnt = getattr(node, "sample_count", None)
        if cnt is None:
            # 退化为 label_distribution 求和
            cnt = _sum_value_from_label_dist(getattr(node, "label_distribution", None)) or 0
        local_count[node] = float(cnt)

        ld = getattr(node, "label_distribution", None)
        label_dist[node] = ld if isinstance(ld, dict) else None
        mj, ratio = _major_and_ratio(ld)
        major_label[node] = mj

    # ——2) 计算 subtree 样本数 & 叶子数量
    subtree_count, leaf_count = {}, {}
    for node in PostOrderIter(root):
        if node.is_leaf:
            subtree_count[node] = local_count[node]
            leaf_count[node] = 1
        else:
            subtree_count[node] = local_count[node] + sum(subtree_count[ch] for ch in node.children)
            leaf_count[node] = sum(leaf_count[ch] for ch in node.children)

    # ——3) 选取绘制节点（逐层，不过滤）
    nodes_in_scope = []
    for n in LevelOrderIter(root):
        d = len(n.ancestors)
        if (max_depth is not None) and (d > max_depth):
            continue
        nodes_in_scope.append(n)
    if not nodes_in_scope:
        raise ValueError("No nodes to draw.")

    # ——4) 固定布局：每层 x 固定；y 在层内等距；根在最左
    def _depth(n): return len(n.ancestors)
    maxd = max(_depth(n) for n in nodes_in_scope)
    level_buckets = {}
    for n in nodes_in_scope:
        level_buckets.setdefault(_depth(n), []).append(n)

    xs, ys, labels = [], [], []
    for d in range(0, maxd + 1):
        bucket = level_buckets.get(d, [])
        m = max(1, len(bucket))
        for i, n in enumerate(bucket):
            x = 0.0 if n.is_root else (d / max(1, maxd))
            y = (i + 0.5) / m
            xs.append(x); ys.append(y); labels.append(n.name)

    idx = {n.name: i for i, n in enumerate(nodes_in_scope)}

    # ——5) 颜色映射：major class -> color；返回映射供外部展示 legend
    label2color = {}
    node_colors, link_colors = None, []
    if color_by_major_label:
        node_colors = []
        ci = 0
        for n in nodes_in_scope:
            mj = major_label[n]
            if mj is None:
                node_colors.append("#cccccc")
            else:
                if mj not in label2color:
                    label2color[mj] = color_list[ci % len(color_list)]
                    ci += 1
                node_colors.append(label2color[mj])

    # ——6) 构造节点 hover（自定义列更可控）
    node_custom = []
    for n in nodes_in_scope:
        d = _depth(n)
        path_str = _fmt_path(n)
        mj, maj_ratio = _major_and_ratio(label_dist[n])
        H = _entropy_from_label_dist(label_dist[n])
        ld_str = _label_dist_to_str(label_dist[n], top_k=topk_hover)

        node_custom.append([
            n.name,                      # 0 名称
            d,                           # 1 深度
            path_str,                    # 2 路径
            int(subtree_count[n]),       # 3 子树样本数
            int(local_count[n]),         # 4 本地样本数
            len(n.children),             # 5 子节点数
            leaf_count[n],               # 6 叶子数量
            mj if mj is not None else "—",            # 7 major
            f"{maj_ratio:.2%}" if maj_ratio is not None else "—",  # 8 major占比
            f"{H:.3f}" if H is not None else "—",     # 9 熵
            ld_str                        # 10 label分布摘要
        ])

    node_hover = (
        "<b>%{customdata[0]}</b>"
        "<br>Depth: %{customdata[1]}"
        "<br>Path: %{customdata[2]}"
        "<br>Subtree Samples: %{customdata[3]}"
        "<br>Local Samples: %{customdata[4]}"
        "<br>#Children: %{customdata[5]} | #Leaves: %{customdata[6]}"
        "<br>Major: %{customdata[7]} (%{customdata[8]})"
        "<br>Entropy: %{customdata[9]}"
        "<br>%{customdata[10]}"
        "<extra></extra>"
    )

    # ——7) 构造边（父→子）
    sources, targets, values, link_custom = [], [], [], []
    for child in nodes_in_scope:
        if child.is_root: 
            continue
        parent = child.parent
        if parent not in nodes_in_scope:
            continue

        if flow_mode == "subtree":
            val = subtree_count[child]
        else:
            # 仍用 child 的 subtree 做层间“高度”，确保下层可见；
            # “local” 纯本地高度可参考上一条信息里的 SINK 方案
            val = subtree_count[child]

        v = float(val)
        if v <= 0: v = float(epsilon)

        sources.append(idx[parent.name])
        targets.append(idx[child.name])
        values.append(v)

        cd = [
            parent.name,                       # 0 父
            child.name,                        # 1 子
            _depth(parent),                    # 2 父层
            _depth(child),                     # 3 子层
            int(subtree_count[child]),         # 4 子树样本数
            major_label[child] if major_label[child] is not None else "—",  # 5 子major
            _label_dist_to_str(label_dist[child], top_k=topk_hover)          # 6 子分布
        ]
        link_custom.append(cd)

        if color_by_major_label:
            mj = major_label[child]
            link_colors.append(label2color.get(mj, "#bbbbbb"))

    link_hover = (
        "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b>"
        "<br>Levels: %{customdata[2]} → %{customdata[3]}"
        "<br>Child Subtree Samples: %{value}"
        "<br>Child Major: %{customdata[5]}"
        "<br>Child Dist: %{customdata[6]}"
        "<extra></extra>"
    )

    # ——8) 组装 Sankey
    sankey_node = dict(
        label=labels,
        pad=18, thickness=14,
        x=xs, y=ys,
        line=dict(color="rgba(0,0,0,0.3)", width=border_width),
        customdata=np.array(node_custom, dtype=object),
        hovertemplate=node_hover,
    )
    if node_colors is not None:
        sankey_node["color"] = node_colors

    sankey_link = dict(
        source=sources,
        target=targets,
        value=values,
        customdata=np.array(link_custom, dtype=object),
        hovertemplate=link_hover,
    )
    if color_by_major_label and link_colors:
        sankey_link["color"] = link_colors

    fig = go.Figure([go.Sankey(arrangement="fixed", node=sankey_node, link=sankey_link)])
    fig.update_layout(title_text=title, font_size=12, margin=dict(t=60, l=10, r=10, b=10))

    return fig  # ☆ 返回颜色映射，便于你在 WandB 放个 legend 表



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
            pl_module, tree_rout, label, data_vis, data_high
        )

        log_dict = {}
        fig_sankey = build_sankey_detailed_from_anytree(
            root,
            title="Cell-type Tree (Detailed Sankey)",
            max_depth=5,
            color_by_major_label=True,
            flow_mode="subtree",
            topk_hover=8
        )
        log_dict["sankey/figure"] = fig_sankey


        trainer.logger.experiment.log(log_dict)
