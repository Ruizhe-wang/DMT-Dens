
import scanpy as sc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
# cs.settings.set_figure_params(format='pdf',figsize=[4,3.5],dpi=100,fontsize=15,pointsize=2,dpi_save=300)
# from callbacks.util import plot_arrow
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.offline as pyo
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import uuid

from sklearn.metrics import pairwise_distances


def plot_arrow(fig, x0, y0, x1, y1, arrow_size=0.05, color="rgba(0, 0, 255, 0.5)"):
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

    # import pdb; pdb.set_trace()
    
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

def plot_path(G, tree_node_embedding, i, near_index, end_node, plotly_fig_rute):
    shortest_path = nx.shortest_path(
        G,
        source=f"L{i}/{near_index}",
        target=f"L{i}/{end_node}",
        weight="weight",
    )
    # print(f"shortest_path: {shortest_path}")

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
            # import pdb; pdb.set_trace()
            star_emb = (tree_node_embedding[s_node_level][s_node_index])
            end_emb = (tree_node_embedding[e_node_level][e_node_index])

            plotly_fig_rute = plot_arrow(
                plotly_fig_rute,
                star_emb[0],
                star_emb[1],
                end_emb[0],
                end_emb[1],
                color="rgba(0, 0, 255, 0.5)",
            )

def get_predict_label(tree_list, token_emb, emb_vis):

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



def get_rout_vis(tree_node_embedding, emb_vis, centre_node = np.array([-1.5, 1])):
    # import pdb; pdb.set_trace()

    G = nx.Graph()
    G.add_node("L-1/0")

    path_list = []
    tree_list = []
    plotly_fig_rute_dict = {}
    for i in range(len(tree_node_embedding)):

        # token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
        token_emb = tree_node_embedding[i]
        # print("token_emb.shape:", token_emb.shape)
        # print("emb_vis.shape:", emb_vis.shape)

        # if token_emb.shape[0] > 20000:
        #     idx = np.random.choice(token_emb.shape[0], 20000, replace=False)
        #     token_emb = token_emb[idx]
        # if emb_vis.shape[0] > 20000:
        #     idx = np.random.choice(token_emb.shape[0], 20000, replace=False)
        #     emb_vis = emb_vis[idx]
        # import pdb; pdb.set_trace()
        label_predict = get_predict_label(tree_list, token_emb, emb_vis)
        tree_list.append(label_predict)

        # for label_i in range(label_predict.max() + 1):
        #     print(
        #         f"level: {i}, label: {label_i}, num: {np.sum(label_predict==label_i)}"
        #     )

        plotly_fig_rute = go.Figure()
        d3_colors = [
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
        ] * 1000

        plotly_fig_rute.add_trace(
            go.Scatter(
                x=emb_vis[:, 0],
                y=emb_vis[:, 1],
                mode="markers",
                marker=dict(
                    size=1,
                    # color=label_predict,
                    # color=[d3_colors[c_i] for c_i in label_predict],
                    color='grey',
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
            n_neighbors=min(3, num_nodes - 1),
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

        
        distance_to_zero = np.linalg.norm(token_emb-centre_node, axis=1)
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
            width=800,  # 图表宽度（像素）
            height=600,  # 图表高度（像素）
            title="Tree Visualization",
            xaxis=dict(visible=False),  # 隐藏 x 轴
            yaxis=dict(visible=False),  # 隐藏 y 轴
            xaxis_scaleanchor="y",  # 锁定 x 和 y 的比例
            yaxis_scaleanchor="x",
            legend=dict(title="Legend", x=0.01, y=0.99),
        )

        # plotly_fig_rute_dict[f'tree/tree_{i}'] = plotly_fig_rute
        # fig_graph = plt.figure(figsize=(10, 10))
        # nx.draw(G, with_labels=True, node_size=700, node_color="lightblue")
        # fig_graph.savefig(f"fig_graph_{i}.png")
        # fig_graph.clear()

        # plotly_fig_rute_dict[f'graph/tree_{i}'] = fig_graph
        uuid_str = str(uuid.uuid4())[:16]
        plotly_fig_rute.write_image(f"fig_emb_colored_with_rout_{i}_{uuid_str}.png", scale=3)
        path_list.append(f"fig_emb_colored_with_rout_{i}_{uuid_str}.png")

    return path_list, plotly_fig_rute_dict







num_top_celltype = 7
adata = sc.read("/zangzelin/data/difftreedata/data/celegan/celegan.h5ad")

# top 500 genes
# import pdb; pdb.set_trace()
if filter:

    data = adata.X    
    np.var(data, axis=0)
    top_genes = np.argsort(np.var(data, axis=0))[-500:]
    data = data[:, top_genes].astype(np.float32)

std = data.std(axis=0)
mean = data.mean(axis=0)
data = (data - mean) / std

# import pdb; pdb.set_trace()
label_celltype = pd.read_csv('/zangzelin/data/difftreedata/data/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
label_embryo_time = pd.read_csv('/zangzelin/data/difftreedata/data/celegan/celegan_embryo_time.tsv', sep='\t', header=None)
adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
adata.obs['embryo_time'] = pd.Categorical(np.squeeze(label_embryo_time))
label_train_str = list(np.squeeze(label_celltype.values))
label_train_str_set = list(set(label_train_str))
label_train_str_set = sorted(label_train_str_set)
label = torch.tensor(
    np.array([label_train_str_set.index(i) for i in label_train_str]))

dict_str_num_sample = {}
for str_label in label_train_str_set:
    dict_str_num_sample[str_label] = (label==label_train_str_set.index(str_label)).sum()
    
sorted_dict = sorted(dict_str_num_sample.items(), key=lambda x: x[1], reverse=True)
for i in range(num_top_celltype):
    print(sorted_dict[i])


time_list = []
for str_time in label_embryo_time[0].to_list():
    if '-' in str_time:
        time_list.append(float(str_time.split('-')[1]))
    elif '<' in str_time:
        time_list.append(float(str_time.split('<')[1])-50)
    elif '>' in str_time:
        time_list.append(float(str_time.split('>')[1])+100)
    else:
        print('error', str_time)
        
# # select the top num of sample 5 classes
num_cell_type = len(label_train_str_set)
num_sample_list = []
for i in range(num_cell_type):
    num_sample_list.append((label==i).sum())

# # select the top 5 cell types
top5_cell_type = np.argsort(num_sample_list)[-num_top_celltype:]
mask = np.zeros(len(label)).astype(np.bool_)
for i in top5_cell_type:
    mask[label==i] = 1
data = data[mask]
label = label[mask]

label_str_new = list(set([str(l) for l in label]))
label_str_new = sorted(label_str_new)
label = torch.tensor([label_str_new.index(str(l)) for l in label])

time_list = torch.tensor(time_list)
time_list = time_list[mask]
    
print('data.shape', data.shape)
# time_label = torch.tensor(label_embryo_time)
label = torch.cat((label.reshape(-1,1), time_list.reshape(-1,1)), dim=1)

# vis the data with UMAP

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# reducer = umap.UMAP()
# tsne
import sklearn
reducer = sklearn.manifold.TSNE(n_components=2)
embedding = reducer.fit_transform(data)

c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
c_list += ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"]

label_list = label[:, 0].tolist()
color = [c_list[int(i)] for i in label_list]

plt.figure(figsize=(10, 10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=color,
    s=1
)

mean = embedding.mean(axis=0)
std = embedding.std(axis=0)

embedding = (embedding - mean) / std

plt.gca().set_aspect('equal', 'datalim')
plt.savefig('umap_celltype.png')

import wandb
wandb.init(project="celegan")

plt.close()

plt.figure(figsize=(10, 10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=label[:, 1],
    cmap='RdBu',
    s=1
)

plt.gca().set_aspect('equal', 'datalim')
plt.savefig('umap_celltype_time.png')

from sklearn.cluster import KMeans
tree_node_embedding_tree_list = []
label_list = []
for i in range(10):
    # kmeans clustering
    label_c = np.zeros(len(label), dtype=np.int32)
    emb_c = []
    for j in range(2**(i+1)):
        
        if len(label_list) == 0:
            mask = np.ones(len(label), dtype=np.bool_)
        else:
            mask = (label_list[-1] == j)
        
        select_emb = embedding[mask]
        # import pdb; pdb.set_trace()
        print(mask)
        if mask.sum() > 1:
            # with parallel_backend('loky', n_jobs=10):
            kmeans = KMeans(n_clusters=2, random_state=0).fit(select_emb)
            label_kmeans = kmeans.labels_ + 2*j
            label_c[mask] = label_kmeans

            # update the embedding
            emb_c.append(kmeans.cluster_centers_[0].reshape(1, -1))
            emb_c.append(kmeans.cluster_centers_[1].reshape(1, -1))
            # self.tree_node_embedding[i].weight.data[2*j] = torch.tensor(kmeans.cluster_centers_[0], device=emb.device)
            # self.tree_node_embedding[i].weight.data[2*j+1] = torch.tensor(kmeans.cluster_centers_[1], device=emb.device)
        else:
            label_c[mask] = torch.tensor([2*j] * mask.sum())
            # self.tree_node_embedding[i].weight.data[2*j] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
            # self.tree_node_embedding[i].weight.data[2*j+1] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
            # import pdb; pdb.set_trace()
            e = tree_node_embedding_tree_list[-1][j]
            emb_c.append(e.reshape(1, -1))
            emb_c.append(e.reshape(1, -1))
    # import pdb; pdb.set_trace()
    emb_c_c = np.concatenate(emb_c, axis=0)
    tree_node_embedding_tree_list.append(emb_c_c)
    label_list.append(label_c)

# tree_node_embedding = np.concatenate([t.reshape(-1, 1) for t in tree_node_embedding_tree_list], axis=1)
dict_log = {
    "umap_celltype": wandb.Image('umap_celltype.png'),
    "umap_celltype_time": wandb.Image('umap_celltype_time.png')
    }

# import pdb; pdb.set_trace()
vis_plot_path, plotly_fig_rute_dict = get_rout_vis(
    tree_node_embedding_tree_list, embedding)

for i, path in enumerate(vis_plot_path):
    dict_log[f"rout/emb_colored_with_rout_{i}"] = wandb.Image(path)

dict_log.update(
    plotly_fig_rute_dict
    )

wandb.log(dict_log)

wandb.finish()

    # label_kmeans_list = [torch.zeros(emb.shape[0], dtype=torch.int64)]
    # for i in range(len(self.tree_node_embedding)):
        
    #     label_last = label_kmeans_list[-1]
    #     label_new = torch.zeros(emb.shape[0], dtype=torch.int64, device=emb.device)-1 
    #     for j in range(max(2 ** (i), 1)):
    #         mask = label_last == j
    #         select_emb = emb[mask]
    #         print(
    #             f"The number of the node:{i+1}_{j}", "The number of the data:", 
    #             mask.sum()
    #         )
    #         # import pdb; pdb.set_trace()

    #         if mask.sum() > 1:
    #             with parallel_backend('loky', n_jobs=10):
    #                 kmeans = KMeans(n_clusters=2, random_state=0).fit(select_emb.cpu().detach().numpy())
    #             label_kmeans = kmeans.labels_ + 2*j
    #             label_new[mask] = torch.tensor(label_kmeans, dtype=torch.int64, device=emb.device)

    #             # update the embedding
    #             self.tree_node_embedding[i].weight.data[2*j] = torch.tensor(kmeans.cluster_centers_[0], device=emb.device)
    #             self.tree_node_embedding[i].weight.data[2*j+1] = torch.tensor(kmeans.cluster_centers_[1], device=emb.device)
    #         else:
    #             label_new[mask] = torch.tensor([2*j] * mask.sum() , dtype=torch.int64, device=emb.device)
    #             self.tree_node_embedding[i].weight.data[2*j] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
    #             self.tree_node_embedding[i].weight.data[2*j+1] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
            