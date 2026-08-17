import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import umap, wandb
import networkx as nx
import plotly.graph_objects as go
import attr
from collections import Counter
import typing
from scipy.spatial import ConvexHull
from typing import Optional
from scipy.special import comb
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from sklearn.cluster import AgglomerativeClustering

# from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score


def leaf_purity(tree_root, ground_truth, ind_samples_of_leaves):
	values = [] # purity rate per leaf
	weights = [] # n_samples per leaf
	# For each leaf calculate the maximum over classes for in-leaf purity (i.e. majority class / n_samples_in_leaf)
	def get_leaf_purities(node):
		nonlocal values
		nonlocal weights
		if node.decoder:
			ind_leaf = np.where([node == ind_samples_of_leaves[ind_leaf][0] for ind_leaf in range(len(ind_samples_of_leaves))])[0].item()
			ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
			node_total_dp_count = len(ind_samples_of_leaf)
			node_per_label_counts = count_values_in_sequence(
				[ground_truth[id] for id in ind_samples_of_leaf])
			if node_total_dp_count > 0:
				purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
			else:
				purity_rate = 1.0
			values.append(purity_rate)
			weights.append(node_total_dp_count)
		elif node.router is None and node.decoder is None:
			# We are in an internal node with pruned leaves and thus only have one child.
			node_left, node_right = node.left, node.right
			child = node_left if node_left is not None else node_right
			get_leaf_purities(child)	
		else:
			get_leaf_purities(node.left)
			get_leaf_purities(node.right)

	get_leaf_purities(tree_root)
	assert len(values) == len(ind_samples_of_leaves), "Didn't iterate through all leaves"
	# Return mean leaf_purity
	return np.average(values, weights=weights)


def dendrogram_purity(tree_root, ground_truth, ind_samples_of_leaves):
    
    from scipy.special import comb

    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {k: comb(v, 2, repetition=True) for k, v in total_per_label_frequencies.items()}
    total_n_of_pairs = sum(total_per_label_pairs_count.values())
    one_div_total_n_of_pairs = 1. / total_n_of_pairs
    purity = 0.

    def calculate_purity(node, level):
        nonlocal purity
        if node.decoder:
            # Match node to leaf samples
            ind_leaf = np.where([node == ind_samples_of_leaves[ind_leaf][0] for ind_leaf in range(len(ind_samples_of_leaves))])[0].item()
            ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
            node_total_dp_count = len(ind_samples_of_leaf)
            # Count how many samples of given leaf fall into which ground-truth class (-> For treevae make use of ground_truth(to which class a sample belongs)&yy(into which leaf a sample falls))
            node_per_label_frequencies = count_values_in_sequence(
            	[ground_truth[id] for id in ind_samples_of_leaf])
            # From above, deduct how many pairs will fall into same leaf
            node_per_label_pairs_count = {k: comb(v, 2, repetition=True) for k, v in node_per_label_frequencies.items()}

        elif node.router is None and node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            node_per_label_frequencies, node_total_dp_count = calculate_purity(child, level + 1)	
            return node_per_label_frequencies, node_total_dp_count
        else:
            # it is an inner splitting node
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(node.left, level + 1)
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(node.right, level + 1)
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            # Count how many samples of given internal node fall into which ground-truth class (=sum of their children's values)
            node_per_label_frequencies = {k: left_child_per_label_freq.get(k, 0) + right_child_per_label_freq.get(k, 0) \
                                            for k in set(left_child_per_label_freq) | set(right_child_per_label_freq)}

            # Class-wisedly count how many pairs of samples of a class will have this node as least common ancestor (=mult. of their children's values, bcs this is all possible pairs coming from different sides)
            node_per_label_pairs_count = {k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k) \
                                            for k in set(left_child_per_label_freq) & set(right_child_per_label_freq)}

		# Given the class-wise number of pairs with given node as least common ancestor node, calculate their purity
        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = node_per_label_pairs_count[label]
            purity += one_div_total_n_of_pairs * label_freq / node_total_dp_count * label_pairs # (1/n_all_pairs) * purity(=n_samples_of_this_class_in_node/n_samples) * n_class_pairs_with_this_node_being_least_common_ancestor(this last term represents sum over pairs with this node being least common ancestor)
        return node_per_label_frequencies, node_total_dp_count
    
    calculate_purity(tree_root, 0)
    return purity

def compute_leaves(tree):
    list_nodes = [{'node': tree, 'depth': 0}]
    nodes_leaves = []
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        node, depth_level = current_node['node'], current_node['depth']
        print('len(list_nodes)', len(list_nodes))
        # import pdb; pdb.set_trace()
        if node.router is not None:
            node_left, node_right = node.left, node.right
            list_nodes.append(
                {'node': node_left, 'depth': depth_level + 1})
            list_nodes.append({'node': node_right, 'depth': depth_level + 1})
        elif node.router is None and node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            list_nodes.append(
                {'node': child, 'depth': depth_level + 1})
        else:
            nodes_leaves.append(current_node)
    return nodes_leaves



def cluster_acc(y_true, y_pred, return_index=False):
	"""
	Calculate clustering accuracy.
	# Arguments
		y: true labels, numpy.array with shape `(n_samples,)`
		y_pred: predicted labels, numpy.array with shape `(n_samples,)`
	# Return
		accuracy, in [0,1]
	"""
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
	w = np.zeros((int(D), (D)), dtype=np.int64)
	for i in range(y_pred.size):
		w[int(y_pred[i]), int(y_true[i])] += 1
	ind = np.array(linear_assignment(w.max() - w))
	if return_index:
		assert all(ind[0] == range(len(ind[0])))  # Assert rows don't change order
		cluster_acc = sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size
		return cluster_acc, ind[1]
	else:
		return sum([w[ind[0,i], ind[1,i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size


def build_tree_zl(gathered_rout, gathered_vector_rout, num_clusters):
    
    index_tree_select = int(np.sqrt(num_clusters))+2
    tree_note_list = []
    for m_level in range(index_tree_select+1):
        level = index_tree_select - m_level
        label_tree_leaf = gathered_rout[:, level]
        # import pdb; pdb.set_trace()
        num_cluster_tree_leaf = 2**(level+1)
        tree_note_level_list = []
        for j in range(num_cluster_tree_leaf):
            if m_level == 0:
                node_index = np.where(label_tree_leaf==j)[0]
                node = TreeNode(
                    decoder=True, 
                    samples=node_index,
                    name=f'L{level}/{j}',
                    ) 
                tree_note_level_list.append(node)
            else:
                left_node = tree_note_list[-1][j*2]
                right_node = tree_note_list[-1][j*2+1]
                node = TreeNode(
                    router=True, 
                    left=left_node, 
                    right=right_node,
                    name=f'L{level}/{j}',
                    samples=np.array(left_node.samples.tolist() + right_node.samples.tolist()),
                )
                left_node.father = node
                right_node.father = node
                tree_note_level_list.append(node)
        tree_note_list.append(tree_note_level_list)
    
    node_root = TreeNode(router=True, left=tree_note_level_list[0], right=tree_note_level_list[1])
    tree_note_level_list[0].father = node_root
    tree_note_level_list[1].father = node_root
    
    for i in range((2**(index_tree_select+1))-num_clusters):
        num_sample_list = [ len(l.samples) + 1e5 * (1-l.bool_mergeable()) for l in reversed(node_root.list_all_leaf())]
        l_i = np.argsort(num_sample_list)[0]
        leave = node_root.list_all_leaf()[::-1][l_i]
        # print([ f'name:{n.name}, bool_mergeable:{n.bool_mergeable()},len: {len(n.samples)}' for n in reversed(node_root.list_all_leaf())])
        # print('removing node', leave.name)
        father = leave.father   
        
        # if father.right.samples is None:
        #     father.right.samples = np.array([])
        # if father.left.samples is None:
        #     father.left.samples = np.array([])
        # if leave.samples is None:
        #     leave.samples = np.array([])
        # print('leave', leave.name)
        # print('leave.father', leave.father.name)
        
        # print('father.right.samples', father.right.samples)
        # print('father.left.samples', father.left.samples)
        # import pdb; pdb.set_trace()
        
        if father.left.name == leave.name and father.right.samples is not None:                
            father.samples = np.array(father.right.samples.tolist() + leave.samples.tolist())
            father.decoder = True
            father.left = None
            father.right = None            
        elif father.right.name == leave.name and father.left.samples is not None:
            # import pdb; pdb.set_trace()
            # print('father.left.samples', father.left.samples)
            # print('leave.samples', leave.samples)
            father.samples = np.array(father.left.samples.tolist() + leave.samples.tolist())
            father.decoder = True
            father.left = None
            father.right = None 

            
    
    return node_root   



def count_values_in_sequence(sequence):
    counts = {}
    for value in sequence:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    return counts


@attr.define()
class DpNode(object):
    """
    node_id should be in such a way that a smaller number means split before a larger number in a top-down manner
    That is the root should have node_id = 0 and the children of the last split should have node id
    2*n_dps-2 and 2*n_dps-1

    """

    left_child: typing.Any = None
    right_child: typing.Any = None
    node_id: Optional[int] = None

    @property
    def children(self):
        return [self.left_child, self.right_child]

    @property
    def is_leaf(self):
        return False


@attr.s(cmp=False)
class DpLeaf(object):
    dp_ids = attr.ib()
    node_id = attr.ib()

    @property
    def children(self):
        return []

    @property
    def is_leaf(self):
        return True


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

def plot_scatterRdBu(vis_data, labels):
    fig = plt.figure(figsize=(10, 10))
    s = 1 if vis_data.shape[0] >= 10000 else 3
    plt.scatter(vis_data[:, 0], vis_data[:, 1], c=labels, cmap="RdBu", s=s)
    return fig



def plot_scatter(vis_data, labels):
    c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    c_list += ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"]
    c_list = c_list * 1000
    
    labels = labels.astype(int)
    
    color = [c_list[label] for label in labels]
    fig = plt.figure(figsize=(10, 10))
    s = 1 if vis_data.shape[0] >= 10000 else 3
    plt.scatter(vis_data[:, 0], vis_data[:, 1], c=color, s=s)
    return fig

def plot_scatter_rout(vis_data, labels_real, gathered_rout):

    c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    c_list += ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"]
    
    
    # if len(unique_labels) > len(c_list):
    #     raise ValueError("Not enough colors in c_list for the number of unique labels.")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # import pdb; pdb.set_trace()
    
    for i in range(4):
        labels = gathered_rout[:,3-i]
        unique_labels = np.unique(labels)
        for label in unique_labels:
            points = vis_data[labels == label]
            if points.shape[0] < 3:
                continue
            
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            ax.fill(hull_points[:, 0], hull_points[:, 1], c=c_list[label], alpha=0.05)
            
            center = points.mean(axis=0)
            ax.scatter(center[0], center[1], c=c_list[label], s=10, marker='x')
    
    ax.scatter(
        vis_data[:, 0],
        vis_data[:, 1],
        c=[c_list[label] for label in labels_real],
        s=1,)
   
    return fig


def euclidean_to_hyperbolic_matrix(u, c=0.5, min_norm=1e-15):
    u = torch.tensor(u).float()
    u = 1.5 * (u - u.mean(dim=0)) / u.std(dim=0)
    sqrt_c = c**0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()


def plot_scatter_hyper(vis_data, labels):
    vis_hyper = euclidean_to_hyperbolic_matrix(vis_data)
    s = 1 if vis_data.shape[0] >= 10000 else 3
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(vis_hyper[:, 0], vis_hyper[:, 1], c=labels, cmap="rainbow", s=s)
    return fig


def get_svc_acc(vis_data, labels):
    method = SVC(kernel="linear", max_iter=900000)
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
    n_scores = cross_val_score(
        method,
        StandardScaler().fit_transform(vis_data),
        labels,
        scoring="accuracy",
        cv=cv,
        n_jobs=5,
    )
    return np.mean(n_scores)


def get_svc_acc_rbf(vis_data, labels):
    method = SVC(kernel="rbf", max_iter=900000)
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
    n_scores = cross_val_score(
        method,
        StandardScaler().fit_transform(vis_data),
        labels,
        scoring="accuracy",
        cv=cv,
        n_jobs=5,
    )
    return np.mean(n_scores)


def plot_vis_diff_gen_umap(data, label, dict):

    # import pdb; pdb.set_trace()
    if data.shape[0] > 10000:
        # idx = np.random.choice(data.shape[0], 60000, replace=False)
        data = data[:10000]
        label = label[:10000]
    # umap_vis = pacmap.PaCMAP(n_components=2, n_neighbors=100, random_state=0).fit_transform(data)
    # umap_vis = TSNE(n_components=2, perplexity=100, random_state=0).fit_transform(data)
    umap_vis = umap.UMAP(n_components=2, random_state=0).fit_transform(data)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(umap_vis[:, 0], umap_vis[:, 1], cmap="rainbow", s=1, c=label)
    dict.update({"rout/umap_gen": wandb.Image(fig)})
    plt.close()
    return None


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
            star_emb = (
                tree_node_embedding[s_node_level]
                .weight.detach()
                .cpu()
                .numpy()[s_node_index]
            )
            end_emb = (
                tree_node_embedding[e_node_level]
                .weight.detach()
                .cpu()
                .numpy()[e_node_index]
            )

            plotly_fig_rute = plot_arrow(
                plotly_fig_rute,
                star_emb[0],
                star_emb[1],
                end_emb[0],
                end_emb[1],
                color="rgba(0, 0, 255, 0.5)",
            )


def calculate_hierarchical_integrity(feature, labels):    
    # 计算 Silhouette 分数
    silhouette_avg = silhouette_score(feature, labels)
    
    # 将 Silhouette 分数转换为百分比
    integrity_percentage = ((silhouette_avg + 1) / 2)
    
    return integrity_percentage

def to_dendrogram_purity_tree(children_array):
    """
    Can convert the children_ matrix of a  sklearn.cluster.hierarchical.AgglomerativeClustering outcome to a dendrogram_purity tree
    :param children_array:  array-like, shape (n_samples-1, 2)
        The children of each non-leaf nodes. Values less than `n_samples`
            correspond to leaves of the tree which are the original samples.
            A node `i` greater than or equal to `n_samples` is a non-leaf
            node and has children `children_[i - n_samples]`. Alternatively
            at the i-th iteration, children[i][0] and children[i][1]
            are merged to form node `n_samples + i`
    :return:
    """
    n_samples = children_array.shape[0] + 1
    max_id = 2 * n_samples - 2
    node_map = {dp_id: DpLeaf([dp_id], max_id - dp_id) for dp_id in range(n_samples)}
    next_id = max_id - n_samples

    for idx in range(n_samples - 1):
        next_fusion = children_array[idx, :]
        child_a = node_map.pop(next_fusion[0])
        child_b = node_map.pop(next_fusion[1])
        node_map[n_samples + idx] = DpNode(child_a, child_b, next_id)
        next_id -= 1
    if len(node_map) != 1:
        raise RuntimeError(
            "tree must be fully developed! Use ompute_full_tree=True for AgglomerativeClustering"
        )
    root = node_map[n_samples + n_samples - 2]
    return root


def calculate_tree_leaf_purity_mean(children_array, true_labels):
    """
    Calculate the mean leaf purity across all nodes in the tree.
    
    :param children_array: Children array from sklearn Agglomerative Clustering
    :param true_labels: Ground truth labels (n_samples,)
    :return: Mean leaf purity across all nodes
    """
    n_samples = len(true_labels)  # Number of samples
    n_nodes = children_array.shape[0] + n_samples  # Total number of nodes

    # Initialize a dictionary to store the leaves covered by each node
    node_to_leaves = {i: [i] for i in range(n_samples)}  # Leaf nodes

    # Build node-to-leaves mapping from the children array
    for node_id, (child_a, child_b) in enumerate(children_array, start=n_samples):
        node_to_leaves[node_id] = node_to_leaves[child_a] + node_to_leaves[child_b]

    # Calculate purity for each node
    purities = []
    for node_id, leaves in node_to_leaves.items():
        leaf_labels = true_labels[leaves]
        most_common_label, count = Counter(leaf_labels).most_common(1)[0]
        purity = count / len(leaves)
        purities.append(purity)

    return np.mean(purities)

# def dendrogram_purity(tree_root, ground_truth):
#     total_per_label_frequencies = count_values_in_sequence(ground_truth)
#     total_per_label_pairs_count = {
#         k: comb(v, 2, exact=True) for k, v in total_per_label_frequencies.items()
#     }
#     total_n_of_pairs = sum(total_per_label_pairs_count.values())

#     one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

#     purity = 0.0

#     def calculate_purity(node, level):
#         nonlocal purity
#         if node.is_leaf:
#             node_total_dp_count = len(node.dp_ids)
#             node_per_label_frequencies = count_values_in_sequence(
#                 [ground_truth[id] for id in node.dp_ids]
#             )
#             node_per_label_pairs_count = {
#                 k: comb(v, 2, exact=True) for k, v in node_per_label_frequencies.items()
#             }

#         elif node.left_child is None or node.right_child is None:
#             # We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
#             node_left, node_right = node.left_child, node.right_child
#             child = node_left if node_left is not None else node_right
#             node_per_label_frequencies, node_total_dp_count = calculate_purity(
#                 child, level + 1
#             )
#             return node_per_label_frequencies, node_total_dp_count

#         else:  # it is an inner node
#             left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
#                 node.left_child, level + 1
#             )
#             right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
#                 node.right_child, level + 1
#             )
#             node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
#             node_per_label_frequencies = {
#                 k: left_child_per_label_freq.get(k, 0)
#                 + right_child_per_label_freq.get(k, 0)
#                 for k in set(left_child_per_label_freq)
#                 | set(right_child_per_label_freq)
#             }

#             node_per_label_pairs_count = {
#                 k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
#                 for k in set(left_child_per_label_freq)
#                 & set(right_child_per_label_freq)
#             }

#         for label, pair_count in node_per_label_pairs_count.items():
#             label_freq = node_per_label_frequencies[label]
#             label_pairs = node_per_label_pairs_count[label]
#             purity += (
#                 one_div_total_n_of_pairs
#                 * label_freq
#                 / node_total_dp_count
#                 * label_pairs
#             )
#         return node_per_label_frequencies, node_total_dp_count

#     calculate_purity(tree_root, 0)
#     return purity

def cluster_acc(y_true, y_pred, return_index=False):
	"""
	Calculate clustering accuracy.
	# Arguments
		y: true labels, numpy.array with shape `(n_samples,)`
		y_pred: predicted labels, numpy.array with shape `(n_samples,)`
	# Return
		accuracy, in [0,1]
	"""
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
	w = np.zeros((int(D), (D)), dtype=np.int64)
	for i in range(y_pred.size):
		w[int(y_pred[i]), int(y_true[i])] += 1
	ind = np.array(linear_assignment(w.max() - w))
	if return_index:
		assert all(ind[0] == range(len(ind[0])))  # Assert rows don't change order
		cluster_acc = sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size
		return cluster_acc
	else:
		return sum([w[ind[0,i], ind[1,i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size




class TreeNode:
    def __init__(self, decoder=False, router=None, left=None, right=None, samples=None, name=''):
        self.decoder = decoder  # Whether this node is a leaf node
        self.router = router    # Stores the router condition if this node is an internal split node
        self.left = left        # Left child
        self.right = right      # Right child
        self.samples = samples  # Stores sample indices for leaf nodes (valid only if decoder=True)
        self.father = None      # Father node (optional, if needed)
        self.name = name        # Node name
        
    def print_tree(self, depth=0):
        # print('name', self.name)
        if self.decoder:
            # import pdb; pdb.set_trace()
            print('leaf node, number of samples:', self.samples.shape[0])
        else:
            print('router node:', self.router, 
                  'left:', self.left.name if self.left else None, 
                  'right:', self.right.name if self.right else None)
            if self.left:
                self.left.print_tree(depth + 1)
            if self.right:
                self.right.print_tree(depth + 1)

    def bool_mergeable(self):
        if self.father.left.name == self.name and self.father.right.left is None:
            return True
        elif self.father.right.name == self.name and self.father.left.left is None:
            return True
        else:
            return False
        

    def list_all_leaf(self):
        if self.decoder:
            return [self]
        else:
            leaves = []
            if self.left:
                leaves.extend(self.left.list_all_leaf())
            if self.right:
                leaves.extend(self.right.list_all_leaf())
            return leaves
    
    def plot_all_tree_with_matplotlib(self):
        
        def plot_tree(node, ax, x, y, x_offset, y_offset):
            """
            node     : current node
            ax       : matplotlib Axes object
            x, y     : coordinates (x, y) of the current node on the canvas
            x_offset : horizontal offset for child nodes
            y_offset : vertical offset for child nodes
            """
            if node is None:
                return

            # If node is a leaf, use red; otherwise use blue
            color = 'red' if node.decoder else 'blue'
            
            # Plot current node at (x, y)
            ax.text(
                x, y, 
                node.name,
                ha='center', va='center',
                bbox=dict(facecolor=color, alpha=0.5)
            )

            # If there is a left child, draw a line and recurse
            if node.left:
                # Draw a line from current node (x, y) to (x - x_offset, y - y_offset)
                ax.plot([x, x - x_offset], [y, y - y_offset], 'k-')
                plot_tree(node.left, ax, x - x_offset, y - y_offset, x_offset*0.5, y_offset)
            
            # If there is a right child, draw a line and recurse
            if node.right:
                ax.plot([x, x + x_offset], [y, y - y_offset], 'k-')
                plot_tree(node.right, ax, x + x_offset, y - y_offset, x_offset*0.5, y_offset)

        # Create a figure and an Axes
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust according to your needs

        # Start plotting from the root node (self), with given coordinates and offsets
        # x=0, y=0       : root node at the origin
        # x_offset=3     : horizontal offset for the children
        # y_offset=1.5   : vertical offset for the children
        plot_tree(self, ax, x=0, y=0, x_offset=5, y_offset=1.5)

        # Manually set axis limits so that the entire tree is visible
        # Increase or decrease the range if the tree is very large or very small
        ax.set_xlim(-15, 15)
        ax.set_ylim(-10, 5)

        # Alternatively, you could use ax.autoscale() if you prefer automatic scaling,
        # but you may need to remove bbox_inches='tight' to avoid text clipping
        # ax.autoscale(enable=True, axis='both', tight=True)

        # Hide the axes (optional)
        plt.axis('off')

        # Save the figure and close
        # plt.savefig(path, bbox_inches='tight')
        # plt.close(fig)
        return fig
    
    def output_label_list(self):
        
        num_sample_list = [l.samples.shape[0] for l in  reversed(self.list_all_leaf()) ]
        num_sample_all = np.sum(num_sample_list)
        predict_label = np.zeros(num_sample_all)
        for i, l in enumerate(reversed(self.list_all_leaf())):
            print(f'leaf {l.name} has {l.samples.shape[0]} samples')
            if l.samples is not None and l.samples.shape[0] > 0:
                predict_label[l.samples] = i
        return predict_label


def calculate_metrics_Embedding(children_array, labels, y_true):
    root = to_dendrogram_purity_tree(children_array)
    leaf_purity_value = calculate_tree_leaf_purity_mean(children_array, y_true)
    dendrogram_purity_value = dendrogram_purity(root, y_true) if root else None
    acc_value = cluster_acc(y_true, labels)
    nmi_value = normalized_mutual_info_score(y_true, labels)
    ari_value = adjusted_rand_score(y_true, labels)
    return {
        "Leaf Purity": leaf_purity_value,
        "Dendrogram Purity": dendrogram_purity_value,
        "Accuracy": acc_value,
        "NMI": nmi_value,
        "ARI": ari_value
    }


def build_tree(feature, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None, compute_full_tree=True)
    clustering.fit(feature)
    return clustering.children_, clustering.labels_


def metric_difftree(feature, label, class_num, tree=None, set_str='train'):
    if tree is None:
        children_array, preds = build_tree(feature, n_clusters=class_num)
    else:
        children_array, preds = None, tree

    # import pdb; pdb.set_trace()
    # Compute evaluation metrics
    metrics = calculate_metrics_Embedding(children_array, preds, label)
    HI_metrics = calculate_hierarchical_integrity(feature, preds)       
    
    # print(f"Metrics for {args.embedding_method}:")
    # print(f"Hierarchical Integrity: {HI_metrics}")
    for metric, value in metrics.items():
        metric_str = f"{set_str}/{metric}"
        # print(f"{set_str}/{metric}: {value:.4f}")
        # Log metrics to wandb
        wandb.log({metric_str: value})

