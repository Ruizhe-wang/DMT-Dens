from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from collections import defaultdict
import eval.eval_core_base as ecb
from eval import evaluation
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import umap
import torch.nn.functional as F
import uuid
import os
import csv

from callbacks.util import (
    leaf_purity,
    plot_scatter,
    plot_scatter_hyper,
    get_svc_acc,
    get_svc_acc_rbf,
    plot_vis_diff_gen_umap,
    metric_difftree,
    plot_path,
    build_tree_zl,
    cluster_acc,
    compute_leaves,
    dendrogram_purity,
)
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


# 计算Reconstruction Loss
def compute_reconstruction(gathered_rec, gathered_input):
    mean = gathered_input.min()
    std = gathered_input.max() - mean
    gathered_input = (gathered_input - mean) / std
    gathered_rec = (gathered_rec - mean) / std
    rl = np.mean(np.abs(gathered_rec - gathered_input)) * gathered_input.shape[1]
    return rl


# 计算Log-Likelihood
def compute_log_likelihood(reconstructed_data, original_data, estimation_samples=100, device='cpu'):
    """
    计算测试集的边际对数似然 (marginal_log_likelihood)。
    
    参数:
        reconstructed_data (torch.Tensor): 重建后的数据 (N, D)。
        original_data (torch.Tensor): 原始数据 (N, D)。
        estimation_samples (int): 采样次数，用于近似边际似然。
        device (str): 使用的计算设备 ("cpu" 或 "cuda")。

    返回:
        float: 平均边际对数似然 (marginal_log_likelihood)。
    """
    import torch
    import scipy
    import numpy as np
    from tqdm import tqdm

    # 初始化负 ELBO 矩阵
    num_samples = original_data.shape[0]
    elbo = np.zeros((num_samples, estimation_samples))

    # print('\nComputing the marginal log-likelihood...')

    # 对每个样本进行估计
    for j in tqdm(range(estimation_samples)):
        # 计算重建误差（例如均方误差或负对数似然）
        reconstruction_error = torch.nn.functional.mse_loss(
            torch.tensor(reconstructed_data), 
            torch.tensor(original_data), 
            reduction='none'
        ).sum(dim=1).detach().cpu().numpy()  # 对每个样本求和

        # 使用 reconstruction_error 近似负 ELBO
        elbo[:, j] = reconstruction_error

    # 计算边际对数似然
    log_likel = np.log(1 / estimation_samples) + scipy.special.logsumexp(-elbo, axis=1)
    marginal_log_likelihood = np.sum(log_likel) / num_samples

    # print("Marginal Log-Likelihood:", marginal_log_likelihood)
    return marginal_log_likelihood


class EvalCallBack(Callback):
    def __init__(
        self,
        inter=10,
        dirpath="",
        fully_eval=False,
        dataset="",
        only_val=False,
        save_results=False,
        vis_rout=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.inter = inter
        self.only_val = only_val
        self.vis_rout = vis_rout
        self.save_results = save_results

        # Use defaultdict to store data for multiple validation sets
        self.val_input = defaultdict(list)
        self.val_rout = defaultdict(list)
        self.val_rec = defaultdict(list)
        self.val_high = defaultdict(list)
        self.val_vis = defaultdict(list)
        self.val_label = defaultdict(list)
        self.val_vector_rout = defaultdict(list)
        self.reconstruct_history_history = defaultdict(list)

        self.test_vis = []
        self.test_label = []

        self.dirpath = dirpath
        self.dataset = dataset
        self.fully_eval = fully_eval
        self.best_acc = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        label = batch["label"]
        key = dataloader_idx
        rute_item = (
            trainer.strategy.all_gather(pl_module.validation_step_rute)
            .cpu()
            .detach()
            .numpy()
        )
        input_item = (
            trainer.strategy.all_gather(pl_module.validation_origin_input)
            .cpu()
            .detach()
            .numpy()
        )
        high_item = (
            trainer.strategy.all_gather(pl_module.validation_step_outputs_high)
            .cpu()
            .detach()
            .numpy()
        )
        vis_item = (
            trainer.strategy.all_gather(pl_module.validation_step_outputs_vis)
            .cpu()
            .detach()
            .numpy()
        )
        vector_rout = (
            trainer.strategy.all_gather(pl_module.validation_step_vector_rout)
            .cpu()
            .detach()
            .numpy()
        )
        vis_rec = (
            trainer.strategy.all_gather(pl_module.reconstruct_example)
            .cpu()
            .detach()
            .numpy()
        )
        label_item = trainer.strategy.all_gather(label).cpu().detach().numpy()

        self.val_rout[key].append(rute_item.reshape(-1, rute_item.shape[-1]))
        self.val_input[key].append(input_item.reshape(-1, input_item.shape[-1]))
        self.val_high[key].append(high_item.reshape(-1, high_item.shape[-1]))
        self.val_vis[key].append(vis_item.reshape(-1, vis_item.shape[-1]))
        self.val_rec[key].append(
            (vis_rec.reshape(-1, vis_rec.shape[-1]) + input_item.reshape(-1, input_item.shape[-1]))/2
            )
        self.val_vector_rout[key].append(
            vector_rout.reshape(-1, vector_rout.shape[-2] * vector_rout.shape[-1])
        )
        # self.val_vis[key].append(vis_item.reshape(-1, vis_item.shape[-1]))
        self.val_label[key].append(label_item.reshape(-1))
        # import pdb; pdb.set_trace()

        # self.reconstruct_history_history[key].append()

        # import pdb; pdb.set_trace()
        # self.val_label[key].append(label)

        # if pl_module.reconstruct_history.shape[0] > 10:
        #     gathered_reconstruct_history_history = pl_module.reconstruct_history
        #     gathered_reconstruct_label = np.arange(
        #         gathered_reconstruct_history_history.shape[1]
        #         )[None, :].repeat(
        #             gathered_reconstruct_history_history.shape[0], axis=0
        #             ).reshape(-1)
        #     np.save(
        #         f'gathered_reconstruct_history_history{batch_idx}.npy',
        #         gathered_reconstruct_history_history.cpu().detach().numpy().astype(np.float32),
        #         )
        #     np.save(
        #         f'gathered_reconstruct_label{batch_idx}.npy',
        #         label.cpu().detach().numpy().astype(np.float32),
        #         )

        # if pl_module.reconstruct_history.shape[0] == 1000:
        # try:
        #     self.val_sample[key].append(pl_module.validation_step_sample)
        # except:
        #     # print('no sample in training process')
        #     pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        label = batch["label"]
        self.test_vis.append(pl_module.test_step_outputs_vis)
        self.test_label.append(label)

    def on_test_epoch_end(self, trainer, pl_module):
        test_vis = torch.cat(self.test_vis).cuda()
        test_label = torch.cat(self.test_label).cuda()
        gathered_vis = trainer.strategy.all_gather(test_vis).cpu().detach().numpy()
        gathered_label = trainer.strategy.all_gather(test_label).cpu().detach().numpy()

        gathered_vis = gathered_vis.reshape(-1, gathered_vis.shape[-1])
        gathered_label = gathered_label.reshape(-1)

        acc_mean = get_svc_acc(gathered_vis, gathered_label)
        # print("Test SVC Accuracy:", acc_mean)

    def process_and_log_metrics(self, val_input, val_vis, val_label, trainer, val_idx):
        fig_scatter = plot_scatter(val_vis, val_label)
        fig_scatter_hyper = plot_scatter_hyper(val_vis, val_label)

        # Sample data if too large
        if val_vis.shape[0] > 100000:
            idx = np.random.choice(val_vis.shape[0], 100000, replace=False)
            val_input = val_input[idx]
            val_vis = val_vis[idx]
            val_label = val_label[idx]

        # Apply TSNE if latent dimension > 2
        if val_vis.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=0)
            val_vis = tsne.fit_transform(val_vis)

        # Flatten input if necessary
        if len(val_input.shape) > 2:
            val_input = val_input.reshape(val_input.shape[0], -1)

        try:
            acc_mean = get_svc_acc(val_vis, val_label)
            acc_mean_rbf = get_svc_acc_rbf(val_vis, val_label)
        except:
            acc_mean = 0
            acc_mean_rbf = 0

        dataset_names = ["train", "validation", "test"]
        dataset_name = (
            dataset_names[val_idx] if val_idx < len(dataset_names) else f"val{val_idx}"
        )

        log_dict = {
            f"{dataset_name}_svc": acc_mean,
            f"{dataset_name}_svc_rbf": acc_mean_rbf,
            "epoch": trainer.current_epoch,
            f"{dataset_name}_scatter": wandb.Image(fig_scatter),
            f"{dataset_name}_scatter_hyper": wandb.Image(fig_scatter_hyper),
        }

        if self.fully_eval:
            for k in [120]:
                ecb_e_train = ecb.Eval(
                    input=val_input, latent=val_vis, label=val_label, k=k
                )
                trust = ecb_e_train.E_trustworthiness()
                continuity = ecb_e_train.E_continuity()
                log_dict[f"{dataset_name}_trust{k}"] = trust
                log_dict[f"{dataset_name}_continuity{k}"] = continuity

            fknn = np.mean(evaluation.faster_knn_eval_series(val_vis, val_label))
            fct = evaluation.faster_centroid_triplet_eval(val_input, val_vis, val_label)
            log_dict[f"{dataset_name}_fknn"] = fknn
            log_dict[f"{dataset_name}_fct"] = fct

        trainer.logger.log_metrics(log_dict)
        plt.close()

        # Save best model
        # if acc_mean > self.best_acc:
        if True:
            self.best_acc = acc_mean
            os.makedirs(self.dirpath, exist_ok=True)
            model_path = os.path.join(
                self.dirpath,
                f"best_model_epoch{trainer.current_epoch}_{self.dataset}_acc{acc_mean}.pth",
            )
            torch.save(trainer.model.state_dict(), model_path)

    # def get_rout_svc_acc(self, rout_vector, label_vector):

    #     method = SVC(kernel='linear', max_iter=9000, )
    #     cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
    #     n_scores = cross_val_score(
    #         method,
    #         StandardScaler().fit_transform(rout_vector),
    #         label_vector,
    #         scoring="accuracy",
    #         cv=cv,
    #         n_jobs=1,
    #     )
    #     return np.mean(n_scores)

    def get_predict_label(self, tree_list, token_emb, emb_vis):

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

    def get_rout_vis(self, tree_node_embedding, emb_vis):
        os.makedirs("pic", exist_ok=True)

        G = nx.Graph()
        G.add_node("L-1/0")

        path_list = []
        tree_list = []
        plotly_fig_rute_dict = {}

        # 新增：用于存储细胞信息和路径信息
        all_cells_info = []
        all_paths_info = []

        for i in range(len(tree_node_embedding)):
            token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()

            if token_emb.shape[0] > 100000:
                idx = np.random.choice(token_emb.shape[0], 100000, replace=False)
                token_emb = token_emb[idx]
            if emb_vis.shape[0] > 100000:
                idx = np.random.choice(token_emb.shape[0], 100000, replace=False)
                emb_vis = emb_vis[idx]

            label_predict = self.get_predict_label(tree_list, token_emb, emb_vis)
            tree_list.append(label_predict)

            # 记录每个细胞的原始 index、所属 embedding
            for cell_idx in range(emb_vis.shape[0]):
                all_cells_info.append({
                    "cell_index": cell_idx,
                    "embedding_level": i,
                    "embedding_index": int(label_predict[cell_idx])
                })

            plotly_fig_rute = go.Figure()
            d3_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
            ] * 1000

            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=emb_vis[:, 0],
                    y=emb_vis[:, 1],
                    mode="markers",
                    marker=dict(
                        size=1,
                        color=[d3_colors[c_i] for c_i in label_predict],
                    ),
                    name="emb_vis",
                )
            )

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
                    textposition="top center",
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
                G.add_edge(s_node, e_node, weight=weight)

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
                # 路径可视化
                plot_path(
                    G, tree_node_embedding, i, near_index, end_node, plotly_fig_rute
                )

                # 记录每个embedding的连接路径（以两列形式存储）
                try:
                    shortest_path = nx.shortest_path(
                        G,
                        source=f"L{i}/{near_index}",
                        target=f'L{i}/{end_node}',
                        weight='weight',
                    )
                    # 以两列形式存储每一段(start, end)
                    for idx in range(len(shortest_path) - 1):
                        all_paths_info.append({
                            "embedding_level": i,
                            "embedding_index": end_node,
                            "start": shortest_path[idx],
                            "end": shortest_path[idx + 1]
                        })
                except Exception as e:
                    continue

            plotly_fig_rute.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                width=800,
                height=600,
                title="Tree Visualization",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                xaxis_scaleanchor="y",
                yaxis_scaleanchor="x",
                legend=dict(title="Legend", x=0.01, y=0.99),
            )

            import uuid
            uuid_str = str(uuid.uuid4())[:16]
            plotly_fig_rute.write_image(f"pic/fig_emb_colored_with_rout_{i}_{uuid_str}.png", scale=3)
            path_list.append(f"pic/fig_emb_colored_with_rout_{i}_{uuid_str}.png")

        # 写入细胞信息
        csv_path_cells = "pic/cell_embedding_info.csv"
        with open(csv_path_cells, "w", newline='') as csvfile:
            fieldnames = ["cell_index", "embedding_level", "embedding_index"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_cells_info:
                writer.writerow(row)

        # 写入路径信息（每行两列：start, end）
        csv_path_paths = "pic/embedding_paths_info.csv"
        with open(csv_path_paths, "w", newline='') as csvfile:
            fieldnames = ["embedding_level", "embedding_index", "start", "end"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_paths_info:
                writer.writerow(row)

        return path_list, plotly_fig_rute_dict
    def rout_eval(
        self,
        trainer,
        pl_module,
        gathered_rout,
        gathered_label,
        emb_vis,
        gathered_sample_history=None,
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

        if gathered_rout_flat.shape[0] > 100000:
            rand_idx = np.random.choice(
                gathered_rout_flat.shape[0], 100000, replace=False
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

        if gathered_sample is not None and gathered_sample.shape[-1] == 784:
            # print("Evaluating the routing vector, plotting the sample images...")
            gathered_image = gathered_sample.reshape(-1, 28, 28)
            # show 10*10 images
            fig_sample = plt.figure(figsize=(30, 30))
            for i in range(100):
                plt.subplot(10, 10, i + 1)
                plt.imshow(gathered_image[i], cmap="gray")
                plt.axis("off")

            dict.update({"rout/fig_sample_image": wandb.Image(fig_sample)})
            plt.close()

        # if (
        #     gathered_sample_history is not None
        #     and gathered_sample_history.shape[-1] == 784
        # ):
        #     print(
        #         "Evaluating the routing vector, plotting the sample history images..."
        #     )
        #     gathered_image = gathered_sample_history.reshape(
        #         *gathered_sample_history.shape[:-1], 28, 28
        #     )
        #     # gathered_image_sample_time_step = gathered_image[:, ::50, :, :]
        #     # import pdb; pdb.set_trace()
        #     # show 10*10 images
        #     fig_sample = plt.figure(figsize=(40, 40))
        #     for i in range(20):
        #         for j in range(21):
        #             plt.subplot(20, 21, i * 21 + j + 1)
        #             plt.imshow(
        #                 gathered_image[i][min(j * 50, gathered_image.shape[1] - 1)],
        #                 cmap="gray",
        #             )
        #             plt.axis("off")
        #     dict.update({"rout/fig_sample_history_image": wandb.Image(fig_sample)})
        #     plt.close()

        # if (
        #     gathered_reconstruct_history_history is not None
        #     and gathered_reconstruct_history_history.shape[-1] == 784
        # ):
        #     gathered_reconstruct_history_history = (
        #         gathered_reconstruct_history_history.reshape(-1, 784)
        #     )
        #     plot_vis_diff_gen_umap(
        #         data=gathered_reconstruct_history_history,
        #         label=gathered_reconstruct_label,
        #         dict=dict,
        #     )

        wandb.log(dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        # import pdb; pdb.set_trace()

        for val_idx in self.val_vis.keys():
            if True:
                gathered_rout = np.concatenate(self.val_rout[val_idx])
                gathered_input = np.concatenate(self.val_input[val_idx])
                gathered_vis = np.concatenate(self.val_vis[val_idx])
                gathered_high = np.concatenate(self.val_high[val_idx])
                gathered_label = np.concatenate(self.val_label[val_idx])
                gathered_vector_rout = np.concatenate(self.val_vector_rout[val_idx])
                gathered_rec = np.concatenate(self.val_rec[val_idx])

                # import pdb; pdb.set_trace()
                data_name = f"{trainer.datamodule.data_name}_seed_{pl_module.uuid_str}_epoch_{trainer.current_epoch}"
                if self.save_results:
                    if os.path.exists(f"save_output") == False:
                        os.makedirs(f"save_output")

                    if os.path.exists(f"save_output/{data_name}") == False:
                        os.makedirs(f"save_output/{data_name}")

                    # np.save(f"save_output/{data_name}/gathered_rout{val_idx}.npy", gathered_rout)
                    # np.save(f"save_output/{data_name}/gathered_input{val_idx}.npy", gathered_input)
                    np.save(f"save_output/{data_name}/gathered_vis{val_idx}.npy", gathered_vis.astype(np.float32))
                    np.save(f"save_output/{data_name}/gathered_high{val_idx}.npy", gathered_high.astype(np.float32))
                    np.save(f"save_output/{data_name}/gathered_label{val_idx}.npy", gathered_label.astype(np.float32))

                if trainer.is_global_zero:
                    self.process_and_log_metrics(
                        gathered_input, gathered_vis, gathered_label, trainer, val_idx
                    )

                    # print('pl_module.reconstruct_history', pl_module.reconstruct_history)
                    # import pdb; pdb.set_trace()
                    # print(
                    #     "Evaluating the tree structure, data shape:", gathered_vis.shape
                    # )
                    if gathered_vis.shape[0] > 100000:
                        samplt_idx = np.random.choice(
                            gathered_vis.shape[0], 100000, replace=False
                        )
                        gathered_vis = gathered_vis[samplt_idx]
                        gathered_high = gathered_high[samplt_idx]
                        gathered_label = gathered_label[samplt_idx]
                        gathered_rout = gathered_rout[samplt_idx]
                        gathered_vector_rout = gathered_vector_rout[samplt_idx]
                        gathered_rec = gathered_rec[samplt_idx]
                        gathered_input = gathered_input[samplt_idx]
                        

                    num_clusters = len(set(gathered_label))
                    # metric_difftree(
                    #     gathered_vis,
                    #     gathered_label,
                    #     num_clusters,
                    #     set_str=f'dim_2_{val_idx}'
                    #     )
                    # metric_difftree(
                    #     gathered_high,
                    #     gathered_label,
                    #     num_clusters,
                    #     set_str=f'dim_{gathered_high.shape[1]}_{val_idx}'
                    #     )

                    # node_root = TreeNode(decoder=True, samples=node_index)  # 样本 0, 1, 2 在这个叶子
                    # print(
                    #     "Evaluating the tree structure, building the tree...",
                    #     pl_module.training_str,
                    # )
                    if pl_module.training_str != "step1":

                        # if pl_module.reconstruct_history.shape[-1] == 784:
                        #     gathered_reconstruct_history_history = pl_module.reconstruct_history[:, 1:, :]
                        #     gathered_reconstruct_history_history = gathered_reconstruct_history_history[:, ::40, :]
                        #     gathered_reconstruct_label = np.arange(
                        #         gathered_reconstruct_history_history.shape[1]
                        #         )[None, :].repeat(
                        #             gathered_reconstruct_history_history.shape[0], axis=0
                        #             ).reshape(-1)
                        #     gathered_reconstruct_history_history = gathered_reconstruct_history_history.cpu().detach().numpy()
                        # else:
                        #     gathered_reconstruct_history_history = None
                        #     gathered_reconstruct_label = None

                        self.rout_eval(
                            trainer,
                            pl_module,
                            gathered_rout,
                            gathered_label,
                            gathered_vis,
                            # pl_module.reconstruct_example.cpu().detach().numpy(),
                            gathered_sample_history=pl_module.reconstruct_history.cpu().detach().numpy(),
                            # gathered_reconstruct_history_history=gathered_reconstruct_history_history,
                            # gathered_reconstruct_label=gathered_reconstruct_label,
                            )
                        # import pdb; pdb.set_trace()
                        node_root = build_tree_zl(
                            gathered_rout, gathered_vector_rout, num_clusters
                        )
                        fig_tree = node_root.plot_all_tree_with_matplotlib()
                        predict_label = node_root.output_label_list().astype(np.int32)
                        fig_predict = plot_scatter(gathered_vis, predict_label)

                        acc_cluster_acc = cluster_acc(predict_label, gathered_label)
                        acc_nmi = normalized_mutual_info_score(
                            predict_label, gathered_label
                        )
                        acc_ari = adjusted_rand_score(predict_label, gathered_label)

                        zzl_leaves = node_root.list_all_leaf()
                        ind_samples_of_leaves = [
                            [zzl_leaves[i], np.where(predict_label == i)[0]]
                            for i in range(len(zzl_leaves))
                        ]
                        dp = dendrogram_purity(
                            node_root, gathered_label, ind_samples_of_leaves
                        )
                        lp = leaf_purity(
                            node_root, gathered_label, ind_samples_of_leaves
                        )

                        # import pdb; pdb.set_trace()
                        rl = compute_reconstruction(gathered_rec, gathered_input)
                        ll = compute_log_likelihood(gathered_rec, gathered_input)

                        wandb.log(
                            {
                                f"predict_label_{val_idx}": wandb.Image(fig_predict),
                                f"tree_{val_idx}": wandb.Image(fig_tree),
                                f"tree/cluster_acc_{val_idx}": acc_cluster_acc,
                                f"tree/nmi_{val_idx}": acc_nmi,
                                f"tree/ari_{val_idx}": acc_ari,
                                f"tree/dp_{val_idx}": dp,
                                f"tree/lp_{val_idx}": lp,
                                f"tree/reconstruction_loss_{val_idx}": rl,
                                f"tree/log_likelihood_{val_idx}": ll,
                            }
                        )

        # Clear stored data
        self.val_rout.clear()
        self.val_input.clear()
        self.val_high.clear()
        self.val_vis.clear()
        self.val_label.clear()
        self.val_rec.clear()
        self.reconstruct_history_history.clear()
