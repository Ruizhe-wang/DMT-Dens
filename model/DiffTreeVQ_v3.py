import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy
from lightning import LightningModule
import uuid
from model.CosineAnnealingSchedule import CosineAnnealingSchedule

from model.encoder import DMTEncoder as TransformerEncoder
from model.encoder import NN_FCBNRL_MM
import math


class DMTEVT_model(LightningModule):
    """
    DMTEVT_model is a PyTorch Lightning module that implements the training and evaluation of the model.
    """

    def __init__(
        self,
        lr=0.005,
        sigma=0.05,
        sample_rate_feature=0.6,
        num_input_dim=64,
        num_train_data=60000,
        weight_decay=0.0001,
        exaggeration_lat=1,
        exaggeration_emb=1,
        weight_mse=2,
        weight_nepo=1,
        nu_lat=0.1,
        nu_emb=0.1,
        tau=1,
        T_num_layers=2,
        T_num_attention_heads=6,
        T_hidden_size=240,
        T_intermediate_size=300,
        t_output_dim=512,
        T_hidden_dropout_prob=0.1,
        T_attention_probs_dropout_prob=0.1,
        ckpt_path=None,
        use_orthogonal=False,
        num_use_moe=1,
        vis_dim=2,
        trans_out_dim=50,
        max_epochs=600,
        ec_ce_weight=10.0,
        n_neg_sample=4,
        test_noise=False,
        training_str=None,
        tree_depth=10,
        n_timestep=1000,
        epoch_num_base=0,
        validate_bool=False,
        weight_e_latent=0.25,
        step2_epoch=2000,
        step2_r_epoch=4000,
        use_tree_rout=False,
        gen_data_bool=False,
        weightrout=0.1,
        loss_type="G",
        all_g_l_weight=0.1,
        exp_b=0.3,
        exp_min=1e-5,
        t_mul=1.0,  # 距离缩放系数（此片段中 Global loss 的 dis_ab 乘法）
        num_use_mlevel_list=[0.5, 1, 2, 5],  # 多尺度/多 level 的距离缩放列表
        use_lg_loss=False,
        lg_weight=0.1,
        lg_teacher_momentum=0.99,
        lg_num_anchors=4,
        lg_subset_k=32,
        lg_distance_weight=1.0,
        lg_position_weight=0.2,
        lg_warmup_epochs=0,        # lg_loss 在前 N 个 epoch 不启用
        **kwargs,
    ):
        """
        Initializes the model with given hyperparameters.

        Args:
            lr (float): Learning rate.
            sigma (float): Sigma parameter for similarity function.
            sample_rate_feature (float): Sampling rate for features.
            num_input_dim (int): Input dimension size.
            num_train_data (int): Number of training data samples.
            weight_decay (float): Weight decay for optimizer.
            exaggeration_lat (float): Exaggeration parameter for latent space.
            exaggeration_emb (float): Exaggeration parameter for embedding space.
            weight_mse (float): Weight for MSE loss.
            weight_nepo (float): Weight for NEPO loss.
            nu_lat (float): Degrees of freedom for t-distribution in latent space.
            nu_emb (float): Degrees of freedom for t-distribution in embedding space.
            tau (float): Temperature parameter.
            T_num_layers (int): Number of layers in Transformer.
            T_num_attention_heads (int): Number of attention heads in Transformer.
            T_hidden_size (int): Hidden size in Transformer.
            T_intermediate_size (int): Intermediate size in Transformer.
            T_hidden_dropout_prob (float): Dropout probability in Transformer.
            T_attention_probs_dropout_prob (float): Dropout probability for attention in Transformer.
            ckpt_path (str): Path to checkpoint for loading model.
            use_orthogonal (bool): Whether to use orthogonal loss.
            num_use_moe (int): Number of experts in Mixture of Experts.
            vis_dim (int): Dimension of visualization space.
            trans_out_dim (int): Output dimension of Transformer.
            max_epochs (int): Maximum number of epochs.
            v_latent (float): Variance parameter in latent space.
            n_neg_sample (int): Number of negative samples.
            test_noise (bool): Whether to test with noise.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.setup_bool_zzl = False
        # self.save_hyperparameters()
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        self.lat_vis_mean = nn.Parameter(torch.zeros(2))
        self.lat_vis_std = nn.Parameter(torch.zeros(2))
        self.init_imge = None
        self.set_mean_bool = False

        self.uuid_str = str(uuid.uuid4())[:10]

        # 多投影头最佳选择：追踪当前最优投影头索引
        # -1 表示尚未选择，默认使用最后一个头
        self.best_head_index = -1
        # 用于在 validation epoch 中累积各投影头的嵌入数据
        self._val_head_data = []

        if self.hparams.nu_emb < 0:
            self.hparams.nu_emb = self.hparams.nu_lat
        if self.hparams.exaggeration_emb < 0:
            self.hparams.exaggeration_emb = self.hparams.exaggeration_lat

        # Initialize the encoder
        self.enc = TransformerEncoder(
            num_layers=T_num_layers,
            num_attention_heads=T_num_attention_heads,
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
            output_dim=t_output_dim,
        )

        self.vis_list = nn.ModuleList(
            self.InitNetworkMLP(
                # NS=[t_output_dim * num_use_moe, 500, vis_dim], last_relu=False
                NS=[t_output_dim, 500, vis_dim],
                last_relu=False,
            )
            for _ in num_use_mlevel_list
        )

        self.enc_teacher = None
        self.vis_list_teacher = None
        if self.hparams.use_lg_loss:
            self.enc_teacher = copy.deepcopy(self.enc)
            self.vis_list_teacher = copy.deepcopy(self.vis_list)
            self._freeze_lg_teacher()
            self._sync_lg_teacher(momentum=0.0)

        if training_str == None:
            self.training_str = "step1"
        else:
            self.training_str = training_str

        self.validate_bool = validate_bool

    def InitNetworkMLP(self, NS, last_relu=True, use_DO=True, use_BN=True, use_RL=True):
        """
        Initializes a multi-layer perceptron (MLP) network.

        Args:
            NS (list): List of layer sizes.
            last_relu (bool): Whether to use ReLU activation on the last layer.
            use_DO (bool): Whether to use Dropout.
            use_BN (bool): Whether to use BatchNorm.
            use_RL (bool): Whether to use LeakyReLU activation.

        Returns:
            model_pat (nn.Sequential): The MLP network.
        """
        layers = []
        for i in range(len(NS) - 1):
            # Determine if last layer should have activation
            if i == len(NS) - 2 and not last_relu:
                layers.append(
                    NN_FCBNRL_MM(
                        NS[i], NS[i + 1], use_RL=False, use_DO=use_DO, use_BN=use_BN
                    )
                )
            else:
                layers.append(
                    NN_FCBNRL_MM(
                        NS[i], NS[i + 1], use_RL=use_RL, use_DO=use_DO, use_BN=use_BN
                    )
                )
        model_pat = nn.Sequential(*layers)
        return model_pat

    def _freeze_lg_teacher(self):
        if self.enc_teacher is None or self.vis_list_teacher is None:
            return
        self.enc_teacher.eval()
        self.vis_list_teacher.eval()
        for param in self.enc_teacher.parameters():
            param.requires_grad = False
        for param in self.vis_list_teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _sync_lg_teacher(self, momentum=None):
        if self.enc_teacher is None or self.vis_list_teacher is None:
            return

        if momentum is None:
            momentum = self.hparams.lg_teacher_momentum

        for teacher_param, student_param in zip(
            self.enc_teacher.parameters(), self.enc.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )

        for teacher_param, student_param in zip(
            self.vis_list_teacher.parameters(), self.vis_list.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )

        for teacher_buffer, student_buffer in zip(
            self.enc_teacher.buffers(), self.enc.buffers()
        ):
            teacher_buffer.copy_(student_buffer)

        for teacher_buffer, student_buffer in zip(
            self.vis_list_teacher.buffers(), self.vis_list.buffers()
        ):
            teacher_buffer.copy_(student_buffer)

        self.enc_teacher.eval()
        self.vis_list_teacher.eval()

    @staticmethod
    def _normalize_embedding_positions(emb):
        emb = emb.float()
        emb = emb - emb.mean(dim=0, keepdim=True)
        std = emb.std(dim=0, unbiased=False, keepdim=True)
        scale = std.clamp_min(std.max() * 0.01 + 1e-8)
        return emb / scale

    @staticmethod
    def _normalize_distance_vector(dist_vec):
        if dist_vec.numel() == 0:
            return dist_vec
        dist_vec = dist_vec.float()
        mean = dist_vec.mean()
        std = dist_vec.std(unbiased=False).clamp_min(mean.abs() * 0.01 + 1e-8)
        return (dist_vec - mean) / std

    @staticmethod
    def _pairwise_distance_vector(emb):
        n = emb.shape[0]
        if n < 2:
            return emb.new_zeros((0,))
        emb = emb.float()
        dist_mat = torch.cdist(emb, emb)
        tri_idx = torch.triu_indices(n, n, offset=1, device=emb.device)
        return dist_mat[tri_idx[0], tri_idx[1]]

    def _sample_local_subsets(self, batch_data):
        batch_size = batch_data.shape[0]
        if batch_size < 4:
            return []

        num_anchors = min(int(self.hparams.lg_num_anchors), batch_size)
        subset_k = min(int(self.hparams.lg_subset_k), batch_size)
        if num_anchors <= 0 or subset_k < 3:
            return []

        anchor_indices = torch.randperm(batch_size, device=batch_data.device)[
            :num_anchors
        ]
        anchor_points = batch_data[anchor_indices].float()
        distances = torch.cdist(anchor_points, batch_data.float())
        neighbor_indices = torch.topk(
            distances, k=subset_k, dim=1, largest=False
        ).indices
        return [idx for idx in neighbor_indices]

    def _compute_local_global_consistency_loss(self, batch_data, student_lat_vis_list):
        if (
            not self.hparams.use_lg_loss
            or self.enc_teacher is None
            or self.vis_list_teacher is None
        ):
            zero = batch_data.new_zeros(())
            return zero

        # Warmup: 前 lg_warmup_epochs 个 epoch 不启用 lg_loss，
        # 让 manifold loss 先建立基本的 2D 结构，避免早期锁定退化模式
        if self.current_epoch < self.hparams.lg_warmup_epochs:
            return batch_data.new_zeros(())

        flat_batch = batch_data.reshape(batch_data.shape[0], -1)
        subsets = self._sample_local_subsets(flat_batch)
        if not subsets:
            return flat_batch.new_zeros(())

        with torch.no_grad():
            teacher_hidden = self.enc_teacher(batch_data)
            teacher_lat_vis_list = [
                head(teacher_hidden) for head in self.vis_list_teacher
            ]

        total_loss = flat_batch.new_zeros(())
        n_terms = 0

        for subset_idx in subsets:
            for student_emb, teacher_emb in zip(
                student_lat_vis_list, teacher_lat_vis_list
            ):
                student_subset = student_emb[subset_idx]
                teacher_subset = teacher_emb[subset_idx]

                student_norm = self._normalize_embedding_positions(student_subset)
                teacher_norm = self._normalize_embedding_positions(teacher_subset)

                pos_loss = F.mse_loss(student_norm, teacher_norm)

                student_dist = self._pairwise_distance_vector(student_norm)
                teacher_dist = self._pairwise_distance_vector(teacher_norm)
                dist_loss = F.mse_loss(
                    self._normalize_distance_vector(student_dist),
                    self._normalize_distance_vector(teacher_dist),
                )

                total_loss = total_loss + (
                    self.hparams.lg_distance_weight * dist_loss
                    + self.hparams.lg_position_weight * pos_loss
                )
                n_terms += 1

        if n_terms == 0:
            return flat_batch.new_zeros(())
        return total_loss / n_terms

    def align_loss(
        self,
        rooter_input,
        emb_level_item,
        distances,
    ):
        """
        对齐损失：类似 VQ-VAE 的向量量化损失

        将连续的输入向量量化到最近的嵌入向量，并计算重构损失。
        这是实现离散潜在空间的标准方法。

        1. 找到最近的嵌入：k* = argmin_k ||z - e_k||²
        2. 量化：z_q = e_{k*}
        3. 损失：
           - q_latent_loss: ||z_q - sg[z]||² (更新嵌入)
           - e_latent_loss: ||sg[z_q] - z||² (更新编码器)

        其中 sg[·] 表示停止梯度。


        直通估计器 (Straight-Through Estimator)
        z_out = z + (z_q - z).detach()

        rooter_input : torch.Tensor
            编码器输出的连续向量，形状 [B, D]

        emb_level_item : torch.Tensor
            嵌入表（聚类中心），形状 [K, D]

        distances : torch.Tensor
            预计算的距离矩阵，形状 [B, K]

        ===================================================================================
        返回值
        encoding_indices : torch.Tensor
            最近嵌入的索引，形状 [B, 1]

        quantized : torch.Tensor
            量化后的向量，形状 [B, D]

        loss : torch.Tensor
            总损失（标量）
        """
        num_embeddings = emb_level_item.shape[0]  # 嵌入数量 K

        # 对每个输入，找到距离最近的嵌入索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)

        encodings = torch.zeros(
            encoding_indices.size(0), num_embeddings, device=rooter_input.device
        )
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encoding

        # Quantize and reshape
        quantized = torch.matmul(encodings.detach(), emb_level_item).view(
            rooter_input.shape
        )  # Reshape back
        quantized = quantized.contiguous()  # (B, C, H, W)

        # import pdb; pdb.set_trace()
        e_latent_loss = F.mse_loss(quantized.detach(), rooter_input)
        q_latent_loss = F.mse_loss(quantized, rooter_input.detach())
        loss = q_latent_loss + self.hparams.weight_e_latent * e_latent_loss

        quantized = (
            rooter_input + (quantized - rooter_input).detach()
        )  # Straight-through estimator

        return encoding_indices, quantized, loss

    def cal_distance_matrix_with_tree(
        self,
        rooter_input,
        emb_level_item,
        last_tree_node_idx=None,
        tree_rout_bool=False,
    ):
        batch_size = rooter_input.shape[0]
        # 计算平方欧氏距离
        # 公式：d² = ||x||² + ||e||² - 2⟨x, e⟩
        distances = (
            (rooter_input**2).sum(dim=1, keepdim=True)
            + (emb_level_item**2).sum(dim=1)
            - 2 * torch.matmul(rooter_input, emb_level_item.t())
        )
        # 应用树结构约束
        if last_tree_node_idx is not None and tree_rout_bool:
            # 创建掩码矩阵：所有位置初始化为无穷大
            distances_plus = torch.full_like(distances, float("inf"))

            # 计算每个样本可以访问的节点（二叉树的两个子节点）
            row_indices = torch.arange(
                batch_size, device=rooter_input.device
            ).repeat_interleave(2)
            index_s = last_tree_node_idx * 2
            col_indices = torch.arange(2, device=rooter_input.device).repeat(
                batch_size
            ) + index_s.repeat_interleave(2)
            distances_plus[row_indices, col_indices] = 0
            distances_on_tree = distances + distances_plus
        else:
            distances_on_tree = distances

        return distances, distances_on_tree

    def router_forward(self, rooter_input, tree_rout_bool=False, ec_ce_weight=10):
        """
        层次化树路由前向传播
        参数:
        - rooter_input: 输入向量 [B, D]，需要被量化的连续表示
        - tree_rout_bool: 是否启用树约束（True=只能选子节点，False=可选任意节点）
        - ec_ce_weight: 码本损失的权重
        """
        tree_rout_list = []  # 存储每层选择的节点索引
        vector_list = []  # 存储每层量化后的向量
        loss_list = []  # 存储每层的量化损失

        for i in range(len(self.tree_node_embedding)):
            # 获取当前层的嵌入表（码本）
            # tree_node_embedding[i] 是 nn.Embedding，.weight 是 [K_i, D] 的矩阵
            emb_level_item = self.tree_node_embedding[i].weight
            if i > 0:
                last_tree_node_idx = tree_rout_list[-1]  # 上一层选的节点
            else:
                last_tree_node_idx = None  # 根层没有父节点

            distances, distances_on_tree = self.cal_distance_matrix_with_tree(
                rooter_input, emb_level_item, last_tree_node_idx, tree_rout_bool
            )
            # 原始距离:           树约束后:
            # [0.5, 0.3, 0.8, 0.2]  →  [0.5, 0.3, ∞, ∞] 不可选

            if last_tree_node_idx is not None:
                # 1. 编码器→码本方向的损失
                # 让输入向量接近选中的嵌入
                encoding_indices, quantized, loss_ec_tree = self.align_loss(
                    rooter_input, emb_level_item, distances_on_tree
                )
                # 2. 码本→编码器方向的损失
                # 让嵌入向量接近输入（反向对齐）
                _, _, loss_ce_tree = self.align_loss(
                    emb_level_item, rooter_input, distances_on_tree.t()
                )
                loss = loss_ec_tree + loss_ce_tree * ec_ce_weight
            else:
                # 根层没有树约束，使用全部距离计算损失
                encoding_indices, quantized, loss_ec = self.align_loss(
                    rooter_input, emb_level_item, distances
                )
                _, _, loss_ce = self.align_loss(
                    emb_level_item, rooter_input, distances.t()
                )
                loss = loss_ec + loss_ce * ec_ce_weight

            tree_rout_list.append(encoding_indices.reshape(-1))
            vector_list.append(quantized)
            loss_list.append(loss)

        tree_rout = torch.stack(tree_rout_list, axis=1)
        vector_rout = torch.stack(vector_list, axis=1)
        loss = torch.stack(loss_list).mean()
        return tree_rout, vector_rout, loss

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        """
        Computes squared Euclidean distance between samples.

        Args:
            x (Tensor): Input tensor of shape (n_samples, n_features).
            y (Tensor): Optional second input tensor.
            metric (str): Distance metric to use ('euclidean').

        Returns:
            dist (Tensor): Distance matrix.
        """
        dist = None
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0], dtype=torch.bool)] = 1e-12
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return dist

    def _CalGamma(self, v):
        """
        Calculates the gamma function value.
        计算 t分布归一化常数中的 Gamma 函数部分

        Γ((v+1)/2) / [√(vπ) · Γ(v/2)]

        这是 t分布概率密度函数的归一化因子的一部分。
        Args:
            v (float): Degrees of freedom.

        Returns:
            out (float): Gamma function value.
        """
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity(self, dist, sigma=0.3):
        """
        Computes similarity using Gaussian kernel.
        使用高斯核计算相似度

        P_ij = exp(-d_ij / (2σ²))

        这是经典的高斯核相似度，用于 SNE 等方法。
        Args:
            dist (Tensor): Distance matrix.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            Pij (Tensor): Similarity matrix.
        """
        dist = dist.clamp(min=0)
        Pij = torch.exp(-dist / (2 * sigma**2))
        return Pij

    def t_distribution_similarity(self, distance_matrix, df):
        """
        Computes similarity matrix using t-distribution kernel.
        使用 t分布核计算相似度矩阵

        t-SNE 的核心思想是在低维空间使用 t分布核来计算相似度：

        q_ij ∝ (1 + ||y_i - y_j||² / v)^(-(v+1)/2)

        Args:
            distance_matrix (Tensor): Distance matrix.
            df (float): Degrees of freedom for t-distribution.

        Returns:
            similarity_matrix (Tensor): Similarity matrix.
        """
        distance_matrix = torch.nan_to_num(
            distance_matrix, nan=1e6, posinf=1e6, neginf=0.0
        )
        distance_matrix = distance_matrix + 1e-9
        # 计算 t分布核
        # (1 + d²/v)^(-(v+1)/2)
        numerator = (1 + distance_matrix**2 / df) ** (-(df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) - torch.diagonal(
            numerator, 0
        ).unsqueeze(1)
        denominator = denominator.clamp_min(1e-12)
        similarity_matrix = numerator / denominator
        return torch.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    def UMAPNoSigmaSimilarity(self, dist, nu=100):
        a = scipy.special.gamma((nu + 1) / 2)
        b = np.sqrt(nu * np.pi) * scipy.special.gamma(nu / 2)
        gamma = a / b

        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / nu), exponent=-1 * (nu + 1))
        )
        return Pij

    def cal_dis_to_p(self, dis_input_ab):
        """
        将距离矩阵转换为基于排名的相似度矩阵
        P_ij = b^{rank(j)}，其中 b = 0.3（默认）
        """
        batch_size = dis_input_ab.size(0)
        sorted_indices = torch.argsort(dis_input_ab, dim=1)
        ranks = torch.zeros_like(
            dis_input_ab, dtype=torch.long, device=dis_input_ab.device
        )
        order = torch.arange(batch_size, device=dis_input_ab.device).expand_as(
            dis_input_ab
        )
        ranks.scatter_(1, sorted_indices, order)
        # P = torch.pow(self.hparams.exp_b, ranks.float())

        logb = math.log(self.hparams.exp_b)
        P = torch.exp(ranks.float() * logb)
        return P

    def LossManifold_Global(
        self,
        input_data,
        latent_data,
        temperature=1.0,
        exaggeration=1.0,
        nu=0.1,
        t_mul=1.0,
    ):
        """
        Computes the manifold loss between two views of the data.
        这是 DMTEVT 模型的核心损失函数，用于学习保持数据结构的低维嵌入。
        核心思想是让低维空间的相似度分布 Q 匹配高维空间的相似度分布 P。
        使用二元交叉熵损失:
            L = -Σ_ij [P_ij · log(Q_ij) + (1-P_ij) · log(1-Q_ij)]

        其中:
            - P_ij: 高维输入空间中样本 i 和 j 的相似度（基于距离排名）
            - Q_ij: 低维嵌入空间中样本 i 和 j 的相似度（基于 t 分布核）
        Args:
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        # if self.hparams.norm_loss:
        #     mean = latent_data.mean().detach()
        #     std = latent_data.std().detach()
        #     latent_data = (latent_data - mean) / std

        batch_size = latent_data.shape[0] // 2
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        dis_ab = torch.cdist(features_a, features_b) * t_mul
        Q = self.t_distribution_similarity(dis_ab, df=nu)
        Q = torch.nan_to_num(Q, nan=1e-8, posinf=1.0 - 1e-8, neginf=1e-8)
        Q = Q.clamp(min=1e-8, max=1.0 - 1e-8)

        with torch.no_grad():
            features_input_a = input_data[:batch_size]
            features_input_b = input_data[batch_size:]
            dis_input_ab = self._DistanceSquared(features_input_a, features_input_b)
            # dis_input_ab[torch.eye(dis_input_ab.shape[0]) == 1] = 0
            dis_input_ab.fill_diagonal_(0)

            # P_x: 从视图 A 的视角看（行视角）
            # P_x[i,j] = b^{rank_i(j)}，即从样本 i 看，j 是第几近的
            P_x = self.cal_dis_to_p(dis_input_ab)

            # P_y: 从视图 B 的视角看（列视角）
            # P_y[i,j] = b^{rank_j(i)}，即从样本 j 看，i 是第几近的
            P_y = self.cal_dis_to_p(dis_input_ab.T).T

            # 结合双视角信息，得到对称的 P 矩阵
            P = torch.sqrt(P_x * P_y)
            # 将过小的 P 值截断到 exp_min（默认 1e-5）
            # 避免后续计算 log(P) 或 log(1-P) 时出现数值问题
            P[P < self.hparams.exp_min] = self.hparams.exp_min

        EPS = 1e-8

        # 正样本项：P_ij · log(Q_ij)
        # 当 P_ij 大（高维相似）时，希望 Q_ij 也大（低维也相似）
        # 否则 log(Q_ij) 会是很大的负数，造成高损失
        losssum1 = P * torch.log(Q + EPS)

        # 负样本项：(1-P_ij) · log(1-Q_ij)
        # 当 P_ij 小（高维不相似）时，希望 Q_ij 也小（低维也不相似）
        # 即 (1-Q_ij) 应该大，log(1-Q_ij) 接近 0
        losssum2 = (1 - P) * torch.log(1 - Q + EPS)
        loss = -1 * (losssum1 + losssum2)

        # Top-K 硬负样本挖掘
        with torch.no_grad():
            # 找到每行损失最大的 k 个
            # topk_values[i] = 第 i 行最大的 k 个损失值
            # Fix: Ensure k does not exceed the number of samples in the batch (loss.shape[1])
            k_safe = min(100, loss.shape[1])
            topk_values, _ = torch.topk(loss, k=k_safe, dim=1)

            # 获取阈值：每行第 k 大的损失值
            # threshold[i] = 第 i 行要被选中的最小损失
            threshold = topk_values[:, -1].unsqueeze(1)

            mask = loss >= threshold
        # import pdb; pdb.set_trace()
        loss = loss[mask]

        return loss.mean()

    def LossManifold_All(
        self,
        input_data,
        latent_data,
        temperature=1.0,
        exaggeration=1.0,
        nu=0.1,
        t_mul=1.0,
    ):
        """
        Computes the manifold loss between two views of the data.

        Args:
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        loss1 = self.LossManifold_Global(
            input_data=input_data,
            latent_data=latent_data,
            temperature=temperature,
            exaggeration=exaggeration,
            nu=nu,
            t_mul=t_mul,
        )

        loss1 = loss1 / loss1.detach()

        loss2 = self.LossManifold(
            input_data=input_data,
            latent_data=latent_data,
            temperature=temperature,
            exaggeration=exaggeration,
            nu=nu,
        )
        loss2 = loss2 / loss2.detach()
        loss = loss1 * self.hparams.all_g_l_weight + loss2 * (
            1 - self.hparams.all_g_l_weight
        )

        return loss

    def LossManifold(
        self,
        input_data,
        latent_data,
        temperature=1.0,
        exaggeration=1.0,
        nu=0.1,
        t_mul=1.0,
    ):
        """
        Computes the manifold loss between two views of the data.

        Args:
            input_data (Tensor): Input data.
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        batch_size = latent_data.shape[0] // 2
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        # Compute pairwise distances
        dis_aa = torch.cdist(features_a, features_a) * temperature
        dis_bb = torch.cdist(features_b, features_b) * temperature
        dis_ab = torch.cdist(features_a, features_b) * temperature

        # Compute similarity matrices using t-distribution
        sim_aa = self.t_distribution_similarity(dis_aa, df=nu)
        sim_bb = self.t_distribution_similarity(dis_bb, df=nu)
        sim_ab = self.t_distribution_similarity(dis_ab, df=nu)

        # Compute alignment term
        tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # Exclude self similarities
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        # Compute uniformity terms
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        # Compute final loss
        loss = -(exaggeration * tempered_alignment - raw_uniformity / 2)

        return loss

    def batch_patten_loss(self, feature_tra, mask):
        """
        Computes orthogonal loss to encourage diversity among experts.

        Args:
            feature_tra (Tensor): Transformed features.
            mask (Tensor): Masks indicating selected features.

        Returns:
            loss (Tensor): Computed loss.
        """
        # Add small noise to features
        feature_tra = (
            feature_tra + torch.randn_like(feature_tra) * 0.001 * feature_tra.std()
        )
        batch_size = feature_tra.shape[0] // 8
        feature_tra = feature_tra[:batch_size]
        mask = mask[:batch_size]

        mean_value_list = []
        for i in range(feature_tra.shape[1]):
            fea_ins = feature_tra[:, i, :]
            mask_ins = mask[:, i, :] == 1
            fea_ins_umask = fea_ins[mask_ins == 1].reshape((feature_tra.shape[0], -1))
            # Compute cosine similarity
            cosine_similarity_matrix = torch.nn.functional.cosine_similarity(
                fea_ins_umask.unsqueeze(1), fea_ins_umask.unsqueeze(0), dim=2
            )
            upper_triangular_matrix_no_diag = torch.triu(
                cosine_similarity_matrix, diagonal=1
            )
            mean_value = upper_triangular_matrix_no_diag.mean()
            mean_value_list.append(mean_value)

        # Return the mean of the mean values
        return 1 + torch.stack(mean_value_list).mean()

    def forward(self, x, tau=100.0):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input data.
            tau (float): Temperature parameter for Gumbel softmax.

        Returns:
            x_masked (Tensor): Masked input data.
            lat_higt_dim_out (Tensor): High-dimensional latent outputs.
            lat_vis (Tensor): Low-dimensional visualization outputs.
            lat_high_dim (Tensor): High-dimensional latent representations.
        """
        batch_size = x.shape[0] // 2
        x_masked = x

        # Pass through encoder
        lat_high_dim_out = self.enc(x_masked)

        lat_vis_list = []
        for i in range(len(self.hparams.num_use_mlevel_list)):
            lat_vis = self.vis_list[i](lat_high_dim_out)
            lat_vis_list.append(lat_vis)

        # 选择最佳投影头的输出作为主要二维表征
        # best_head_index == -1 时使用最后一个头（默认/未选择状态）
        best_idx = (
            self.best_head_index
            if 0 <= self.best_head_index < len(lat_vis_list)
            else -1
        )
        lat_vis_best = lat_vis_list[best_idx]

        return x_masked, lat_high_dim_out, lat_vis_best, lat_vis_list

    def get_weight(self):
        """
        Retrieves and processes the expert weights.

        Returns:
            weight (Tensor): Processed weights.
        """
        w = self.exp(torch.arange(self.hparams.num_use_moe).to(self.device)).reshape(
            1, self.hparams.num_use_moe, -1
        )
        weight = F.tanh(w) * 10
        return weight

    def get_tau(self, epoch, total_epochs=900, tau_start=100, tau_end=1.001):
        """
        Computes the temperature parameter tau for Gumbel softmax.

        Args:
            epoch (int): Current epoch.
            total_epochs (int): Total number of epochs.
            tau_start (float): Initial tau value.
            tau_end (float): Final tau value.

        Returns:
            tau (float): Computed tau value.
        """
        if epoch >= total_epochs:
            return tau_end
        else:
            return tau_start * (tau_end / tau_start) ** (epoch / (total_epochs - 1))

    def forward_train_enc(self, data_input_item, data_input_aug):
        data_input = torch.cat([data_input_item, data_input_aug])
        x_masked, lat_high_dim, lat_vis, lat_vis_list = self(
            data_input,
            tau=self.hparams.tau,
        )

        # Compute orthogonal loss if required
        if self.hparams.use_orthogonal:
            orthogonal_loss = self.batch_patten_loss(x_masked, self.mask)
        else:
            orthogonal_loss = 0

        loss_emb_list = []
        for i, lat_vis in enumerate(lat_vis_list):
            with torch.no_grad():
                mean = lat_vis.mean(dim=0).detach()
                std = lat_vis.std(dim=0).detach() + 1e-8

            lat_vis_n = (lat_vis - mean) / std

            if self.hparams.loss_type == "G":
                LossFunc = self.LossManifold_Global
            elif self.hparams.loss_type == "L":
                LossFunc = self.LossManifold
            elif self.hparams.loss_type == "A":
                LossFunc = self.LossManifold_All
            else:
                raise ValueError(f"Unsupported loss_type: {self.hparams.loss_type}")

            # Compute manifold losses
            # loss_lat = LossFunc(
            #     input_data=input_data.reshape(lat_high_dim.shape[0], -1),
            #     latent_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            #     temperature=1,
            #     exaggeration=self.hparams.exaggeration_lat,
            #     nu=self.hparams.nu_lat,
            # )
            loss_emb = LossFunc(
                input_data=data_input.reshape(lat_vis_n.shape[0], -1),
                latent_data=lat_vis_n.reshape(lat_vis_n.shape[0], -1),
                temperature=0.2,
                exaggeration=self.hparams.exaggeration_emb,
                nu=self.hparams.nu_emb,
                t_mul=self.hparams.num_use_mlevel_list[i],
            )
            loss_emb_list.append(loss_emb)

        loss_emb_mean = torch.stack(loss_emb_list).mean()

        lg_loss = self._compute_local_global_consistency_loss(
            data_input_item.reshape(data_input_item.shape[0], -1),
            [lat_vis[: data_input_item.shape[0]] for lat_vis in lat_vis_list],
        )

        loss_total = loss_emb_mean + self.hparams.lg_weight * lg_loss

        return loss_total, loss_emb_list, orthogonal_loss, lg_loss

    def update_training_str(self, epoch):
        """
        Updates the training string based on the current epoch.

        If the current epoch is greater than 20, the training string is set to
        'step2', indicating that the model is in the second stage of training.
        """

        if epoch > self.hparams.step2_epoch:
            self.training_str = "step2_s"
        if epoch > self.hparams.step2_r_epoch:
            self.training_str = "step2_r"

        # print(f"self.training_str {self.training_str}, epoch {epoch}")

    def validation_step(self, batch, batch_idx):
        data_input_item = batch["data_input_item"].float()
        data_input_aug = batch["data_input_aug"].float()
        index = batch["index"]

        self.update_training_str(self.current_epoch)

        loss_emb_mean, loss_emb_list, orthogonal_loss, lg_loss = self.forward_train_enc(
            data_input_item=data_input_item, data_input_aug=data_input_aug
        )

        # 收集高维输入和各 head 的 2D 嵌入，用于 epoch 结束时计算 kNN 质量
        with torch.no_grad():
            x_masked, lat_high_dim, lat_vis_best, lat_vis_list = self(
                data_input_item, tau=self.hparams.tau
            )
            self._val_head_data.append(
                {
                    "high_dim": data_input_item.detach().cpu(),
                    "head_embs": [lv.detach().cpu() for lv in lat_vis_list],
                }
            )

        self.log("val_loss_lg", lg_loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss_emb_mean

    def _compute_knn_preservation(self, hd_neighbors, emb_data, k):
        """计算 kNN Preservation Rate（局部召回）。

        高维 k 近邻在嵌入空间中被保留的比例。
        值域 [0, 1]，越高越好。
        """
        from sklearn.neighbors import NearestNeighbors

        nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb_data)
        emb_neighbors = nn_emb.kneighbors(emb_data, return_distance=False)[:, 1:]
        n = len(hd_neighbors)
        overlap = sum(
            len(np.intersect1d(hd_neighbors[i], emb_neighbors[i])) for i in range(n)
        )
        return overlap / (n * k)

    def _compute_trustworthiness(self, hd_data, emb_data, k):
        """计算 Trustworthiness（局部精度）。

        嵌入中的 k 近邻在高维中也是近邻的程度。
        惩罚那些在嵌入中"闯入"邻域的假近邻。
        值域 [0, 1]，越高越好。
        """
        from sklearn.manifold import trustworthiness

        return trustworthiness(hd_data, emb_data, n_neighbors=k)

    def _compute_distance_correlation(self, hd_data, emb_data, n_pairs=5000):
        """计算采样 Spearman 距离秩相关性（全局结构）。

        随机采样点对，比较高维距离排名与嵌入距离排名的相关性。
        值域 [-1, 1]，越高越好。
        """
        n = len(hd_data)
        # 随机采样点对
        idx_a = np.random.randint(0, n, size=n_pairs)
        idx_b = np.random.randint(0, n, size=n_pairs)
        # 避免自身比较
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        dist_hd = np.sqrt(((hd_data[idx_a] - hd_data[idx_b]) ** 2).sum(axis=1))
        dist_emb = np.sqrt(((emb_data[idx_a] - emb_data[idx_b]) ** 2).sum(axis=1))

        corr = scipy.stats.spearmanr(dist_hd, dist_emb).correlation
        return corr if not np.isnan(corr) else 0.0

    def on_validation_epoch_end(self):
        """在每个验证 epoch 结束时，综合多角度指标选择最佳投影头。

        综合三个互补的、尺度无关的指标：
        1. kNN Preservation（局部召回）：高维近邻在嵌入中保留了多少
        2. Trustworthiness（局部精度）：嵌入近邻在高维中是否真实
        3. Spearman 距离相关性（全局结构）：整体距离排名关系是否保持

        三者均基于排名，不受 t_mul 尺度影响，是公平的跨 head 比较。
        最终得分 = 三者均值，得分最高的 head 被选为最优。
        """
        if not self._val_head_data:
            return

        # 合并所有 batch 的数据
        high_dim_all = torch.cat(
            [d["high_dim"] for d in self._val_head_data], dim=0
        ).numpy()
        n_heads = len(self._val_head_data[0]["head_embs"])
        head_embs_all = []
        for h in range(n_heads):
            emb = torch.cat(
                [d["head_embs"][h] for d in self._val_head_data], dim=0
            ).numpy()
            head_embs_all.append(emb)

        # 采样以控制计算量
        n_samples = len(high_dim_all)
        max_samples = 3000
        if n_samples > max_samples:
            idx = np.random.choice(n_samples, max_samples, replace=False)
            high_dim_all = high_dim_all[idx]
            head_embs_all = [emb[idx] for emb in head_embs_all]

        # 高维 kNN（只算一次，所有 head 共用）
        from sklearn.neighbors import NearestNeighbors

        k = min(12, len(high_dim_all) - 1)
        nn_hd = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(high_dim_all)
        hd_neighbors = nn_hd.kneighbors(high_dim_all, return_distance=False)[:, 1:]

        # 评估各 head
        log_dict = {}
        combined_scores = []
        for h in range(n_heads):
            knn = self._compute_knn_preservation(hd_neighbors, head_embs_all[h], k)
            trust = self._compute_trustworthiness(high_dim_all, head_embs_all[h], k)
            dist_corr = self._compute_distance_correlation(
                high_dim_all, head_embs_all[h]
            )

            # 综合得分：三个指标均值
            score = (knn + trust + dist_corr) / 3.0
            combined_scores.append(score)

            log_dict[f"val_knn_head_{h}"] = knn
            log_dict[f"val_trust_head_{h}"] = trust
            log_dict[f"val_dist_corr_head_{h}"] = dist_corr
            log_dict[f"val_score_head_{h}"] = score

        # 选择综合得分最高的投影头
        best_idx = int(np.argmax(combined_scores))
        self.best_head_index = best_idx
        log_dict["global_best_head"] = best_idx
        self.log_dict(log_dict)

        self._val_head_data = []

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            loss_all (Tensor): Computed loss.
        """
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        index = batch["index"]

        self.update_training_str(self.current_epoch)
        log_dict = {}

        loss_emb_mean, loss_emb_list, orthogonal_loss, lg_loss = self.forward_train_enc(
            data_input_item, data_input_aug
        )

        for i in range(len(loss_emb_list)):
            log_dict[f"loss_emb_{i}"] = loss_emb_list[i]

        loss_all = loss_emb_mean

        log_dict.update(
            {
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "global_best_head": self.best_head_index,
                "loss_lg": lg_loss,
                "loss_all": loss_all,
            }
        )

        self.log_dict(log_dict)
        # Warmup 结束时硬同步 teacher，避免 teacher 持有训练初期的陈旧权重
        if self.current_epoch == self.hparams.lg_warmup_epochs and batch_idx == 0:
            self._sync_lg_teacher(momentum=0.0)
        else:
            self._sync_lg_teacher()
        return loss_all

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr,
        )
        lrsched = CosineAnnealingSchedule(
            optimizer, n_epochs=self.hparams.max_epochs, warmup_epochs=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lrsched,
                "interval": "epoch",
            },  # interval "step" for batch update
        }
