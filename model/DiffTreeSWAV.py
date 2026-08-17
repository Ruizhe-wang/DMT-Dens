import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.cluster import KMeans
import scipy
from lightning import LightningModule
from model.diffmodel.model import AE_CNN_bottleneck_deep, AE, AE_layer2
from model.diffmodel.diffusion import GaussianDiffusion, make_beta_schedule
from joblib import parallel_backend
from collections import OrderedDict
import time
import uuid
from model.CosineAnnealingSchedule import CosineAnnealingSchedule

from model.encoder import TransformerEncoder
from model.encoder import NN_FCBNRL_MM


class SwAVHead(nn.Module):
    """
    SwAV (Swapping Assignments between Views) 聚类头
    
    核心思想：
    - 使用一组可学习的"原型"(prototypes)作为聚类中心
    - 通过 Sinkhorn-Knopp 算法生成软聚类分配（避免所有样本坍缩到同一个原型）
    - 交叉预测：用视图1的分配来监督视图2的预测，反之亦然
    
    参考文献：
    - Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments" (NeurIPS 2020)
    """
    def __init__(
        self,
        embedding_dim: int = 2,          # 嵌入维度（与可视化空间维度一致）
        n_prototypes: int = 1024,        # 原型数量（聚类中心数）
        temperature: float = 1,          # 温度参数，控制softmax 分布的锐度 τ→0 logits 被极度放大接近 one-hot（硬分配）, τ=1,标准softmax正常概率分布, τ→∞, 平滑化接近均匀分布
        sinkhorn_epsilon: float = 0.05,  # Sinkhorn 正则化系数
        sinkhorn_iters: int = 3,         # Sinkhorn 迭代次数
        metric: str = "euclidean_t",     # 距离度量："cosine" 或 "euclidean_t"（t分布核）
        t_df: float = 1.0,               # t分布自由度，1.0 对应柯西分布
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.metric = metric
        self.t_df = t_df

        # 原型参数 (K x D)
        # 原型参数 (K x D)：可学习的聚类中心


        # 标准化的目的：
        # - 使原型在特征空间中分布更均匀
        # - 避免初始化时某些原型过于集中
        # - 有助于训练早期的稳定性
        self.prototypes = nn.Parameter(
            self._standardize(
                torch.randn(
                    n_prototypes,
                    embedding_dim
                    ))
        )


    @torch.no_grad()
    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp 算法：生成均衡的软聚类分配
        
        目的：
        - 避免模式坍缩（所有样本分配到同一个原型）
        - 确保每个原型被均匀使用
        
        数学原理：
        - 通过交替归一化行和列，使分配矩阵接近双随机矩阵
        - Q[k,b] 表示样本 b 被分配到原型 k 的概率
        
        参数:
            logits (Tensor): 样本到原型的相似度得分 [B, K]
        
        返回:
            Q (Tensor): 归一化后的软分配矩阵 [B, K]
        """
        B, K = logits.shape
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits / self.sinkhorn_epsilon).t()  # [K, B]
        
        # r：原型的目标边际分布（均匀分布，每个原型被 1/K 的样本使用）
        # c：样本的目标边际分布（均匀分布，每个样本分配 1/B 的权重）
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B
        for _ in range(self.sinkhorn_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12); Q = Q * r.view(-1, 1)
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12); Q = Q * c.view(1, -1)
        Q = (Q / (Q.sum(dim=0, keepdim=True) + 1e-12)).t()
        return Q

    def _pairwise_sqeuclidean(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: [B, D], B: [K, D] -> d2: [B, K]
        d^2 = ||A||^2 + ||B||^2 - 2 A B^T

        计算成对平方欧氏距离
        
        数学公式：
            d²(a, b) = ||a||² + ||b||² - 2⟨a, b⟩
        
        参数:
            A (Tensor): 第一组向量 [B, D]
            B (Tensor): 第二组向量 [K, D]
        
        返回:
            d2 (Tensor): 平方距离矩阵 [B, K]
        """
        A2 = (A * A).sum(dim=1, keepdim=True)          # [B, 1]
        B2 = (B * B).sum(dim=1, keepdim=True).t()      # [1, K]
        d2 = A2 + B2 - 2.0 * (A @ B.t())               # [B, K]
        return d2.clamp_min_(0.0)

    def _logits(self, z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        给定 z, P 计算 logits（未过 softmax）
        - cosine: (z·P^T)/T
        - euclidean_t: 以 t 核的 log 相似度为 logits： -((ν+1)/2)*log(1 + d^2/ν)/T

        计算样本到原型的 logits（未归一化的相似度得分）
        
        支持两种度量：
        1. cosine: 余弦相似度，适合高维稀疏表示
        2. euclidean_t: t分布核，适合低维可视化空间（如 t-SNE 的思想）
        
        参数:
            z (Tensor): 样本嵌入 [B, D]
            P (Tensor): 原型 [K, D]
        
        返回:
            logits (Tensor): 相似度得分 [B, K]
        """
        if self.metric == "cosine":
            # 余弦相似度：先L2归一化，再内积
            z_norm = F.normalize(z, dim=1)
            P_norm = F.normalize(P, dim=1)
            return (z_norm @ P_norm.t()) / self.temperature  # [B, K]
        elif self.metric == "euclidean_t":
            # 不强制单位范数；用欧氏距离
            # t分布核相似度（类似 t-SNE 的低维相似度计算）

            d2 = self._pairwise_sqeuclidean(z, P)           # [B, K]
            # log t-kernel 相似度（数值稳定）
            # 2. 计算 t分布核的对数相似度
            # 公式：log_sim = -((ν+1)/2) * log(1 + d²/ν)
            # 
            # 数学解释：
            # t分布的 PDF: f(d) ∝ (1 + d²/ν)^(-(ν+1)/2)
            # 对数形式: log(f(d)) = -((ν+1)/2) * log(1 + d²/ν) + const
            logits = -0.5 * (self.t_df + 1.0) * torch.log1p(d2 / self.t_df)
            return logits / self.temperature                 # [B, K]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        按 batch 内做 StandardScaler: 减均值 / 除标准差
        
        标准化：减均值，除以标准差
        
        目的：使特征分布更稳定，有助于训练收敛
        
        参数:
            x (Tensor): 输入张量
        
        返回:
            标准化后的张量
        """
        mean = x.mean(dim=0, keepdim=True)                  # [1, D]
        std = x.std(dim=0, keepdim=True, unbiased=False)    # [1, D]
        return 1 * (x - mean) / (std + 1e-6)

    def forward(self, lat_vis: torch.Tensor, rep=False):
        """
        SwAV 前向传播
        
        核心流程：
        1. 将输入分成两个视图（假设输入是两个增强视图的拼接）
        2. 计算每个视图到原型的 logits
        3. 用 Sinkhorn 生成软分配 q1, q2
        4. 交叉熵损失：q1 监督 logits2，q2 监督 logits1
        
        参数:
            lat_vis (Tensor): 可视化空间的嵌入 [2B, D]，两个视图拼接
            rep (bool): 保留参数，未使用
        
        返回:
            loss (Tensor): SwAV 损失
            stats (dict): 统计信息字典
        """
        assert lat_vis.dim() == 2 and lat_vis.size(1) == self.prototypes.size(1), \
            "lat_vis dim must match prototype dim"
        # 分割两个视图
        lat_vis = self._standardize(lat_vis)

        B2 = lat_vis.size(0)
        assert B2 % 2 == 0, "Expect even batch (2 views concatenated)"
        B = B2 // 2
        z1, z2 = lat_vis[:B], lat_vis[B:]                     # [B, D] each

        # logits to prototypes for each view
        # 计算到原型的 logits
        logits1 = self._logits(z1, self.prototypes)           # [B, K]
        logits2 = self._logits(z2, self.prototypes)           # [B, K]

        # 用 Sinkhorn 生成软分配（停止梯度，作为伪标签）
        with torch.no_grad():
            q1 = self._sinkhorn(logits1).detach()
            q2 = self._sinkhorn(logits2).detach()

        # 交叉熵损失：用一个视图的分配监督另一个视图的预测
        logp2 = F.log_softmax(logits2, dim=1)
        logp1 = F.log_softmax(logits1, dim=1)

        # loss12: 用 q1 监督 logp2（视图1的分配指导视图2的预测）
        loss12 = (-q1 * logp2).sum(dim=1).mean()
        # loss21: 用 q2 监督 logp1（视图2的分配指导视图1的预测）
        loss21 = (-q2 * logp1).sum(dim=1).mean()
        loss = 0.5 * (loss12 + loss21)

        # 统计信息（用于监控训练）
        with torch.no_grad():

            # 分配熵：衡量分配的均匀程度（越高越均匀）
            ent1 = -(q1 * (q1.clamp_min(1e-12).log())).sum(dim=1).mean()
            ent2 = -(q2 * (q2.clamp_min(1e-12).log())).sum(dim=1).mean()

            # 监控原型向量的尺度
            # 如果范数过大或过小，可能表明训练不稳定
            proto_norm = self.prototypes.norm(dim=1).mean()

        stats = {
             "swav/loss": loss.item(),           # 总损失
            "swav/loss12": loss12.item(),       # 视图1→视图2 方向的损失
            "swav/loss21": loss21.item(),       # 视图2→视图1 方向的损失
            "swav/q_entropy_v1": ent1.item(),   # 视图1分配的熵
            "swav/q_entropy_v2": ent2.item(),   # 视图2分配的熵
            "swav/proto_norm": proto_norm.item(),  # 原型平均范数
        }
        return loss, stats



class DMTEVT_model(LightningModule):
    """
    DMTEVT_model is a PyTorch Lightning module that implements the training and evaluation of the model.

    这是一个用于高维数据可视化的深度学习模型，结合了：
    1. Transformer 编码器：提取高级语义特征
    2. MLP 投影头：将高维特征映射到低维可视化空间
    3. SwAV 聚类头：自监督学习聚类结构
    4. 流形学习损失：保持数据的局部和全局结构
    
    训练策略：
    1. 数据增强：生成同一样本的两个不同视图
    2. 流形损失：保持数据的局部/全局结构（类似 t-SNE/UMAP）
    3. SwAV 损失：自监督聚类，发现数据结构
    4. 多阶段训练：不同阶段使用不同损失组合


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
        nu_lat=0.1,                        # 潜在空间 t分布自由度
        nu_emb=0.1,                        # 嵌入空间 t分布自由度
        tau=1,
        T_num_layers=2,                    # Transformer 层数
        T_num_attention_heads=6,           # 注意力头数
        T_hidden_size=240,                 # 隐藏层大小
        T_intermediate_size=300,           # FFN 中间层大小
        T_hidden_dropout_prob=0.1,         # 隐藏层 Dropout 率
        T_attention_probs_dropout_prob=0.1, # 注意力 Dropout 率
        t_output_dim=512,
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
        loss_type='G',                    # 损失类型：'G'全局/'L'局部/'A'全部
        all_g_l_weight=0.1,
        av_temperature=0.02,               # SwAV 温度
        av_t_df=1.0,                       # SwAV t分布自由度
        av_n_prototypes=1024,              # SwAV 原型数量
        av_sinkhorn_epsilon=0.1,           # Sinkhorn 正则化系数
        av_sinkhorn_iters=5,               # Sinkhorn 迭代次数
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
        self.save_hyperparameters(locals())

        num_input_dim = self.hparams.num_input_dim

        # 用于跟踪可视化空间的统计量
        self.lat_vis_mean = nn.Parameter(torch.zeros(2))
        self.lat_vis_std = nn.Parameter(torch.zeros(2))
        self.init_imge = None
        self.set_mean_bool = False

        # 唯一标识符（用于区分不同实验）
        self.uuid_str = str(uuid.uuid4())[:10]

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

        self.swav_head = SwAVHead(
            metric="euclidean_t",
            temperature=av_temperature,
            t_df=av_t_df,           # Cauchy
            n_prototypes=av_n_prototypes,
            sinkhorn_epsilon=av_sinkhorn_epsilon,
            sinkhorn_iters=av_sinkhorn_iters,
        )

        self.val_vis_list = []  # 用于存储验证时的可视化结果

        self.clustering_embedding = nn.Embedding(1024, 2)

        self.vis = self.InitNetworkMLP(
            NS=[t_output_dim * num_use_moe, 500, vis_dim], last_relu=False
        )

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
            encoding_indices.size(0),     
            num_embeddings, 
            device=rooter_input.device
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
        """
        计算输入向量到嵌入向量的距离矩阵

        """

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

            # 只有可访问的节点距离设为 0
            distances_plus[row_indices, col_indices] = 0
            distances_on_tree = distances + distances_plus
        else:
            distances_on_tree = distances

        return distances, distances_on_tree

    # def router_forward(self, rooter_input, tree_rout_bool=False, ec_ce_weight=10):
    #     tree_rout_list = []
    #     vector_list = []
    #     loss_list = []

    #     for i in range(len(self.tree_node_embedding)):
    #         emb_level_item = self.tree_node_embedding[i].weight
    #         if i > 0:
    #             last_tree_node_idx = tree_rout_list[-1]
    #         else:
    #             last_tree_node_idx = None

    #         distances, distances_on_tree = self.cal_distance_matrix_with_tree(
    #             rooter_input, emb_level_item, last_tree_node_idx, tree_rout_bool
    #         )

    #         if last_tree_node_idx is not None:
    #             encoding_indices, quantized, loss_ec_tree = self.align_loss(
    #                 rooter_input, emb_level_item, distances_on_tree
    #             )
    #             _, _, loss_ce_tree = self.align_loss(
    #                 emb_level_item, rooter_input, distances_on_tree.t()
    #             )
    #             loss = loss_ec_tree + loss_ce_tree * ec_ce_weight
    #         else:
    #             encoding_indices, quantized, loss_ec = self.align_loss(
    #                 rooter_input, emb_level_item, distances
    #             )
    #             _, _, loss_ce = self.align_loss(
    #                 emb_level_item, rooter_input, distances.t()
    #             )
    #             loss = loss_ec + loss_ce * ec_ce_weight

    #         tree_rout_list.append(encoding_indices.reshape(-1))
    #         vector_list.append(quantized)
    #         loss_list.append(loss)

    #     tree_rout = torch.stack(tree_rout_list, axis=1)
    #     vector_rout = torch.stack(vector_list, axis=1)
    #     loss = torch.stack(loss_list).mean()
    #     return tree_rout, vector_rout, loss

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
        distance_matrix = distance_matrix + 1e-6
        # 计算 t分布核
        # (1 + d²/v)^(-(v+1)/2)
        numerator = (1 + distance_matrix**2 / df) ** (-(df + 1) / 2)

        denominator = torch.sum(numerator, dim=1, keepdim=True) - torch.diagonal(
            numerator, 0
        ).unsqueeze(1)
        similarity_matrix = numerator / denominator
        return similarity_matrix

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


    def LossManifold_Global(self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1):
        """
        Computes the manifold loss between two views of the data.
        计算两个增强视图之间的流形保持损失。
        目标是使低维空间中视图1的样本 i 与视图2的样本 j 之间的相似度 (Q_ij)
        匹配高维输入空间中的相似度 (P_ij)。
        
        这是一种全局损失，考虑所有样本对之间的关系。
        L = -Σ_ij [P_ij · log(Q_ij) + (1-P_ij) · log(1-Q_ij)]
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
        # 分割两个视图
        batch_size = latent_data.shape[0] // 2        
        features_a = latent_data[:batch_size]# 视图1的低维表示
        features_b = latent_data[batch_size:]# 视图2的低维表示

        # 计算低维空间相似度 (Q)
        # 视图1到视图2的距离
        dis_ab = self._DistanceSquared(features_a, features_b) * temperature
        Q = self.UMAPNoSigmaSimilarity(dis_ab, nu=nu)

        # 计算高维空间相似度 (P)
        features_input_a = input_data[:batch_size]# 视图1的高维输入
        features_input_b = input_data[batch_size:]# 视图2的高维输入
        dis_input_ab = self._DistanceSquared(features_input_a, features_input_b) * temperature

        # 减去对角线距离的一半
        # 这相当于将同一样本两个视图之间的距离作为基准
        diag_dis_input_ab = torch.diagonal(dis_input_ab).detach()
        dis_input_ab = dis_input_ab - diag_dis_input_ab.unsqueeze(1)/2 - diag_dis_input_ab.unsqueeze(0)/2

         # 高维相似度（使用较大的 nu，更接近高斯分布）
        P = self.UMAPNoSigmaSimilarity(dis_input_ab, nu=100)

        # 对角线（同一样本的两个视图）设为最大相似度
        P[torch.eye(P.shape[0]) == 1] = 1.0

        EPS = 1e-8

        # P · log(Q)：相似样本应该在低维空间中也相似
        losssum1 = P * torch.log(Q + EPS)

        # (1-P) · log(1-Q)：不相似样本应该在低维空间中也不相似
        losssum2 = (1 - P) * torch.log(1 - Q + EPS)
        loss = -1 * (losssum1 + losssum2).mean()
        
        # print('loss:', loss.item())

        return loss

    def LossManifold_All(self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1):
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
            nu=nu
        ) 
        
        loss1 = loss1/loss1.detach()
        
        loss2 = self.LossManifold(
            input_data=input_data,
            latent_data=latent_data,
            temperature=temperature,
            exaggeration=exaggeration,
            nu=nu
        ) 
        loss2 = loss2/loss2.detach()
        loss = loss1 * self.hparams.all_g_l_weight + loss2 * (1 - self.hparams.all_g_l_weight)

        return loss


    def LossManifold(self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1):
        """
        Computes the manifold loss between two views of the data.
        局部流形损失（类似 t-SNE 的对比损失）
        计算基于 t-SNE 思想的流形保持损失。
        
        核心思想：
        1. 对齐 (Alignment)：同一样本的两个视图应该映射到相近的位置
        2. 均匀性 (Uniformity)：不同样本应该在空间中均匀分布

        损失 = -exaggeration · E[log(q_{ii'})] + 1/2 · [log(Σ_j q_{ij}) + log(Σ_j q_{i'j})]
        
        其中：
        - q_{ii'} 是同一样本两个视图之间的相似度（对齐项）
        - Σ_j q_{ij} 是样本 i 与所有其他样本的相似度之和（均匀性项）

        Args:
            input_data (Tensor): Input data.
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        # 分割两个视图
        batch_size = latent_data.shape[0] // 2
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        # Compute pairwise distances
        # 计算所有成对距离
        dis_aa = torch.cdist(features_a, features_a) * temperature
        dis_bb = torch.cdist(features_b, features_b) * temperature
        dis_ab = torch.cdist(features_a, features_b) * temperature

        # Compute similarity matrices using t-distribution
        # 计算相似度矩阵
        sim_aa = self.t_distribution_similarity(dis_aa, df=nu)
        sim_bb = self.t_distribution_similarity(dis_bb, df=nu)
        sim_ab = self.t_distribution_similarity(dis_ab, df=nu)

        # Compute alignment term
        # 计算对齐项
        # 对角线元素 sim_ab[i,i] 是同一样本两个视图之间的相似度
        # 我们希望这个相似度尽可能高
        tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # Exclude self similarities
        # 创建单位矩阵掩码，排除自相似项
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        # 将自相似度设为 0（排除在均匀性计算之外）
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        # Compute uniformity terms
        # 对于视图2的每个样本，计算它与所有样本的相似度之和
        # [sim_ab.T, sim_bb] 将视图间相似度和视图2内部相似度拼接
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()

        # 对于视图1的每个样本
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        # 均匀性项是两个方向的平均
        raw_uniformity = logsumexp_1 + logsumexp_2

        # Compute final loss
        loss = -(exaggeration * tempered_alignment - raw_uniformity / 2)

        return loss

    def batch_patten_loss(self, feature_tra, mask):
        """
        正交损失：鼓励 MoE 中不同专家产生多样化的输出
        Computes orthogonal loss to encourage diversity among experts.

        对于每个专家 i：
        1. 提取该专家处理所有样本后的特征 f_i
        2. 计算所有样本对之间的余弦相似度
        3. 损失 = 1 + mean(相似度)

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
        # 只使用 1/8 的样本（节省计算）
        batch_size = feature_tra.shape[0] // 8
        feature_tra = feature_tra[:batch_size]
        mask = mask[:batch_size]

        mean_value_list = []

        # 对每个专家计算
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
        模型前向传播
        
        数据流

        输入 x [2B, D_input]
            ↓
        Transformer 编码器 (self.enc)
            ↓
        高维潜在表示 lat_high_dim [2B, D_high]
            ↓
        MLP 投影头 (self.vis)
            ↓
        低维可视化表示 lat_vis [2B, D_vis]

        Args:
            x (Tensor): Input data.
            tau (float): Temperature parameter for Gumbel softmax.

        Returns:
            x_masked (Tensor): Masked input data.
            lat_higt_dim_out (Tensor): High-dimensional latent outputs.
            lat_vis (Tensor): Low-dimensional visualization outputs.
            lat_high_dim (Tensor): High-dimensional latent representations.
        """
        batch_size = x.shape[0] // 2 # 单个视图的批次大小
        x_masked = x

        # Pass through encoder
        # 将输入数据编码为高维潜在表示
        lat_higt_dim_out = self.enc(x_masked)

        # 使用 MLP 将高维表示映射到低维空间
        lat_vis = self.vis(lat_higt_dim_out)

        # import pdb; pdb.set_trace()

        return x_masked, lat_higt_dim_out, lat_vis, lat_higt_dim_out

    def get_weight(self):
        """
        Retrieves and processes the expert weights.
        获取并处理专家权重

        Returns:
            weight (Tensor): Processed weights.
        """
        # 获取专家嵌入
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

    def forward_train_enc(self, x_masked, input_data, lat_high_dim, lat_vis):
        """
        编码器训练的前向传播（计算流形损失）

        计算用于训练编码器的各种损失：
        1. 正交损失（可选）：鼓励 MoE 专家多样化
        2. 潜在空间流形损失：保持高维潜在表示的结构
        3. 可视化空间流形损失：保持低维可视化的结构
        
        """

        # Compute orthogonal loss if required
        if self.hparams.use_orthogonal:
            orthogonal_loss = self.batch_patten_loss(x_masked, self.mask)
        else:
            orthogonal_loss = 0

        with torch.no_grad():
            mean = lat_vis.mean(dim=0).detach()
            std = lat_vis.std(dim=0).detach()

        # print(f"mean: {mean.shape}, std: {std.shape}")
        lat_vis = (lat_vis - mean) / std

        if self.hparams.loss_type == 'G':
            LossFunc = self.LossManifold_Global
        elif self.hparams.loss_type == 'L':
            LossFunc = self.LossManifold
        elif self.hparams.loss_type == 'A':
            LossFunc = self.LossManifold_All


        # Compute manifold losses
        loss_lat = LossFunc(
            input_data=input_data.reshape(lat_high_dim.shape[0], -1),
            latent_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_lat,
            nu=self.hparams.nu_lat,
        )
        loss_emb = LossFunc(
            input_data=input_data.reshape(lat_vis.shape[0], -1),
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_emb,
            nu=self.hparams.nu_emb,
        )

        return loss_emb, loss_lat, orthogonal_loss

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
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        index = batch["index"]

        self.update_training_str(self.current_epoch)
        log_dict = {}
        # Concatenate original and augmented data
        data_input = torch.cat([data_input_item, data_input_aug])
        # Forward pass
        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input,
            tau=self.hparams.tau,
        )        
        
        loss_emb, loss_lat, orthogonal_loss = self.forward_train_enc(
            x_masked, data_input, lat_high_dim, lat_vis
        )
        # Compute total loss

        # Log losses
        log_dict.update(
            {
                "loss_emb": loss_emb,
                "loss_lat": loss_lat,
                "orthogonal_loss": orthogonal_loss,
            }
        )
        loss_all = (loss_emb + loss_lat) / 2 + orthogonal_loss * 10
        
        return loss_all


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
        data_input = torch.cat([data_input_item, data_input_aug])
        
        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input,
            tau=self.hparams.tau,
        )
        
        rep = True if self.current_epoch > 100 else False
        swav_loss, swav_stats = self.swav_head(lat_vis, rep=rep)

        log_dict.update(swav_stats)

        # self.clustering_embedding = nn.Embedding(1024, 2)


        self.log_dict(log_dict)
        return swav_loss

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


