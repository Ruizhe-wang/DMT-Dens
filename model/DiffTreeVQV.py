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
    def __init__(
        self,
        embedding_dim: int = 2,
        n_prototypes: int = 1024,
        temperature: float = 1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        metric: str = "euclidean_t",    # 新增: "cosine" 或 "euclidean_t"
        t_df: float = 1.0,              # t 分布自由度 ν，常用 1.0（Cauchy）
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.metric = metric
        self.t_df = t_df

        # 原型参数 (K x D)
        self.prototypes = nn.Parameter(
            self._standardize(
                torch.randn(
                    n_prototypes,
                    embedding_dim
                    ))
        )


    @torch.no_grad()
    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        B, K = logits.shape
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        Q = torch.exp(logits / self.sinkhorn_epsilon).t()  # [K, B]
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
        """
        if self.metric == "cosine":
            z_norm = F.normalize(z, dim=1)
            P_norm = F.normalize(P, dim=1)
            return (z_norm @ P_norm.t()) / self.temperature  # [B, K]
        elif self.metric == "euclidean_t":
            # 不强制单位范数；用欧氏距离
            d2 = self._pairwise_sqeuclidean(z, P)           # [B, K]
            # log t-kernel 相似度（数值稳定）
            logits = -0.5 * (self.t_df + 1.0) * torch.log1p(d2 / self.t_df)
            return logits / self.temperature                 # [B, K]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        按 batch 内做 StandardScaler: 减均值 / 除标准差
        """
        mean = x.mean(dim=0, keepdim=True)                  # [1, D]
        std = x.std(dim=0, keepdim=True, unbiased=False)    # [1, D]
        return 1 * (x - mean) / (std + 1e-6)

    def forward(self, lat_vis: torch.Tensor, rep=False):
        assert lat_vis.dim() == 2 and lat_vis.size(1) == self.prototypes.size(1), \
            "lat_vis dim must match prototype dim"

        lat_vis = self._standardize(lat_vis)

        B2 = lat_vis.size(0)
        assert B2 % 2 == 0, "Expect even batch (2 views concatenated)"
        B = B2 // 2
        z1, z2 = lat_vis[:B], lat_vis[B:]                     # [B, D] each

        # logits to prototypes for each view
        logits1 = self._logits(z1, self.prototypes)           # [B, K]
        logits2 = self._logits(z2, self.prototypes)           # [B, K]

        with torch.no_grad():
            q1 = F.softmax(logits1 / self.temperature, dim=1).detach()
            q2 = F.softmax(logits2 / self.temperature, dim=1).detach()

        logp2 = F.log_softmax(logits2, dim=1)
        logp1 = F.log_softmax(logits1, dim=1)
        loss12 = (-q1 * logp2).sum(dim=1).mean()
        loss21 = (-q2 * logp1).sum(dim=1).mean()
        loss = 0.5 * (loss12 + loss21)

        with torch.no_grad():
            ent1 = -(q1 * (q1.clamp_min(1e-12).log())).sum(dim=1).mean()
            ent2 = -(q2 * (q2.clamp_min(1e-12).log())).sum(dim=1).mean()
            proto_norm = self.prototypes.norm(dim=1).mean()

        stats = {
            "swav/loss": loss.item(),
            "swav/loss12": loss12.item(),
            "swav/loss21": loss21.item(),
            "swav/q_entropy_v1": ent1.item(),
            "swav/q_entropy_v2": ent2.item(),
            "swav/proto_norm": proto_norm.item(),
        }
        return loss, stats


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
        nu_lat=[0.1, 0.1],
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
        loss_type='G',
        all_g_l_weight=0.1,
        av_temperature=0.02,
        av_t_df=1.0,           # Cauchy
        av_n_prototypes=1024,
        av_sinkhorn_epsilon=0.2,
        av_sinkhorn_iters=2,
        zzlfct=2.0,
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
        self.lat_vis_mean = nn.Parameter(torch.zeros(2))
        self.lat_vis_std = nn.Parameter(torch.zeros(2))
        self.init_imge = None
        self.set_mean_bool = False

        self.uuid_str = str(uuid.uuid4())[:10]

        # if self.hparams.nu_emb < 0:
        #     self.hparams.nu_emb = self.hparams.nu_lat
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

        self.val_vis_list = []

        self.tree_node_embedding = nn.ModuleList(
            [nn.Embedding(2 ** (i + 1), 2) for i in range(self.hparams.tree_depth)]
        )

        self.vis = self.InitNetworkMLP(
            NS=[32 * num_use_moe, 500, vis_dim], last_relu=False
        )
        training_str = None

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

        num_embeddings = emb_level_item.shape[0]
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
        distances = (
            (rooter_input**2).sum(dim=1, keepdim=True)
            + (emb_level_item**2).sum(dim=1)
            - 2 * torch.matmul(rooter_input, emb_level_item.t())
        )
        if last_tree_node_idx is not None and tree_rout_bool:
            distances_plus = torch.full_like(distances, float("inf"))

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
        tree_rout_list = []
        vector_list = []
        loss_list = []

        for i in range(len(self.tree_node_embedding)):
            emb_level_item = self.tree_node_embedding[i].weight
            if i > 0:
                last_tree_node_idx = tree_rout_list[-1]
            else:
                last_tree_node_idx = None

            distances, distances_on_tree = self.cal_distance_matrix_with_tree(
                rooter_input, emb_level_item, last_tree_node_idx, tree_rout_bool
            )

            if last_tree_node_idx is not None:
                encoding_indices, quantized, loss_ec_tree = self.align_loss(
                    rooter_input, emb_level_item, distances_on_tree
                )
                _, _, loss_ce_tree = self.align_loss(
                    emb_level_item, rooter_input, distances_on_tree.t()
                )
                loss = loss_ec_tree + loss_ce_tree * ec_ce_weight
            else:
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

        Args:
            distance_matrix (Tensor): Distance matrix.
            df (float): Degrees of freedom for t-distribution.

        Returns:
            similarity_matrix (Tensor): Similarity matrix.
        """
        distance_matrix = distance_matrix + 1e-6
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
        num_neg_sample = 200
        EPS = 1e-12
        
        batch_size = latent_data.shape[0] // 2        
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]
        
        dis_ab = self._DistanceSquared(features_a, features_b) * temperature
        Q = self.UMAPNoSigmaSimilarity(dis_ab, nu=nu)
        
        with torch.no_grad():

            features_input_a = input_data[:batch_size]
            features_input_b = input_data[batch_size:]
            dis_input_ab = self._DistanceSquared(features_input_a, features_input_b) * temperature
            # diag_dis_input_ab = torch.diagonal(dis_input_ab).detach()

            # dis_input_ab = dis_input_ab - \
            #     self.hparams.zzlfct*diag_dis_input_ab.unsqueeze(1) / 2 - \
            #     self.hparams.zzlfct*diag_dis_input_ab.unsqueeze(0) / 2
            
            dis_input_ab[torch.eye(dis_input_ab.shape[0]) == 1] = 0
            
            dis_input_ab[dis_input_ab < EPS] = EPS
        
            P = self.UMAPNoSigmaSimilarity(dis_input_ab, nu=100)

        losssum1 = P * torch.log(Q + EPS)
        losssum2 = (1 - P) * torch.log(1 - Q + EPS)
        loss = losssum1 + losssum2

        # with torch.no_grad():
        #     loss_sort, idx = torch.sort(loss, dim=1)
        #     mask = torch.zeros_like(loss, dtype=bool, device=latent_data.device)
        #     idx_y = idx[:, :num_neg_sample + 1].reshape(-1)
        #     idx_x = torch.arange(batch_size, device=latent_data.device).repeat_interleave(num_neg_sample + 1)
            
        #     mask[idx_x, idx_y] = True


        # loss = loss[mask].mean()
        loss = loss.mean() 

        # import pdb; pdb.set_trace()
        # print('loss:', loss.item())

        return -1 * loss

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
        
        # loss1 = loss1/loss1.detach()
        
        loss2 = self.LossManifold(
            input_data=input_data,
            latent_data=latent_data,
            temperature=temperature,
            exaggeration=exaggeration,
            nu=nu
        ) 
        # loss2 = loss2/loss2.detach()
        loss = loss1 * self.hparams.all_g_l_weight + loss2 * (1 - self.hparams.all_g_l_weight)

        return loss


    def LossManifold(self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1):
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

        return loss - dis_ab.mean()

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
        
        x_masked = x.to(dtype=torch.float64)

        # Pass through encoder
        lat_higt_dim_out = self.enc(x_masked)
        lat_vis = self.vis(lat_higt_dim_out)

        # import pdb; pdb.set_trace()

        return x_masked, lat_higt_dim_out, lat_vis, lat_higt_dim_out

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

    def forward_train_enc(self, data_input, nu=0.1):

        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input,
            tau=self.hparams.tau,
        )

        # Compute orthogonal loss if required
        # if self.hparams.use_orthogonal:
        #     orthogonal_loss = self.batch_patten_loss(x_masked, self.mask)
        # else:
        #     orthogonal_loss = 0

        # with torch.no_grad():
        #     mean = lat_vis.mean(dim=0).detach()
        #     std = lat_vis.std(dim=0).detach()

        # # print(f"mean: {mean.shape}, std: {std.shape}")
        # normalized_lat_vis = (lat_vis - mean) / (std + 1e-8)
        
        # 随着训练进行逐渐减少的噪声量
        # if hasattr(self, 'current_epoch') and hasattr(self.hparams, 'max_epochs'):
        #     noise_factor = max(0.0, 0.001 * (1.0 - self.current_epoch / self.hparams.max_epochs))
        # else:
        #     noise_factor = 0.05
            
        # print(f"noise_factor: {noise_factor}, current_epoch: {self.current_epoch}, max_epochs: {self.hparams.max_epochs}")
        
        # 在训练时添加噪声，采用重参数化技巧
        # if self.training:
        #     epsilon = torch.randn_like(normalized_lat_vis) * noise_factor
        #     normalized_lat_vis = normalized_lat_vis + epsilon
        # else:
        normalized_lat_vis = lat_vis
        # import pdb; pdb.set_trace()
        

        if self.hparams.loss_type == 'G':
            LossFunc = self.LossManifold_Global
        elif self.hparams.loss_type == 'L':
            LossFunc = self.LossManifold
        elif self.hparams.loss_type == 'A':
            LossFunc = self.LossManifold_All

        # Compute manifold losses
        loss_lat = LossFunc(
            input_data=data_input.reshape(data_input.shape[0], -1),
            latent_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_lat,
            nu=nu,
        )
        loss_emb = LossFunc(
            input_data=data_input.reshape(normalized_lat_vis.shape[0], -1),
            latent_data=normalized_lat_vis.reshape(normalized_lat_vis.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_emb,
            nu=nu,
        )
        


        return loss_emb, loss_lat, 0

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
            data_input
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
        label = batch["label"]
        level = batch["data_input_item_list"].shape[1]
        
        # import pdb; pdb.set_trace()
        
        self.label = label

        self.update_training_str(self.current_epoch)
        log_dict = {}

        loss_list = []

        nu_lat_list = self.hparams.nu_lat
        
        if self.training_str == "step1":
            
            for level in range(level):
                data_input_item = batch["data_input_item_list"][:, level, :]
                data_input_aug = batch["data_input_aug_list"][:, level, :]
                
                data_input = torch.cat([data_input_item, data_input_aug])
                loss_emb, loss_lat, orthogonal_loss = self.forward_train_enc(
                    data_input, nu=nu_lat_list[level])
                
                loss_list.append(loss_lat)
                log_dict.update(
                    {
                        f"loss_emb_{level}": loss_emb,
                        f"loss_lat_{level}": loss_lat,
                        f"orthogonal_loss_{level}": orthogonal_loss,
                        # "swav_loss": swav_loss,
                    }
                )
            # log_dict.update(swav_stats)
            # loss_all = torch.stack(loss_list).mean() 
            
            loss_all = loss_list[1]/loss_list[1].detach() 

        elif "step2" in self.training_str:

            if self.set_mean_bool == False:
                print("Set the mean and std of the lat_vis")
                mean = lat_vis.mean(dim=0)
                std = lat_vis.std(dim=0)
                device = lat_vis.device
                self.lat_vis_mean.data = torch.tensor(mean).to(device)
                self.lat_vis_std.data = torch.tensor(std).to(device)
                self.set_mean_bool = True

            lat_vis = (lat_vis - self.lat_vis_mean) / (self.lat_vis_std + 1e-8)
            cond = lat_vis.detach()

            tree_rout, vector_rout, loss_rout = self.router_forward(
                cond.float().detach(),
                tree_rout_bool=True,
                ec_ce_weight=self.hparams.ec_ce_weight,
            )

            loss_all = loss_rout

            log_dict.update(
                {
                    "loss_rute": loss_rout,
                    "epoch": self.current_epoch,
                }
            )

        log_dict.update(
            {
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "loss_all": loss_all,
            }
        )

        self.log_dict(log_dict)
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


