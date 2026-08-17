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

from model.encoder import DMTEncoder
from model.encoder import NN_FCBNRL_MM


class Tree_learn(LightningModule):
    """
    Tree_learn is a PyTorch Lightning module that implements the training and evaluation of the model.
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
        t_mul=1.0,
        num_use_mlevel_list=[0.5, 1, 2, 5],
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

        if self.hparams.nu_emb < 0:
            self.hparams.nu_emb = self.hparams.nu_lat
        if self.hparams.exaggeration_emb < 0:
            self.hparams.exaggeration_emb = self.hparams.exaggeration_lat

        # Initialize the encoder
        self.enc = DMTEncoder(
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
        state_dict = torch.load(ckpt_path+"enc-latest.pt", map_location="cpu")  
        self.enc.load_state_dict(state_dict)
        self.enc.to(self.device)

        self.vis_list = nn.ModuleList(
            self.InitNetworkMLP(
                # NS=[t_output_dim * num_use_moe, 500, vis_dim], last_relu=False
                NS=[t_output_dim, 500, vis_dim], last_relu=False
            ) for _ in num_use_mlevel_list
        )
        state_dict = torch.load(ckpt_path+"vis_list-latest.pt", map_location="cpu")
        self.vis_list.load_state_dict(state_dict)
        self.vis_list.to(self.device)


        self.tree_node_embedding = nn.ModuleList(
            [nn.Embedding(2 ** (i + 1), t_output_dim) for i in range(self.hparams.tree_depth)]
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

            row_indices = torch.arange(batch_size, device=rooter_input.device).repeat_interleave(2)
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
        distance_matrix = distance_matrix + 1e-9
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

    def cal_dis_to_p(self, dis_input_ab):
        batch_size = dis_input_ab.size(0)
        sorted_indices = torch.argsort(dis_input_ab, dim=1)
        ranks = torch.zeros_like(
            dis_input_ab, dtype=torch.long, device=dis_input_ab.device
        )
        order = torch.arange(batch_size, device=dis_input_ab.device).expand_as(
            dis_input_ab
        )
        ranks.scatter_(1, sorted_indices, order)
        P = torch.pow(self.hparams.exp_b, ranks.float())
        return P

    def LossManifold_Global(
        self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1, t_mul=1
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
        # if self.hparams.norm_loss:
        #     mean = latent_data.mean().detach()
        #     std = latent_data.std().detach()
        #     latent_data = (latent_data - mean) / std

        batch_size = latent_data.shape[0] // 2
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        dis_ab = torch.cdist(features_a, features_b) * t_mul
        Q = self.t_distribution_similarity(dis_ab, df=nu)

        with torch.no_grad():
            features_input_a = input_data[:batch_size]
            features_input_b = input_data[batch_size:]
            dis_input_ab = self._DistanceSquared(features_input_a, features_input_b)
            dis_input_ab[torch.eye(dis_input_ab.shape[0]) == 1] = 0
            P_x = self.cal_dis_to_p(dis_input_ab)
            P_y = self.cal_dis_to_p(dis_input_ab.T).T
            P = torch.sqrt(P_x * P_y)
            P[P < self.hparams.exp_min] = self.hparams.exp_min

        # P = self.UMAPNoSigmaSimilarity(dis_input_ab, nu=100)

        # import pdb; pdb.set_trace()

        #     with torch.no_grad():
        # dis_input_ab[torch.eye(dis_input_ab.shape[0]) == 1] = 0
        # ranks_x = self.cal_ranks(dis_input_ab)
        # # ranks_y = self.cal_ranks(dis_input_ab.T).T
        # P_x = torch.pow(self.hparams.exp_b, ranks_x.float())
        # # P_y = torch.pow(self.hparams.exp_b, ranks_y.float())
        # # P = (P_x * P_y)
        # P = P_x
        # P[P < self.hparams.exp_min] = self.hparams.exp_min

        EPS = 1e-8
        losssum1 = P * torch.log(Q + EPS)
        losssum2 = (1 - P) * torch.log(1 - Q + EPS)
        loss = -1 * (losssum1 + losssum2).mean()
        return loss

    def LossManifold_All(
        self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1
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
        self, input_data, latent_data, temperature=1, exaggeration=1, nu=0.1
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
            # import pdb; pdb.set_trace()
            lat_vis = self.vis_list[i](lat_high_dim_out)
            lat_vis_list.append(lat_vis)

        # import pdb; pdb.set_trace()

        return x_masked, lat_high_dim_out, lat_vis_list[-1], lat_vis_list

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
                std = lat_vis.std(dim=0).detach()

            lat_vis_n = (lat_vis - mean) / std

            if self.hparams.loss_type == "G":
                LossFunc = self.LossManifold_Global
            elif self.hparams.loss_type == "L":
                LossFunc = self.LossManifold
            elif self.hparams.loss_type == "A":
                LossFunc = self.LossManifold_All

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

        return loss_emb_mean, loss_emb_list, orthogonal_loss


        # print(f"self.training_str {self.training_str}, epoch {epoch}")

    def validation_step(self, batch, batch_idx):
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        index = batch["index"]

        log_dict = {}
        # Concatenate original and augmented data
        data_input = torch.cat([data_input_item, data_input_aug])
        # Forward pass
        # x_masked, lat_high_dim, lat_vis, _ = self(
        #     data_input,
        #     tau=self.hparams.tau,
        # )

        loss_emb_mean, loss_emb_list, orthogonal_loss = self.forward_train_enc(
            data_input_item=data_input_item, data_input_aug=data_input_aug
        )
        # Compute total loss

        # Log losses
        # log_dict.update(
        #     {
        #         "loss_emb": loss_emb,
        #         "orthogonal_loss": orthogonal_loss,
        #     }
        # )
        # loss_all = (loss_emb + loss_lat) / 2 + orthogonal_loss * 10

        return loss_emb_mean

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

        data_input = torch.cat([data_input_item, data_input_aug])
        x_masked, lat_high_dim, lat_vis, lat_vis_list = self(
            data_input,
            tau=self.hparams.tau,
        )

        # self.update_training_str(self.current_epoch)
        log_dict = {}
        if self.set_mean_bool == False:
            print("Set the mean and std of the lat_vis")
            mean = lat_vis.mean(dim=0)
            std = lat_vis.std(dim=0)
            device = lat_vis.device
            self.lat_vis_mean.data = torch.tensor(mean).to(device)
            self.lat_vis_std.data = torch.tensor(std).to(device)
            self.set_mean_bool = True

        lat_vis = (lat_vis - self.lat_vis_mean) / (self.lat_vis_std + 1e-8)
        cond = lat_high_dim.detach()

        tree_rout, vector_rout, loss_rout = self.router_forward(
            cond.float().detach(),
            tree_rout_bool=True,
            ec_ce_weight=self.hparams.ec_ce_weight,
        )

        log_dict["loss_all"] = loss_rout
        log_dict["loss_route"] = loss_rout
        log_dict["epoch"] = self.current_epoch
        
        self.log_dict(log_dict)
        return loss_rout

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
