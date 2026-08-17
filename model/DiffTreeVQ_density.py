import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy
from lightning import LightningModule
import uuid
from model.CosineAnnealingSchedule import CosineAnnealingSchedule

from model.encoder import DMTEncoder
from model.encoder import NN_FCBNRL_MM
from model.encoder_factory import build_encoder, describe_capacity
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
        # --- Experiment 1 mechanism-ablation toggles (defaults reproduce the
        #     original hard-coded behaviour of LossManifold_Global) ---
        manifold_affinity="rank",  # "rank" (b^{rank}) | "distance" (Gaussian on distance)
        manifold_symmetric="bidirectional",  # "bidirectional" (geom. mean) | "unidirectional"
        manifold_hardpair=True,  # True: hard-pair mining | False: mean over all pairs
        hardpair_k=100,  # number of hardest pairs kept per row when manifold_hardpair
        # Causal diagnostic for the distance/all-pair collapse. The default
        # preserves historical configs. When enabled, only the final
        # distance-derived P has its off-diagonal mass normalized row-wise.
        distance_p_row_normalize=False,
        t_mul=1.0,  # 距离缩放系数（此片段中 Global loss 的 dis_ab 乘法）
        num_use_mlevel_list=[1],  # 当前实验默认使用单个 2D projection head
        density_weight=0.1,
        density_k=12,
        density_num_anchors=256,
        density_scale_mode="multi",
        # Numerical-control switch for the collapse diagnostic.  False keeps
        # the historical detached-statistics backward pass byte-for-byte.
        # True preserves the same standardized forward objective while
        # allowing the batch mean/std to participate in autograd and imposing
        # a small floor before division.
        stable_embedding_standardization=False,
        embedding_std_floor=1e-4,
        # Explicit replacement for the legacy boolean above. ``None`` keeps
        # historical configs reproducible; new experiments should select
        # ``differentiable_global`` so both axes share one autograd-visible
        # scale and a line collapse is not hidden by per-axis rescaling.
        embedding_standardization=None,
        density_distance_floor=None,
        density_pearson_floor=None,
        use_lg_loss=False,
        lg_weight=0.1,
        lg_teacher_momentum=0.99,
        lg_num_anchors=4,
        lg_subset_k=32,
        lg_distance_weight=1.0,
        lg_position_weight=0.2,
        # --- Encoder-ablation benchmark ---
        # "mlp" reproduces the historical DMTEncoder exactly; the other types
        # swap only the encoder, leaving the projection head, losses,
        # augmentation and callbacks untouched.
        encoder_type="mlp",
        encoder_kwargs=None,
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
            T_num_layers (int): Legacy encoder depth parameter kept for config compatibility.
            T_num_attention_heads (int): Legacy compatibility parameter; not used by the default MLP encoder.
            T_hidden_size (int): Hidden size in the DMT/MLP encoder.
            T_intermediate_size (int): Legacy compatibility parameter.
            T_hidden_dropout_prob (float): Legacy compatibility parameter.
            T_attention_probs_dropout_prob (float): Legacy compatibility parameter.
            ckpt_path (str): Path to checkpoint for loading model.
            use_orthogonal (bool): Whether to use orthogonal loss.
            num_use_moe (int): Number of experts in Mixture of Experts.
            vis_dim (int): Dimension of visualization space.
            trans_out_dim (int): Legacy compatibility parameter.
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
        # Enabled and consumed by ManifoldDiagnosticsCallback.  Only scalar
        # summaries are exposed; full pairwise matrices never leave this model.
        self._manifold_diagnostics_enabled = False
        self._manifold_diagnostics_every_n_steps = 1
        self._latest_manifold_diagnostics = None

        if self.hparams.nu_emb < 0:
            self.hparams.nu_emb = self.hparams.nu_lat
        if self.hparams.exaggeration_emb < 0:
            self.hparams.exaggeration_emb = self.hparams.exaggeration_lat

        # Initialize the encoder. The T_* argument names are preserved in
        # configs; with encoder_type="mlp" (the default) this is byte-identical
        # to the historical DMTEncoder construction.
        self.enc = build_encoder(
            encoder_type=encoder_type,
            num_input_dim=num_input_dim,
            output_dim=t_output_dim,
            encoder_kwargs=encoder_kwargs,
            T_num_layers=T_num_layers,
            T_num_attention_heads=T_num_attention_heads,
            T_hidden_size=T_hidden_size,
            T_intermediate_size=T_intermediate_size,
            T_hidden_dropout_prob=T_hidden_dropout_prob,
            T_attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
        )
        # Capacity accounting is a required column of the benchmark result
        # table. It is computed here so that every run records it, rather than
        # relying on the write-up to remember an out-of-band encoder.
        self.encoder_capacity = describe_capacity(
            self.enc,
            num_input_dim=num_input_dim,
            output_dim=t_output_dim,
            T_num_layers=T_num_layers,
            T_num_attention_heads=T_num_attention_heads,
            T_hidden_size=T_hidden_size,
            T_intermediate_size=T_intermediate_size,
            T_hidden_dropout_prob=T_hidden_dropout_prob,
            T_attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
        )
        self.encoder_num_params = self.encoder_capacity["params"]
        print(
            f"[encoder] type={encoder_type} in_dim={num_input_dim} "
            f"out_dim={t_output_dim} params={self.encoder_num_params:,} "
            f"({self.encoder_capacity['param_ratio']:.3f}x baseline MLP "
            f"{self.encoder_capacity['baseline_params']:,})"
        )
        if not self.encoder_capacity["param_in_band"]:
            print(
                f"[encoder] WARNING capacity out of the 0.5-2.0x band: "
                f"{self.encoder_capacity['param_ratio']:.3f}x. This is a capacity "
                f"mismatch, not a pure architecture comparison -- it must be "
                f"stated in the summary table and in the conclusions."
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

        if training_str == None:
            self.training_str = "step1"
        else:
            self.training_str = training_str

        self.validate_bool = validate_bool

    def on_fit_start(self):
        """Records encoder capacity in the run config.

        Written to the logger rather than only to stdout so that an encoder
        sitting outside the 0.5-2x parameter band is attached to every run and
        cannot be lost when the summary table is assembled.
        """
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        config = getattr(experiment, "config", None)
        if config is None:
            return
        payload = {
            f"encoder_{key}": value for key, value in self.encoder_capacity.items()
        }
        payload["encoder_type"] = self.hparams.encoder_type
        try:
            config.update(payload, allow_val_change=True)
        except Exception:  # noqa: BLE001 - logging must never break training
            pass

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
        emb = emb - emb.mean(dim=0, keepdim=True)
        scale = emb.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return emb / scale

    @staticmethod
    def _normalize_distance_vector(dist_vec):
        if dist_vec.numel() == 0:
            return dist_vec
        mean = dist_vec.mean()
        std = dist_vec.std(unbiased=False).clamp_min(1e-6)
        return (dist_vec - mean) / std

    @staticmethod
    def _pearson_correlation_loss(x, y, denominator_floor=None):
        """
        Pearson 相关系数损失：1 - r(x, y)

        直接优化两个密度向量之间的线性相关性，
        不要求绝对数值匹配，只要求相对排序/形状一致。

        等价关系：归一化 MSE = 2 * (1 - Pearson)，但显式 Pearson 对尺度不变，
        梯度行为更稳定，且与顶会论文表述一致。
        参考：densMAP (Narayan et al., NeurIPS 2021)
        """
        xc = x - x.mean()
        yc = y - y.mean()
        denominator = xc.norm() * yc.norm()
        if denominator_floor is None:
            # Byte-compatible legacy path for already queued experiments.
            denominator = denominator + 1e-8
        else:
            denominator = denominator.clamp_min(float(denominator_floor))
        r = (xc * yc).sum() / denominator
        return 1.0 - r

    @staticmethod
    def _compute_knn_log_density(query, reference, k, distance_floor=None):
        """
        标准 kNN 对数密度估计器。

        理论依据：
            f(x) ∝ k / (n · V_d · r_k^d)
            log f(x) ∝ -log(r_k)  （k, n, V_d 为常数，省略）

        其中 r_k 为第 k 个最近邻距离，是 densMAP 等方法使用的标准形式。
        比"RMS 均值距离"更接近真实密度估计量。

        query:     [A, D] 查询点（锚点）
        reference: [N, D] 参考点集（query 是其子集时自身距离为 0 会被跳过）
        k:         近邻数

        返回：[A] 对数密度（未归一化，值越大密度越高）
        """
        # Keep the estimator in float32 under AMP. If r_k is zero, the
        # historical 1e-8 epsilon underflows in fp16 and log(0) becomes inf.
        dist = torch.cdist(query.float(), reference.float())  # [A, N]
        k_fetch = min(k + 1, dist.shape[1])
        # values[:, -1] 是第 k 个近邻距离（跳过距离=0 的自身）
        rk = torch.topk(dist, k=k_fetch, dim=1, largest=False).values[:, -1]  # [A]
        if distance_floor is None:
            # Byte-compatible legacy path for already queued experiments.
            rk_safe = rk + 1e-8
        else:
            # A true floor also stops gradients from growing without bound once
            # neighbours are numerically coincident.
            rk_safe = rk.clamp_min(float(distance_floor))
        return -torch.log(rk_safe)  # log-density ∝ -log(r_k)

    def _embedding_standardization_mode(self):
        """Resolve the explicit mode without changing historical configs."""
        mode = getattr(self.hparams, "embedding_standardization", None)
        if mode is None:
            return (
                "differentiable_axis"
                if self.hparams.stable_embedding_standardization
                else "legacy_detached_axis"
            )
        mode = str(mode).lower()
        valid = {
            "legacy_detached_axis",
            "differentiable_axis",
            "differentiable_global",
        }
        if mode not in valid:
            raise ValueError(
                f"embedding_standardization must be one of {sorted(valid)}, "
                f"got {mode!r}"
            )
        return mode

    def _standardize_embedding(self, embedding):
        """Standardize a 2-D embedding according to the configured mode."""
        mode = self._embedding_standardization_mode()
        if mode == "legacy_detached_axis":
            with torch.no_grad():
                mean = embedding.mean(dim=0).detach()
                std = embedding.std(dim=0).detach() + 1e-8
            return (embedding - mean) / std

        embedding_fp32 = embedding.float()
        centered = embedding_fp32 - embedding_fp32.mean(dim=0, keepdim=True)
        if mode == "differentiable_axis":
            scale = embedding_fp32.std(dim=0, keepdim=True).clamp_min(
                float(self.hparams.embedding_std_floor)
            )
        else:
            # One RMS scale shared by both axes. This removes translation and
            # isotropic scale while preserving the raw axis ratio, so a line
            # collapse remains visible to both losses and diagnostics.
            # Floor the mean square *before* sqrt. Clamping sqrt(0) only makes
            # the forward value finite: its backward still encounters the
            # singular derivative of sqrt at zero and can produce 0 * inf =
            # NaN when an fp16 embedding has quantized to a single point.
            scale_floor = float(self.hparams.embedding_std_floor)
            mean_square = centered.square().mean()
            scale = mean_square.clamp_min(scale_floor**2).sqrt()
        return centered / scale

    @staticmethod
    def _raw_embedding_diagnostics(embedding):
        """Compute raw-coordinate health metrics in explicit float32.

        Lightning runs ``training_step`` under AMP autocast.  Merely converting
        the embedding to float32 is not sufficient because the covariance
        matrix multiplication is autocast back to float16 on CUDA, while
        ``torch.linalg.eigvalsh`` has no float16 CUDA kernel.  These metrics are
        detached diagnostics, so disabling autocast here preserves the training
        objective and makes the eigensolver portable across precision modes.
        """
        with torch.no_grad(), torch.autocast(
            device_type=embedding.device.type, enabled=False
        ):
            raw_embedding = embedding.detach().float()
            raw_centered = raw_embedding - raw_embedding.mean(dim=0, keepdim=True)
            raw_std = raw_centered.std(dim=0, unbiased=False)
            raw_global_scale = raw_centered.square().mean().sqrt()
            raw_coordinate_ratio = raw_std.min() / raw_std.max().clamp_min(1e-12)
            covariance = raw_centered.T @ raw_centered / max(
                1, raw_centered.shape[0]
            )
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            raw_axis_ratio = eigenvalues[0].sqrt() / eigenvalues[-1].sqrt().clamp_min(
                1e-12
            )
        return {
            "embedding/raw_std_axis0": raw_std[0],
            "embedding/raw_std_axis1": raw_std[1],
            "embedding/raw_global_scale": raw_global_scale,
            "embedding/raw_coordinate_std_ratio": raw_coordinate_ratio,
            "embedding/raw_axis_ratio": raw_axis_ratio,
            # Keep the historical strict collapse threshold unchanged for
            # backward-compatible tables, but expose a practical geometric
            # warning for embeddings that are already visually near-linear.
            "embedding/raw_near_line": (raw_axis_ratio < 5.0e-2).to(
                dtype=raw_std.dtype
            ),
        }

    @staticmethod
    def _pairwise_distance_vector(emb):
        n = emb.shape[0]
        if n < 2:
            return emb.new_zeros((0,))
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
        anchor_points = batch_data[anchor_indices]
        distances = torch.cdist(anchor_points, batch_data)
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

    def _compute_local_density_loss(self, batch_data, student_lat_vis_list):
        """
        多尺度 kNN 密度保持损失。

        改进说明（相对原始实现）：
        1. 使用第 k 个邻居距离 r_k 作为密度估计（标准 kNN 密度估计量）
           原先使用 RMS 均值距离，不对应任何标准估计量。
        2. 在低维空间独立计算 kNN 密度（而非沿用 HD 邻居索引）
           原先固定 HD 邻居导致"幽灵密度"——HD 近邻在 LD 中可能很远，
           计算出的 LD 密度不反映真实 LD 局部结构。
        3. 多尺度：同时在 k_small=k//2 和 k 两个尺度上计算，
           捕捉不同粒度的密度结构（类内精细结构 + 类间粗糙结构）。
        4. Pearson 相关系数损失替代归一化 MSE
           两者数学等价（归一化 MSE = 2*(1-Pearson)），但 Pearson 显式表达
           "保持密度排序"的优化目标，对绝对尺度不变，梯度更稳定。
        参考：densMAP (Narayan et al., NeurIPS 2021)
        """
        flat_batch = batch_data.reshape(batch_data.shape[0], -1)
        batch_size = flat_batch.shape[0]
        if batch_size < 3:
            return flat_batch.new_zeros(())

        k = min(int(self.hparams.density_k), batch_size - 1)
        if k <= 0:
            return flat_batch.new_zeros(())

        num_anchors = int(self.hparams.density_num_anchors)
        if num_anchors > 0:
            num_anchors = min(num_anchors, batch_size)
            anchor_idx = torch.randperm(batch_size, device=flat_batch.device)[
                :num_anchors
            ]
        else:
            anchor_idx = torch.arange(batch_size, device=flat_batch.device)

        anchor_batch = flat_batch[anchor_idx]  # [A, D_hd]

        # 多尺度邻域：k_small 捕捉精细结构，k 捕捉粗粒度结构
        density_scale_mode = getattr(self.hparams, "density_scale_mode", "multi")
        if density_scale_mode == "single":
            k_scales = [k]
        elif density_scale_mode == "multi":
            k_small = max(1, k // 2)
            k_scales = [k_small, k] if k_small != k else [k]
        else:
            raise ValueError(
                "density_scale_mode must be either 'single' or 'multi', "
                f"got {density_scale_mode!r}"
            )

        # 预计算 HD 对数密度（detach：HD 特征不参与梯度回传）
        log_density_hd_scales = [
            self._compute_knn_log_density(
                anchor_batch,
                flat_batch,
                k_s,
                distance_floor=self.hparams.density_distance_floor,
            ).detach()
            for k_s in k_scales
        ]

        loss_terms = []
        for student_emb in student_lat_vis_list:
            anchor_ld = student_emb[anchor_idx]  # [A, D_ld]

            scale_losses = []
            for k_s, log_hd in zip(k_scales, log_density_hd_scales):
                # 在 LD 空间独立计算 kNN 密度
                # 使用全部 student_emb 作为参考集，与 HD 保持相同的参考规模
                log_ld = self._compute_knn_log_density(
                    anchor_ld,
                    student_emb,
                    k_s,
                    distance_floor=self.hparams.density_distance_floor,
                )
                scale_losses.append(
                    self._pearson_correlation_loss(
                        log_ld,
                        log_hd,
                        denominator_floor=self.hparams.density_pearson_floor,
                    )
                )

            loss_terms.append(torch.stack(scale_losses).mean())

        if not loss_terms:
            return flat_batch.new_zeros(())
        return torch.stack(loss_terms).mean()

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

    @staticmethod
    def t_distribution_similarity(distance_matrix, df):
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
        if df <= 0:
            raise ValueError(
                f"Student-t degrees of freedom must be positive, got {df}"
            )
        # Preserve row normalization, but compute in float32 under AMP. With
        # nu=0.01, fp16 affinities can underflow and make the denominator zero.
        distance_matrix = distance_matrix.float().clamp_min(0.0) + 1e-9
        # 计算 t分布核
        # (1 + d²/v)^(-(v+1)/2)
        numerator = (1.0 + distance_matrix.square() / float(df)).pow(
            -(float(df) + 1.0) / 2.0
        )
        # Sum off-diagonal affinities directly. ``sum(row) - diagonal`` loses
        # meaningful digits when the diagonal is 1 and all other affinities
        # are tiny, even in float32.
        off_diagonal = numerator.clone()
        off_diagonal.fill_diagonal_(0.0)
        denominator = off_diagonal.sum(dim=1, keepdim=True)
        denominator = denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
        similarity_matrix = numerator / denominator
        return similarity_matrix

    @staticmethod
    def _bernoulli_pair_loss(P, Q):
        """Reference pairwise BCE with safe row-normalized boundaries."""
        eps = 1e-8
        q_positive = Q + eps
        q_negative = Q.clone()
        q_negative.fill_diagonal_(0.0)
        q_negative = q_negative.clamp(min=0.0, max=1.0)
        positive = P * torch.log(q_positive)
        negative = (1.0 - P) * torch.log(1.0 - q_negative + eps)
        return -(positive + negative)

    @staticmethod
    def student_t_pairwise_affinity_raw(distance_matrix, df):
        """Compute the unnormalized Student-t kernel in float32."""
        if df <= 0:
            raise ValueError(f"Student-t degrees of freedom must be positive, got {df}")

        distance_fp32 = distance_matrix.float().clamp_min(0.0)
        return (1.0 + distance_fp32.square() / float(df)).pow(
            -(float(df) + 1.0) / 2.0
        )

    @staticmethod
    def student_t_pairwise_affinity(distance_matrix, df, eps=1e-7):
        """Compute an unnormalized Student-t affinity for pairwise BCE.

        Each entry is an independent, bounded pairwise affinity rather than a
        row-wise categorical probability.  This matches the elementwise
        Bernoulli targets used by ``LossManifold_Global``.

        Compute in float32 under mixed precision so that both ``eps`` and
        ``1 - eps`` remain representable before evaluating the BCE.
        """
        affinity = DMTEVT_model.student_t_pairwise_affinity_raw(distance_matrix, df)
        return affinity.clamp(min=eps, max=1.0 - eps)

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

    def cal_dis_to_p_distance(self, dis_input_ab):
        """
        绝对距离亲和度（Experiment 1 因子 M1-a 的对照项）。
        不使用排名，而是对（平方）距离施加逐行带宽自适应的高斯核：
            P_ij = exp(-d_ij / bandwidth_i),  bandwidth_i = mean_j d_ij
        自身距离为 0 → P=1（最近邻最大），与排序核结构可比但直接依赖绝对距离。
        带宽逐行自适应，避免引入额外超参数并对尺度差异保持稳健。
        """
        bandwidth = dis_input_ab.mean(dim=1, keepdim=True) + 1e-8
        P = torch.exp(-dis_input_ab / bandwidth)
        return P

    @staticmethod
    def _row_normalize_off_diagonal_affinity(affinity):
        """Normalize each row's off-diagonal mass and preserve its diagonal."""
        affinity = affinity.float()
        diagonal = torch.diagonal(affinity).clone()
        off_diagonal = affinity.clone()
        off_diagonal.fill_diagonal_(0.0)
        denominator = off_diagonal.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(off_diagonal.dtype).tiny
        )
        normalized = off_diagonal / denominator
        normalized.diagonal().copy_(diagonal)
        return normalized

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
        self._latest_manifold_diagnostics = None

        # Row-normalized Student-t similarity, as in the reference results.
        #
        # An earlier change replaced this with an unnormalized pairwise
        # affinity plus F.binary_cross_entropy, on the argument that row
        # normalization is inconsistent with elementwise Bernoulli targets and
        # that Q_ii is not guaranteed <= 1.  That reasoning is about formal
        # consistency, but the substitution changes what the loss optimizes:
        # row normalization makes Q_ij the *relative* weight of j among i's
        # neighbours, so the objective matches neighbourhood ordering, while
        # the unnormalized form ties each pair to an absolute distance.
        # Empirically it destroyed NG20 (density correlation 0.74 -> 0.21-0.35,
        # kNN preservation 0.275 -> 0.017), so the reference behaviour is
        # restored here.
        Q = self.t_distribution_similarity(dis_ab, df=nu)

        with torch.no_grad():
            features_input_a = input_data[:batch_size]
            features_input_b = input_data[batch_size:]
            dis_input_ab = self._DistanceSquared(features_input_a, features_input_b)
            # dis_input_ab[torch.eye(dis_input_ab.shape[0]) == 1] = 0
            dis_input_ab.fill_diagonal_(0)

            # 高维亲和度构造（Experiment 1 因子 M1-a：排序 vs 绝对距离）
            # P_x: 从视图 A 的视角看（行视角）；P_y: 从视图 B 的视角看（列视角）
            if self.hparams.manifold_affinity == "rank":
                # P_x[i,j] = b^{rank_i(j)}，即从样本 i 看，j 是第几近的
                P_x = self.cal_dis_to_p(dis_input_ab)
                P_y = self.cal_dis_to_p(dis_input_ab.T).T
            elif self.hparams.manifold_affinity == "distance":
                # 绝对距离高斯核（逐行带宽自适应），作为排序核的对照
                P_x = self.cal_dis_to_p_distance(dis_input_ab)
                P_y = self.cal_dis_to_p_distance(dis_input_ab.T).T
            else:
                raise ValueError(
                    f"Unsupported manifold_affinity: {self.hparams.manifold_affinity}"
                )

            # 对称化（Experiment 1 因子 M1-b：双向几何平均 vs 单侧）
            if self.hparams.manifold_symmetric == "bidirectional":
                P = torch.sqrt(P_x * P_y)
            elif self.hparams.manifold_symmetric == "unidirectional":
                P = P_x
            else:
                raise ValueError(
                    f"Unsupported manifold_symmetric: {self.hparams.manifold_symmetric}"
                )

            # 将过小的 P 值截断到 exp_min（默认 1e-5）
            # 避免后续计算 log(P) 或 log(1-P) 时出现数值问题
            P[P < self.hparams.exp_min] = self.hparams.exp_min

            if self.hparams.distance_p_row_normalize:
                if self.hparams.manifold_affinity != "distance":
                    raise ValueError(
                        "distance_p_row_normalize requires manifold_affinity='distance'"
                    )
                P = self._row_normalize_off_diagonal_affinity(P)

        # 正样本项：P_ij · log(Q_ij)
        # 当 P_ij 大（高维相似）时，希望 Q_ij 也大（低维也相似）
        # 否则 log(Q_ij) 会是很大的负数，造成高损失

        # 负样本项：(1-P_ij) · log(1-Q_ij)
        # 当 P_ij 小（高维不相似）时，希望 Q_ij 也小（低维也不相似）
        # 即 (1-Q_ij) 应该大，log(1-Q_ij) 接近 0
        # The helper preserves this expression while masking its mathematically
        # zero diagonal negative term before evaluating log(1 - Q_ii).
        loss = self._bernoulli_pair_loss(P.float(), Q)
        # 配对选择（Experiment 1 因子 M2：hard-pair 挖掘 vs 全对平均）
        if self.hparams.manifold_hardpair:
            # Top-K 硬对挖掘：每行只保留损失最大的 hardpair_k 个
            with torch.no_grad():
                # k 不能超过 batch 内样本对数（loss.shape[1]）
                k_safe = min(self.hparams.hardpair_k, loss.shape[1])
                topk_values, _ = torch.topk(loss, k=k_safe, dim=1)
                # threshold[i] = 第 i 行要被选中的最小损失
                threshold = topk_values[:, -1].unsqueeze(1)
                mask = loss >= threshold
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

        # 默认使用最后一个投影头作为主要二维表征。
        lat_vis_best = lat_vis_list[-1]

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
        normalized_lat_vis_list = []
        for i, lat_vis in enumerate(lat_vis_list):
            lat_vis_n = self._standardize_embedding(lat_vis)
            normalized_lat_vis_list.append(lat_vis_n)

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

        # Raw-coordinate health signals: global standardization intentionally
        # preserves this axis ratio, so these values reveal real point/line
        # collapse instead of reporting the rescaled coordinates as healthy.
        self._latest_embedding_diagnostics = self._raw_embedding_diagnostics(
            lat_vis_list[-1]
        )

        # Current ablation configs use a single projection head
        # (num_use_mlevel_list=[1]). Averaging keeps backward compatibility if
        # older configs request multiple heads.
        loss_emb_mean = torch.stack(loss_emb_list).mean()

        if float(self.hparams.density_weight) == 0.0:
            # Do not evaluate a disabled objective. Multiplying a non-finite
            # density value or derivative by zero does not sanitize it
            # (0 * NaN and 0 * inf are still NaN).
            density_loss = loss_emb_mean.new_zeros(())
        else:
            # The new global mode deliberately feeds the exact same normalized
            # coordinates to manifold and density losses. Historical modes
            # retain their old raw-density path for reproducibility.
            density_lat_vis_list = (
                normalized_lat_vis_list
                if self._embedding_standardization_mode()
                == "differentiable_global"
                else lat_vis_list
            )
            density_loss = self._compute_local_density_loss(
                data_input_item.reshape(data_input_item.shape[0], -1),
                [
                    lat_vis[: data_input_item.shape[0]]
                    for lat_vis in density_lat_vis_list
                ],
            )

        loss_total = loss_emb_mean + self.hparams.density_weight * density_loss

        return loss_total, loss_emb_mean, loss_emb_list, orthogonal_loss, density_loss

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

        (
            loss_total,
            loss_emb_mean,
            loss_emb_list,
            orthogonal_loss,
            density_loss,
        ) = self.forward_train_enc(
            data_input_item=data_input_item, data_input_aug=data_input_aug
        )

        self.log(
            "val_loss_density",
            density_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss_total

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

        (
            loss_total,
            loss_emb_mean,
            loss_emb_list,
            orthogonal_loss,
            density_loss,
        ) = self.forward_train_enc(data_input_item, data_input_aug)

        for i in range(len(loss_emb_list)):
            log_dict[f"loss_emb_{i}"] = loss_emb_list[i]

        loss_all = loss_total

        log_dict.update(
            {
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "loss_emb_mean": loss_emb_mean,
                "loss_density": density_loss,
                "loss_all": loss_all,
            }
        )
        log_dict.update(getattr(self, "_latest_embedding_diagnostics", {}) or {})

        # Read-only projection diagnostics for the collapse causal test.  The
        # standard projection head is [40 -> 500 -> 2], with BatchNorm in each
        # block.  Log the final affine scale without changing the forward pass.
        try:
            final_bn = self.vis_list[-1][-1].block[1]
            gamma = final_bn.weight.detach().float()
            if gamma.numel() >= 2:
                gamma_abs = gamma.abs()
                log_dict.update(
                    {
                        "projection_bn/gamma_axis0": gamma[0],
                        "projection_bn/gamma_axis1": gamma[1],
                        "projection_bn/gamma_abs_min": gamma_abs.min(),
                        "projection_bn/gamma_abs_ratio": gamma_abs.min()
                        / gamma_abs.max().clamp_min(1e-12),
                    }
                )
        except (AttributeError, IndexError, TypeError):
            # Diagnostics must remain non-invasive if a future projection head
            # does not contain the historical final BatchNorm layer.
            pass

        # Read-only diagnostics hook.  The benchmark callback consumes detached
        # values from here so it can compute final-epoch means, non-finite flags
        # and loss-oscillation statistics without changing any training target.
        self._latest_training_losses = {
            "manifold_loss": loss_emb_mean.detach(),
            "density_loss": density_loss.detach(),
            "total_loss": loss_all.detach(),
        }

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
