"""Alternative encoders for the encoder-ablation benchmark.

All encoders in this module share the same contract as ``DMTEncoder`` in
``model/encoder.py``:

    (B, num_input_dim) -> (B, output_dim)

so that the 2D projection head, the manifold/density losses, the augmentation
pipeline and the evaluation callbacks stay untouched.

Two properties are enforced on purpose and should not be relaxed:

* **No row attention.** Attention is applied over feature/latent tokens only.
  A sample's embedding never depends on which other samples share its inference
  batch (the final BatchNorm head is inherited from the baseline and uses
  running statistics at eval time, exactly like the current MLP encoder).
* **No label input.** ``forward`` takes the feature tensor only; ``svc_acc`` is
  a post-hoc evaluation metric, never a training signal.

Hardware note: the target machine is 8x RTX 2080 Ti (Turing, SM75). Turing has
no FlashAttention kernel, so ``scaled_dot_product_attention`` is pinned to the
memory-efficient (cutlass) backend. Without the pin PyTorch would silently fall
back to the math backend, which materialises the full (rows, heads, L, L)
attention matrix -- about 40 GB for MNIST at 8192 rows -- and OOMs the card.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import DMTEncoder, NN_FCBNRL_MM

try:  # torch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _SDPA_CTX_AVAILABLE = True
except ImportError:  # pragma: no cover - torch 2.2 fallback
    _SDPA_CTX_AVAILABLE = False


def _mem_efficient_sdpa_ctx(enabled=True):
    """Context manager pinning SDPA to the memory-efficient backend.

    Raises inside ``scaled_dot_product_attention`` if the backend cannot serve
    the call, which is the intended behaviour: failing loudly is much better
    than a silent fallback to the math backend on a 11 GB card.
    """
    if not enabled:
        return nullcontext()
    if _SDPA_CTX_AVAILABLE:
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    # torch 2.2 style API
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=True, enable_math=False
    )


class TokenBatchNorm(nn.Module):
    """BatchNorm1d over the channel axis of a ``(B, L, C)`` token tensor.

    Normalizes each channel across the ``B * L`` (sample, token) pairs, the
    formulation used by TST (Zerveas et al., 2021) in place of LayerNorm for
    numerical/tabular sequences.

    The distinction matters for this project specifically: LayerNorm normalizes
    *within* a sample across channels, which removes that sample's overall
    magnitude. The manifold and kNN-density objectives here are computed against
    raw-input distances, and local density is carried by exactly that magnitude.
    BatchNorm normalizes per channel across samples instead, so relative
    per-sample magnitude survives -- the same property the baseline MLP encoder
    gets from its ``BatchNorm1d`` layers.

    Inference is unaffected by batch composition: BatchNorm uses running
    statistics in eval mode, exactly like the baseline encoder, so a sample's
    embedding never depends on which other samples share its inference batch.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, eps=eps, momentum=momentum)

    def forward(self, x):  # (B, L, C)
        B, L, C = x.shape
        return self.norm(x.reshape(B * L, C)).reshape(B, L, C)


def _build_token_norm(norm, dim):
    """Token-axis normalization layer selected by name."""
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    if norm == "batchnorm":
        return TokenBatchNorm(dim)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unsupported token norm: {norm}")


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention over the token axis."""

    def __init__(self, dim, num_heads, attn_dropout=0.0, force_mem_efficient=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.force_mem_efficient = force_mem_efficient

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):  # (B, L, C)
        B, L, C = x.shape
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))

        dropout_p = self.attn_dropout if self.training else 0.0
        with _mem_efficient_sdpa_ctx(self.force_mem_efficient and x.is_cuda):
            # No mask and no causal flag: feature tokens are a set, not a sequence.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block (LN -> attn -> residual -> LN -> FFN -> residual)."""

    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
        force_mem_efficient=True,
        norm="layernorm",
        norm_position="pre",
    ):
        super().__init__()
        if norm_position not in ("pre", "post"):
            raise ValueError(f"unsupported norm_position: {norm_position}")
        self.norm_position = norm_position
        hidden = int(dim * ffn_ratio)
        self.norm1 = _build_token_norm(norm, dim)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            attn_dropout=attn_dropout,
            force_mem_efficient=force_mem_efficient,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = _build_token_norm(norm, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        if self.norm_position == "pre":
            x = x + self.drop1(self.attn(self.norm1(x)))
            x = x + self.drop2(self.ffn(self.norm2(x)))
            return x
        # Post-norm (the original Vaswani ordering). E10-E12 found that what
        # helps density preservation is normalizing the representation that is
        # actually carried forward, not the one fed into a sub-block: on the
        # ResMLP side, BatchNorm applied after every Linear beat both LayerNorm
        # blocks and a pre-norm input BatchNorm. Pre-norm leaves the residual
        # stream itself unnormalized, which is exactly the quantity that gets
        # pooled and projected here. A4's null result for token BatchNorm was
        # measured in the pre-norm slot, so this is the untested cell that
        # evidence predicts should work.
        x = self.norm1(x + self.drop1(self.attn(x)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class AttentionPooling(nn.Module):
    """Learn a sample-dependent weighted mean over tokens.

    The query starts close to zero, so the initial softmax is close to uniform
    mean pooling. Training can then emphasize different feature/latent tokens
    for each sample without introducing row attention.
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.scale = dim**-0.5

    def forward(self, x):
        logits = torch.einsum("bld,d->bl", x, self.query.to(x.dtype)) * self.scale
        weights = logits.softmax(dim=1)
        return torch.einsum("bl,bld->bd", weights, x)


class ResidualMLPEncoder(nn.Module):
    """Non-attention control group: pre-LN residual MLP with GELU.

    Width 512 keeps the parameter count at 1.2-1.5x the current MLP encoder;
    width 256 would land at ~0.35x, below the 0.5-2x band required by the
    experiment protocol.
    """

    def __init__(
        self,
        num_input_dim,
        output_dim=40,
        width=512,
        num_blocks=3,
        dropout=0.0,
        norm="layernorm",
        activation="gelu",
        input_norm=None,
        norm_position="pre",
    ):
        super().__init__()
        if norm not in ("layernorm", "batchnorm"):
            raise ValueError(f"unsupported resmlp norm: {norm}")
        if norm_position not in ("pre", "post"):
            raise ValueError(f"unsupported resmlp norm_position: {norm_position}")
        if activation not in ("gelu", "leaky_relu"):
            raise ValueError(f"unsupported resmlp activation: {activation}")

        # (B, width) tensors, so BatchNorm needs no reshape here -- unlike the
        # token-axis case in TokenBatchNorm.
        def make_norm():
            if norm == "layernorm":
                return nn.LayerNorm(width)
            return nn.BatchNorm1d(width)

        def make_activation():
            if activation == "gelu":
                return nn.GELU()
            # 0.1 slope: identical to the baseline MLP encoder's LeakyReLU.
            return nn.LeakyReLU(0.1)

        # E10 found the active ingredient to be cross-sample normalization at
        # the input, not the token-axis norm inside the blocks. This switch
        # tests that claim on the non-attention side: LayerNorm blocks kept,
        # one BatchNorm1d(D) added in front of the stem. Default None keeps
        # every existing config exact.
        if input_norm is None or input_norm == "none":
            self.input_norm = nn.Identity()
        elif input_norm == "batchnorm":
            self.input_norm = nn.BatchNorm1d(num_input_dim)
        else:
            raise ValueError(f"unsupported input_norm: {input_norm}")

        self.stem = nn.Linear(num_input_dim, width)
        self.norm_position = norm_position
        if norm_position == "pre":
            self.blocks = nn.ModuleList(
                nn.Sequential(
                    make_norm(),
                    nn.Linear(width, width),
                    make_activation(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(width, width),
                )
                for _ in range(num_blocks)
            )
            self.post_norms = None
            self.norm = make_norm()
        else:
            # Post-norm: the norm moves after the residual add, so every
            # representation that is carried forward is normalized -- the
            # discipline the baseline MLP encoder follows with a BatchNorm after
            # every Linear. The trailing norm is dropped because the last
            # block's post-norm already normalizes the output, and the shared
            # `fc` head ends in BatchNorm regardless.
            self.blocks = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(width, width),
                    make_activation(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(width, width),
                )
                for _ in range(num_blocks)
            )
            self.post_norms = nn.ModuleList(make_norm() for _ in range(num_blocks))
            self.norm = nn.Identity()
        # Same output head as the baseline encoder, so output scale and BN
        # behaviour are identical across all encoders under comparison.
        self.fc = nn.Sequential(NN_FCBNRL_MM(width, output_dim, use_RL=False))

    def forward(self, input_x):
        x = self.stem(self.input_norm(input_x))
        if self.post_norms is None:
            for block in self.blocks:
                x = x + block(x)
        else:
            for block, norm in zip(self.blocks, self.post_norms):
                x = norm(x + block(x))
        x = self.norm(x)
        return self.fc(x)


class MLPResidualFusionEncoder(nn.Module):
    """MLP-preserving hybrid with a learnable Transformer correction.

    The gate uses the ReZero idea: at initialization the encoder is exactly the
    historical MLP, while the Transformer branch is allowed to contribute only
    when the unsupervised TopoBranch objective moves the scalar gate away from
    zero.  This is an engineering candidate, not evidence that a *pure*
    Transformer replaces the MLP; results must be reported as a hybrid.
    """

    def __init__(
        self,
        num_input_dim,
        output_dim=40,
        mlp_num_layers=2,
        mlp_hidden_size=512,
        residual_init=0.0,
        transformer=None,
        transformer_factory=None,
    ):
        super().__init__()
        if (transformer is None) == (transformer_factory is None):
            raise ValueError(
                "provide exactly one of transformer or transformer_factory"
            )
        # Construct the baseline first. With a factory this consumes the same
        # RNG sequence as a standalone same-seed Current MLP, making the
        # initialization comparison exact rather than merely output-equivalent.
        self.mlp = DMTEncoder(
            num_layers=mlp_num_layers,
            hidden_size=mlp_hidden_size,
            num_input_dim=num_input_dim,
            output_dim=output_dim,
        )
        self.transformer = (
            transformer_factory() if transformer_factory is not None else transformer
        )
        self.residual_logit = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, input_x):
        baseline = self.mlp(input_x)
        correction = self.transformer(input_x)
        # tanh keeps the correction bounded to one normalized branch while
        # retaining an exact MLP initialization when residual_init == 0.
        return baseline + torch.tanh(self.residual_logit) * correction


class TransformerReadout(nn.Module):
    """Maps a pooled token representation to the fixed-size latent.

    Transformer attention activations dominate memory for the large real batch
    used by TopoBranch.  Optional hidden layers therefore add capacity *after*
    pooling, where memory scales with ``B * hidden_dim`` instead of
    ``B * num_tokens * d_token``.  An empty hidden-dimension list reproduces
    the historical single Linear + BatchNorm output head exactly.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=None,
        hidden_norm="batchnorm",
        activation="gelu",
        dropout=0.0,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        if hidden_norm not in ("batchnorm", "layernorm", "none"):
            raise ValueError(f"unsupported readout hidden_norm: {hidden_norm}")
        if activation not in ("gelu", "leaky_relu"):
            raise ValueError(f"unsupported readout activation: {activation}")
        if dropout < 0 or dropout >= 1:
            raise ValueError("readout dropout must be in [0, 1)")

        if not hidden_dims:
            self.network = nn.Sequential(
                NN_FCBNRL_MM(input_dim, output_dim, use_RL=False)
            )
            return

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("readout hidden dimensions must be positive")
            layers.append(nn.Linear(in_dim, hidden_dim))
            if hidden_norm == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif hidden_norm == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.LeakyReLU(0.1))
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Keep the same normalized 40-D output contract as all existing
        # encoders; the shared 40 -> 500 -> 2 projection head is untouched.
        layers.append(NN_FCBNRL_MM(in_dim, output_dim, use_RL=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FTTransformerEncoder(nn.Module):
    """Feature tokenizer + transformer (Gorishniy et al., 2021).

    Each numerical feature i gets its own learnable parameters,
    ``z_i = x_i * W_i + b_i``. Feature identity therefore lives in independent
    per-feature parameters, not in positional embeddings, and no causal mask is
    applied. ``pooling="cls"`` reproduces the standard baseline; mean and
    attention pooling are exposed for controlled aggregation experiments.

    Memory scales as rows * (D + 1) * d_token, which is why this encoder is only
    run on the low/medium dimensional datasets.
    """

    def __init__(
        self,
        num_input_dim,
        output_dim=40,
        d_token=32,
        num_layers=2,
        num_heads=4,
        ffn_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
        pooling="cls",
        flatten_init="random",
        final_norm=True,
        block_norm="layernorm",
        block_norm_position="pre",
        input_norm=None,
        force_mem_efficient=True,
        tokenizer_basis="linear",
        readout_hidden_dims=None,
        readout_hidden_norm="batchnorm",
        readout_activation="gelu",
        readout_dropout=0.0,
    ):
        super().__init__()
        if pooling not in ("cls", "mean", "attention", "flatten"):
            raise ValueError(f"unsupported pooling: {pooling}")
        if tokenizer_basis not in ("linear", "linear_tanh_square"):
            raise ValueError(f"unsupported tokenizer_basis: {tokenizer_basis}")
        self.num_input_dim = num_input_dim
        self.d_token = d_token
        self.pooling = pooling
        self.tokenizer_basis = tokenizer_basis

        # E10-E12 found cross-sample normalization of the *input* to be the one
        # change that helps this objective (+0.05 on the latent encoder). FT's
        # last formal run predates that finding, so it has never been combined
        # with it. Default None keeps every historical FT config exact.
        if input_norm is None or input_norm == "none":
            self.input_norm = nn.Identity()
        elif input_norm == "batchnorm":
            self.input_norm = nn.BatchNorm1d(num_input_dim)
        else:
            raise ValueError(f"unsupported input_norm: {input_norm}")

        # Per-feature tokenizer: independent weight and bias per feature.
        self.token_weight = nn.Parameter(torch.empty(num_input_dim, d_token))
        self.token_bias = nn.Parameter(torch.empty(num_input_dim, d_token))
        bound = 1.0 / math.sqrt(d_token)
        nn.init.uniform_(self.token_weight, -bound, bound)
        nn.init.uniform_(self.token_bias, -bound, bound)
        if tokenizer_basis == "linear_tanh_square":
            # Start exactly as the standard FT tokenizer while allowing each
            # feature to learn bounded nonlinear value embeddings immediately.
            # Zero initialization is safe here: these additive weights receive
            # direct gradients on the first backward pass (unlike a gated
            # branch whose own gradients are initially blocked).
            self.token_tanh_weight = nn.Parameter(
                torch.zeros(num_input_dim, d_token)
            )
            self.token_square_weight = nn.Parameter(
                torch.zeros(num_input_dim, d_token)
            )
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
            nn.init.uniform_(self.cls_token, -bound, bound)
        elif pooling == "attention":
            self.attention_pool = AttentionPooling(d_token)
        elif pooling == "flatten":
            # Feature-specific learned aggregation. Unlike mean or a shared
            # attention query, every feature/output-channel pair has its own
            # coefficient. This adds capacity without increasing token
            # activations, which is critical at batch 4096 on an 11 GB GPU.
            self.flatten_pool = nn.Linear(num_input_dim * d_token, d_token)
            if flatten_init not in ("random", "mean"):
                raise ValueError(f"unsupported flatten_init: {flatten_init}")
            if flatten_init == "mean":
                # Start exactly equivalent to x.mean(dim=1), then let every
                # feature/channel coefficient specialize during training.
                with torch.no_grad():
                    self.flatten_pool.weight.zero_()
                    self.flatten_pool.bias.zero_()
                    channels = torch.arange(d_token)
                    for feature_idx in range(num_input_dim):
                        inputs = feature_idx * d_token + channels
                        self.flatten_pool.weight[channels, inputs] = (
                            1.0 / num_input_dim
                        )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=d_token,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                force_mem_efficient=force_mem_efficient,
                norm=block_norm,
                norm_position=block_norm_position,
            )
            for _ in range(num_layers)
        )
        self.norm = (
            _build_token_norm(block_norm, d_token) if final_norm else nn.Identity()
        )
        if readout_hidden_dims:
            self.fc = TransformerReadout(
                input_dim=d_token,
                output_dim=output_dim,
                hidden_dims=readout_hidden_dims,
                hidden_norm=readout_hidden_norm,
                activation=readout_activation,
                dropout=readout_dropout,
            )
        else:
            # Preserve both computation and state-dict keys for historical
            # Transformer checkpoints when the optional readout is disabled.
            self.fc = nn.Sequential(
                NN_FCBNRL_MM(d_token, output_dim, use_RL=False)
            )

    def forward(self, input_x):
        if input_x.shape[1] != self.num_input_dim:
            raise ValueError(
                f"expected {self.num_input_dim} features, got {input_x.shape[1]}"
            )
        # (B, D) -> (B, D, d_token)
        values = self.input_norm(input_x).unsqueeze(-1)
        tokens = values * self.token_weight + self.token_bias
        if self.tokenizer_basis == "linear_tanh_square":
            tokens = (
                tokens
                + values.tanh() * self.token_tanh_weight
                + values.square().div(1.0 + values.square())
                * self.token_square_weight
            )
        if self.pooling == "cls":
            cls = self.cls_token.expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            x = torch.cat([cls, tokens], dim=1)  # (B, 1 + D, d_token)
        else:
            x = tokens

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "attention":
            pooled = self.attention_pool(x)
        elif self.pooling == "flatten":
            pooled = self.flatten_pool(x.flatten(1))
        else:
            pooled = x.mean(dim=1)
        return self.fc(pooled)


class LatentTokenTransformerEncoder(nn.Module):
    """Scalable transformer-like encoder for high dimensional inputs.

    Tokenization follows the protocol literally:

    1. a learnable low-rank map compresses D features into M short vectors of
       width ``latent_rank`` (``latent_rank=1`` gives the "M scalars" variant),
    2. independent per-latent parameters expand each short vector to
       ``d_token``,
    3. each latent receives its own identity embedding.

    Attention then runs over M tokens only (M = 16 by default), so cost is
    independent of D apart from the compression matmul -- this is what makes
    HCL (3038-d) and MCA (9120-d) tractable.

    ``latent_rank=None`` switches to a single full map ``Linear(D, M*d_token)``.
    That variant is *not* a compression when M*d_token > D (true for NG20, ACT
    and MNIST), so it is not the default; it is kept because it is the only
    configuration whose parameter count lands inside the 0.5-2x band required
    by the benchmark protocol, and is therefore run as a capacity control.

    Per the protocol this encoder is explicitly *not* required to do
    per-feature attention.
    """

    def __init__(
        self,
        num_input_dim,
        output_dim=40,
        num_latents=16,
        d_token=64,
        num_layers=2,
        num_heads=4,
        ffn_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
        latent_rank=4,
        pooling="mean",
        final_norm=True,
        block_norm="layernorm",
        block_norm_position="pre",
        input_norm=None,
        force_mem_efficient=True,
        readout_hidden_dims=None,
        readout_hidden_norm="batchnorm",
        readout_activation="gelu",
        readout_dropout=0.0,
        input_residual=False,
    ):
        super().__init__()
        if pooling not in (
            "mean",
            "cls",
            "attention",
            "mean_std",
            "mean_std_residual",
            "weighted_mean",
        ):
            raise ValueError(f"unsupported pooling: {pooling}")
        self.num_input_dim = num_input_dim
        self.num_latents = num_latents
        self.d_token = d_token
        self.latent_rank = latent_rank
        self.pooling = pooling

        # Optional input-side normalization. The baseline MLP encoder's first
        # operation is Linear -> BatchNorm1d -> LeakyReLU, so raw feature-scale
        # imbalance never reaches its hidden layers; this encoder's first
        # operation is a bare Linear. Default None keeps existing configs exact.
        if input_norm is None or input_norm == "none":
            self.input_norm = nn.Identity()
        elif input_norm == "batchnorm":
            self.input_norm = nn.BatchNorm1d(num_input_dim)
        else:
            raise ValueError(f"unsupported input_norm: {input_norm}")

        if latent_rank is None:
            # Capacity-control variant: one full map straight to M * d_token.
            self.compress = nn.Linear(num_input_dim, num_latents * d_token)
            self.expand_weight = None
        else:
            # Low-rank compression D -> M * latent_rank, i.e. M short vectors.
            self.compress = nn.Linear(
                num_input_dim, num_latents * latent_rank, bias=False
            )
            # Independent expansion parameters per latent: (M, latent_rank, d).
            # No bias here -- the additive identity embedding below plays that
            # role, and two additive per-latent terms would be redundant.
            self.expand_weight = nn.Parameter(
                torch.empty(num_latents, latent_rank, d_token)
            )
            bound = 1.0 / math.sqrt(latent_rank)
            nn.init.uniform_(self.expand_weight, -bound, bound)

        self.latent_id = nn.Parameter(torch.zeros(1, num_latents, d_token))
        nn.init.trunc_normal_(self.latent_id, std=0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        elif pooling == "attention":
            self.attention_pool = AttentionPooling(d_token)
        elif pooling == "mean_std_residual":
            # Starts exactly as mean pooling. The model can add or subtract
            # channel-wise token dispersion only when training finds it useful.
            self.std_scale = nn.Parameter(torch.zeros(d_token))
        elif pooling == "weighted_mean":
            # Preserve the known-good mean-pooling initialization while
            # allowing each latent/channel pair to retain its own identity.
            # This is substantially cheaper than flatten -> Linear:
            # M*d parameters instead of M*d*d.
            self.pool_weight = nn.Parameter(
                torch.full((num_latents, d_token), 1.0 / num_latents)
            )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=d_token,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                force_mem_efficient=force_mem_efficient,
                norm=block_norm,
                norm_position=block_norm_position,
            )
            for _ in range(num_layers)
        )
        self.norm = (
            _build_token_norm(block_norm, d_token) if final_norm else nn.Identity()
        )
        pooled_dim = 2 * d_token if pooling == "mean_std" else d_token
        if readout_hidden_dims:
            self.fc = TransformerReadout(
                input_dim=pooled_dim,
                output_dim=output_dim,
                hidden_dims=readout_hidden_dims,
                hidden_norm=readout_hidden_norm,
                activation=readout_activation,
                dropout=readout_dropout,
            )
        else:
            self.fc = nn.Sequential(
                NN_FCBNRL_MM(pooled_dim, output_dim, use_RL=False)
            )
        self.input_residual = (
            nn.Linear(num_input_dim, output_dim, bias=False)
            if input_residual
            else None
        )
        if self.input_residual is not None:
            # The arm starts exactly equal to its Transformer control. Unlike
            # a zero scalar gate, the projection itself receives gradients on
            # the first backward pass.
            nn.init.zeros_(self.input_residual.weight)

    def forward(self, input_x):
        if input_x.shape[1] != self.num_input_dim:
            raise ValueError(
                f"expected {self.num_input_dim} features, got {input_x.shape[1]}"
            )
        B = input_x.shape[0]
        input_x = self.input_norm(input_x)
        if self.latent_rank is None:
            x = self.compress(input_x).view(B, self.num_latents, self.d_token)
        else:
            # (B, M, r) short vectors -> per-latent expansion -> (B, M, d)
            short = self.compress(input_x).view(B, self.num_latents, self.latent_rank)
            x = torch.einsum("bmr,mrd->bmd", short, self.expand_weight.to(short.dtype))
        x = x + self.latent_id.to(x.dtype)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1).to(x.dtype)
            x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        elif self.pooling == "attention":
            pooled = self.attention_pool(x)
        elif self.pooling == "mean_std":
            pooled = torch.cat(
                [x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1
            )
        elif self.pooling == "mean_std_residual":
            pooled = x.mean(dim=1) + self.std_scale.to(x.dtype) * x.std(
                dim=1, unbiased=False
            )
        elif self.pooling == "weighted_mean":
            pooled = (x * self.pool_weight.to(x.dtype)).sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        output = self.fc(pooled)
        if self.input_residual is not None:
            output = output + self.input_residual(input_x)
        return output
