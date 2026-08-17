"""Encoder factory for the encoder-ablation benchmark.

Every encoder returned by :func:`build_encoder` obeys the same contract:

    (B, num_input_dim) -> (B, output_dim)

``encoder_type="mlp"`` reproduces the current ``DMTEncoder`` exactly (same
class, same arguments), so existing configs and results are unaffected.

Architecture hyper-parameters belong in the YAML under ``encoder_kwargs`` --
never hard-coded in a training script.
"""

from model.encoder import DMTEncoder
from model.encoder_transformer import (
    FTTransformerEncoder,
    LatentTokenTransformerEncoder,
    MLPResidualFusionEncoder,
    ResidualMLPEncoder,
)

ENCODER_TYPES = (
    "mlp",
    "resmlp",
    "ft_transformer",
    "latent_transformer",
    "ft_transformer_mlp_residual",
    "latent_transformer_mlp_residual",
)


def build_encoder(
    encoder_type="mlp",
    num_input_dim=784,
    output_dim=40,
    encoder_kwargs=None,
    # Legacy T_* arguments, consumed by the baseline MLP encoder only.
    T_num_layers=2,
    T_num_attention_heads=6,
    T_hidden_size=240,
    T_intermediate_size=300,
    T_hidden_dropout_prob=0.1,
    T_attention_probs_dropout_prob=0.1,
    num_use_moe=1,
):
    """Builds the encoder selected by ``encoder_type``.

    Args:
        encoder_type (str): One of :data:`ENCODER_TYPES`.
        num_input_dim (int): Input feature dimension D.
        output_dim (int): Latent dimension (40 in the current experiments).
        encoder_kwargs (dict): Architecture overrides for the non-baseline
            encoders. Ignored by ``mlp``.

    Returns:
        nn.Module: encoder mapping (B, D) -> (B, output_dim).
    """
    encoder_kwargs = dict(encoder_kwargs or {})

    if encoder_type == "mlp":
        if encoder_kwargs:
            raise ValueError(
                "encoder_type='mlp' reproduces the baseline exactly and takes no "
                f"encoder_kwargs; got {sorted(encoder_kwargs)}"
            )
        return DMTEncoder(
            num_layers=T_num_layers,
            num_attention_heads=T_num_attention_heads,
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
            output_dim=output_dim,
        )

    if encoder_type == "resmlp":
        return ResidualMLPEncoder(
            num_input_dim=num_input_dim, output_dim=output_dim, **encoder_kwargs
        )

    if encoder_type == "ft_transformer":
        return FTTransformerEncoder(
            num_input_dim=num_input_dim, output_dim=output_dim, **encoder_kwargs
        )

    if encoder_type == "latent_transformer":
        return LatentTokenTransformerEncoder(
            num_input_dim=num_input_dim, output_dim=output_dim, **encoder_kwargs
        )

    if encoder_type in (
        "ft_transformer_mlp_residual",
        "latent_transformer_mlp_residual",
    ):
        residual_init = encoder_kwargs.pop("residual_init", 0.0)
        if encoder_type == "ft_transformer_mlp_residual":
            transformer_factory = lambda: FTTransformerEncoder(
                num_input_dim=num_input_dim, output_dim=output_dim, **encoder_kwargs
            )
        else:
            transformer_factory = lambda: LatentTokenTransformerEncoder(
                num_input_dim=num_input_dim,
                output_dim=output_dim,
                **encoder_kwargs,
            )
        return MLPResidualFusionEncoder(
            num_input_dim=num_input_dim,
            output_dim=output_dim,
            mlp_num_layers=T_num_layers,
            mlp_hidden_size=T_hidden_size,
            residual_init=residual_init,
            transformer_factory=transformer_factory,
        )

    raise ValueError(
        f"unknown encoder_type {encoder_type!r}; expected one of {ENCODER_TYPES}"
    )


def count_parameters(module, trainable_only=True):
    """Counts parameters, for the engineering-cost column of the result table."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


# The benchmark protocol asks every encoder to stay within 0.5-2x of the
# current MLP encoder, and requires any deviation to be stated in the results
# rather than silently absorbed.
PARAM_BAND = (0.5, 2.0)


def baseline_param_count(num_input_dim, output_dim=40, **legacy):
    """Parameter count of the baseline MLP encoder for the same input size.

    Built and discarded on the fly rather than hard-coded, so the reference
    can never drift away from the actual baseline encoder.
    """
    reference = build_encoder(
        encoder_type="mlp",
        num_input_dim=num_input_dim,
        output_dim=output_dim,
        **legacy,
    )
    n_params = count_parameters(reference)
    del reference
    return n_params


def describe_capacity(encoder, num_input_dim, output_dim=40, **legacy):
    """Reports encoder size against the baseline MLP.

    Returns a dict with ``params``, ``baseline_params``, ``param_ratio`` and
    ``param_in_band``, ready to be logged with the run so that a capacity
    mismatch is visible in every record instead of only in the final write-up.
    """
    n_params = count_parameters(encoder)
    n_baseline = baseline_param_count(num_input_dim, output_dim, **legacy)
    ratio = n_params / max(n_baseline, 1)
    return {
        "params": n_params,
        "baseline_params": n_baseline,
        "param_ratio": ratio,
        "param_in_band": bool(PARAM_BAND[0] <= ratio <= PARAM_BAND[1]),
    }
