"""Fast invariants for the E14 latent-Transformer arms."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.encoder_transformer import LatentTokenTransformerEncoder


BASE = {
    "num_input_dim": 100,
    "output_dim": 40,
    "num_latents": 32,
    "d_token": 224,
    "num_layers": 2,
    "num_heads": 4,
    "ffn_ratio": 4.0,
    "dropout": 0.0,
    "attn_dropout": 0.0,
    "latent_rank": 16,
    "pooling": "mean",
    "final_norm": True,
    "block_norm": "layernorm",
    "input_norm": "batchnorm",
}


def build(**overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return LatentTokenTransformerEncoder(**kwargs)


def main():
    torch.manual_seed(42)
    control = build().eval()
    weighted = build(pooling="weighted_mean").eval()
    weighted.load_state_dict(control.state_dict(), strict=False)
    residual = build(input_residual=True).eval()
    residual.load_state_dict(control.state_dict(), strict=False)
    x = torch.randn(64, 100)

    with torch.no_grad():
        reference = control(x)
        torch.testing.assert_close(reference, weighted(x), rtol=1e-5, atol=1e-6)
        assert torch.equal(reference, residual(x))

    for name, model in {
        "m16": build(num_latents=16),
        "weighted_mean": weighted,
        "input_residual": residual,
    }.items():
        model.train()
        output = model(x)
        assert output.shape == (64, 40)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        print(
            name,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            "PASS",
        )

    assert residual.input_residual.weight.grad is not None
    assert residual.input_residual.weight.grad.abs().sum() > 0
    assert weighted.pool_weight.grad is not None
    assert weighted.pool_weight.grad.abs().sum() > 0


if __name__ == "__main__":
    main()
