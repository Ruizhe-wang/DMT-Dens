"""E15 isolated check: post-norm arms on both branches at HCL dimension.

    python scripts/e15_isolated_check.py

Verifies, in order:

1. the new ``norm_position`` switches are inert at their defaults, against the
   published HCL parameter counts from E13;
2. every E15 arm builds, returns (rows, 40), and produces finite non-zero
   gradients at the real row count (2 x batch 4096);
3. post-norm really moved the normalization: a pre-norm block's output is not
   normalized while a post-norm block's is, measured rather than assumed;
4. the inference contract still holds for the BatchNorm arms.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.encoder_factory import build_encoder, count_parameters  # noqa: E402
from model.encoder_transformer import TransformerBlock  # noqa: E402

D = 3038  # HCL
OUTPUT_DIM = 40
LEGACY = dict(T_num_layers=3, T_hidden_size=512, T_intermediate_size=300)

LATENT_BN = {
    "num_latents": 32, "d_token": 224, "num_layers": 2, "num_heads": 4,
    "ffn_ratio": 4.0, "dropout": 0.0, "attn_dropout": 0.0, "latent_rank": 16,
    "pooling": "mean", "final_norm": True, "input_norm": "batchnorm",
}
RESMLP_BN = {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"}


def latent(**kw):
    out = dict(LATENT_BN)
    out.update(kw)
    return out


def resmlp(**kw):
    out = dict(RESMLP_BN)
    out.update(kw)
    return out


# E13 HCL counts that must not move when the new defaults are unchanged.
PUBLISHED = {
    "mlp": (2_632_312, "mlp", {}),
    "resmlp-bn": (3_156_600, "resmlp", RESMLP_BN),
    "latent-bn": (2_902_964, "latent_transformer", LATENT_BN),
}

ARMS = {
    "T1 latent-postnorm-ln": ("latent_transformer", latent(block_norm_position="post")),
    "T2 latent-postnorm-bn": (
        "latent_transformer",
        latent(block_norm_position="post", block_norm="batchnorm"),
    ),
    "T3 latent-rank32": ("latent_transformer", latent(latent_rank=32)),
    "C1 resmlp-postnorm-bn": ("resmlp", resmlp(norm_position="post")),
}
BN_ARMS = ["T2 latent-postnorm-bn", "C1 resmlp-postnorm-bn"]


def make(enc_type, kwargs):
    return build_encoder(
        encoder_type=enc_type, num_input_dim=D, output_dim=OUTPUT_DIM,
        encoder_kwargs=dict(kwargs) or None, **LEGACY,
    )


def check_defaults_inert():
    print("== 1. defaults inert (HCL counts from E13) ==")
    ok = True
    for label, (expected, enc_type, kwargs) in PUBLISHED.items():
        actual = count_parameters(make(enc_type, kwargs))
        good = actual == expected
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {label:12s} {actual:,} expected {expected:,}")
    rm = make("resmlp", RESMLP_BN)
    lat = make("latent_transformer", LATENT_BN)
    checks = [
        ("resmlp default is pre-norm", rm.post_norms is None),
        ("resmlp default keeps trailing norm", not isinstance(rm.norm, nn.Identity)),
        ("latent default block is pre-norm", lat.blocks[0].norm_position == "pre"),
    ]
    for name, value in checks:
        ok &= value
        print(f"  [{'OK ' if value else 'FAIL'}] {name}")
    return ok


def check_arms(device, rows):
    print(f"== 2. arms at {rows} rows (batch 4096, two views) ==")
    baseline = count_parameters(make("mlp", {}))
    models, ok = {}, True
    for label, (enc_type, kwargs) in ARMS.items():
        model = make(enc_type, kwargs).to(device)
        model.train()
        n = count_parameters(model)
        ratio = n / baseline
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        out = model(torch.randn(rows, D, device=device))
        assert out.shape == (rows, OUTPUT_DIM), f"{label}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{label}: non-finite output"
        loss = out.pow(2).mean()
        loss.backward()
        missing = [k for k, p in model.named_parameters() if p.grad is None]
        assert not missing, f"{label}: no grad for {missing[:4]}"
        bad = [k for k, p in model.named_parameters() if not torch.isfinite(p.grad).all()]
        assert not bad, f"{label}: non-finite grad in {bad[:4]}"
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        assert gn > 0, f"{label}: zero gradient"
        peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
        in_band = 0.5 <= ratio <= 2.0
        ok &= in_band
        models[label] = model
        print(
            f"  [{'OK ' if in_band else 'BAND'}] {label:24s} {n:>9,} ({ratio:.3f}x) "
            f"grad={gn:.2e} peak={peak:7.1f} MB"
        )
    return models, ok


def check_position_took_effect(device):
    """Measure that post-norm normalizes the block output and pre-norm does not."""
    print("== 3. norm position actually moved ==")
    torch.manual_seed(0)
    x = 7.0 * torch.randn(64, 32, 224, device=device) + 4.0
    ok = True
    for position, expect_normalized in (("pre", False), ("post", True)):
        block = TransformerBlock(
            dim=224, num_heads=4, dropout=0.0, attn_dropout=0.0,
            norm="layernorm", norm_position=position,
        ).to(device)
        block.eval()
        with torch.no_grad():
            y = block(x)
        std = y.std(dim=-1).mean().item()
        mean = y.mean(dim=-1).abs().mean().item()
        normalized = abs(std - 1.0) < 0.1 and mean < 0.1
        good = normalized == expect_normalized
        ok &= good
        print(
            f"  [{'OK ' if good else 'FAIL'}] {position:4s} output |mean|={mean:6.3f} "
            f"std={std:6.3f} -> normalized={normalized} (expected {expect_normalized})"
        )
    return ok


def check_inference_contract(models, device, rel_tol=1e-5):
    print("== 4. inference contract for the BatchNorm arms ==")
    ok = True
    probe = torch.randn(1, D, device=device)
    a = torch.randn(255, D, device=device)
    b = 5.0 * torch.randn(255, D, device=device) + 3.0
    for label in BN_ARMS:
        model = models[label]
        model.eval()
        with torch.no_grad():
            oa = model(torch.cat([probe, a]))[:1]
            ob = model(torch.cat([probe, b]))[:1]
            delta = ((oa - ob).abs().max() / oa.abs().max().clamp_min(1e-12)).item()
        good = delta < rel_tol
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {label:24s} eval rel-delta={delta:.2e}")
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = 8192 if device == "cuda" else 256
    print(f"[env] torch={torch.__version__} device={device} D={D}")
    ok = check_defaults_inert()
    models, arms_ok = check_arms(device, rows)
    ok &= arms_ok
    ok &= check_position_took_effect(device)
    ok &= check_inference_contract(models, device)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
