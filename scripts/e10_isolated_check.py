"""E10 isolated check: backward compatibility, per-arm health, inference contract.

Run before the full-pipeline smoke test:

    python scripts/e10_isolated_check.py

Three things are verified.

1. **Backward compatibility.** The new ``norm`` / ``activation`` / ``block_norm``
   / ``input_norm`` switches must be inert at their defaults. Verified against
   the published parameter counts of the existing NG20 encoders, and by
   asserting the default module types are still LayerNorm / GELU / Identity.

2. **Per-arm health.** Every E10 arm builds, returns (rows, 40), produces finite
   output and finite non-zero gradients, and reports its capacity ratio against
   the baseline MLP (protocol band 0.5-2.0x).

3. **Inference contract.** For the BatchNorm arms, a sample's eval-mode
   embedding must not depend on which other samples share its batch. BatchNorm
   uses running statistics in eval mode, exactly like the baseline MLP encoder,
   but this is the property a reviewer will question, so it is measured rather
   than asserted in prose.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.encoder_factory import build_encoder, count_parameters  # noqa: E402
from model.encoder_transformer import TokenBatchNorm  # noqa: E402

NUM_INPUT_DIM = 100  # NG20
OUTPUT_DIM = 40
BASELINE_LEGACY = dict(T_num_layers=3, T_hidden_size=512, T_intermediate_size=300)

# Published NG20 counts that must not move when the defaults are unchanged.
# MLP and ResMLP: experiment_logs/ng20_encoder_report.md section 4.
# Latent M32/r16/d224: E03 run d9830su3.
PUBLISHED = {
    "mlp": (1_128_056, "mlp", {}),
    "resmlp": (1_652_344, "resmlp", {"width": 512, "num_blocks": 3, "dropout": 0.0}),
    "latent_m32_r16_d224": (
        1_392_632,
        "latent_transformer",
        {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "final_norm": True,
        },
    ),
}

LATENT_CONTROL = PUBLISHED["latent_m32_r16_d224"][2]


def latent(**overrides):
    kwargs = dict(LATENT_CONTROL)
    kwargs.update(overrides)
    return kwargs


ARMS = {
    "A1 resmlp-bn": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "norm": "batchnorm"},
    ),
    "A2 resmlp-lrelu": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "activation": "leaky_relu"},
    ),
    "A3 latent-nodrop": (
        "latent_transformer",
        latent(dropout=0.0, attn_dropout=0.0),
    ),
    "A4 latent-nodrop-bn": (
        "latent_transformer",
        latent(dropout=0.0, attn_dropout=0.0, block_norm="batchnorm"),
    ),
    "A5 latent-nodrop-bn-input": (
        "latent_transformer",
        latent(
            dropout=0.0,
            attn_dropout=0.0,
            block_norm="batchnorm",
            input_norm="batchnorm",
        ),
    ),
    # E11: isolate the input norm from the token-axis norm.
    "A6 latent-nodrop-inputbn": (
        "latent_transformer",
        latent(dropout=0.0, attn_dropout=0.0, input_norm="batchnorm"),
    ),
    "A7 resmlp-inputbn": (
        "resmlp",
        {"width": 512, "num_blocks": 3, "dropout": 0.0, "input_norm": "batchnorm"},
    ),
}

BN_ARMS = {
    "A1 resmlp-bn",
    "A4 latent-nodrop-bn",
    "A5 latent-nodrop-bn-input",
    "A6 latent-nodrop-inputbn",
    "A7 resmlp-inputbn",
}


def make(encoder_type, encoder_kwargs):
    return build_encoder(
        encoder_type=encoder_type,
        num_input_dim=NUM_INPUT_DIM,
        output_dim=OUTPUT_DIM,
        encoder_kwargs=dict(encoder_kwargs) or None,
        **BASELINE_LEGACY,
    )


def check_backward_compatibility():
    print("== 1. backward compatibility (defaults must be inert) ==")
    ok = True
    for label, (expected, encoder_type, kwargs) in PUBLISHED.items():
        actual = count_parameters(make(encoder_type, kwargs))
        status = "OK " if actual == expected else "FAIL"
        ok &= actual == expected
        print(f"  [{status}] {label:22s} params={actual:,} expected={expected:,}")

    resmlp = make("resmlp", PUBLISHED["resmlp"][2])
    resmlp_norm_ok = isinstance(resmlp.norm, nn.LayerNorm) and isinstance(
        resmlp.blocks[0][0], nn.LayerNorm
    )
    resmlp_act_ok = isinstance(resmlp.blocks[0][2], nn.GELU)
    latent = make("latent_transformer", LATENT_CONTROL)
    latent_norm_ok = isinstance(latent.blocks[0].norm1, nn.LayerNorm) and isinstance(
        latent.norm, nn.LayerNorm
    )
    latent_input_ok = isinstance(latent.input_norm, nn.Identity)
    for name, value in [
        ("resmlp default norm is LayerNorm", resmlp_norm_ok),
        ("resmlp default activation is GELU", resmlp_act_ok),
        ("latent default block/final norm is LayerNorm", latent_norm_ok),
        ("latent default input_norm is Identity", latent_input_ok),
    ]:
        ok &= value
        print(f"  [{'OK ' if value else 'FAIL'}] {name}")
    return ok


def check_arm(label, encoder_type, encoder_kwargs, device, rows=8192):
    model = make(encoder_type, encoder_kwargs).to(device)
    model.train()
    n_params = count_parameters(model)
    baseline = count_parameters(make("mlp", {}))
    ratio = n_params / baseline

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x = torch.randn(rows, NUM_INPUT_DIM, device=device)
    out = model(x)
    assert out.shape == (rows, OUTPUT_DIM), f"{label}: shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{label}: non-finite output"
    loss = out.pow(2).mean()
    assert torch.isfinite(loss), f"{label}: non-finite loss"
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{label}: no gradient for {missing[:5]}"
    nonfinite = [
        n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"{label}: non-finite gradient in {nonfinite[:5]}"
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
    assert grad_norm > 0, f"{label}: all-zero gradient"

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    in_band = 0.5 <= ratio <= 2.0
    print(
        f"  [{'OK ' if in_band else 'BAND'}] {label:26s} params={n_params:>9,} "
        f"({ratio:.3f}x) grad_norm={grad_norm:.3e} peak={peak_mb:7.1f} MB"
    )
    return model, in_band


def check_inference_contract(label, model, device, rel_tol=1e-5):
    """Eval-mode embedding of one sample must not depend on its batch mates.

    Compared against two *different* sets of batch mates, so the test measures
    dependence on the mates' statistics rather than on batch size alone. Exact
    bitwise equality is not the right bar: float32 reductions differ with tensor
    shape, so a residual around 1e-7 is kernel noise, not batch dependence. The
    train-mode number is printed alongside to show the check has power -- there
    BatchNorm does use batch statistics and the difference is orders larger.
    """
    probe = torch.randn(1, NUM_INPUT_DIM, device=device)
    mates_a = torch.randn(255, NUM_INPUT_DIM, device=device)
    mates_b = 5.0 * torch.randn(255, NUM_INPUT_DIM, device=device) + 3.0

    def probe_delta():
        with torch.no_grad():
            out_a = model(torch.cat([probe, mates_a]))[:1]
            out_b = model(torch.cat([probe, mates_b]))[:1]
            scale = out_a.abs().max().clamp_min(1e-12)
            return ((out_a - out_b).abs().max() / scale).item()

    model.eval()
    eval_delta = probe_delta()
    model.train()
    train_delta = probe_delta()
    model.eval()

    ok = eval_delta < rel_tol
    print(
        f"  [{'OK ' if ok else 'FAIL'}] {label:26s} eval rel-delta={eval_delta:.2e} "
        f"(tol {rel_tol:.0e})  train rel-delta={train_delta:.2e}"
    )
    return ok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch={torch.__version__} device={device}")
    if device == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    ok = check_backward_compatibility()

    print("== 2. per-arm forward/backward at 8192 rows (batch 4096, two views) ==")
    models = {}
    for label, (encoder_type, kwargs) in ARMS.items():
        model, in_band = check_arm(label, encoder_type, kwargs, device)
        models[label] = model
        ok &= in_band

    print("== 3. inference contract for the BatchNorm arms (eval mode) ==")
    for label in BN_ARMS:
        ok &= check_inference_contract(label, models[label], device)

    print("== 4. token BatchNorm shape handling ==")
    tbn = TokenBatchNorm(16).to(device)
    y = tbn(torch.randn(4, 7, 16, device=device))
    shape_ok = y.shape == (4, 7, 16)
    ok &= shape_ok
    print(f"  [{'OK ' if shape_ok else 'FAIL'}] (4,7,16) -> {tuple(y.shape)}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
