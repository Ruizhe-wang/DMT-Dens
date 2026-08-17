"""Encoder-level smoke test: 2 forward/backward passes per encoder.

Checks, per the benchmark protocol:
  * output shape == (rows, output_dim)
  * no NaN/Inf in the output or in the loss
  * every trainable parameter receives a finite, non-zero-everywhere gradient
  * peak VRAM and per-step wall time
  * parameter count vs. the current MLP baseline (0.5-2x band)

Row count matters: the model concatenates the item view and the augmented view
before the encoder (DiffTreeVQ_density.forward_train_enc), so the encoder sees
2 * batch_size rows.

Usage:
    python scripts/smoke_encoder.py --dataset mnist
    python scripts/smoke_encoder.py --dataset mnist --encoder ft_transformer --batch-size 2048
    python scripts/smoke_encoder.py --all

This validates the encoder in isolation. The full-pipeline 2-batch check is:
    python main.py fit -c configs/encoder_bench/<cfg>.yaml --trainer.fast_dev_run 2
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.encoder_factory import build_encoder, count_parameters  # noqa: E402

# Mirrors configs/dmtme_dataset_weighted/*.yaml (the TopoBranch mainline).
DATASETS = {
    "ng20": {"num_input_dim": 100, "batch_size": 4096},
    "act": {"num_input_dim": 561, "batch_size": 5000},
    "mnist": {"num_input_dim": 784, "batch_size": 4096},
    "hcl": {"num_input_dim": 3038, "batch_size": 4096},
}

# Mirrors the encoder_kwargs blocks of configs/encoder_bench/*.yaml.
# label -> (encoder_type, encoder_kwargs)
ENCODER_SPECS = {
    "mlp": ("mlp", {}),
    "resmlp": ("resmlp", {"width": 512, "num_blocks": 3}),
    "ft_transformer": (
        "ft_transformer",
        {"d_token": 32, "num_layers": 2, "num_heads": 4},
    ),
    # Protocol default: low-rank compression to M short vectors.
    "latent_transformer": (
        "latent_transformer",
        {
            "num_latents": 16,
            "d_token": 64,
            "num_layers": 2,
            "num_heads": 4,
            "latent_rank": 4,
        },
    ),
    # Capacity control: full map, the only variant inside the 0.5-2x band.
    "latent_transformer_full": (
        "latent_transformer",
        {
            "num_latents": 16,
            "d_token": 64,
            "num_layers": 2,
            "num_heads": 4,
            "latent_rank": None,
        },
    ),
}

OUTPUT_DIM = 40


def report_sdpa_backends():
    if not torch.cuda.is_available():
        print("[env] cuda unavailable, running on cpu")
        return
    cap = torch.cuda.get_device_capability()
    print(
        f"[env] torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"sm={cap[0]}{cap[1]}"
    )
    if cap[0] < 8:
        print(
            "[env] pre-Ampere GPU: FlashAttention is unavailable, SDPA is pinned "
            "to the memory-efficient backend (math fallback would OOM)"
        )


def smoke_one(dataset, label, batch_size=None, device="cuda", steps=2):
    spec = DATASETS[dataset]
    num_input_dim = spec["num_input_dim"]
    batch_size = batch_size or spec["batch_size"]
    rows = 2 * batch_size  # item view + augmented view

    encoder_type, encoder_kwargs = ENCODER_SPECS[label]
    encoder_kwargs = dict(encoder_kwargs)
    model = build_encoder(
        encoder_type=encoder_type,
        num_input_dim=num_input_dim,
        output_dim=OUTPUT_DIM,
        encoder_kwargs=encoder_kwargs or None,
        # Baseline T_* values from the mainline configs.
        T_num_layers=3,
        T_hidden_size=512,
        T_intermediate_size=300,
    ).to(device)
    model.train()

    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step_times = []
    # steps + 1 iterations: the first is a warmup (kernel autotune, allocator
    # growth) and is excluded from the reported time.
    for step in range(steps + 1):
        x = torch.randn(rows, num_input_dim, device=device)
        opt.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = out.pow(2).mean()

        if out.shape != (rows, OUTPUT_DIM):
            raise AssertionError(f"bad output shape {tuple(out.shape)} != {(rows, OUTPUT_DIM)}")
        if not torch.isfinite(out).all():
            raise AssertionError(f"non-finite output at step {step}")
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss at step {step}")

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise AssertionError(f"parameters without gradient: {missing[:5]}")
        nonfinite = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        if nonfinite:
            raise AssertionError(f"non-finite gradient in: {nonfinite[:5]}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        if grad_norm == 0.0:
            raise AssertionError(f"all-zero gradient at step {step}")

        scaler.step(opt)
        scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
        if step > 0:
            step_times.append(time.perf_counter() - t0)

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return {
        "dataset": dataset,
        "encoder": label,
        "D": num_input_dim,
        "batch": batch_size,
        "rows": rows,
        "params": n_params,
        "peak_mb": peak_mb,
        "step_ms": 1000 * sum(step_times) / len(step_times),
        "grad_norm": grad_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ng20")
    parser.add_argument("--encoder", choices=sorted(ENCODER_SPECS), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="every dataset x encoder pair")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    report_sdpa_backends()

    if args.all:
        pairs = [(d, e) for d in DATASETS for e in ENCODER_SPECS]
    elif args.encoder:
        pairs = [(args.dataset, args.encoder)]
    else:
        pairs = [(args.dataset, e) for e in ENCODER_SPECS]

    header = f"{'dataset':<8}{'encoder':<20}{'batch':>7}{'rows':>8}{'params':>12}{'peak MB':>10}{'ms/step':>10}  status"
    print("\n" + header)
    print("-" * len(header))

    baseline = {}
    failures = 0
    for dataset, encoder_type in pairs:
        try:
            r = smoke_one(
                dataset, encoder_type, args.batch_size, args.device, args.steps
            )
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never crashes early
            failures += 1
            short = type(exc).__name__
            print(
                f"{dataset:<8}{encoder_type:<20}{'-':>7}{'-':>8}{'-':>12}{'-':>10}{'-':>10}  FAIL {short}: {exc}"
            )
            if args.device == "cuda":
                torch.cuda.empty_cache()
            continue

        if encoder_type == "mlp":
            baseline[dataset] = r["params"]
        ratio = ""
        if dataset in baseline and encoder_type != "mlp":
            factor = r["params"] / baseline[dataset]
            flag = "" if 0.5 <= factor <= 2.0 else "  OUT-OF-BAND"
            ratio = f"  {factor:.2f}x baseline{flag}"

        print(
            f"{r['dataset']:<8}{r['encoder']:<20}{r['batch']:>7}{r['rows']:>8}"
            f"{r['params']:>12,}{r['peak_mb']:>10.0f}{r['step_ms']:>10.1f}  ok{ratio}"
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
