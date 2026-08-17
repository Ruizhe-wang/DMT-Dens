"""Generate standalone TopoBranch encoder-benchmark YAML configs.

The generated files intentionally inherit each dataset's current recommended
TopoBranch config and change only:

* encoder selection and encoder architecture arguments;
* the provisional learning rate for new encoders;
* run names and output paths, so benchmark runs never overwrite one another;
* logging-only callback settings required by the benchmark protocol.

Run from the repository root:

    python configs/encoder_bench/generate_configs.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "configs" / "encoder_bench"

PROJECT_PREFIX = "TopoBranch_encoder_bench"


def project_name(dataset: str) -> str:
    """One wandb project per dataset, so runs never mix across datasets."""
    return f"{PROJECT_PREFIX}_{dataset.upper()}"


# ``max_epochs`` overrides the dataset baseline where set. It applies to every
# encoder of that dataset, so the comparison stays fair; it does mean the
# absolute numbers are no longer directly comparable to historical
# dmtme_dataset_weighted runs at the baseline epoch count.
DATASETS = {
    "ng20": {
        "source": ROOT / "configs" / "dmtme_dataset_weighted" / "NG20.yaml",
        "max_epochs": 1000,  # baseline is 500
    },
    "act": {
        "source": ROOT / "configs" / "dmtme_dataset_weighted" / "ACT.yaml",
    },
    "mnist": {
        "source": ROOT / "configs" / "dmtme_dataset_weighted" / "mnist.yaml",
    },
    "hcl": {
        "source": ROOT / "configs" / "dmtme_dataset_weighted" / "HCL.yaml",
    },
    "mca": {
        "source": ROOT / "configs" / "dmtme_dataset_weighted" / "MCA.yaml",
    },
}

# Fairness rule 1 requires an identical learning-rate search range for every
# model, the baseline included. The search runs on NG20 only (sweep/encoder_bench_ng20_lr_seed42.yaml),
# where every architecture gets the same three candidates; the winner is then
# fixed for the other datasets.
LR_SEARCH_GRID = (5e-3, 1e-3, 3e-4)

# NG20 configs are launched through the sweep, which overrides lr. The value
# written into the file is the dataset's mainline lr, identical for every
# encoder so a standalone NG20 launch is still a fair comparison.
NG20_LR = 5e-3

# Selected from the NG20 search (sweep v92rl72v, seed 42, restored baseline).
#
# Criterion, fixed before looking at the numbers: the mean of the three metrics
# the protocol calls core -- density_correlation, local_density_correlation and
# svc_acc. Trustworthiness, kNN preservation and the distance correlations are
# secondary there, so they do not decide the learning rate. Picking by the
# secondary composite instead would move resmlp to 3e-4 and ft_transformer to
# 3e-4; both alternatives are recorded here rather than quietly dropped.
#
# The MLP is nearly flat over the whole range (core3 0.6125 / 0.6163 / 0.6165
# for 5e-3 / 1e-3 / 3e-4), so its entry is a near-tie rather than a real
# preference; note that 5e-3 is the historical mainline value.
#
# latent_transformer_full was not part of the four-encoder sweep; it inherits
# the latent_transformer choice as the closest architecture.
SELECTED_LR: dict[str, float] = {
    "mlp": 3e-4,
    "resmlp": 1e-3,
    "ft_transformer": 5e-3,
    "latent_transformer": 5e-3,
    "latent_transformer_full": 5e-3,
    # Capacity controls inherit the FT-Transformer rate so that width is
    # the only thing that changes between them and ft_transformer.
    "ft_transformer_d64": 5e-3,
    "ft_transformer_d96": 5e-3,
}

# Per-config batch overrides, used only where the baseline batch does not fit in
# 11 GB. Protocol rule 4: keep the real batch size wherever possible, reduce it
# step by step only after a genuine OOM, and record the value that was used.
# ACT feature tokenization needs 562 tokens; measured with
# scripts/smoke_encoder.py, batch 5000 and 2500 both OOM and 2048 peaks at
# 7.8 GB. Every other ACT encoder keeps the baseline 5000, so the FT-Transformer
# row of the ACT table is not batch-matched with the rest and must say so.
BATCH_OVERRIDE: dict[tuple[str, str], int] = {
    ("act", "ft_transformer"): 2048,
}

LR_SENTINEL = "SET_FROM_NG20_LR_SEARCH"

ENCODERS = {
    "mlp": {
        "encoder_type": "mlp",
        "encoder_kwargs": None,
    },
    "resmlp": {
        "encoder_type": "resmlp",
        "encoder_kwargs": {
            "width": 512,
            "num_blocks": 3,
            "dropout": 0.0,
        },
    },
    "ft_transformer": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 32,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "force_mem_efficient": True,
        },
    },
    # Capacity controls for the FT-Transformer, NG20 only.
    #
    # The protocol fixes d_token=32, which lands at 0.03x the baseline parameter
    # count, so a poor result cannot be separated from simple under-capacity.
    # Widening is the only way to add capacity here: about 76% of the parameters
    # sit in the blocks, which scale as 24*d^2. On an 11 GB card at the aligned
    # batch of 4096, d_token=96 (0.22x) is the widest that fits; 128 and above
    # OOM, so genuine parameter matching at 0.5-2x is not reachable without
    # shrinking the batch for every encoder.
    #
    # d_token=64 is the informative one: it lands at ~114k parameters, which is
    # within a percent of latent_transformer's 114,296. Same width, same depth,
    # same heads, same parameter count, same training setup - the only
    # difference left is per-feature tokenization versus low-rank latent
    # tokenization. If this one still collapses while latent_transformer does
    # not, capacity is ruled out as the explanation.
    "ft_transformer_d64": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "ft_transformer_d96": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 4,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
    },
    "latent_transformer_full": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": None,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
    },
    # --- Capacity controls -------------------------------------------------
    # At the protocol widths both transformers sit far below the 0.5-2x band
    # (ft_transformer 0.03x, latent_transformer 0.10x), so a poor result cannot
    # be separated from simple under-parameterisation. These two widen the token
    # dimension and change nothing else.
    #
    # Measured on NG20 at the aligned batch of 4096 (8192 rows through the
    # encoder), one process per configuration:
    #
    #   latent d=160 0.57x 2.2 GB | d=192 0.82x 2.6 GB | d=224 1.10x 3.1 GB
    #   ft     d=64  0.10x 5.5 GB | d=96  0.22x 8.2 GB | d=128 does not fit
    #
    # latent_transformer_wide takes d=224, the middle of the band. ft
    # transformer_wide takes d=96, the widest that fits without reducing the
    # batch -- 7.4x the parameters of the protocol width but still 0.22x the
    # baseline, which the results table has to state. Reaching parity for
    # feature tokenization would need a smaller batch, and that would have to
    # apply to every encoder to stay like-for-like.
    #
    # d_token must be divisible by 8 * num_heads: the memory-efficient SDPA
    # kernel requires a head dimension that is a multiple of 8, and with flash
    # and math disabled a bad width fails hard with "No available kernel"
    # (d_token=112 with 4 heads gives head_dim 28 and does exactly that).
    "ft_transformer_wide": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "force_mem_efficient": True,
        },
    },
    "latent_transformer_wide": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 4,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
    },
    # E02 architecture controls. E01 showed that width alone does not fix FT
    # collapse and does not help latent-token density preservation when the
    # rank-4 tokenizer is unchanged.
    "ft_transformer_wide_mean": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r8": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 8,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r16": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    # E03 controls: E02 identified r16/mean as the strongest latent model and
    # showed that FT aggregation matters more than width. Change one structural
    # axis at a time from the corresponding E02 winner.
    "ft_transformer_wide_attention": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "pooling": "attention",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r16_attention": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "attention",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_m32_r16": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_r16_l3": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 3,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    # E04 normalization controls. Final token-wise LayerNorm discards absolute
    # token scale immediately before pooling; these variants preserve it while
    # leaving every block and all training components unchanged.
    "ft_transformer_wide_mean_no_final_norm": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "pooling": "mean",
            "final_norm": False,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r16_no_final_norm": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "final_norm": False,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_m32_r16_no_final_norm": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean",
            "final_norm": False,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    # E05 pooling-capacity controls. Flatten pooling gives FT feature-specific
    # output coefficients and reaches parameter parity without increasing
    # token activations. Mean+std pooling exposes token dispersion to the
    # latent output head as a low-cost density cue.
    "ft_transformer_wide_flatten": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "pooling": "flatten",
            "final_norm": True,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r16_mean_std": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean_std",
            "final_norm": True,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_m32_r16_mean_std": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean_std",
            "final_norm": True,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    # E06 conservative adaptive pooling. Each model starts exactly at its best
    # known E04/E03 pooling function and learns only a residual specialization.
    "ft_transformer_wide_flatten_mean_init": {
        "encoder_type": "ft_transformer",
        "encoder_kwargs": {
            "d_token": 96,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "pooling": "flatten",
            "flatten_init": "mean",
            "final_norm": False,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_wide_r16_mean_std_residual": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 16,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean_std_residual",
            "final_norm": True,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
    "latent_transformer_m32_r16_mean_std_residual": {
        "encoder_type": "latent_transformer",
        "encoder_kwargs": {
            "num_latents": 32,
            "d_token": 224,
            "num_layers": 2,
            "num_heads": 4,
            "ffn_ratio": 4.0,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "latent_rank": 16,
            "pooling": "mean_std_residual",
            "final_norm": True,
            "force_mem_efficient": True,
        },
        "only_datasets": ("ng20",),
    },
}


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return loaded


def _callback(config: dict, class_path: str) -> dict:
    callbacks = config["trainer"]["callbacks"]
    for item in callbacks:
        if item.get("class_path") == class_path:
            return item
    raise KeyError(f"callback {class_path!r} not found in source config")


def _resolve_lr(dataset: str, label: str):
    """Learning rate for one config, or a sentinel that refuses to launch.

    Once an encoder has a selected value it applies everywhere, NG20 included,
    so a standalone launch uses the searched rate. The NG20 search sweep
    overrides it anyway while the search is being run or repeated.
    """
    if label in SELECTED_LR:
        return SELECTED_LR[label]
    if dataset == "ng20":
        return NG20_LR
    return LR_SENTINEL


def _plot_style(config: dict) -> dict:
    """Marker style scaled to dataset size.

    Fairness requires the style to be identical across encoders of the same
    dataset, not across datasets. Fixed s=6 / alpha=0.85 overplots a 60k-point
    embedding into a solid blob and hides exactly the density structure being
    evaluated.
    """
    n_samples = int(config["model"]["init_args"].get("num_train_data", 0) or 0)
    if n_samples > 50_000:
        return {"point_size": 1.5, "alpha": 0.5}
    if n_samples > 20_000:
        return {"point_size": 3.0, "alpha": 0.7}
    return {"point_size": 6.0, "alpha": 0.85}


def _configure_run(
    config: dict, dataset: str, label: str, spec: dict, dataset_spec: dict
) -> None:
    run_id = f"{dataset}_{label}"
    method_name = label.replace("_", "-")

    model_args = config["model"]["init_args"]
    model_args["encoder_type"] = spec["encoder_type"]
    model_args["encoder_kwargs"] = copy.deepcopy(spec["encoder_kwargs"])
    model_args["lr"] = _resolve_lr(dataset, label)

    batch_override = BATCH_OVERRIDE.get((dataset, label))
    if batch_override is not None:
        config["data"]["init_args"]["batch_size"] = int(batch_override)

    max_epochs = dataset_spec.get("max_epochs")
    if max_epochs is not None:
        # Both are set: main.py links trainer.max_epochs to the model argument,
        # and leaving a stale model value behind would be confusing to read.
        config["trainer"]["max_epochs"] = int(max_epochs)
        model_args["max_epochs"] = int(max_epochs)

    logger_args = config["trainer"]["logger"]["init_args"]
    logger_args["name"] = f"{method_name}_{dataset}_seed42"
    logger_args["project"] = project_name(dataset)
    logger_args["save_dir"] = "wandb/encoder_bench"

    checkpoint = _callback(
        config, "lightning.pytorch.callbacks.ModelCheckpoint"
    )["init_args"]
    checkpoint["dirpath"] = f"outputs/encoder_bench/checkpoints/{run_id}"
    checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

    visualization = _callback(
        config, "callbacks.xc_plot_callback.VisualizationCallback"
    )["init_args"]
    visualization["output_dir"] = f"outputs/encoder_bench/plots/{run_id}"
    visualization["save_embeddings"] = True
    visualization["embedding_method_name"] = method_name

    heterogeneity = _callback(
        config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
    )["init_args"]
    heterogeneity["output_dir"] = (
        f"outputs/encoder_bench/plots/{run_id}/heterogeneity"
    )

    config["trainer"]["callbacks"].append(
        {
            "class_path": "callbacks.runtime_profiler.RuntimeProfilerCallback",
            "init_args": {
                "output_path": "results/encoder_bench/runtime_runs.csv",
            },
        }
    )
    config["trainer"]["callbacks"].append(
        {
            "class_path": "callbacks.encoder_benchmark.EncoderBenchmarkCallback",
            "init_args": {
                "tail_epochs": 20,
                "cv_threshold": 0.25,
                "direction_change_threshold": 0.5,
                "spike_ratio_threshold": 3.0,
                "check_grad_every_n_steps": 50,
            },
        }
    )
    config["trainer"]["callbacks"].append(
        {
            "class_path": (
                "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            ),
            "init_args": {
                "output_dir": f"outputs/encoder_bench/plots/{run_id}/paper",
                "method_name": method_name,
                "color_key": "final_annotation",
                "every_n_epochs": None,
                **_plot_style(config),
                "cmap": "tab20",
                # Datasets with more classes than tab20 holds fall back to a
                # continuous map so no two classes share a colour.
                "overflow_cmap": "gist_ncar",
                "figsize": 4.0,
                "dpi": 300,
                "formats": ["png"],
            },
        }
    )
    config["trainer"]["callbacks"].append(
        {
            "class_path": "callbacks.runtime_info_callback.RuntimeInfoCallback",
            "init_args": {
                "output_dir": "outputs/encoder_bench/runtime_info",
            },
        }
    )


def _header(dataset: str, label: str) -> str:
    notes = [
        "# Generated by configs/encoder_bench/generate_configs.py.",
        f"# Dataset: {dataset}; encoder configuration: {label}; seed: 42.",
        "# Projection head, augmentation, sampling, losses and fidelity callback",
        "# are inherited unchanged from the current dataset baseline.",
    ]
    if dataset == "ng20":
        notes.append(
            f"# lr={NG20_LR:g} is the dataset mainline value and is identical for "
            "every encoder. The formal NG20 runs go through sweep/encoder_bench_ng20_lr_seed42.yaml, "
            f"which searches {', '.join(f'{v:g}' for v in LR_SEARCH_GRID)} for "
            "every architecture including the baseline."
        )
    elif label not in SELECTED_LR:
        notes.append(
            f"# lr is the sentinel {LR_SENTINEL!r} and this config CANNOT run. "
            "Fill SELECTED_LR in generate_configs.py with the NG20-selected "
            "value for this encoder and regenerate. Running the baseline at its "
            "tuned lr against new encoders at an unsearched lr would violate "
            "fairness rule 1."
        )
    if dataset in {"hcl", "mca"} and label == "ft_transformer":
        notes.append(
            "# DIAGNOSTIC ONLY: standard feature attention is excluded from the "
            "formal high-dimensional matrix."
        )
    if dataset == "mca":
        notes.append(
            "# CONDITIONAL: run MCA only after the HCL latent-token result is "
            "acceptable."
        )
    return "\n".join(notes) + "\n"


def generate() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, dataset_spec in DATASETS.items():
        baseline = _load_yaml(dataset_spec["source"])
        for label, encoder_spec in ENCODERS.items():
            only = encoder_spec.get("only_datasets")
            if only and dataset not in only:
                continue
            config = copy.deepcopy(baseline)
            _configure_run(config, dataset, label, encoder_spec, dataset_spec)
            output_path = OUTPUT_DIR / f"{dataset}_{label}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            output_path.write_text(
                _header(dataset, label) + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_wide_lr_runs() -> list[Path]:
    """Write collision-free configs for the NG20 wide-capacity sweep.

    A W&B grid that varies ``model.init_args.lr`` while reusing one config also
    reuses that config's checkpoint and plot directories. Concurrent agents
    then overwrite one another's artifacts. Materializing one config per
    learning rate makes the command, LR, output paths, runtime CSV, and W&B run
    name unambiguous.
    """
    run_dir = OUTPUT_DIR / "sweep_wide" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label in ("ft_transformer_wide", "latent_transformer_wide"):
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E01/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E01/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E01/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E01/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E01/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E01/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E01 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e02_lr_runs() -> list[Path]:
    """Write collision-free configs for the E02 architecture sweep."""
    run_dir = OUTPUT_DIR / "sweep_e02" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    labels = (
        "ft_transformer_wide_mean",
        "latent_transformer_wide_r8",
        "latent_transformer_wide_r16",
    )
    for label in labels:
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E02/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E02/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E02/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E02/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E02/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E02/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E02 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e03_lr_runs() -> list[Path]:
    """Write collision-free configs for the E03 architecture sweep."""
    run_dir = OUTPUT_DIR / "sweep_e03" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    labels = (
        "ft_transformer_wide_attention",
        "latent_transformer_wide_r16_attention",
        "latent_transformer_m32_r16",
        "latent_transformer_r16_l3",
    )
    for label in labels:
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E03/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E03/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E03/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E03/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E03/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E03/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E03 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e04_lr_runs() -> list[Path]:
    """Write collision-free configs for the E04 final-norm sweep."""
    run_dir = OUTPUT_DIR / "sweep_e04" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    labels = (
        "ft_transformer_wide_mean_no_final_norm",
        "latent_transformer_wide_r16_no_final_norm",
        "latent_transformer_m32_r16_no_final_norm",
    )
    for label in labels:
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E04/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E04/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E04/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E04/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E04/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E04/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E04 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e05_lr_runs() -> list[Path]:
    """Write collision-free configs for the E05 pooling-capacity sweep."""
    run_dir = OUTPUT_DIR / "sweep_e05" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    labels = (
        "ft_transformer_wide_flatten",
        "latent_transformer_wide_r16_mean_std",
        "latent_transformer_m32_r16_mean_std",
    )
    for label in labels:
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E05/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E05/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E05/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E05/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E05/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E05/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E05 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e06_lr_runs() -> list[Path]:
    """Write collision-free configs for the E06 adaptive-pooling sweep."""
    run_dir = OUTPUT_DIR / "sweep_e06" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    labels = (
        "ft_transformer_wide_flatten_mean_init",
        "latent_transformer_wide_r16_mean_std_residual",
        "latent_transformer_m32_r16_mean_std_residual",
    )
    for label in labels:
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        for lr in LR_SEARCH_GRID:
            config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"

            config["model"]["init_args"]["lr"] = float(lr)
            logger_args = config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E06/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"

            visualization = _callback(
                config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]
            visualization["output_dir"] = f"outputs/encoder_tuning/E06/{run_id}/plots"

            heterogeneity = _callback(
                config, "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback"
            )["init_args"]
            heterogeneity["output_dir"] = (
                f"outputs/encoder_tuning/E06/{run_id}/heterogeneity"
            )

            runtime = _callback(
                config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]
            runtime["output_path"] = (
                f"results/encoder_tuning/E06/{run_id}/runtime.csv"
            )

            paper = _callback(
                config, "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback"
            )["init_args"]
            paper["output_dir"] = f"outputs/encoder_tuning/E06/{run_id}/paper"

            runtime_info = _callback(
                config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]
            runtime_info["output_dir"] = (
                f"outputs/encoder_tuning/E06/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            body = yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            )
            header = (
                "# Generated collision-free E06 run config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                "# Every artifact path is unique to this encoder/LR run.\n"
            )
            output_path.write_text(
                header + body,
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e07_hybrid_runs() -> list[Path]:
    """Write base and collision-free configs for the E07 ReZero hybrids."""
    variants = {
        "ft_transformer_mlp_residual": (
            "ng20_ft_transformer_wide_mean_no_final_norm.yaml",
            {
                "d_token": 96,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "pooling": "mean",
                "final_norm": False,
                "force_mem_efficient": True,
                "residual_init": 0.0,
            },
        ),
        "latent_transformer_m32_d192_r16_mlp_residual": (
            "ng20_latent_transformer_m32_r16.yaml",
            {
                "num_latents": 32,
                "d_token": 192,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "latent_rank": 16,
                "pooling": "mean",
                "final_norm": True,
                "force_mem_efficient": True,
                "residual_init": 0.0,
            },
        ),
    }
    run_dir = OUTPUT_DIR / "sweep_e07" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, (source_name, kwargs) in variants.items():
        config = _load_yaml(OUTPUT_DIR / source_name)
        encoder_type = (
            "ft_transformer_mlp_residual"
            if label == "ft_transformer_mlp_residual"
            else "latent_transformer_mlp_residual"
        )
        model_args = config["model"]["init_args"]
        model_args["encoder_type"] = encoder_type
        model_args["encoder_kwargs"] = kwargs
        base_path = OUTPUT_DIR / f"ng20_{label}.yaml"
        base_path.write_text(
            "# E07 ReZero hybrid: exact MLP output at initialization.\n"
            "# This is an engineering hybrid, not a pure-Transformer baseline.\n"
            + yaml.dump(
                config,
                Dumper=_NoAliasDumper,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=100,
            ),
            encoding="utf-8",
            newline="\n",
        )
        written.append(base_path)

        for lr in LR_SEARCH_GRID:
            run_config = _load_yaml(base_path)
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"
            run_config["model"]["init_args"]["lr"] = float(lr)
            logger_args = run_config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                run_config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E07/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"
            _callback(
                run_config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E07/{run_id}/plots"
            )
            _callback(
                run_config,
                "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E07/{run_id}/heterogeneity"
            )
            _callback(
                run_config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]["output_path"] = (
                f"results/encoder_tuning/E07/{run_id}/runtime.csv"
            )
            _callback(
                run_config,
                "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E07/{run_id}/paper"
            )
            _callback(
                run_config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E07/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            output_path.write_text(
                "# Generated collision-free E07 hybrid config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                + yaml.dump(
                    run_config,
                    Dumper=_NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=100,
                ),
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e08_readout_runs() -> list[Path]:
    """Write pure-Transformer configs with capacity added after token pooling.

    This keeps attention activation sizes identical to the strongest E04/E03
    pure models.  The additional parameters live in a nonlinear readout, which
    is substantially safer than widening feature attention at batch size 4096.
    """
    variants = {
        "ft_transformer_d96_readout1024_512": (
            "ng20_ft_transformer_wide_mean_no_final_norm.yaml",
            {
                "d_token": 96,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "pooling": "mean",
                "final_norm": False,
                "force_mem_efficient": True,
                "readout_hidden_dims": [1024, 512],
                "readout_hidden_norm": "batchnorm",
                "readout_activation": "gelu",
                "readout_dropout": 0.0,
            },
        ),
        "latent_transformer_m32_d192_r16_readout512_256": (
            "ng20_latent_transformer_m32_r16.yaml",
            {
                "num_latents": 32,
                "d_token": 192,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "latent_rank": 16,
                "pooling": "mean",
                "final_norm": True,
                "force_mem_efficient": True,
                "readout_hidden_dims": [512, 256],
                "readout_hidden_norm": "batchnorm",
                "readout_activation": "gelu",
                "readout_dropout": 0.0,
            },
        ),
    }
    run_dir = OUTPUT_DIR / "sweep_e08" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, (source_name, kwargs) in variants.items():
        config = _load_yaml(OUTPUT_DIR / source_name)
        model_args = config["model"]["init_args"]
        model_args["encoder_type"] = (
            "ft_transformer"
            if label.startswith("ft_transformer")
            else "latent_transformer"
        )
        model_args["encoder_kwargs"] = kwargs

        for lr in LR_SEARCH_GRID:
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"
            run_config = copy.deepcopy(config)
            run_config["model"]["init_args"]["lr"] = float(lr)
            logger_args = run_config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                run_config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E08/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"
            _callback(
                run_config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E08/{run_id}/plots"
            )
            _callback(
                run_config,
                "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E08/{run_id}/heterogeneity"
            )
            _callback(
                run_config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]["output_path"] = (
                f"results/encoder_tuning/E08/{run_id}/runtime.csv"
            )
            _callback(
                run_config,
                "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E08/{run_id}/paper"
            )
            _callback(
                run_config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E08/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            output_path.write_text(
                "# Generated collision-free E08 pure-Transformer config.\n"
                "# Capacity is added only after token pooling; projection head unchanged.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                + yaml.dump(
                    run_config,
                    Dumper=_NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=100,
                ),
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


def generate_e09_information_runs() -> list[Path]:
    """Write information-preserving tokenizer/resolution experiments."""
    variants = {
        "ft_transformer_d96_nonlinear_tokenizer": (
            "ng20_ft_transformer_wide_mean_no_final_norm.yaml",
            {
                "d_token": 96,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "pooling": "mean",
                "final_norm": False,
                "force_mem_efficient": True,
                "tokenizer_basis": "linear_tanh_square",
                "readout_hidden_dims": [1024, 512],
                "readout_hidden_norm": "batchnorm",
                "readout_activation": "gelu",
                "readout_dropout": 0.0,
            },
        ),
        "latent_transformer_m48_d192_r16": (
            "ng20_latent_transformer_m32_r16.yaml",
            {
                "num_latents": 48,
                "d_token": 192,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_ratio": 4.0,
                "dropout": 0.1,
                "attn_dropout": 0.1,
                "latent_rank": 16,
                "pooling": "mean",
                "final_norm": True,
                "force_mem_efficient": True,
            },
        ),
    }
    run_dir = OUTPUT_DIR / "sweep_e09" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, (source_name, kwargs) in variants.items():
        config = _load_yaml(OUTPUT_DIR / source_name)
        model_args = config["model"]["init_args"]
        model_args["encoder_type"] = (
            "ft_transformer"
            if label.startswith("ft_transformer")
            else "latent_transformer"
        )
        model_args["encoder_kwargs"] = kwargs

        for lr in LR_SEARCH_GRID:
            lr_slug = f"{lr:g}".replace(".", "p").replace("-", "m")
            run_id = f"ng20_{label}_lr{lr_slug}"
            run_config = copy.deepcopy(config)
            run_config["model"]["init_args"]["lr"] = float(lr)
            logger_args = run_config["trainer"]["logger"]["init_args"]
            logger_args["name"] = f"{label.replace('_', '-')}_ng20_lr{lr:g}_seed42"

            checkpoint = _callback(
                run_config, "lightning.pytorch.callbacks.ModelCheckpoint"
            )["init_args"]
            checkpoint["dirpath"] = f"outputs/encoder_tuning/E09/{run_id}/checkpoints"
            checkpoint["filename"] = f"{run_id}-{{epoch:04d}}"
            _callback(
                run_config, "callbacks.xc_plot_callback.VisualizationCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E09/{run_id}/plots"
            )
            _callback(
                run_config,
                "callbacks.xc_plot_heterogeneity.HeterogeneityPlotCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E09/{run_id}/heterogeneity"
            )
            _callback(
                run_config, "callbacks.runtime_profiler.RuntimeProfilerCallback"
            )["init_args"]["output_path"] = (
                f"results/encoder_tuning/E09/{run_id}/runtime.csv"
            )
            _callback(
                run_config,
                "callbacks.xc_paper_embedding_callback.PaperEmbeddingCallback",
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E09/{run_id}/paper"
            )
            _callback(
                run_config, "callbacks.runtime_info_callback.RuntimeInfoCallback"
            )["init_args"]["output_dir"] = (
                f"outputs/encoder_tuning/E09/{run_id}/runtime_info"
            )

            output_path = run_dir / f"{run_id}.yaml"
            output_path.write_text(
                "# Generated collision-free E09 information-preservation config.\n"
                f"# Encoder: {label}; learning rate: {lr:g}; seed: 42.\n"
                + yaml.dump(
                    run_config,
                    Dumper=_NoAliasDumper,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=100,
                ),
                encoding="utf-8",
                newline="\n",
            )
            written.append(output_path)
    return written


if __name__ == "__main__":
    paths = generate()
    paths.extend(generate_wide_lr_runs())
    paths.extend(generate_e02_lr_runs())
    paths.extend(generate_e03_lr_runs())
    paths.extend(generate_e04_lr_runs())
    paths.extend(generate_e05_lr_runs())
    paths.extend(generate_e06_lr_runs())
    paths.extend(generate_e07_hybrid_runs())
    paths.extend(generate_e08_readout_runs())
    paths.extend(generate_e09_information_runs())
    print(f"generated {len(paths)} configs in {OUTPUT_DIR}")
