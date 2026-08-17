from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from .config import load_bio_eval_config, validate_config_against_adata
from .embeddings import MethodEmbedding, align_embedding_to_adata, load_embedding
from .metrics import compute_metrics, write_metrics
from .plots import plot_all_panels
from .report import write_report
from .signals import BioSignals, build_bio_signals


@dataclass(frozen=True)
class BioEvalResult:
    output_dir: Path
    report_path: Path
    metrics_path: Path
    figure_paths: tuple[Path, ...]


def run_bio_eval(
    *,
    config_path: str | Path,
    embedding_paths: list[str | Path],
    output_dir: str | Path,
    methods: list[str] | None = None,
    layer: int | None = None,
) -> BioEvalResult:
    config = load_bio_eval_config(config_path)
    validation = validate_config_against_adata(config)
    signals = build_bio_signals(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(config.adata_path, backed="r")
    try:
        embeddings = _load_and_align_embeddings(adata, embedding_paths, methods=methods, layer=layer)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

    figure_paths: list[Path] = []
    metric_rows = []
    for embedding in embeddings:
        aligned_signals = _subset_signals(signals, embedding.cell_ids)
        figure_paths.extend(plot_all_panels(embedding, aligned_signals, out, enabled_panels=config.panels))
        metric_rows.append(compute_metrics(embedding, aligned_signals))

    metrics_path = out / "metrics_summary.csv"
    write_metrics(metric_rows, metrics_path)
    report_path = write_report(config, validation, figure_paths, metrics_path, out / "report.md")
    return BioEvalResult(
        output_dir=out,
        report_path=report_path,
        metrics_path=metrics_path,
        figure_paths=tuple(figure_paths),
    )


def _load_and_align_embeddings(
    adata,
    embedding_paths: list[str | Path],
    *,
    methods: list[str] | None,
    layer: int | None,
) -> list[MethodEmbedding]:
    if methods and len(methods) not in {1, len(embedding_paths)}:
        raise ValueError("--method must be provided once or once per --embedding")
    resolved: list[MethodEmbedding] = []
    for index, embedding_path in enumerate(embedding_paths):
        method = None
        if methods:
            method = methods[0] if len(methods) == 1 else methods[index]
        embedding = load_embedding(embedding_path, method=method, layer=layer)
        resolved.append(align_embedding_to_adata(adata, embedding, require_all=True))
    return resolved


def _subset_signals(signals: BioSignals, cell_ids: list[str] | tuple[str, ...]) -> BioSignals:
    if signals.cell_ids is None:
        return signals
    order = {cell_id: index for index, cell_id in enumerate(signals.cell_ids.astype(str))}
    indices = [order[str(cell_id)] for cell_id in cell_ids]
    return BioSignals(
        categorical={key: values[indices] for key, values in signals.categorical.items()},
        continuous={key: values[indices] for key, values in signals.continuous.items()},
        masks={key: values[indices] for key, values in signals.masks.items()},
        warnings=list(signals.warnings),
        cell_ids=signals.cell_ids[indices],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run biological information preservation evaluation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--embedding", action="append", required=True, type=Path)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_bio_eval(
        config_path=args.config,
        embedding_paths=args.embedding,
        output_dir=args.output,
        methods=args.method,
        layer=args.layer,
    )
    print(f"Report: {result.report_path}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
