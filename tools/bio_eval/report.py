from __future__ import annotations

from pathlib import Path

from .config import BioEvalConfig, ConfigValidationReport


def write_report(
    config: BioEvalConfig,
    validation: ConfigValidationReport,
    figure_paths: list[Path],
    metrics_path: Path,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Bio Evaluation Report: {config.dataset}",
        "",
        f"- AnnData: `{config.adata_path}`",
        f"- Metrics: `{metrics_path.name}`",
        "",
        "## Signal Support",
        "",
        "| Signal | Supported |",
        "|---|---:|",
    ]
    for key, value in sorted(validation.supported.items()):
        lines.append(f"| {key} | {value} |")
    if validation.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in validation.warnings)
    lines.extend(["", "## Figures", ""])
    for figure in figure_paths:
        try:
            rel = figure.relative_to(path.parent)
        except ValueError:
            rel = figure
        lines.append(f"- `{rel}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
