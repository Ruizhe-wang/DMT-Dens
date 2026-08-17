"""Re-render saved embeddings with the established paper callback.

This is intentionally an offline plotting utility: it reads the consolidated
NPZ produced at the end of a completed run, calls the exact plotting method in
``callbacks.paper_embedding_plot_callback``, saves the PNG on the server, and
optionally resumes the original W&B run to log the same image there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support the documented ``python scripts/<name>.py`` invocation from the
# repository root as well as ``python -m scripts.<name>``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anndata as ad
import numpy as np
import torch
import wandb
import yaml

from callbacks.paper_embedding_plot_callback import VisualizationCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to render; defaults to every manifest entry.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Save server-side figures without updating the original W&B runs.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    required = {
        "entity",
        "project",
        "wandb_base_url",
        "summary_path",
        "plot",
        "runs",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing keys: {sorted(missing)}")
    return manifest


def make_adata(labels: np.ndarray) -> ad.AnnData:
    # The old callback only needs obs[color_key], obsm, slicing and copy.  A
    # zero-column AnnData preserves those semantics without loading raw data.
    return ad.AnnData(
        X=np.empty((len(labels), 0), dtype=np.float32),
        obs={"final_annotation": labels.astype(str)},
    )


def render_one(entry: dict, manifest: dict, upload: bool) -> dict:
    npz_path = Path(entry["npz_path"])
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    archive = np.load(npz_path)
    labels = archive["labels"].astype(str)
    layer_keys = sorted(
        (key for key in archive.files if key.startswith("layer_")),
        key=lambda key: int(key.split("_")[-1]),
    )
    if not layer_keys:
        raise ValueError(f"No layer_* arrays found in {npz_path}")

    # The established callback randomly subsamples very large datasets.  Make
    # an offline replot reproducible instead of changing the selected points
    # every time the command is rerun.
    seed = int(entry.get("seed", manifest.get("seed", 42)))
    np.random.seed(seed)
    torch.manual_seed(seed)

    plot_args = dict(manifest["plot"])
    plot_args.update(
        output_dir=entry["output_dir"],
        dataset_name=entry["dataset"].upper(),
        method_name=entry.get("method_name", "latent-bn"),
    )
    callback = VisualizationCallback(**plot_args)
    adata = make_adata(labels)
    dummy_input = torch.empty((len(labels), 0), dtype=torch.float32)

    run = None
    if upload:
        run = wandb.init(
            entity=manifest["entity"],
            project=manifest["project"],
            id=entry["run_id"],
            resume="must",
            job_type="paper-replot",
            tags=["legacy-paper-callback", "offline-replot"],
        )

    logged_keys: list[str] = []
    saved_files: list[str] = []
    try:
        for layer_key in layer_keys:
            layer_index = int(layer_key.split("_")[-1])
            embedding = torch.from_numpy(archive[layer_key]).float()
            if embedding.shape != (len(labels), 2):
                raise ValueError(
                    f"{npz_path}:{layer_key} has shape {embedding.shape}; "
                    f"expected ({len(labels)}, 2)"
                )
            figure_dict = callback.plot_dmt(
                adata.copy(),
                dummy_input,
                embedding,
                text=f"_layer{layer_index}",
                info="final_annotation",
                log_to_wandb=run is not None,
            )
            if run is not None:
                run.log(figure_dict)
                logged_keys.extend(figure_dict)
            for fmt in callback.save_formats:
                saved_files.append(
                    str(
                        Path(callback.output_dir)
                        / f"final_annotation_batch_layer{layer_index}.{fmt}"
                    )
                )
        if run is not None:
            run.summary["paper_plot/callback"] = (
                "callbacks.paper_embedding_plot_callback.VisualizationCallback"
            )
            run.summary["paper_plot/source_npz"] = str(npz_path)
    finally:
        if run is not None:
            run.finish()

    return {
        "dataset": entry["dataset"],
        "run_id": entry["run_id"],
        "seed": seed,
        "saved_files": saved_files,
        "wandb_keys": logged_keys,
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    os.environ.setdefault("WANDB_BASE_URL", manifest["wandb_base_url"])

    requested = set(args.datasets or [])
    entries = [
        entry
        for entry in manifest["runs"]
        if not requested or entry["dataset"] in requested
    ]
    found = {entry["dataset"] for entry in entries}
    if requested.difference(found):
        raise ValueError(
            f"Datasets not present in manifest: {sorted(requested.difference(found))}"
        )

    results = []
    for entry in entries:
        print(f"[paper-replot] dataset={entry['dataset']} run={entry['run_id']}")
        result = render_one(entry, manifest, upload=not args.no_wandb)
        results.append(result)
        print(
            f"[paper-replot] saved={result['saved_files']} "
            f"wandb_keys={result['wandb_keys']}"
        )

    summary_path = Path(manifest["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"[paper-replot] summary={summary_path}")


if __name__ == "__main__":
    main()
