"""
Consolidated embeddings save callback.

Saves final embeddings at training end in two formats:

  NPZ: {dataset_name}_{method}_embeddings.npz
       Keys: cell_ids, labels, method, dataset_name, num_layers,
             layer_0 … layer_N (each shape [n_cells, 2])

  CSV: {dataset_name}_{method}_embeddings.csv
       Flat format compatible with tools/pancreas_cellrank_plots.py:
         cell_id, method, layer, x, y
       One row per (cell × layer); layers stacked vertically.

Compatible with all baseline methods (t-SNE, UMAP, PaCMAP, densMAP, densNE,
PHATE, densSNE) and TopoBranch — auto-detects output format (tuple vs array).

Configure in YAML:
  - class_path: callbacks.xc_save_consolidated_embeddings.SaveConsolidatedEmbeddingsCallback
    init_args:
      dataset_name: pancreas
      output_dir: outputs/embeddings
      save_format: both  # "npz", "csv", or "both"
      embedding_method_name: topobranch
"""

import os
from contextlib import contextmanager

import lightning as pl
import numpy as np
import pandas as pd
import torch


@contextmanager
def _eval_mode(module):
    """Temporarily force deterministic inference and restore caller state."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class SaveConsolidatedEmbeddingsCallback(pl.Callback):
    def __init__(
        self,
        dataset_name: str,
        output_dir: str = "outputs/embeddings",
        save_format: str = "both",
        embedding_method_name: str | None = None,
        save_on_train_end: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.save_format = save_format
        self.embedding_method_name = embedding_method_name
        self.save_on_train_end = save_on_train_end
        self._saved = False

    # ------------------------------------------------------------------
    #  Internal filenames
    # ------------------------------------------------------------------
    def _npz_path(self, method: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{self.dataset_name}_{method}_embeddings.npz")

    def _csv_path(self, method: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{self.dataset_name}_{method}_embeddings.csv")

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _resolve_method_name(self, pl_module) -> str:
        """Determine the method label for output files.

        Priority: explicit config > pl_module.method (baselines) >
        class name heuristic > dataset name fallback.
        """
        if self.embedding_method_name:
            return self.embedding_method_name

        # Baseline model: pl_module.method is e.g. '_tsne', '_umap'
        if hasattr(pl_module, "method"):
            return str(pl_module.method).lstrip("_")

        # Try class name heuristics for DiffTree variants
        cls_name = type(pl_module).__name__.lower()
        for keyword in ("topobranch", "difftree", "mldr"):
            if keyword in cls_name:
                return keyword

        return self.dataset_name

    def _gather_embeddings(self, trainer, pl_module):
        """Run inference on full dataset; returns all-layer embeddings + metadata.

        Handles both TopoBranch (tuple output with list of tensors) and
        baseline methods (single numpy array / tensor).
        """
        import inspect

        forward_params = inspect.signature(pl_module.forward).parameters
        supports_tau = "tau" in forward_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
        )

        dataloader = trainer.datamodule.val_dataloader()
        adata = trainer.datamodule.adata

        data_list: list[torch.Tensor] = []
        cell_id_list: list[np.ndarray] = []
        label_list: list[np.ndarray] = []

        device = pl_module.device

        with _eval_mode(pl_module), torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                data_input_item = batch["data_input_item"].to(device)
                batch_size = data_input_item.shape[0]

                # Forward pass
                if supports_tau:
                    tau_val = (
                        getattr(pl_module.hparams, "tau", 1.0)
                        if hasattr(pl_module, "hparams")
                        else 1.0
                    )
                    model_output = pl_module(data_input_item, tau=tau_val)
                else:
                    model_output = pl_module(data_input_item)

                # Extract 2D embeddings — handles 3 output shapes:
                #   TopoBranch:  tuple[4+]  → model_output[3] is list of tensors
                #   TopoBranch:  tuple[3]   → model_output[2] is tensor
                #   Baseline:    ndarray    → single tensor
                if isinstance(model_output, tuple) and len(model_output) >= 4:
                    lat_vis_list = model_output[3]
                elif isinstance(model_output, tuple) and len(model_output) >= 3:
                    lat_vis_list = [model_output[2]]
                else:
                    # Baseline: numpy array or torch tensor
                    lat_vis = torch.as_tensor(model_output, dtype=torch.float32, device=device)
                    lat_vis_list = [lat_vis]

                # Stack layers into (num_layers, batch_size, 2)
                lat_stack = torch.stack([lv.detach().cpu() for lv in lat_vis_list], dim=0)
                data_list.append(lat_stack.float())

                # Per-batch cell IDs and labels (aligned with dataloader order)
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + batch_size, adata.n_obs)
                cell_id_list.append(adata.obs_names[start_idx:end_idx].to_numpy(dtype=str))
                if "final_annotation" in adata.obs:
                    label_list.append(adata.obs["final_annotation"].values[start_idx:end_idx].astype(str))
                elif "batch" in adata.obs:
                    label_list.append(adata.obs["batch"].values[start_idx:end_idx].astype(str))
                else:
                    label_list.append(np.full(batch_size, "unknown", dtype=str))

        all_lat = torch.cat(data_list, dim=1)  # (num_layers, total_samples, 2)
        all_cell_ids = np.concatenate(cell_id_list, axis=0)
        all_labels = np.concatenate(label_list, axis=0)

        num_layers = all_lat.shape[0]
        all_layers = [all_lat[i].numpy() for i in range(num_layers)]

        return all_layers, all_cell_ids, all_labels

    def _save_npz(self, all_layers, cell_ids, labels, method):
        save_dict = {
            "cell_ids": np.array(cell_ids, dtype=str),
            "labels": np.array(labels, dtype=str),
            "num_layers": len(all_layers),
            "method": method,
            "dataset_name": self.dataset_name,
        }
        for i, emb in enumerate(all_layers):
            save_dict[f"layer_{i}"] = emb

        path = self._npz_path(method)
        np.savez(path, **save_dict)
        return path

    def _save_csv(self, all_layers, cell_ids, labels, method):
        """Flat CSV: cell_id, method, layer, x, y — one row per (cell, layer)."""
        frames = []
        for i, emb in enumerate(all_layers):
            frames.append(pd.DataFrame({
                "cell_id": cell_ids,
                "method": method,
                "layer": i,
                "x": emb[:, 0],
                "y": emb[:, 1],
            }))

        frame = pd.concat(frames, ignore_index=True)
        path = self._csv_path(method)
        frame.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------
    def on_train_end(self, trainer, pl_module):
        if not self.save_on_train_end or self._saved:
            return
        if not trainer.is_global_zero:
            return

        method = self._resolve_method_name(pl_module)
        print(f"\n[ConsolidatedEmbeddings] dataset={self.dataset_name}  method={method}")

        all_layers, cell_ids, labels = self._gather_embeddings(trainer, pl_module)
        print(f"  layers={len(all_layers)}  cells={len(cell_ids)}")

        saved = []
        if self.save_format in ("npz", "both"):
            saved.append(self._save_npz(all_layers, cell_ids, labels, method))
        if self.save_format in ("csv", "both"):
            saved.append(self._save_csv(all_layers, cell_ids, labels, method))

        for p in saved:
            print(f"  -> {p}")
        self._saved = True

    def on_fit_end(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)
