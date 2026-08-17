"""Ground-truth trajectory metrics logged live during training (dynGen fate).

Computes the same case-study metrics as
``tools/dyngen_case_study/compute_metrics.py`` — but every ``every_n_epochs``
on the current embedding, logging them to W&B so each run (e.g. every cell of
the hyperparameter sweep) is directly rankable by trajectory preservation
against the dynGen ground truth, instead of running the offline script per run.

Logged keys (prefix ``gt`` by default), per embedding layer:
  * ``gt/demap``               — Spearman(ambient geodesic, embedding Euclidean)
  * ``gt/branch_monotonicity`` — mean |Spearman(pseudotime, branch PC1)|
  * ``gt/topology_f1``         — MST-over-centroids edges vs true edges (~edgeF1)
  * ``gt/fate_corr``           — true fate probs vs distance-softmax probs
  * ``gt/knn_label_acc``       — 5-fold kNN acc of the terminal-fate label (~term)

Reuses the metric functions from tools.dyngen_case_study.compute_metrics so the
live numbers match the offline script (modulo the ambient representation: here
the ambient PCA is taken on the datamodule's feature matrix, which is consistent
across runs and so valid for ranking).
"""

import lightning as pl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from tools.dyngen_case_study.compute_metrics import (
    branch_monotonicity,
    demap,
    fate_recovery,
    knn_label_acc,
    topology_f1,
)


class DyngenGroundTruthMetricsCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs=50,
        pca_dim=50,
        time_key="pseudotime",
        from_key="from",
        to_key="to",
        terminal_key="final_annotation",
        fate_prefix="fate_prob_",
        demap_landmarks=1000,
        demap_knn=15,
        prefix="gt",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.pca_dim = pca_dim
        self.time_key = time_key
        self.from_key = from_key
        self.to_key = to_key
        self.terminal_key = terminal_key
        self.fate_prefix = fate_prefix
        self.demap_landmarks = demap_landmarks
        self.demap_knn = demap_knn
        self.prefix = prefix
        self._gt = None  # cached ground-truth arrays (data is fixed across epochs)

    # ------------------------------------------------------------------ utils
    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _build_gt(self, adata):
        """Assemble ground-truth arrays from the datamodule's adata (cached)."""
        obs = adata.obs
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        X = np.asarray(X, dtype=np.float32)
        n_comp = min(self.pca_dim, X.shape[0], X.shape[1])
        ambient = PCA(n_components=n_comp, random_state=0).fit_transform(X)

        fate_cols = sorted(c for c in obs.columns if str(c).startswith(self.fate_prefix))
        fate_probs = (
            np.column_stack([pd.to_numeric(obs[c], errors="coerce").to_numpy(float) for c in fate_cols])
            if fate_cols else None
        )
        gt = {
            "ambient": ambient,
            "pseudotime": pd.to_numeric(obs[self.time_key], errors="coerce").to_numpy(float)
            if self.time_key in obs else None,
            "from": obs[self.from_key].astype(str).to_numpy() if self.from_key in obs else None,
            "to": obs[self.to_key].astype(str).to_numpy() if self.to_key in obs else None,
            "terminal_label": obs[self.terminal_key].astype(str).to_numpy()
            if self.terminal_key in obs else None,
            "fate_names": [c[len(self.fate_prefix):] for c in fate_cols],
            "fate_probs": fate_probs,
        }
        return gt

    def _get_embedding(self, trainer, pl_module):
        """Return [n_layers, n_cells, dim] embedding in dataloader (== adata) order."""
        import inspect

        data_list = []
        with torch.inference_mode():
            forward_params = inspect.signature(pl_module.forward).parameters
            supports_tau = "tau" in forward_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_params.values()
            )
            for batch in trainer.datamodule.val_dataloader():
                x = batch["data_input_item"].to(pl_module.device)
                out = pl_module(x, tau=getattr(pl_module.hparams, "tau", 1.0)) if supports_tau else pl_module(x)
                if isinstance(out, tuple) and len(out) >= 4:
                    lat_list = out[3]
                elif isinstance(out, tuple) and len(out) >= 3:
                    lat_list = [out[2]]
                else:
                    lat_list = [torch.as_tensor(out, dtype=torch.float32, device=x.device)]
                data_list.append(torch.stack([lv.detach().cpu() for lv in lat_list], dim=0).float())
        return torch.cat(data_list, dim=1)

    def _compute(self, gt, emb):
        keep = np.isfinite(emb).all(axis=1)
        e = emb[keep]
        amb = gt["ambient"][keep]
        pt = gt["pseudotime"][keep] if gt["pseudotime"] is not None else None
        fr = gt["from"][keep] if gt["from"] is not None else None
        to = gt["to"][keep] if gt["to"] is not None else None
        lab = gt["terminal_label"][keep] if gt["terminal_label"] is not None else None
        fp = gt["fate_probs"][keep] if gt["fate_probs"] is not None else None
        branch = (
            (pd.Series(fr).astype(str) + "->" + pd.Series(to).astype(str)).to_numpy()
            if fr is not None and to is not None else None
        )
        _, _, f1 = topology_f1(e, fr, to)
        return {
            "demap": demap(amb, e, n_landmarks=self.demap_landmarks, knn=self.demap_knn),
            "branch_monotonicity": branch_monotonicity(e, pt, branch),
            "topology_f1": f1,
            "fate_corr": fate_recovery(e, fr, to, gt["fate_names"], fp),
            "knn_label_acc": knn_label_acc(e, lab),
        }

    # ----------------------------------------------------------------- hooks
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if (
            trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING
            and trainer.current_epoch == 0
            and not self._is_baseline_model(pl_module)
        ):
            return
        epoch_num = trainer.current_epoch + 1
        if self.every_n_epochs and self.every_n_epochs > 0 and epoch_num % self.every_n_epochs != 0:
            return

        adata = trainer.datamodule.adata
        if self._gt is None:
            self._gt = self._build_gt(adata)

        lat = self._get_embedding(trainer, pl_module)  # [layers, n, dim]
        log = {}
        for i in range(lat.shape[0]):
            emb = lat[i].detach().cpu().numpy()
            m = self._compute(self._gt, emb)
            suffix = "" if lat.shape[0] == 1 else f"_layer{i}"
            for k, v in m.items():
                if v is not None and np.isfinite(v):
                    log[f"{self.prefix}/{k}{suffix}"] = float(v)

        # Log through Lightning so the keys land in the WandbLogger history and
        # are sortable/optimizable in the sweep table (single logging path to
        # avoid wandb step conflicts with the image callbacks' manual run.log).
        for k, v in log.items():
            pl_module.log(k, v, rank_zero_only=True, prog_bar=False, on_epoch=True, on_step=False)
