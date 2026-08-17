"""Developmental-time trajectory metrics logged live during training (C. elegans).

C. elegans has no simulated lineage backbone (no from/to tree, no fate
probabilities) -- the only ground truth is the per-cell embryonic time
(``embryo_time_numeric``, derived from the binned ``embryo_time``) and the
cell-type label. This callback therefore logs, every ``every_n_epochs`` on the
current embedding, three time-preservation metrics plus DEMaP, so each run is
directly comparable in W&B:

  * ``gt/pseudotime_corr``  -- Spearman(embedding pseudotime, GT time)
  * ``gt/time_ordering_acc`` -- tie-aware order concordance vs GT time
  * ``gt/time_continuity``  -- smoothness of GT time over the embedding kNN graph
  * ``gt/demap``            -- Spearman(ambient geodesic, embedding Euclidean)
  * ``gt/pseudotime_reachable_frac`` -- diagnostic: fraction of cells reachable
    from the earliest-time root on the embedding kNN graph (cross-method
    comparisons of pseudotime_corr are only fair when this is ~1.0)

Embedding extraction mirrors DyngenGroundTruthMetricsCallback so it works for
both the parametric model and the (non-parametric) baseline_tri models. Metric
definitions live in tools.celegan_case_study.compute_metrics.
"""

import lightning as pl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from tools.celegan_case_study.compute_metrics import (
    pseudotime_corr,
    time_continuity,
    time_ordering_acc,
)
from tools.dyngen_case_study.compute_metrics import demap


class CeleganGroundTruthMetricsCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs=50,
        pca_dim=50,
        time_key="embryo_time_numeric",
        knn=15,
        demap_landmarks=1000,
        demap_knn=15,
        prefix="gt",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.pca_dim = pca_dim
        self.time_key = time_key
        self.knn = knn
        self.demap_landmarks = demap_landmarks
        self.demap_knn = demap_knn
        self.prefix = prefix
        self._gt = None  # cached (data is fixed across epochs)

    # ------------------------------------------------------------------ utils
    def _is_baseline_model(self, pl_module):
        return hasattr(pl_module, "method") and hasattr(pl_module, "validation_step_outputs_vis")

    def _build_gt(self, adata):
        obs = adata.obs
        time = (
            pd.to_numeric(obs[self.time_key], errors="coerce").to_numpy(float)
            if self.time_key in obs else None
        )
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        X = np.asarray(X, dtype=np.float32)
        n_comp = min(self.pca_dim, X.shape[0], X.shape[1])
        ambient = PCA(n_components=n_comp, random_state=0).fit_transform(X)
        return {"time": time, "ambient": ambient}

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
        time = gt["time"][keep] if gt["time"] is not None else None
        amb = gt["ambient"][keep] if gt["ambient"] is not None else None
        if time is None:
            return {}
        rho, cover = pseudotime_corr(e, time, knn=self.knn)
        out = {
            "pseudotime_corr": rho,
            "pseudotime_reachable_frac": cover,
            "time_ordering_acc": time_ordering_acc(e, time, knn=self.knn),
            "time_continuity": time_continuity(e, time, knn=self.knn),
        }
        if amb is not None:
            out["demap"] = demap(amb, e, n_landmarks=self.demap_landmarks, knn=self.demap_knn)
        return out

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

        for k, v in log.items():
            pl_module.log(k, v, rank_zero_only=True, prog_bar=False, on_epoch=True, on_step=False)
