"""Dump the trained model's 2-D embedding (plus the high-dimensional input and
labels) to an ``.npz`` at the end of training, for offline density analysis
(e.g. the Mellon ground-truth comparison in ``scripts/run_mellon_eval.py``).

Reuses :meth:`FidelityEvalCallback.get_embeddings` so the dumped embedding is
exactly what the fidelity metrics are computed on.
"""
import os
import numpy as np

from callbacks.Eval_density import FidelityEvalCallback


class DumpEmbeddingCallback(FidelityEvalCallback):
    def __init__(self, out_path, dump_on_validation=False, **kwargs):
        super().__init__(**kwargs)
        self.out_path = out_path
        self.dump_on_validation = dump_on_validation

    def _dump(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        lat_vis, data_input, labels = self.get_embeddings(trainer, pl_module)
        emb = lat_vis[0].numpy()                       # first vis layer -> (N, d)
        hd = data_input.numpy()                        # (N, D)
        lab = labels.numpy() if labels is not None else None
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        np.savez(self.out_path, emb=emb, hd=hd, labels=lab)
        print(f"[DumpEmbeddingCallback] saved {self.out_path} emb={emb.shape} hd={hd.shape}")

    def on_fit_end(self, trainer, pl_module):
        self._dump(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Do not run the (wandb-bound) parent metric logging from this callback;
        # optionally dump the current embedding for periodic inspection.
        if self.dump_on_validation:
            self._dump(trainer, pl_module)
