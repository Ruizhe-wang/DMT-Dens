import os
import torch
import lightning as pl

class SaveLatestSubmodulesCallback(pl.Callback):
    def __init__(self, output_dir="zzl_checkpoints"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        enc_path = os.path.join(self.output_dir, "enc-latest.pt")
        vis_path = os.path.join(self.output_dir, "vis_list-latest.pt")

        torch.save(pl_module.enc.state_dict(), enc_path)
        torch.save(pl_module.vis_list.state_dict(), vis_path)