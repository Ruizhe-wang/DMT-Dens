"""LightningCLI entry point for DMT-Dens experiments."""

import numpy as np
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI


torch.set_float32_matmul_precision("medium")

# NumPy objects occur in historical checkpoints. Register only the specific
# safe globals needed to load those checkpoints with modern PyTorch.
try:
    from numpy.core.multiarray import scalar as numpy_scalar

    torch.serialization.add_safe_globals([numpy_scalar, np.dtype])
except (AttributeError, ImportError):
    pass


class DMTDensCLI(LightningCLI):
    """Link shared data/model settings while retaining YAML overrides."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.num_positive_samples",
            "model.init_args.num_positive_samples",
        )
        parser.link_arguments("data.init_args.data_name", "model.init_args.data_name")
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")


def main():
    DMTDensCLI(
        LightningModule,
        LightningDataModule,
        save_config_callback=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )

    # W&B is optional when trainer.logger=false. Close a run cleanly when the
    # package is installed and a W&B logger was selected.
    try:
        import wandb

        wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
