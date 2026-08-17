import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import lightning as pl
import wandb

class VisualizationCallback(pl.Callback):
    def __init__(self, output_dir="output"):
        # adata = adata
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_our_visualization(self, trainer, pl_module):
        
        data_list = []
        for batch in trainer.datamodule.val_dataloader():
            # for batch in val_loader:
                # import pdb; pdb.set_trace()

                # print(f"Processing batch with size: {batch['data_input_item'].shape}")
                
            data_input_item = batch["data_input_item"].to(pl_module.device)
            index = batch["index"]
            
            # Pass data through the model (use pl_module here)
            x_masked, lat_high_dim, lat_vis, _ = pl_module(
                data_input_item,
                tau=pl_module.hparams.tau,  # Access hyperparameters from the model
            )
            
            # Collect the latent visualizations
            data_list.append(lat_vis)
        
        # Concatenate the latent visualizations along the batch dimension
        data = torch.cat(data_list, dim=0)
        return data
            

    def on_validation_epoch_end(self, trainer, pl_module):
        
        adata = trainer.datamodule.adata
        
        sc.set_figure_params(dpi=100, facecolor='white', frameon=False)
            
        # if 'PCA' not in adata.uns and :
        #     sc.tl.pca(adata, n_comps=50)
        #     adata.uns['PCA'] = adata.obsm['X_pca']
        
        lat_vis = self.get_our_visualization(trainer, pl_module)
        adata.obsm['X_dmtevt'] = lat_vis.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        sc.pl.scatter(
            adata,
            basis='dmtevt',  # 指定二维坐标
            color='batch',   # 按批次着色
            ax=ax,
            show=False,
            frameon=True,
            size=5,
            alpha=0.8,
            palette='viridis',  # 使用Viridis色图
        )

        ax.tick_params(
            axis='both',
            which='both',
            direction='in',   # 刻度方向，可选 'out' / 'inout'
            length=4,
            labelsize=10,      # 刻度字体大小
            palette='viridis',  # 使用Viridis色图
            size=5,
        )
        ax.set_xlabel("dmtevt1")
        ax.set_ylabel("dmtevt2")

        plt.tight_layout()
        
        trainer.logger.experiment.log({
            "batch_effect_correction_comparison": wandb.Image(fig)
        })
        