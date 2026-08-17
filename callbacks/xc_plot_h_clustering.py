import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import lightning as pl
import wandb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


class VisualizationHCCallback(pl.Callback):
    def __init__(self, output_dir="output"):
        # adata = adata
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_tree_and_emb(self, trainer, pl_module):
        
        data_list = []
        tree_rout_list = []
        vector_rout_list = []
        
        for batch in trainer.datamodule.val_dataloader():
                
            data_input_item = batch["data_input_item"].to(pl_module.device)
            index = batch["index"]
            
            x_masked, lat_high_dim, lat_vis, lat_vis_list = pl_module(
                data_input_item,
                tau=pl_module.hparams.tau,  # Access hyperparameters from the model
            )
            
            tree_rout, vector_rout, loss_rout = pl_module.router_forward(
                lat_high_dim.float().detach(),
                tree_rout_bool=True,
                ec_ce_weight=pl_module.hparams.ec_ce_weight,
            )
            
            
            # Collect the latent visualizations
            data_list.append(lat_vis_list[0])
            tree_rout_list.append(tree_rout)
            vector_rout_list.append(vector_rout)

        # Concatenate the latent visualizations along the batch dimension
        data = torch.cat(data_list, dim=0)
        tree_rout = torch.cat(tree_rout_list, dim=0)
        vector_rout = torch.cat(vector_rout_list, dim=0)

        return data, tree_rout, vector_rout
            

    def on_validation_epoch_end(self, trainer, pl_module):

        # if in training and epoch == 0, return
        if trainer.state.fn == pl.pytorch.trainer.states.TrainerFn.FITTING and trainer.current_epoch == 0:
            return

        adata = trainer.datamodule.adata
        
        lat_vis, tree_rout, vector_rout = self.get_tree_and_emb(trainer, pl_module)
        
        lat_vis_numpy = lat_vis.detach().cpu().numpy()
        tree_rout_numpy = tree_rout.detach().cpu().numpy()
        vector_rout_numpy = vector_rout.detach().cpu().numpy()
        
        # import pdb; pdb.set_trace()
        data_x = np.concatenate([lat_vis_numpy, tree_rout_numpy], axis=1)
        columns = ['x', 'y'] + [f'tree_level_{i}' for i in range(tree_rout.shape[1])]

        pd_data = pd.DataFrame(data=data_x, columns=columns)


        dict_log = {}
        
        # plot with plotly
        for i in range(10):
            level_col = columns[i+2]
            fig = px.scatter(
                pd_data, 
                x='x', 
                y='y', 
                color=level_col, 
                title=f"Tree Level {i}",
            )
            # 原始点固定大小
            fig.update_traces(marker=dict(size=2))

            # 计算各簇中心
            centers = pd_data.groupby(level_col)[['x', 'y']].mean().reset_index()

            # 添加中心点（更大、更醒目）
            fig.add_trace(
                go.Scatter(
                    x=centers['x'],
                    y=centers['y'],
                    mode='markers',
                    name='centers',
                    text=[f'Center {lvl}' for lvl in centers[level_col]],
                    marker=dict(size=10, symbol='x', line=dict(width=1))
                )
            )

            dict_log[f"clustering/tree_level_{i}"] = fig

        trainer.logger.experiment.log(dict_log)


        