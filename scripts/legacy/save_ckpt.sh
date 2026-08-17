scp zzl_checkpoints/best_model_epoch999_dral_4gpu_acc0.8313499999999999.pth home_root:/volume1/zelin_big/cellTree/ckpt/

scp -r save_output/Limb_seed_8c1a1513-3_epoch_0 home_root:/volume1/zelin_big/cellTree/

scp home_root:/volume1/zelin_big/cellTree/ckpt/best_model_epoch999_dral_4gpu_acc0.8313499999999999.pth  zzl_checkpoints/


scp -r save_output/DARLINAll_seed_4b939902-f_epoch_0 home_root:/volume1/zelin_big/cellTree/

scp -r home_root:/volume1/zelin_big/cellTree/data/celegan ./