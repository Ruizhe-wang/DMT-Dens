export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

CUDA_VISIBLE_DEVICES=0 python main.py validate -c conf/difftree/ckpt/G_fmnist_1gpu.yaml --seed_everything 41 &

sleep 300

CUDA_VISIBLE_DEVICES=0 python main.py validate -c conf/difftree/ckpt/G_fmnist_1gpu.yaml --seed_everything 40 &

