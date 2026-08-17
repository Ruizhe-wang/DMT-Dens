#!/bin/bash 
#! this code is used to run wandb agent on specified GPUs and CPUs, zelin 2024-2-16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080
# wandb

python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 30 --model.nu_lat 1 --model.nu_emb 1
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 30 --model.nu_lat 0.5 --model.nu_emb 0.5
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 30 --model.nu_lat 0.2 --model.nu_emb 0.2


python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 50 --model.nu_lat 1 --model.nu_emb 1
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 50 --model.nu_lat 0.5 --model.nu_emb 0.5
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 50 --model.nu_lat 0.2 --model.nu_emb 0.2


python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 70 --model.nu_lat 1 --model.nu_emb 1
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 70 --model.nu_lat 0.5 --model.nu_emb 0.5
python main.py fit -c conf/v1/vq_mouse_8gpu.yaml --data.K 70 --model.nu_lat 0.2 --model.nu_emb 0.2
