export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

# python main.py fit -c conf/mnistsmall.yaml
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.2
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.1
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.05
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.02
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.01
CUDA_VISIBLE_DEVICES=5 python main.py fit -c conf/difftree/G_xinli_1gpu.yaml --model.nu_lat 0.005