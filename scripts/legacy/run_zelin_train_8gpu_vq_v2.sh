export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

# python main.py fit -c conf/mnistsmall.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py fit -c conf/mnistsmall.yaml
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py fit -c conf/mnist_4gpu.yaml
python main.py fit -c conf/vq_mnist_8gpu_v2.yaml