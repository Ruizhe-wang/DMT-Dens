#!/bin/bash 
#! this code is used to run wandb agent on specified GPUs and CPUs, zelin 2024-2-16

# wandb login --host=https://api.wandb.ai 1bfeca96066bdd13c317111756395d792b64ce56
wandb login --relogin --host=http://www.zangzelin.fun:4080 local-04561b3685afca039aca56f554efe6a008118c01
# alias wandb='/mnt/workspace/workgroup/zangzelin.zzl/conda/envs/torch2/bin/wandb'
# alias python='/mnt/workspace/workgroup/zangzelin.zzl/conda/envs/torch2/bin/python3'

export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

# wandb

# Check if at least three arguments were provided (script name, agent ID, and GPU list are always present)
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <agent_id> <gpu_list> <cpus_per_program>"
    echo "Example: $0 AGENT_ID '0,1,2' 2"
    exit 1
fi

# The first argument is the agent ID
AGENT_ID=$1

# The second argument is the list of GPUs
GPU_LIST=$2

# The third argument is the number of CPUs per program
CPUS_PER_PROGRAM=$3

# Split the GPU list into an array using comma as a delimiter
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

# Initialize the starting CPU ID
CPU_START=0

# Launch wandb agents on specified CUDA devices and CPUs
for GPU_ID in "${GPUS[@]}"
do
    echo "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
    echo "Starting wandb agent $AGENT_ID on GPU $GPU_ID with $CPUS_PER_PROGRAM CPUs"
    echo "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
    # Calculate the CPU end index based on the number of CPUs per program
    let CPU_END=CPU_START+CPUS_PER_PROGRAM-1

    # Use taskset with a range of CPUs
    CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CPU_START-$CPU_END wandb agent $AGENT_ID &

    # Update the starting CPU ID for the next program
    let CPU_START+=CPUS_PER_PROGRAM

    sleep 30
done

# Wait for all background processes to finish
wait


python main.py fit --config=model/tran_nnet_exp_cond_pacmap.py --data.init_args.batch_size=500 --model.T_attention_probs_dropout_prob=0 --model.T_hidden_dropout_prob=0 --model.T_hidden_size=784 
# --model.T_intermediate_size=300 --model.T_num_attention_heads=14 --model.init_args.T_num_layers=4 --model.init_args.w_fp=1 --model.init_args.w_nb=1 --model.init_args.weight_mse=0 --model.init_args.weight_nepo=0.5 --model.lr=0.001