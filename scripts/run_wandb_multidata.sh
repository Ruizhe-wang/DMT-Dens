#!/bin/bash
wandb login --relogin --host=http://www.zangzelin.fun:4080 local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080
# 定义你的 yaml 文件所在的目录
YAML_DIR="configs/dmtme_dataset_weighted/sweep/run_celegan_baseline_structural"  # 将这里替换为你的 yaml 文件夹路径（repo 相对路径，从仓库根目录运行）

# 遍历目录下的所有 yaml 文件
for yaml_file in "$YAML_DIR"/*.yaml; do
    echo "==================================================================="
    echo "Processing yaml file: $yaml_file"
    echo "==================================================================="
    # 获取 sweep ID
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oE "wandb [^ ]+" | awk '{print $2}')
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oP "Run sweep agent with: wandb agent \K[^ ]+")
    # sweep_id=$(wandb sweep "$yaml_file" | grep -o "wandb agent [^ ]*" | awk '{print $3}')
    # sweep_id=$(wandb sweep "$yaml_file" | grep -oP "wandb agent \K.*")
    # sweep_id=$(wandb sweep "$yaml_file" | sed -n 's/.*wandb agent //p')
    # output=$(wandb sweep sweep/nml5_val/ACT.yaml 2>&1)
    output=$(wandb sweep "$yaml_file" 2>&1)
    echo "------------------- wandb sweep raw output BEGIN ------------------"
    echo "$output"
    echo "------------------- wandb sweep raw output END   ------------------"
    sweep_id=$(echo "$output" | grep -oP '(?<=wandb agent ).*')
    echo "Parsed sweep_id: '$sweep_id'"


    # 启动 wandb agent
    # wandb agent "$sweep_id" &
    echo '=================='
    echo "$sweep_id"

    # bash "run_wandb.sh $sweep_id 0,1,2,3 4"
    bash run_wandb.sh "$sweep_id" 0,1,2,3,4,5,6,7 6
    # echo '=================='

    # sleep 2m
    echo '=================='
    echo '=================='
    response=$(curl -s -H "Authorization: Bearer $API_KEY" \
  "https://api.wandb.ai/sweeps/$sweep_id")
    echo "$sweep_id"
    echo '=================='
    echo '=================='

    
    # 让 wandb agent 以后台进程的形式运行
    echo "Started wandb agent for sweep $sweep_id"
done
