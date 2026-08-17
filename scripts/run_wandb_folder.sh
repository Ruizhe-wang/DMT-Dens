#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Usage instruction
usage() {
    echo "Usage: $0 <yaml_directory> <gpu_list> <cpus_per_program>"
    echo "Example: $0 configs/dmtme_dataset '0,1,2,3' 6"
    exit 1
}

# Check if correct number of arguments are provided
if [ "$#" -lt 3 ]; then
    usage
fi

YAML_DIR=$1
GPU_LIST=$2
CPUS_PER_PROGRAM=$3

# Check if the given directory exists
if [ ! -d "$YAML_DIR" ]; then
    echo "Error: Directory '$YAML_DIR' does not exist."
    exit 1
fi

# Authenticate and set up wandb environment variables
wandb login --relogin --host=http://www.zangzelin.fun:4080 local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

# Get the directory where this script is located so we can call run_wandb.sh relative to it
SCRIPT_DIR=$(dirname "$0")

# Loop over all .yaml files in the specified directory
for yaml_file in "$YAML_DIR"/*.yaml; do
    # Ensure there are actually yaml files matching the pattern
    if [ ! -f "$yaml_file" ]; then
        echo "No .yaml files found in '$YAML_DIR'."
        exit 0
    fi

    echo "==================================================================="
    echo "Processing yaml file: $yaml_file"
    echo "==================================================================="
    
    # Run wandb sweep and capture the raw output
    output=$(wandb sweep "$yaml_file" 2>&1)
    
    echo "------------------- wandb sweep raw output BEGIN ------------------"
    echo "$output"
    echo "------------------- wandb sweep raw output END   ------------------"
    
    # Extract the sweep_id from the output
    sweep_id=$(echo "$output" | grep -oP '(?<=wandb agent ).*')
    
    if [ -z "$sweep_id" ]; then
        echo "Error: Failed to parse sweep_id from wandb output for $yaml_file"
        continue
    fi
    
    echo "Parsed sweep_id: '$sweep_id'"
    echo '==================================================================='
    
    # Run the wandb agent using the existing run_wandb.sh script
    bash "$SCRIPT_DIR/run_wandb.sh" "$sweep_id" "$GPU_LIST" "$CPUS_PER_PROGRAM"
    
    echo '==================================================================='
    echo "Completed processing and running sweep $sweep_id for $yaml_file"
    echo '==================================================================='
done

echo "All YAML files in '$YAML_DIR' have been processed."
