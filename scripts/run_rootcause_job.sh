#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <config.yaml> <gpu-index> <log-path> [checkpoint-path]" >&2
  exit 2
fi

config_path="$1"
gpu_index="$2"
log_path="$3"
checkpoint_path="${4:-}"

mkdir -p "$(dirname "$log_path")"
export WANDB_API_KEY="$(grep -oP 'export WANDB_API_KEY=\K.*' scripts/run_wandb_multidata.sh | head -1)"
export WANDB_BASE_URL="http://www.zangzelin.fun:4080"

fit_args=(fit -c "$config_path")
if [ -n "$checkpoint_path" ]; then
  fit_args+=(--ckpt_path "$checkpoint_path")
fi

CUDA_VISIBLE_DEVICES="$gpu_index" \
  python main.py "${fit_args[@]}" 2>&1 | tee -a "$log_path"
