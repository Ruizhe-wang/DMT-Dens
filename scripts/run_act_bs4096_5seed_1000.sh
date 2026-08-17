#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0,1,2,3,4}"
MODE="${2:-train}"
MAX_BASELINE_MB="${MAX_BASELINE_MB:-1500}"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"

if [[ "${#GPUS[@]}" -ne 5 ]]; then
  echo "Expected exactly five comma-separated GPU IDs, got: ${GPU_CSV}" >&2
  exit 2
fi
if [[ "${MODE}" != "train" && "${MODE}" != "dry-run" ]]; then
  echo "Mode must be 'train' or 'dry-run', got: ${MODE}" >&2
  exit 2
fi
if [[ "${MODE}" == "train" && (-z "${WANDB_API_KEY:-}" || -z "${WANDB_BASE_URL:-}") ]]; then
  echo "WANDB_API_KEY and WANDB_BASE_URL must be set" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_DIR="configs/encoder_bench/act_bs4096_5seed_1000/runs"
LOG_DIR="logs/encoder_tuning/ACT_bs4096_5seed_1000_20260814/${MODE}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [[ "${MODE}" == "dry-run" ]]; then
  EXTRA_ARGS=(--trainer.fast_dev_run 2)
fi

SEEDS=(42 43 44 45 46)

for gpu in "${GPUS[@]}"; do
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${used}" -gt "${MAX_BASELINE_MB}" ]]; then
    echo "gpu_busy gpu=${gpu} memory_used_mb=${used} limit_mb=${MAX_BASELINE_MB}" >&2
    exit 3
  fi
done

echo "commit=$(git rev-parse HEAD)"
echo "mode=${MODE}"
echo "wandb_base_url=${WANDB_BASE_URL:-disabled}"
echo "gpus=${GPU_CSV}"
echo "batch_size=4096"
echo "paper_plot_callback=enabled"

pids=()
names=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[${index}]}"
  gpu="${GPUS[${index}]}"
  name="act_latent_bn_bs4096_seed${seed}_c90_20260814"
  config="${CONFIG_DIR}/${name}.yaml"
  log="${LOG_DIR}/${name}.log"
  echo "launch seed=${seed} gpu=${gpu} config=${config}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py fit -c "${config}" "${EXTRA_ARGS[@]}" \
    > >(tee "${log}") 2>&1 &
  pids+=("$!")
  names+=("${name}")
done

total_failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "completed run=${names[${index}]}"
  else
    status=$?
    echo "failed run=${names[${index}]} exit=${status}" >&2
    total_failed=$((total_failed + 1))
  fi
done

echo "all_act_bs4096_runs_processed=true total_failed=${total_failed}"
if [[ "${total_failed}" -ne 0 ]]; then
  exit 1
fi
