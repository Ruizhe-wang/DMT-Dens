#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0,1,3,4,5}"
MODE="${2:-train}"
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
  echo "WANDB_API_KEY and WANDB_BASE_URL must be set in the runtime environment" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_DIR="configs/encoder_bench/ablation_single_factor_loo_1000/runs"
LOG_DIR="logs/ablation/single_factor_loo_1000_20260811/${MODE}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [[ "${MODE}" == "dry-run" ]]; then
  EXTRA_ARGS=(--trainer.fast_dev_run 2)
fi

DATASETS=(hcl epi mnist)
VARIANTS=(
  r1_distance
  r2_unidirectional
  r3_allpair
  r4_single_density
  r5_no_density
)

echo "commit=$(git rev-parse HEAD)"
echo "mode=${MODE}"
echo "wandb_base_url=${WANDB_BASE_URL:-disabled}"
echo "gpus=${GPU_CSV}"

for dataset in "${DATASETS[@]}"; do
  echo "starting_dataset=${dataset}"
  pids=()
  names=()

  for index in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[${index}]}"
    gpu="${GPUS[${index}]}"
    name="singlefactor_loo1000_${dataset}_${variant}_seed42_20260811"
    config="${CONFIG_DIR}/${name}.yaml"
    log="${LOG_DIR}/${name}.log"

    echo "launch dataset=${dataset} variant=${variant} gpu=${gpu} config=${config}"
    CUDA_VISIBLE_DEVICES="${gpu}" python main.py fit -c "${config}" "${EXTRA_ARGS[@]}" \
      > >(tee "${log}") 2>&1 &
    pids+=("$!")
    names+=("${name}")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
      echo "completed run=${names[${index}]}"
    else
      status=$?
      echo "failed run=${names[${index}]} exit=${status}" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "dataset_group_failed=${dataset}; stopping before the next dataset" >&2
    exit 1
  fi
  echo "completed_dataset=${dataset}"
done

echo "all_single_factor_runs_completed=true"
