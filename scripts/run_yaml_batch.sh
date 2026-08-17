#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash/run_yaml_batch.sh <yaml_path> <gpu_list> <cpus_per_job> [cli_command] [-- extra args]

Arguments:
  yaml_path       Single yaml file or a directory that contains *.yaml
  gpu_list        Comma-separated GPU ids, for example: 0,1,3
  cpus_per_job    Number of CPU cores bound to each launched process
  cli_command     Lightning CLI subcommand, default: fit

Examples:
  bash bash/run_yaml_batch.sh conf/consistency_eval 0,1 8
  bash bash/run_yaml_batch.sh conf/consistency_eval/demo.yaml 2 6 fit
  bash bash/run_yaml_batch.sh conf/dr_nn 0,1,2 4 fit -- --trainer.max_epochs 100
EOF
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

YAML_PATH=$1
GPU_LIST=$2
CPUS_PER_JOB=$3
CLI_COMMAND=${4:-fit}

shift 3
if [ "$#" -gt 0 ]; then
    shift 1 || true
fi

EXTRA_ARGS=()
if [ "$#" -gt 0 ]; then
    if [ "$1" = "--" ]; then
        shift
    fi
    EXTRA_ARGS=("$@")
fi

if ! [[ "$CPUS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "cpus_per_job must be a positive integer: $CPUS_PER_JOB" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "gpu_list must contain at least one GPU id" >&2
    exit 1
fi

YAML_FILES=()
if [ -d "$YAML_PATH" ]; then
    while IFS= read -r yaml_file; do
        YAML_FILES+=("$yaml_file")
    done < <(find "$YAML_PATH" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)
elif [ -f "$YAML_PATH" ]; then
    YAML_FILES+=("$YAML_PATH")
else
    echo "yaml_path does not exist: $YAML_PATH" >&2
    exit 1
fi

if [ "${#YAML_FILES[@]}" -eq 0 ]; then
    echo "No yaml files found under: $YAML_PATH" >&2
    exit 1
fi

run_job() {
    local yaml_file=$1
    local gpu_id=$2
    local cpu_start=$3
    local cpu_end=$4

    echo "==================================================================="
    echo "Launching: $yaml_file"
    echo "GPU: $gpu_id | CPUs: $cpu_start-$cpu_end | Command: $CLI_COMMAND"
    echo "==================================================================="

    CUDA_VISIBLE_DEVICES="$gpu_id" \
    OMP_NUM_THREADS="$CPUS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$CPUS_PER_JOB" \
    MKL_NUM_THREADS="$CPUS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$CPUS_PER_JOB" \
    taskset -c "$cpu_start-$cpu_end" \
    python main.py "$CLI_COMMAND" -c "$yaml_file" "${EXTRA_ARGS[@]}"
}

declare -a SLOT_PIDS=()
declare -a SLOT_YAMLS=()
declare -a SLOT_GPUS=()

for idx in "${!GPUS[@]}"; do
    SLOT_PIDS[$idx]=""
    SLOT_YAMLS[$idx]=""
    SLOT_GPUS[$idx]="${GPUS[$idx]}"
done

active_jobs=0
yaml_index=0
total_yamls=${#YAML_FILES[@]}

while [ "$yaml_index" -lt "$total_yamls" ] || [ "$active_jobs" -gt 0 ]; do
    for slot in "${!GPUS[@]}"; do
        pid=${SLOT_PIDS[$slot]}
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            echo "Finished: ${SLOT_YAMLS[$slot]}"
            SLOT_PIDS[$slot]=""
            SLOT_YAMLS[$slot]=""
            active_jobs=$((active_jobs - 1))
        fi
    done

    for slot in "${!GPUS[@]}"; do
        if [ "$yaml_index" -ge "$total_yamls" ]; then
            break
        fi

        if [ -z "${SLOT_PIDS[$slot]}" ]; then
            cpu_start=$((slot * CPUS_PER_JOB))
            cpu_end=$((cpu_start + CPUS_PER_JOB - 1))
            yaml_file=${YAML_FILES[$yaml_index]}
            gpu_id=${SLOT_GPUS[$slot]}

            run_job "$yaml_file" "$gpu_id" "$cpu_start" "$cpu_end" &
            SLOT_PIDS[$slot]=$!
            SLOT_YAMLS[$slot]="$yaml_file"
            active_jobs=$((active_jobs + 1))
            yaml_index=$((yaml_index + 1))

            sleep 2
        fi
    done

    if [ "$active_jobs" -gt 0 ]; then
        sleep 5
    fi
done

echo "All yaml jobs finished."
