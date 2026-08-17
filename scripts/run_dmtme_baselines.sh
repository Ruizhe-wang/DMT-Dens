#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_dmtme_baselines.sh [options]

Options:
  --methods tsne,umap     Methods to run. Default: tsne,umap
  --datasets a,b,c        Optional dataset stems to run, e.g. ACT,MCA,gast10k
  --jobs N                Parallel jobs. Default: 1
  --python PATH           Python executable. Default: python
  --source-dir DIR        Source config dir. Default: configs/dmtme_dataset
  --output-dir DIR        Generated baseline config dir. Default: configs/dmtme_dataset_baselines
  --omp-threads N         OMP/BLAS threads per job. Default: 8

Examples:
  bash scripts/run_dmtme_baselines.sh
  bash scripts/run_dmtme_baselines.sh --methods tsne --datasets ACT,MCA --jobs 2
  bash scripts/run_dmtme_baselines.sh --python /opt/miniforge3/envs/mldr/bin/python --jobs 4
EOF
}

METHODS="tsne,umap"
DATASETS=""
JOBS=1
PYTHON_BIN="python"
SOURCE_DIR="configs/dmtme_dataset"
OUTPUT_DIR="configs/dmtme_dataset_baselines"
OMP_THREADS=8

while [ "$#" -gt 0 ]; do
    case "$1" in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --omp-threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
DATASET_ARGS=()
if [ -n "$DATASETS" ]; then
    IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
    DATASET_ARGS=(--datasets "${DATASET_LIST[@]}")
fi

echo "=========================================================="
echo "Generating dmtme baseline configs"
echo "Source dir: $SOURCE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Methods: $METHODS"
echo "Datasets: ${DATASETS:-all}"
echo "=========================================================="

"$PYTHON_BIN" scripts/generate_dmtme_baselines.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --methods "${METHOD_LIST[@]}" \
    "${DATASET_ARGS[@]}"

CONFIGS=()
for method in "${METHOD_LIST[@]}"; do
    method_dir="$OUTPUT_DIR/$method"
    if [ ! -d "$method_dir" ]; then
        continue
    fi
    while IFS= read -r config_path; do
        CONFIGS+=("$config_path")
    done < <(find "$method_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
done

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "No generated baseline configs found." >&2
    exit 1
fi

mkdir -p outputs/logs/dmtme_baselines

run_one() {
    local config_path=$1
    local config_name
    config_name=$(basename "${config_path%.yaml}")
    local log_path="outputs/logs/dmtme_baselines/${config_name}.log"

    echo "----------------------------------------------------------"
    echo "Running $config_path"
    echo "Log: $log_path"
    echo "----------------------------------------------------------"

    (
        export CUDA_VISIBLE_DEVICES=""
        export OMP_NUM_THREADS="$OMP_THREADS"
        export OPENBLAS_NUM_THREADS="$OMP_THREADS"
        export MKL_NUM_THREADS="$OMP_THREADS"
        export NUMEXPR_NUM_THREADS="$OMP_THREADS"
        "$PYTHON_BIN" main.py fit -c "$config_path"
    ) 2>&1 | tee "$log_path"
}

active_jobs=0
failed=0

for config_path in "${CONFIGS[@]}"; do
    run_one "$config_path" &
    active_jobs=$((active_jobs + 1))

    if [ "$active_jobs" -ge "$JOBS" ]; then
        if ! wait -n; then
            failed=$((failed + 1))
        fi
        active_jobs=$((active_jobs - 1))
    fi
done

while [ "$active_jobs" -gt 0 ]; do
    if ! wait -n; then
        failed=$((failed + 1))
    fi
    active_jobs=$((active_jobs - 1))
done

echo "=========================================================="
echo "Finished running ${#CONFIGS[@]} baseline configs"
echo "Failed jobs: $failed"
echo "=========================================================="

if [ "$failed" -gt 0 ]; then
    exit 1
fi
