#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_dmtme_new_baselines.sh [options]

Options:
  --methods densmap,phate,denssne   Methods to run. Default: densmap,phate,denssne
  --datasets a,b,c                  Optional dataset stems to run, e.g. ACT,MCA,gast10k
  --jobs N                          Parallel jobs. Default: 1
  --python PATH                     Python executable. Default: python
  --config-root DIR                 Config root. Default: configs/dmtme_dataset_baselines
  --omp-threads N                   OMP/BLAS threads per job. Default: 8
  --skip-missing-deps               Skip configs whose Python dependency is missing

Examples:
  bash scripts/run_dmtme_new_baselines.sh
  bash scripts/run_dmtme_new_baselines.sh --methods densmap,phate --datasets ACT,MCA --jobs 2
  bash scripts/run_dmtme_new_baselines.sh --python /opt/miniforge3/envs/mldr/bin/python --jobs 3
EOF
}

METHODS="densmap,phate,denssne"
DATASETS=""
JOBS=1
PYTHON_BIN="python"
CONFIG_ROOT="configs/dmtme_dataset_baselines"
OMP_THREADS=8
SKIP_MISSING_DEPS=0

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
        --config-root)
            CONFIG_ROOT="$2"
            shift 2
            ;;
        --omp-threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        --skip-missing-deps)
            SKIP_MISSING_DEPS=1
            shift
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

declare -A DATASET_FILTER=()
if [ -n "$DATASETS" ]; then
    IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
    for dataset in "${DATASET_LIST[@]}"; do
        key=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
        DATASET_FILTER["$key"]=1
    done
fi

dependency_for_method() {
    case "$1" in
        densmap)
            echo "umap"
            ;;
        phate)
            echo "phate"
            ;;
        denssne)
            echo ""
            ;;
        *)
            echo ""
            ;;
    esac
}

check_python_dependency() {
    local module_name="$1"
    if [ -z "$module_name" ]; then
        return 0
    fi

    "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module_name') else 1)"
}

CONFIGS=()
SKIPPED_CONFIGS=()

for method in "${METHOD_LIST[@]}"; do
    method_dir="$CONFIG_ROOT/$method"
    if [ ! -d "$method_dir" ]; then
        echo "Warning: method directory not found: $method_dir" >&2
        continue
    fi

    dep_module=$(dependency_for_method "$method")
    if ! check_python_dependency "$dep_module"; then
        if [ "$SKIP_MISSING_DEPS" -eq 1 ]; then
            echo "Skipping method '$method' because dependency '$dep_module' is unavailable."
            continue
        fi
        echo "Missing Python dependency '$dep_module' required by method '$method'." >&2
        echo "Rerun with --skip-missing-deps to skip it." >&2
        exit 1
    fi

    while IFS= read -r config_path; do
        dataset_name=$(basename "${config_path%.yaml}")
        dataset_key=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')

        if [ "${#DATASET_FILTER[@]}" -gt 0 ] && [ -z "${DATASET_FILTER[$dataset_key]:-}" ]; then
            continue
        fi

        CONFIGS+=("$config_path")
    done < <(find "$method_dir" -maxdepth 1 -type f -name '*.yaml' | sort)
done

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "No matching baseline configs found under $CONFIG_ROOT." >&2
    exit 1
fi

mkdir -p outputs/logs/dmtme_new_baselines

echo "=========================================================="
echo "Running new DMTME baselines"
echo "Config root: $CONFIG_ROOT"
echo "Methods: $METHODS"
echo "Datasets: ${DATASETS:-all}"
echo "Jobs: $JOBS"
echo "Python: $PYTHON_BIN"
echo "OMP threads/job: $OMP_THREADS"
echo "Configs: ${#CONFIGS[@]}"
echo "=========================================================="

run_one() {
    local config_path=$1
    local method_name
    method_name=$(basename "$(dirname "$config_path")")
    local config_name
    config_name=$(basename "${config_path%.yaml}")
    local log_path="outputs/logs/dmtme_new_baselines/${method_name}_${config_name}.log"

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
echo "Finished running ${#CONFIGS[@]} configs"
echo "Failed jobs: $failed"
echo "Logs: outputs/logs/dmtme_new_baselines"
echo "=========================================================="

if [ "$failed" -gt 0 ]; then
    exit 1
fi
