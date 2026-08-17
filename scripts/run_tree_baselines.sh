#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    bash scripts/run_tree_baselines.sh [options] [-- extra Lightning args]

Options:
    --methods all|tsne,umap,pacmap  Methods to run. Default: all
    --jobs N                        Parallel jobs. Default: 5
    --python PATH                   Python executable. Default: python
    --config-root DIR               Config root. Default: configs/dmtme_dataset_baselines
  -h, --help                      Show this help message

Examples:
  bash scripts/run_tree_baselines.sh
  bash scripts/run_tree_baselines.sh --methods tsne,umap,pacmap
  bash scripts/run_tree_baselines.sh --methods tsne -- --trainer.max_epochs 1
EOF
}

METHODS="all"
JOBS=5
PYTHON_BIN="python"
CONFIG_ROOT="configs/dmtme_dataset_baselines"
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --methods)
            METHODS="$2"
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
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "jobs must be a positive integer: $JOBS" >&2
    exit 1
fi

if [ "$METHODS" = "all" ] || [ -z "$METHODS" ]; then
    METHOD_LIST=(tsne umap pacmap densmap phate denssne)
else
    IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
fi

CONFIGS=()

for raw_method in "${METHOD_LIST[@]}"; do
    method=$(echo "$raw_method" | tr '[:upper:]' '[:lower:]' | xargs)
    [ -z "$method" ] && continue

    config_path="$CONFIG_ROOT/$method/tree.yaml"
    if [ ! -f "$config_path" ]; then
        echo "Warning: missing tree config for '$method': $config_path" >&2
        continue
    fi

    CONFIGS+=("$config_path")
done

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "No tree baseline configs were found under $CONFIG_ROOT." >&2
    exit 1
fi

echo "=========================================================="
echo "Tree baseline runner"
echo "Repo root: $REPO_ROOT"
echo "Config root: $CONFIG_ROOT"
echo "Methods: ${METHODS}"
echo "Jobs: $JOBS"
echo "Python: $PYTHON_BIN"
echo "Configs: ${#CONFIGS[@]}"
echo "=========================================================="

mkdir -p outputs/logs/tree_baselines

run_one() {
    local config_path=$1
    local method_name
    method_name=$(basename "$(dirname "$config_path")")
    local log_path="outputs/logs/tree_baselines/${method_name}_tree.log"

    echo "----------------------------------------------------------"
    echo "Running $config_path"
    echo "Log: $log_path"
    echo "----------------------------------------------------------"

    "$PYTHON_BIN" main.py fit -c "$config_path" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log_path"
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
echo "Finished running ${#CONFIGS[@]} tree baseline configs"
echo "Failed jobs: $failed"
echo "Logs: outputs/logs/tree_baselines"
echo "=========================================================="

if [ "$failed" -gt 0 ]; then
    exit 1
fi