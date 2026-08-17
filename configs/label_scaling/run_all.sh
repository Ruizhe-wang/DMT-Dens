#!/bin/bash
# =============================================================================
# Label Scaling Consistency Experiment — Multi-GPU Parallel
#
# All methods (DiffTree/t-SNE/UMAP) share the same parallelism limit (MAX_PARALLEL).
# DiffTree: round-robin across GPUs
# Baselines: CPU
#
# Usage:
#   bash conf/label_scaling/run_all.sh "0,1,2"              # All methods
#   bash conf/label_scaling/run_all.sh "0,1,2" difftree      # DiffTree only
#   bash conf/label_scaling/run_all.sh "0"     baselines     # Baselines only
# =============================================================================

set -e

GPU_LIST=${1:?"Usage: $0 <gpu_list> [difftree|baselines|all]"}
METHOD=${2:-all}
MAX_PARALLEL=8

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

CONF_DIR="conf/label_scaling"

TAGS=(
    # --- size 2: visually similar pairs ---
    s2_1_7  s2_4_9
    # --- size 3: thin/angular vs curvy ---
    s3_1_4_7  s3_3_5_8
    # --- size 4: mixed ---
    s4_0_1_4_7  s4_0_6_8_9
    # --- size 5: contiguous vs confusable cluster ---
    s5_01234  s5_34589
    # --- size 6: separable (diverse archetypes) vs confusable (all confusion clusters) ---
    s6_012467  s6_134589
    # --- size 7: mixed vs confusable-heavy ---
    s7_0134579  s7_1345689
    # --- size 8: drop confusable (5,8) vs drop distinctive (0,2) ---
    s8_01234679  s8_13456789
    # --- size 9: drop most-confusable (8) vs drop most-distinctive (0) ---
    s9_012345679  s9_123456789
    # --- size 10: full ---
    s10_all
)

# Helper: wait for any one background job to finish when we hit the limit
wait_for_slot() {
    while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
        # Wait for any single child to finish
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid" || echo "  WARNING: PID $pid failed"
            fi
        done
        if (( ${#new_pids[@]} == ${#PIDS[@]} )); then
            # All still running, wait briefly
            sleep 1
        fi
        PIDS=("${new_pids[@]}")
    done
}

wait_all() {
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  WARNING: PID $pid failed"
    done
    PIDS=()
}

run_baselines() {
    echo "========== Running baselines (CPU, max ${MAX_PARALLEL} parallel) =========="
    PIDS=()
    for TAG in "${TAGS[@]}"; do
        wait_for_slot
        echo "  t-SNE ${TAG}"
        python main.py fit -c ${CONF_DIR}/tsne_${TAG}.yaml &
        PIDS+=($!)

        wait_for_slot
        echo "  UMAP  ${TAG}"
        python main.py fit -c ${CONF_DIR}/umap_${TAG}.yaml &
        PIDS+=($!)
    done
    wait_all
    echo "========== Baselines complete =========="
}

run_difftree() {
    echo "========== Running ${#TAGS[@]} DiffTree configs (max ${MAX_PARALLEL} parallel, ${NUM_GPUS} GPUs) =========="
    PIDS=()
    local gpu_idx=0
    for TAG in "${TAGS[@]}"; do
        wait_for_slot
        GPU_ID=${GPUS[$((gpu_idx % NUM_GPUS))]}
        echo "  [GPU ${GPU_ID}] diff_${TAG}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py fit -c ${CONF_DIR}/diff_${TAG}.yaml &
        PIDS+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
    wait_all
    echo "========== DiffTree complete =========="
}

case "$METHOD" in
    difftree)  run_difftree ;;
    baselines) run_baselines ;;
    all)
        run_baselines &
        PID_BL=$!
        run_difftree
        wait $PID_BL
        ;;
    *)
        echo "Unknown: $METHOD. Use: difftree|baselines|all"
        exit 1
        ;;
esac

echo "========== All experiments complete =========="
