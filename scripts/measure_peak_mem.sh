#!/usr/bin/env bash
# Measure per-dataset TRAINING peak GPU memory for TopoBranch (DiffTreeVQ_density).
#
# Peak GPU memory is hit on the first training steps (it is batch-bounded), so we
# run each dataset's *_full config for only a few training batches and read
# torch.cuda.max_memory_allocated via callbacks.peakmem_print.PeakMemPrint.
# We REPLACE the config's callbacks with just PeakMemPrint, which drops the
# ModelCheckpoint / visualization / eval callbacks (faster, no checkpoint
# conflict, clean training-only peak), and disable the wandb logger.
#
# Usage:
#   bash scripts/measure_peak_mem.sh [GPU_ID] [LIMIT_TRAIN_BATCHES]
# Example:
#   bash scripts/measure_peak_mem.sh 7 15
#
# Output:
#   results/peak_mem/peak_mem_summary.csv   (dataset,peak_mem_mb,exit)
#   results/peak_mem/<dataset>.log          (full run log per dataset)
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dmt
cd /usr/storage/ruizhe/mldr_rz
export WANDB_MODE=offline

GPU="${1:-7}"
LBATCH="${2:-15}"
OUTDIR=results/peak_mem
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/peak_mem_summary.csv"
echo "dataset,peak_mem_mb,exit" > "$SUMMARY"

DATASETS=(act aqc emnist epi gast10k hcl mca ng20 tree)

for ds in "${DATASETS[@]}"; do
  cfg="configs/ablation/component/${ds}_full.yaml"
  log="$OUTDIR/${ds}.log"
  echo "=================== ${ds} ==================="
  CUDA_VISIBLE_DEVICES="$GPU" timeout 1200 python main.py fit -c "$cfg" \
    --trainer.max_epochs=1 \
    --trainer.limit_train_batches="$LBATCH" \
    --trainer.limit_val_batches=0 \
    --trainer.num_sanity_val_steps=0 \
    --trainer.enable_checkpointing=false \
    --trainer.logger=false \
    --trainer.callbacks=callbacks.peakmem_print.PeakMemPrint \
    > "$log" 2>&1
  ec=$?
  peak=$(grep -oE 'PEAKMEM_MB [0-9.]+' "$log" | tail -1 | awk '{print $2}')
  [ -z "$peak" ] && peak="NA"
  echo "${ds},${peak},${ec}" >> "$SUMMARY"
  echo "${ds} -> peak=${peak} MB exit=${ec}"
  if [ "$peak" = "NA" ]; then
    echo "  error tail:"
    grep -iE 'Error|Traceback|out of memory|No such file|KeyError|FileNotFound|Misconfiguration' "$log" | tail -4
  fi
done

echo "=================== SUMMARY ==================="
cat "$SUMMARY"
echo "DONE_MEASURE_PEAK"
