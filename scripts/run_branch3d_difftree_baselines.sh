#!/usr/bin/env bash
# Train DiffTree on the diverse-density 3D branching dataset, export its 2D
# embedding, then build a comparison figure of DiffTree vs all baselines
# (PCA / t-SNE / densNE / UMAP / densMAP / PHATE) on the SAME points.
#
# DiffTree training needs a GPU; the baseline + comparison step runs on CPU.
# Run from the repo root:  bash scripts/run_branch3d_difftree_baselines.sh
#
# Skip the (GPU) training and only (re)build the baseline comparison:
#   SKIP_DIFFTREE=1 bash scripts/run_branch3d_difftree_baselines.sh
set -euo pipefail

CONFIG=configs/difftree_branch3d.yaml
CKPT=outputs/checkpoints/difftree_branch3d/last.ckpt
EMB_DIR=outputs/embeddings
EMB=${EMB_DIR}/branch3d_topobranch_embeddings.npz
DATASET=data/synthetic_branch3d/branch3d_diverse_density.npz

DIFFTREE_ARG=()

if [[ "${SKIP_DIFFTREE:-0}" != "1" ]]; then
  echo "==> [1/3] Training DiffTree:  ${CONFIG}"
  python main.py fit -c "${CONFIG}"

  echo "==> [2/3] Exporting DiffTree 2D embedding -> ${EMB}"
  python scripts/export_embeddings.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --output "${EMB_DIR}" \
    --dataset-name branch3d \
    --format both
  DIFFTREE_ARG=(--difftree-embedding "${EMB}")
else
  echo "==> SKIP_DIFFTREE=1 : skipping training/export"
  [[ -f "${EMB}" ]] && DIFFTREE_ARG=(--difftree-embedding "${EMB}")
fi

echo "==> [3/3] Running baselines + comparison figure"
python scripts/branch3d_compare.py \
  --dataset "${DATASET}" \
  "${DIFFTREE_ARG[@]}"

echo "Done. See outputs/branch3d/branch3d_baselines_2d.png and _compare_report.txt"
