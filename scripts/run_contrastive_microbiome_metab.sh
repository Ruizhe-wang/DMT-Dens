#!/bin/bash
# 微生物 - 代谢组跨模态对比学习训练脚本
# 用法：bash scripts/run_contrastive_microbiome_metab.sh

set -e

# ==================== 配置参数 ====================

# 数据路径 (根据实际情况修改)
TRAIN_DATA="${TRAIN_DATA:-./data/microbiome_metab_train.pt}"
EVAL_DATA="${EVAL_DATA:-./data/microbiome_metab_eval.pt}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/contrastive_microbiome_metab}"

# 预训练 RoBERTa 模型路径 (根据实际情况修改)
ROBERTA_PATH="${ROBERTA_PATH:-./pretrained_roberta}"

# 模型参数
METAB_INPUT_DIM="${METAB_INPUT_DIM:-128}"   # 代谢数据特征维度
PROJ_DIM="${PROJ_DIM:-256}"                 # 投影维度
FREEZE_LAYERS="${FREEZE_LAYERS:-8}"         # 冻结 RoBERTa 前 N 层
TEMPERATURE="${TEMPERATURE:-0.07}"          # 对比学习温度

# 训练参数
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
METAB_NOISE_STD="${METAB_NOISE_STD:-0.01}"

# 评估与保存
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SEED="${SEED:-42}"

# 其他
FP16="${FP16:---fp16}"
NUM_GPUS="${NUM_GPUS:-1}"

# ==================== 打印配置 ====================

echo "=============================================="
echo "Microbiome-Metabolite Contrastive Learning"
echo "=============================================="
echo "Train Data: $TRAIN_DATA"
echo "Eval Data: $EVAL_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "RoBERTa Path: $ROBERTA_PATH"
echo "Metab Input Dim: $METAB_INPUT_DIM"
echo "Proj Dim: $PROJ_DIM"
echo "Freeze Layers: $FREEZE_LAYERS"
echo "Temperature: $TEMPERATURE"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Num Epochs: $NUM_EPOCHS"
echo "Num GPUs: $NUM_GPUS"
echo "=============================================="

# ==================== 创建输出目录 ====================

mkdir -p "$OUTPUT_DIR"

# ==================== 运行训练 ====================

cd "$(dirname "$0")/.."  # 进入项目根目录

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running with $NUM_GPUS GPUs (DDP mode)..."
    torchrun --nproc_per_node=$NUM_GPUS scripts/train_contrastive_microbiome_metab.py \
        --train-data "$TRAIN_DATA" \
        --eval-data "$EVAL_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --roberta-path "$ROBERTA_PATH" \
        --metab-input-dim "$METAB_INPUT_DIM" \
        --proj-dim "$PROJ_DIM" \
        --freeze-layers "$FREEZE_LAYERS" \
        --temperature "$TEMPERATURE" \
        --max-length "$MAX_LENGTH" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --warmup-ratio "$WARMUP_RATIO" \
        --metab-noise-std "$METAB_NOISE_STD" \
        --eval-steps "$EVAL_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --seed "$SEED" \
        $FP16
else
    echo "Running with single GPU..."
    python scripts/train_contrastive_microbiome_metab.py \
        --train-data "$TRAIN_DATA" \
        --eval-data "$EVAL_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --roberta-path "$ROBERTA_PATH" \
        --metab-input-dim "$METAB_INPUT_DIM" \
        --proj-dim "$PROJ_DIM" \
        --freeze-layers "$FREEZE_LAYERS" \
        --temperature "$TEMPERATURE" \
        --max-length "$MAX_LENGTH" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --warmup-ratio "$WARMUP_RATIO" \
        --metab-noise-std "$METAB_NOISE_STD" \
        --eval-steps "$EVAL_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --seed "$SEED" \
        $FP16
fi

echo "=============================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="
