#!/bin/bash

# 设置环境变量
export WANDB_API_KEY=local-04561b3685afca039aca56f554efe6a008118c01
export WANDB_BASE_URL=http://www.zangzelin.fun:4080

# 创建结果保存目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/step2_ckpt_complete_$TIMESTAMP"
mkdir -p $RESULTS_DIR

echo "=========================================="
echo "🚀 开始运行完整的 step2_ckpt 配置（包含所有可视化）"
echo "开始时间: $(date)"
echo "结果保存目录: $RESULTS_DIR"
echo "=========================================="

# 配置文件列表
CONFIGS=(
    "conf/step2_ckpt/G_xc_1gpu_mnist.yaml"
    "conf/step2_ckpt/G_xc_1gpu_celegan.yaml"
    "conf/step2_ckpt/G_xc_1gpu_weinreb.yaml"
    "conf/step2_ckpt/G_xc_1gpu_Cortex_Var_G.yaml"
    "conf/step2_ckpt/G_xc_1gpu_marrow_Var_G.yaml"
    "conf/step2_ckpt/G_xc_1gpu_huadalai.yaml"
    "conf/step2_ckpt/G_xc_1gpu_jkyliu.yaml"
)

# 主日志文件
MAIN_LOG="$RESULTS_DIR/complete_run_$TIMESTAMP.log"

echo "开始运行所有配置..." | tee -a $MAIN_LOG
echo "开始时间: $(date)" | tee -a $MAIN_LOG

total_configs=${#CONFIGS[@]}
current=0

for config in "${CONFIGS[@]}"; do
    current=$((current + 1))
    config_name=$(basename "$config" .yaml)
    
    echo "" | tee -a $MAIN_LOG
    echo "==========================================" | tee -a $MAIN_LOG
    echo "运行配置 $current/$total_configs: $config_name" | tee -a $MAIN_LOG
    echo "配置文件: $config" | tee -a $MAIN_LOG
    echo "开始时间: $(date)" | tee -a $MAIN_LOG
    echo "==========================================" | tee -a $MAIN_LOG
    
    # 创建该配置的专用结果目录
    config_results_dir="$RESULTS_DIR/$config_name"
    mkdir -p "$config_results_dir"
    
    # 运行训练命令（包含所有可视化callback）
    echo "执行命令: python main.py fit -c $config" | tee -a $MAIN_LOG
    echo "包含所有可视化callback:" | tee -a $MAIN_LOG
    echo "  - 保存模型 (SaveLatestSubmodulesCallback)" | tee -a $MAIN_LOG
    echo "  - 基础可视化 (VisualizationCallback)" | tee -a $MAIN_LOG
    echo "  - 层次聚类 (VisualizationHCCallback)" | tee -a $MAIN_LOG
    echo "  - 树形谱系路径图 (PlotTree) - 多个视角" | tee -a $MAIN_LOG
    echo "  - 径向树图 (PlotTreeMap)" | tee -a $MAIN_LOG
    echo "  - 带谱系的径向树 (PlotTreeMap with lineage)" | tee -a $MAIN_LOG
    echo "  - 桑基图 (Sankey)" | tee -a $MAIN_LOG
    echo "  - 树状图 (Treemap)" | tee -a $MAIN_LOG
    echo "  - 路径追踪 (VisualizationTrace)" | tee -a $MAIN_LOG
    
    # 使用 timeout 防止无限等待，设置最大运行时间为 24 小时
    timeout 86400 python main.py fit -c "$config" 2>&1 | tee -a "$config_results_dir/training.log"
    
    # 检查退出状态
    exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 配置 $config_name 运行成功" | tee -a $MAIN_LOG
        echo "✅ 配置 $config_name 运行成功"
        
        # 复制生成的所有结果到结果目录
        if [ -d "zzl_checkpoints" ]; then
            echo "复制检查点和所有可视化结果..." | tee -a $MAIN_LOG
            cp -r zzl_checkpoints/* "$config_results_dir/" 2>/dev/null || true
        fi
        
        # 复制 wandb 日志
        if [ -d "wandb" ]; then
            echo "复制 wandb 日志..." | tee -a $MAIN_LOG
            mkdir -p "$config_results_dir/wandb_logs"
            cp -r wandb/* "$config_results_dir/wandb_logs/" 2>/dev/null || true
        fi
        
        echo "📊 生成的可视化包括:" | tee -a $MAIN_LOG
        echo "  - 树形谱系路径图 (多个视角)" | tee -a $MAIN_LOG
        echo "  - 径向树图" | tee -a $MAIN_LOG
        echo "  - 带谱系的径向树图" | tee -a $MAIN_LOG
        echo "  - 层次聚类图" | tee -a $MAIN_LOG
        echo "  - 桑基图" | tee -a $MAIN_LOG
        echo "  - 树状图" | tee -a $MAIN_LOG
        echo "  - 路径追踪图" | tee -a $MAIN_LOG
        
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ 配置 $config_name 运行超时（24小时）" | tee -a $MAIN_LOG
        echo "⏰ 配置 $config_name 运行超时（24小时）"
    else
        echo "❌ 配置 $config_name 运行失败，退出码: $exit_code" | tee -a $MAIN_LOG
        echo "❌ 配置 $config_name 运行失败，退出码: $exit_code"
    fi
    
    echo "结束时间: $(date)" | tee -a $MAIN_LOG
    echo "==========================================" | tee -a $MAIN_LOG
    
    # 在配置之间稍作停顿，避免资源冲突
    echo "等待 30 秒后继续下一个配置..." | tee -a $MAIN_LOG
    sleep 30
done

echo "" | tee -a $MAIN_LOG
echo "==========================================" | tee -a $MAIN_LOG
echo "🎉 所有配置运行完成！" | tee -a $MAIN_LOG
echo "结束时间: $(date)" | tee -a $MAIN_LOG
echo "结果保存目录: $RESULTS_DIR" | tee -a $MAIN_LOG
echo "==========================================" | tee -a $MAIN_LOG

# 生成最终摘要
echo "生成最终摘要..." | tee -a $MAIN_LOG
SUMMARY_FILE="$RESULTS_DIR/final_summary.txt"
echo "Step2 Checkpoint 完整运行摘要" > $SUMMARY_FILE
echo "运行时间: $(date)" >> $SUMMARY_FILE
echo "结果目录: $RESULTS_DIR" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "包含的可视化类型:" >> $SUMMARY_FILE
echo "  - 树形谱系路径图 (PlotTree) - 多个视角" >> $SUMMARY_FILE
echo "  - 径向树图 (PlotTreeMap)" >> $SUMMARY_FILE
echo "  - 带谱系的径向树图 (PlotTreeMap with lineage)" >> $SUMMARY_FILE
echo "  - 层次聚类图 (VisualizationHCCallback)" >> $SUMMARY_FILE
echo "  - 桑基图 (Sankey)" >> $SUMMARY_FILE
echo "  - 树状图 (Treemap)" >> $SUMMARY_FILE
echo "  - 路径追踪图 (VisualizationTrace)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "配置列表:" >> $SUMMARY_FILE
for config in "${CONFIGS[@]}"; do
    echo "  - $(basename "$config" .yaml)" >> $SUMMARY_FILE
done

echo "✅ 脚本执行完成！"
echo "📁 所有结果已保存到: $RESULTS_DIR"
echo "📊 查看运行日志: $MAIN_LOG"
echo "📋 查看最终摘要: $SUMMARY_FILE"
echo "🎨 每个配置目录都包含完整的可视化结果"
