#!/bin/bash
# 单独运行 EdgeCloud + Entropy Router 实验
# 使用方法: bash edgecloud_experiments/run_entropy_experiment.sh

# 设置基本参数
TASK="CVDN"
SPLIT="val_unseen"
CFG_FILE="configs/multi.yaml"
DATA_DIR="data"
OUTPUT_DIR="build/edgecloud_results_v2"

# 模型路径
TEACHER_CKPT="build/nav_ckpts/navillm_cvdn_teacher.pt"
STUDENT_DISTILL_CKPT="build/distill_training_stage2/checkpoints/navqwen3_stage2_best.pt"
PRETRAINED_PATH="data/models/Qwen3-1.7B"

# GPU设置（多卡并行）
# 注意：如果之前OOM，可以减少GPU数量
CUDA_VISIBLE_DEVICES="0"  # 根据实际情况修改
NPROC_PER_NODE=1
MASTER_PORT=29501

# τ阈值列表
TAU_LIST="0.3 0.4 0.5 0.6 0.7"

echo "=========================================="
echo "EdgeCloud + Entropy Router 实验"
echo "=========================================="
echo "任务: $TASK, 数据分割: $SPLIT"
echo "Student: distill"
echo "Router: Entropy (baseline)"
echo "τ值: $TAU_LIST"
echo "GPU: $CUDA_VISIBLE_DEVICES (使用 $NPROC_PER_NODE 个进程)"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行实验
echo "开始运行实验..."
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    edgecloud_experiments/eval_edgecloud.py \
    --task $TASK \
    --split $SPLIT \
    --cfg_file $CFG_FILE \
    --data_dir $DATA_DIR \
    --teacher_ckpt $TEACHER_CKPT \
    --student_ckpt $STUDENT_DISTILL_CKPT \
    --pretrained_model_name_or_path $PRETRAINED_PATH \
    --student_type distill \
    --mode edgecloud \
    --router_type entropy \
    --tau_list $TAU_LIST \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --multi_gpu \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✅ 实验完成！"
    echo "结果保存在: $OUTPUT_DIR/results_edgecloud_entropy_distill.json"
else
    echo "❌ 实验失败，请检查错误信息"
fi
echo "=========================================="

