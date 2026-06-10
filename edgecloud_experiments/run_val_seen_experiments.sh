#!/bin/bash
# 运行val_seen数据集的实验
# 使用方法: bash edgecloud_experiments/run_val_seen_experiments.sh

# 设置基本参数
TASK="CVDN"
SPLIT="val_seen"  # 修改为val_seen
CFG_FILE="configs/multi.yaml"
DATA_DIR="data"
OUTPUT_DIR="build/edgecloud_results_v2_val_seen"  # 使用不同的输出目录，避免覆盖val_unseen的结果

# 模型路径
TEACHER_CKPT="build/nav_ckpts/navillm_cvdn_teacher.pt"
STUDENT_DISTILL_CKPT="build/distill_training_stage2/checkpoints/navqwen3_stage2_best.pt"
STUDENT_BASE_CKPT=""
ROUTER_CKPT="build/router_data_v2/checkpoints/router_epoch_50.pt"
PRETRAINED_PATH="data/models/Qwen3-1.7B"

# GPU设置（多卡并行）
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
NPROC_PER_NODE=6
MASTER_PORT_BASE=29500

# τ阈值列表
TAU_LIST="0.3 0.4 0.5 0.6 0.7"

echo "=========================================="
echo "Running All Edge-Cloud Experiments on val_seen"
echo "=========================================="
echo "Dataset Split: $SPLIT"
echo "Output directory: $OUTPUT_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 计数器（用于master_port）
PORT_COUNTER=0

# ============================================
# 1. Baselines
# ===========================================

# 1.1 Teacher-only baseline
echo ""
echo "=========================================="
echo "1.1 Teacher-only baseline..."
echo "=========================================="
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU python edgecloud_experiments/eval_edgecloud.py \
    --task $TASK \
    --split $SPLIT \
    --cfg_file $CFG_FILE \
    --data_dir $DATA_DIR \
    --teacher_ckpt $TEACHER_CKPT \
    --student_ckpt $STUDENT_DISTILL_CKPT \
    --pretrained_model_name_or_path $PRETRAINED_PATH \
    --mode teacher_only \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --gpu 0 \
    --output_dir $OUTPUT_DIR

# 1.2 Student-distill-only
echo ""
echo "=========================================="
echo "1.2 Student-distill-only..."
echo "=========================================="
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU python edgecloud_experiments/eval_edgecloud.py \
    --task $TASK \
    --split $SPLIT \
    --cfg_file $CFG_FILE \
    --data_dir $DATA_DIR \
    --teacher_ckpt $TEACHER_CKPT \
    --student_ckpt $STUDENT_DISTILL_CKPT \
    --pretrained_model_name_or_path $PRETRAINED_PATH \
    --student_type distill \
    --mode student_only \
    --gpu 0 \
    --output_dir $OUTPUT_DIR

# ============================================
# 2. EdgeCloud with Different Routers
# ===========================================

# 2.1 EdgeCloud with Off-course Router
echo ""
echo "=========================================="
echo "2.1 EdgeCloud with Off-course Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
MASTER_PORT=$((MASTER_PORT_BASE + PORT_COUNTER))
PORT_COUNTER=$((PORT_COUNTER + 1))
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
    --router_ckpt $ROUTER_CKPT \
    --pretrained_model_name_or_path $PRETRAINED_PATH \
    --student_type distill \
    --mode edgecloud \
    --router_type offcourse \
    --tau_list $TAU_LIST \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --multi_gpu \
    --output_dir $OUTPUT_DIR

# 2.2 EdgeCloud with Entropy Router
echo ""
echo "=========================================="
echo "2.2 EdgeCloud with Entropy Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
MASTER_PORT=$((MASTER_PORT_BASE + PORT_COUNTER))
PORT_COUNTER=$((PORT_COUNTER + 1))
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

# 2.3 EdgeCloud with Divergence Router
echo ""
echo "=========================================="
echo "2.3 EdgeCloud with Divergence Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
MASTER_PORT=$((MASTER_PORT_BASE + PORT_COUNTER))
PORT_COUNTER=$((PORT_COUNTER + 1))
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
    --router_type divergence \
    --tau_list $TAU_LIST \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --multi_gpu \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "All experiments on val_seen completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="





