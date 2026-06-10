#!/bin/bash
# 运行所有实验配置的脚本（使用新的router v2）
# 使用方法: bash edgecloud_experiments/run_all_experiments_v2.sh

# 设置基本参数
TASK="CVDN"
SPLIT="val_unseen"
CFG_FILE="configs/multi.yaml"
DATA_DIR="data"
OUTPUT_DIR="build/edgecloud_results_v2"

# 模型路径（使用新的router v2）
TEACHER_CKPT="build/nav_ckpts/navillm_cvdn_teacher.pt"
STUDENT_DISTILL_CKPT="build/distill_training_stage2/checkpoints/navqwen3_stage2_best.pt"
STUDENT_BASE_CKPT=""  # 未蒸馏的Student（如果有）
ROUTER_CKPT="build/router_data_v2/checkpoints/router_epoch_50.pt"  # 新的router
PRETRAINED_PATH="data/models/Qwen3-1.7B"

# GPU设置（可以根据实际情况修改）
# 单卡模式
GPU=0
# 多卡模式（推荐，取消注释使用）
# CUDA_VISIBLE_DEVICES="0,1,2,3"
# NPROC_PER_NODE=4
# MASTER_PORT=29500

# τ阈值列表
TAU_LIST="0.3 0.4 0.5 0.6 0.7"

echo "=========================================="
echo "Running All Edge-Cloud Experiments (Router v2)"
echo "=========================================="
echo "Router checkpoint: $ROUTER_CKPT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# ============================================
# 1. Baselines
# ============================================

# 1.1 Teacher-only baseline
echo ""
echo "=========================================="
echo "1.1 Teacher-only baseline..."
echo "=========================================="
python edgecloud_experiments/eval_edgecloud.py \
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
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 1.2 Student-distill-only
echo ""
echo "=========================================="
echo "1.2 Student-distill-only..."
echo "=========================================="
python edgecloud_experiments/eval_edgecloud.py \
    --task $TASK \
    --split $SPLIT \
    --cfg_file $CFG_FILE \
    --data_dir $DATA_DIR \
    --teacher_ckpt $TEACHER_CKPT \
    --student_ckpt $STUDENT_DISTILL_CKPT \
    --pretrained_model_name_or_path $PRETRAINED_PATH \
    --student_type distill \
    --mode student_only \
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 1.3 Student-base-only (if available)
if [ -n "$STUDENT_BASE_CKPT" ] && [ -f "$STUDENT_BASE_CKPT" ]; then
    echo ""
    echo "=========================================="
    echo "1.3 Student-base-only..."
    echo "=========================================="
    python edgecloud_experiments/eval_edgecloud.py \
        --task $TASK \
        --split $SPLIT \
        --cfg_file $CFG_FILE \
        --data_dir $DATA_DIR \
        --teacher_ckpt $TEACHER_CKPT \
        --student_ckpt $STUDENT_BASE_CKPT \
        --pretrained_model_name_or_path $PRETRAINED_PATH \
        --student_type base \
        --mode student_only \
        --gpu $GPU \
        --output_dir $OUTPUT_DIR
fi

# ============================================
# 2. EdgeCloud with Different Routers
# ============================================

# 2.1 EdgeCloud with Off-course Router (我们的方法)
echo ""
echo "=========================================="
echo "2.1 EdgeCloud with Off-course Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
python edgecloud_experiments/eval_edgecloud.py \
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
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 2.2 EdgeCloud with Entropy Router (baseline)
echo ""
echo "=========================================="
echo "2.2 EdgeCloud with Entropy Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
python edgecloud_experiments/eval_edgecloud.py \
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
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 2.3 EdgeCloud with Divergence Router (baseline)
echo ""
echo "=========================================="
echo "2.3 EdgeCloud with Divergence Router (τ sweep: $TAU_LIST)..."
echo "=========================================="
python edgecloud_experiments/eval_edgecloud.py \
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
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 2.4 EdgeCloud with Student-base + Off-course Router (if available)
if [ -n "$STUDENT_BASE_CKPT" ] && [ -f "$STUDENT_BASE_CKPT" ]; then
    echo ""
    echo "=========================================="
    echo "2.4 EdgeCloud with Student-base + Off-course Router (τ sweep: $TAU_LIST)..."
    echo "=========================================="
    python edgecloud_experiments/eval_edgecloud.py \
        --task $TASK \
        --split $SPLIT \
        --cfg_file $CFG_FILE \
        --data_dir $DATA_DIR \
        --teacher_ckpt $TEACHER_CKPT \
        --student_ckpt $STUDENT_BASE_CKPT \
        --router_ckpt $ROUTER_CKPT \
        --pretrained_model_name_or_path $PRETRAINED_PATH \
        --student_type base \
        --mode edgecloud \
        --router_type offcourse \
        --tau_list $TAU_LIST \
        --latency_mode fixed \
        --latency_ms 400.0 \
        --gpu $GPU \
        --output_dir $OUTPUT_DIR
fi

# ============================================
# 3. 结果分析
# ============================================
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Running analysis..."
echo "=========================================="

if [ -f "edgecloud_experiments/analysis/analyze_results.py" ]; then
    python edgecloud_experiments/analysis/analyze_results.py \
        --results_dir $OUTPUT_DIR \
        --output_dir $OUTPUT_DIR/analysis
    echo ""
    echo "Analysis completed! Check $OUTPUT_DIR/analysis for results."
else
    echo "Analysis script not found. Skipping analysis."
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="





