#!/bin/bash
# 运行所有实验配置的脚本（Linux）

# 设置基本参数
TASK="CVDN"
SPLIT="val_unseen"
CFG_FILE="configs/multi.yaml"
DATA_DIR="data"
OUTPUT_DIR="build/edgecloud_results"

# 模型路径（需要根据实际情况修改）
TEACHER_CKPT="build/nav_ckpts/navillm_cvdn_teacher.pt"
STUDENT_DISTILL_CKPT="build/distill_training_stage2/checkpoints/navqwen3_stage2_best.pt"
STUDENT_BASE_CKPT=""  # 未蒸馏的Student（如果有）
ROUTER_CKPT="build/router_data/checkpoints/router_best.pt"
PRETRAINED_PATH="data/models/Qwen3-1.7B"

# GPU设置
GPU=0

echo "=========================================="
echo "Running All Edge-Cloud Experiments"
echo "=========================================="

# 1. Teacher-only baseline
echo ""
echo "1. Teacher-only baseline..."
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

# 2. Student-distill-only
echo ""
echo "2. Student-distill-only..."
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

# 3. Student-base-only (if available)
if [ -n "$STUDENT_BASE_CKPT" ] && [ -f "$STUDENT_BASE_CKPT" ]; then
    echo ""
    echo "3. Student-base-only..."
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

# 4. EdgeCloud with Off-course Router (τ sweep)
echo ""
echo "4. EdgeCloud with Off-course Router (τ sweep)..."
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
    --tau_list 0.3 0.4 0.5 0.6 0.7 \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 5. EdgeCloud with Entropy Router (τ sweep)
echo ""
echo "5. EdgeCloud with Entropy Router (τ sweep)..."
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
    --tau_list 0.3 0.4 0.5 0.6 0.7 \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 6. EdgeCloud with Divergence Router (τ sweep)
echo ""
echo "6. EdgeCloud with Divergence Router (τ sweep)..."
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
    --tau_list 0.3 0.4 0.5 0.6 0.7 \
    --latency_mode fixed \
    --latency_ms 400.0 \
    --gpu $GPU \
    --output_dir $OUTPUT_DIR

# 7. EdgeCloud with Student-base + Router (if available)
if [ -n "$STUDENT_BASE_CKPT" ] && [ -f "$STUDENT_BASE_CKPT" ]; then
    echo ""
    echo "7. EdgeCloud with Student-base + Off-course Router..."
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
        --tau_list 0.3 0.4 0.5 0.6 0.7 \
        --latency_mode fixed \
        --latency_ms 400.0 \
        --gpu $GPU \
        --output_dir $OUTPUT_DIR
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Running analysis..."
python edgecloud_experiments/analysis/analyze_results.py \
    --results_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR/analysis

echo ""
echo "Analysis completed! Check $OUTPUT_DIR/analysis for results."

