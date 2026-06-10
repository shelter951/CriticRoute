#!/bin/bash
# 多GPU蒸馏训练启动脚本
# 使用6张GPU: 1,3,4,5,6,7

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告

# 训练参数
STAGE="stage1"
CFG_FILE="../configs/multi.yaml"
PRETRAINED_MODEL="../data/models/Qwen3-1.7B"
JSONL_PATHS="../build/distill_collection/distill_logs/distill_cvdn_train.jsonl"
OUTPUT_DIR="../build/distill_training"
BATCH_SIZE=2  # 每张GPU的batch size
GRAD_ACCUM=2  # 梯度累积，有效batch size = 2 * 2 * 6 = 24
LR=1e-5
NUM_EPOCHS=10
PRECISION="amp_bf16"
NUM_WORKERS=2

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 进入脚本目录
cd "$(dirname "$0")"

# 使用torchrun启动多GPU训练
# --nnodes=1: 单机
# --nproc_per_node=6: 6张GPU
# --master_port: 主节点端口
torchrun \
    --nnodes=1 \
    --nproc_per_node=6 \
    --master_port=29500 \
    train_distill.py \
    --stage multi \
    --cfg_file $CFG_FILE \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --distill_jsonl_paths $JSONL_PATHS \
    --distill_stage $STAGE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_step $GRAD_ACCUM \
    --lr $LR \
    --num_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --precision $PRECISION \
    --num_workers $NUM_WORKERS \
    --weight_decay 0.01 \
    --lr_scheduler cosine \
    --num_warmup_steps 500

echo "Training started with 6 GPUs"
echo "Logs will be saved to: $OUTPUT_DIR"
echo "To monitor: tail -f $OUTPUT_DIR/train.log"

