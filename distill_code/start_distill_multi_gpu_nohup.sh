#!/bin/bash
# 多GPU蒸馏训练启动脚本（后台运行）
# 使用6张GPU: 1,3,4,5,6,7

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 训练参数
STAGE="stage1"
CFG_FILE="../configs/multi.yaml"
PRETRAINED_MODEL="../data/models/Qwen3-1.7B"
JSONL_PATHS="../build/distill_collection/distill_logs/distill_cvdn_train.jsonl"
OUTPUT_DIR="../build/distill_training"
BATCH_SIZE=2
GRAD_ACCUM=2
LR=1e-5
NUM_EPOCHS=10
PRECISION="amp_bf16"
NUM_WORKERS=2

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 进入脚本目录
cd "$(dirname "$0")"

# 使用nohup后台运行
nohup torchrun \
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
    --num_warmup_steps 500 \
    > $OUTPUT_DIR/train.log 2>&1 &

# 保存进程ID
PID=$!
echo "Training started with PID: $PID"
echo $PID > $OUTPUT_DIR/train.pid

echo "=========================================="
echo "Multi-GPU Distillation Training Started"
echo "=========================================="
echo "GPUs: 1,3,4,5,6,7 (6 GPUs)"
echo "Total batch size: $((BATCH_SIZE * GRAD_ACCUM * 6))"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/train.log"
echo "PID: $PID"
echo ""
echo "To view logs: tail -f $OUTPUT_DIR/train.log"
echo "To stop: kill $PID"
echo "=========================================="

