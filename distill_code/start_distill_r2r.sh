#!/bin/bash
# R2R distillation training script
# Clean, isolated run for reproducible experiments.

set -euo pipefail

CONDA_PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG_FILE="${CFG_FILE:-../configs/ablation/fgr2r.yaml}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-../data/models/Qwen3-1.7B}"
JSONL_PATHS="${JSONL_PATHS:-../build/distill_collection/distill_logs/distill_r2r_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-../build/distill_training_r2r_clean_v1}"
DISTILL_TASKS="R2R"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR_STAGE1="${LR_STAGE1:-1e-5}"
LR_STAGE2="${LR_STAGE2:-5e-6}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-10}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-10}"
PRECISION="${PRECISION:-amp_bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NGPUS="${NGPUS:-4}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29511}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29512}"

mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")"

if pgrep -af "train_distill.py .*--output_dir $OUTPUT_DIR" >/dev/null 2>&1; then
    echo "Another R2R distillation run is already using $OUTPUT_DIR"
    exit 1
fi

echo "============================================"
echo "R2R Distillation Training"
echo "============================================"
echo "Config:       $CFG_FILE"
echo "JSONL:        $JSONL_PATHS"
echo "Output:       $OUTPUT_DIR"
echo "Tasks:        $DISTILL_TASKS"
echo "Base model:   $PRETRAINED_MODEL"
echo "GPUs:         $NGPUS ($CUDA_VISIBLE_DEVICES)"
echo "Batch/GPU:    $BATCH_SIZE"
echo "Grad Accum:   $GRAD_ACCUM"
echo "Precision:    $PRECISION"
echo "Python:       $CONDA_PYTHON"
echo "============================================"

echo ""
echo "[Stage 1] head/vision alignment"
"$CONDA_PYTHON" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT_STAGE1" \
    train_distill.py \
    --stage multi \
    --cfg_file "$CFG_FILE" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --distill_jsonl_paths "$JSONL_PATHS" \
    --distill_tasks "$DISTILL_TASKS" \
    --distill_stage stage1 \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_step "$GRAD_ACCUM" \
    --lr "$LR_STAGE1" \
    --num_epochs "$NUM_EPOCHS_STAGE1" \
    --output_dir "$OUTPUT_DIR" \
    --precision "$PRECISION" \
    --num_workers "$NUM_WORKERS" \
    --weight_decay 0.01 \
    --lr_scheduler cosine \
    --num_warmup_steps 500 \
    2>&1 | tee "$OUTPUT_DIR/train_stage1.log"

RESUME_CKPT="$OUTPUT_DIR/checkpoints/navqwen3_stage1_best.pt"
if [ ! -f "$RESUME_CKPT" ]; then
    echo "Stage 1 finished but checkpoint not found: $RESUME_CKPT"
    exit 1
fi

echo ""
echo "[Stage 2] unfreeze top LLM layers"
"$CONDA_PYTHON" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT_STAGE2" \
    train_distill.py \
    --stage multi \
    --cfg_file "$CFG_FILE" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --distill_jsonl_paths "$JSONL_PATHS" \
    --distill_tasks "$DISTILL_TASKS" \
    --distill_stage stage2 \
    --resume_from_checkpoint "$RESUME_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_step "$GRAD_ACCUM" \
    --lr "$LR_STAGE2" \
    --num_epochs "$NUM_EPOCHS_STAGE2" \
    --output_dir "$OUTPUT_DIR" \
    --precision "$PRECISION" \
    --num_workers "$NUM_WORKERS" \
    --weight_decay 0.01 \
    --lr_scheduler cosine \
    --num_warmup_steps 300 \
    2>&1 | tee "$OUTPUT_DIR/train_stage2.log"

echo ""
echo "============================================"
echo "R2R Student Distillation Completed"
echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "============================================"
