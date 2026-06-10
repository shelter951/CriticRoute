#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}

export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
PY=${PYTHON:-python}
TRAIN_ROOT=build/continuation_router/cv_train_r2r_2000_v1
ROUTER_ROOT=build/continuation_router/cv_router_r2r_2000_v1
EVAL_ROOT=build/continuation_router/cv_eval_val_unseen_v1
mkdir -p "$ROUTER_ROOT" "$EVAL_ROOT"

cat "$TRAIN_ROOT"/shard_*/samples.jsonl > "$TRAIN_ROOT/samples_train_2000_cv.jsonl"

"$PY" edgecloud_experiments/continuation_router/train_cv_group_router.py \
  --samples "$TRAIN_ROOT/samples_train_2000_cv.jsonl" \
  --out_dir "$ROUTER_ROOT/l030_btarget40" \
  --epochs 100 \
  --target_budget 0.40 \
  --cost_lambda 0.30 \
  --budget_penalty 0.30

idx=0
for key in b30 b40 b50; do
  gpu=$((4 + idx))
  out="$EVAL_ROOT/cv_group_${key}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
    --split val_unseen \
    --max_episodes 0 \
    --sample_seed -1 \
    --gpu "$gpu" \
    --teacher_json ${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json \
    --router_mode trained \
    --router_ckpt "$ROUTER_ROOT/l030_btarget40/hetero_router.pt" \
    --budget_key "$key" \
    --out_dir "$out" \
    > "$out/nohup.log" 2>&1 &
  echo "started eval $key gpu=$gpu pid=$!"
  idx=$((idx + 1))
done

oracle="$EVAL_ROOT/cv_oracle"
mkdir -p "$oracle"
CUDA_VISIBLE_DEVICES=7 nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
  --split val_unseen \
  --max_episodes 0 \
  --sample_seed -1 \
  --gpu 7 \
  --teacher_json ${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json \
  --router_mode oracle \
  --continuation_horizon 15 \
  --out_dir "$oracle" \
  > "$oracle/nohup.log" 2>&1 &
echo "started cv oracle pid=$!"
