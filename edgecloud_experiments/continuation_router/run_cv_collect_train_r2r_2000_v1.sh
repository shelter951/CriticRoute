#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}

export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
PY=${PYTHON:-python}
ROOT=build/continuation_router/cv_train_r2r_2000_v1
TEACHER_JSON=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_train_decisions_v2/R2R_train.json
mkdir -p "$ROOT"

for i in 0 1 2 3; do
  start=$((i * 500))
  end=$(((i + 1) * 500))
  gpu=$((4 + i))
  shard="$ROOT/shard_${i}"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
    --split train \
    --max_episodes 0 \
    --start_index "$start" \
    --end_index "$end" \
    --sample_seed 20260512 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER_JSON" \
    --router_mode oracle \
    --continuation_horizon 15 \
    --out_dir "$shard" \
    --samples_out "$shard/samples.jsonl" \
    > "$shard/nohup.log" 2>&1 &
  echo "started shard=$i gpu=$gpu range=[$start,$end) pid=$!"
done
