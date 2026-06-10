#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

RUN_ROOT=build/hetero_router/train_r2r_2000_v1
TEACHER_JSON=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_train_decisions_v2/R2R_train.json
PY=${PYTHON:-python}
SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py

mkdir -p "${RUN_ROOT}"
date > "${RUN_ROOT}/launch_time.txt"

launch_shard() {
  local shard="$1"
  local gpu="$2"
  local start="$3"
  local end="$4"
  local shard_dir="${RUN_ROOT}/shard_${shard}"
  mkdir -p "${shard_dir}"
  {
    echo "shard=${shard}"
    echo "gpu=${gpu}"
    echo "start_index=${start}"
    echo "end_index=${end}"
    echo "launch_time=$(date)"
  } > "${shard_dir}/status.log"

  nohup "${PY}" "${SCRIPT}" \
    --split train \
    --sample_seed 20260505 \
    --start_index "${start}" \
    --end_index "${end}" \
    --max_episodes 0 \
    --teacher_json "${TEACHER_JSON}" \
    --router_mode small \
    --samples_out "${shard_dir}/samples.jsonl" \
    --out_dir "${shard_dir}" \
    --gpu "${gpu}" \
    --max_steps 15 \
    > "${shard_dir}/run.log" 2>&1 &
  echo "$!" > "${shard_dir}/pid"
}

launch_shard 0 4 0 500
launch_shard 1 5 500 1000
launch_shard 2 6 1000 1500
launch_shard 3 7 1500 2000

echo "Launched train_r2r_2000_v1 shards:"
for shard in 0 1 2 3; do
  echo "shard_${shard} pid=$(cat "${RUN_ROOT}/shard_${shard}/pid")"
done

