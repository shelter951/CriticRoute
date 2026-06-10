#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
TRAIN_TEACHER=$ROOT/build/object_router/cloud_navillm_clean_split_v1/reverie_val_seen/REVERIE_val_seen_raw.json
EVAL_TEACHER=$ROOT/build/object_router/cloud_navillm_full_v1/reverie/REVERIE_val_unseen.json
OUT=$ROOT/build/object_router/reverie_router_clean_split_v1
SCRIPT=$ROOT/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py
TRAINER=$ROOT/edgecloud_experiments/hetero_router/train_reward_router.py

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"/samples "$OUT"/logs "$OUT"/models "$OUT"/eval

if [[ ! -s "$TRAIN_TEACHER" ]]; then
  echo "Missing train teacher cache: $TRAIN_TEACHER" >&2
  exit 2
fi
if [[ ! -s "$EVAL_TEACHER" ]]; then
  echo "Missing eval teacher cache: $EVAL_TEACHER" >&2
  exit 2
fi

echo "START REVERIE clean-split router train/eval $(date)" | tee -a "$OUT/pipeline.log"

collect_shard() {
  local shard=$1
  local gpu=$2
  local start=$3
  local end=$4
  local sample_file=$OUT/samples/shard_${shard}.jsonl
  local shard_dir=$OUT/collect_val_seen_shard_${shard}
  if [[ -s "$sample_file" ]]; then
    echo "SKIP collect shard=${shard}; sample file exists" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  echo "START collect val_seen shard=${shard} gpu=${gpu} range=${start}:${end} $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" "$SCRIPT" \
    --task REVERIE \
    --split val_seen \
    --start_index "$start" \
    --end_index "$end" \
    --max_episodes 0 \
    --max_steps 15 \
    --sample_seed 20260508 \
    --teacher_json "$TRAIN_TEACHER" \
    --gpu "$gpu" \
    --router_mode small \
    --strict_teacher_paths \
    --samples_out "$sample_file" \
    --out_dir "$shard_dir" \
    > "$OUT/logs/collect_shard_${shard}.log" 2>&1
  echo "DONE collect shard=${shard} $(date)" | tee -a "$OUT/pipeline.log"
}

if [[ ! -s "$OUT/samples/all_samples.jsonl" ]]; then
  collect_shard 0 4 0 360 &
  p0=$!
  collect_shard 1 5 360 720 &
  p1=$!
  collect_shard 2 6 720 1080 &
  p2=$!
  collect_shard 3 7 1080 1423 &
  p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
  cat "$OUT"/samples/shard_*.jsonl > "$OUT/samples/all_samples.jsonl"
  echo "MERGED samples $(wc -l < "$OUT/samples/all_samples.jsonl") rows" | tee -a "$OUT/pipeline.log"
fi

train_router() {
  local name=$1
  local lambda=$2
  local model_dir=$OUT/models/$name
  if [[ -s "$model_dir/hetero_router.pt" ]]; then
    echo "SKIP train $name; checkpoint exists" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  echo "START train $name lambda=$lambda $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" "$TRAINER" \
    --samples "$OUT/samples/all_samples.jsonl" \
    --out_dir "$model_dir" \
    --epochs 180 \
    --lr 8e-4 \
    --cost_lambda "$lambda" \
    --budgets 0.20,0.30,0.40,0.50 \
    > "$OUT/logs/train_${name}.log" 2>&1
  echo "DONE train $name $(date)" | tee -a "$OUT/pipeline.log"
}

train_router reward_l025 0.25
train_router reward_l035 0.35

eval_router() {
  local name=$1
  local key=$2
  local gpu=$3
  local model=$4
  local eval_dir=$OUT/eval/${name}_${key}
  if find "$eval_dir" -name 'summary_*.json' -type f | grep -q .; then
    echo "SKIP eval ${name}_${key}; summary exists" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  echo "START eval ${name}_${key} gpu=${gpu} $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" "$SCRIPT" \
    --task REVERIE \
    --split val_unseen \
    --max_episodes 300 \
    --max_steps 15 \
    --sample_seed 20260507 \
    --teacher_json "$EVAL_TEACHER" \
    --gpu "$gpu" \
    --router_mode trained \
    --router_checkpoint "$model/hetero_router.pt" \
    --budget_key "$key" \
    --strict_teacher_paths \
    --out_dir "$eval_dir" \
    > "$OUT/logs/eval_${name}_${key}.log" 2>&1
  echo "DONE eval ${name}_${key} $(date)" | tee -a "$OUT/pipeline.log"
}

for name in reward_l025 reward_l035; do
  model=$OUT/models/$name
  eval_router "$name" b30 4 "$model" &
  p0=$!
  eval_router "$name" b40 5 "$model" &
  p1=$!
  eval_router "$name" b50 6 "$model" &
  p2=$!
  wait "$p0" "$p1" "$p2"
done

echo "FINISH REVERIE clean-split router train/eval $(date)" | tee -a "$OUT/pipeline.log"
