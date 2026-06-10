#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
TRAIN_TEACHER=$ROOT/build/object_router/cloud_navillm_clean_split_v1/soon_val_seen/SOON_val_seen_raw.json
OUT=$ROOT/build/object_router/soon_router_clean_split_v2
SOURCE_EVAL_TEACHER=$ROOT/build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json
EVAL_TEACHER=$OUT/SOON_val_unseen_teacher_remapped.json
SCRIPT=$ROOT/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py
TRAINER=$ROOT/edgecloud_experiments/hetero_router/train_reward_router.py
export EVAL_TEACHER

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"/samples "$OUT"/logs "$OUT"/models "$OUT"/eval

if [[ ! -s "$TRAIN_TEACHER" ]]; then
  echo "Missing train teacher cache: $TRAIN_TEACHER" >&2
  exit 2
fi
if [[ ! -s "$SOURCE_EVAL_TEACHER" ]]; then
  echo "Missing source eval teacher cache: $SOURCE_EVAL_TEACHER" >&2
  exit 2
fi

echo "START SOON clean-split router train/eval $(date)" | tee -a "$OUT/pipeline.log"

if [[ ! -s "$EVAL_TEACHER" ]]; then
  echo "BUILD SOON val_unseen teacher remap $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" - <<'PY'
import json
import os
from pathlib import Path
from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import load_episodes

root = Path("${PROJECT_ROOT:-.}")
source = root / "build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json"
target = Path(os.environ.get("EVAL_TEACHER", "")) if os.environ.get("EVAL_TEACHER") else root / "build/object_router/soon_router_clean_split_v2/SOON_val_unseen_teacher_remapped.json"
episodes = load_episodes("SOON", str(root / "data"), "val_unseen", -1, 0)
teacher = json.load(open(source, encoding="utf-8"))
if len(teacher) < len(episodes):
    raise RuntimeError(f"teacher has {len(teacher)} rows but episodes has {len(episodes)}")
rows = []
for ep, rec in zip(episodes, teacher):
    new = dict(rec)
    new["orig_instr_id"] = new.get("instr_id")
    new["instr_id"] = ep["instr_id"]
    rows.append(new)
target.parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(target, "w", encoding="utf-8"), ensure_ascii=False)
print({"episodes": len(episodes), "teacher": len(teacher), "out": str(target)})
PY
fi

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
    --task SOON \
    --split val_seen \
    --start_index "$start" \
    --end_index "$end" \
    --max_episodes 0 \
    --max_steps 20 \
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
  collect_shard 0 4 0 285 &
  p0=$!
  collect_shard 1 5 285 570 &
  p1=$!
  collect_shard 2 6 570 855 &
  p2=$!
  collect_shard 3 7 855 1130 &
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
    --task SOON \
    --split val_unseen \
    --max_episodes 300 \
    --max_steps 20 \
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

echo "FINISH SOON clean-split router train/eval $(date)" | tee -a "$OUT/pipeline.log"
