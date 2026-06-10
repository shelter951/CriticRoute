#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=$ROOT/build/object_router/soon_router_train_300_v1
SOURCE_TEACHER=$ROOT/build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json
TEACHER=$OUT/SOON_val_unseen_teacher_remapped.json
SCRIPT=$ROOT/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py
TRAINER=$ROOT/edgecloud_experiments/hetero_router/train_reward_router.py

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"/samples "$OUT"/logs "$OUT"/models "$OUT"/eval

echo "START SOON router train/eval $(date)" | tee -a "$OUT/pipeline.log"

if [[ ! -s "$TEACHER" ]]; then
  echo "BUILD SOON teacher remap $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" - <<'PY'
import json
from pathlib import Path
from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import load_episodes

root = Path("${PROJECT_ROOT:-.}")
source = root / "build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json"
target = root / "build/object_router/soon_router_train_300_v1/SOON_val_unseen_teacher_remapped.json"
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
  local shard_dir=$OUT/collect_shard_${shard}
  if [[ -s "$sample_file" ]]; then
    echo "SKIP collect shard=${shard}; sample file exists: $sample_file" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  echo "START collect shard=${shard} gpu=${gpu} range=${start}:${end} $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" "$SCRIPT" \
    --task SOON \
    --split val_unseen \
    --sample_seed 20260508 \
    --start_index "$start" \
    --end_index "$end" \
    --max_episodes 0 \
    --max_steps 20 \
    --teacher_json "$TEACHER" \
    --gpu "$gpu" \
    --router_mode small \
    --strict_teacher_paths \
    --samples_out "$sample_file" \
    --out_dir "$shard_dir" \
    > "$OUT/logs/collect_shard_${shard}.log" 2>&1
  echo "DONE collect shard=${shard} $(date)" | tee -a "$OUT/pipeline.log"
}

if [[ ! -s "$OUT/samples/all_samples.jsonl" ]]; then
  collect_shard 0 4 0 250 &
  p0=$!
  collect_shard 1 5 250 500 &
  p1=$!
  collect_shard 2 6 500 750 &
  p2=$!
  collect_shard 3 7 750 1000 &
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
    --epochs 160 \
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
    --teacher_json "$TEACHER" \
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
  eval_router "$name" b20 4 "$model" &
  p0=$!
  eval_router "$name" b30 5 "$model" &
  p1=$!
  eval_router "$name" b40 6 "$model" &
  p2=$!
  eval_router "$name" b50 7 "$model" &
  p3=$!
  wait "$p0" "$p1" "$p2" "$p3"
done

echo "FINISH SOON router train/eval $(date)" | tee -a "$OUT/pipeline.log"
