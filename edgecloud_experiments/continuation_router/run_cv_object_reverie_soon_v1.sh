#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
SCRIPT=$ROOT/edgecloud_experiments/continuation_router/eval_cv_object_edgecloud_nav.py
TRAINER=$ROOT/edgecloud_experiments/continuation_router/train_cv_group_router.py
OUT=$ROOT/build/continuation_router/object_cv_reverie_soon_v1

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"/{logs,samples,models,eval}

echo "START object CV pipeline $(date)" | tee -a "$OUT/pipeline.log"

teacher_json() {
  local task=$1
  local split=$2
  if [[ "$task" == "REVERIE" && "$split" == "val_seen" ]]; then
    echo "$ROOT/build/object_router/cloud_navillm_clean_split_v1/reverie_val_seen/REVERIE_val_seen_raw.json"
  elif [[ "$task" == "REVERIE" && "$split" == "val_unseen" ]]; then
    echo "$ROOT/build/object_router/cloud_navillm_full_v1/reverie/REVERIE_val_unseen.json"
  elif [[ "$task" == "SOON" && "$split" == "val_seen" ]]; then
    echo "$ROOT/build/object_router/cloud_navillm_clean_split_v1/soon_val_seen/SOON_val_seen_raw.json"
  elif [[ "$task" == "SOON" && "$split" == "val_unseen" ]]; then
    local remap="$OUT/SOON_val_unseen_teacher_remapped.json"
    if [[ ! -s "$remap" ]]; then
      echo "BUILD SOON val_unseen teacher remap $(date)" | tee -a "$OUT/pipeline.log" >&2
      "$PY" - <<'PY'
import json
import sys
from pathlib import Path
from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import load_episodes

root = Path("${PROJECT_ROOT:-.}")
source = root / "build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json"
target = root / "build/continuation_router/object_cv_reverie_soon_v1/SOON_val_unseen_teacher_remapped.json"
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
print({"episodes": len(episodes), "teacher": len(teacher), "out": str(target)}, file=sys.stderr)
PY
    fi
    echo "$remap"
  else
    echo "unknown task/split $task $split" >&2
    return 2
  fi
}

collect_task() {
  local task=$1
  local max_steps=$2
  local total=$3
  local task_lc
  task_lc=$(echo "$task" | tr '[:upper:]' '[:lower:]')
  local tjson
  tjson=$(teacher_json "$task" val_seen)
  local task_out="$OUT/$task_lc"
  mkdir -p "$task_out"/{samples,logs,models}
  local all="$task_out/samples/val_seen_all_samples.jsonl"
  if [[ -s "$all" ]]; then
    echo "SKIP collect $task; samples exist $all" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  local spans=("0 $((total/4))" "$((total/4)) $((total/2))" "$((total/2)) $((3*total/4))" "$((3*total/4)) $total")
  local pids=()
  for i in 0 1 2 3; do
    local start end gpu sample shard_dir
    read -r start end <<<"${spans[$i]}"
    gpu=$((4+i))
    sample="$task_out/samples/shard_${i}.jsonl"
    shard_dir="$task_out/collect_shard_${i}"
    echo "START collect $task shard=$i gpu=$gpu range=$start:$end $(date)" | tee -a "$OUT/pipeline.log"
    "$PY" "$SCRIPT" \
      --task "$task" \
      --split val_seen \
      --sample_seed -1 \
      --start_index "$start" \
      --end_index "$end" \
      --max_episodes 0 \
      --max_steps "$max_steps" \
      --continuation_horizon "$max_steps" \
      --teacher_json "$tjson" \
      --gpu "$gpu" \
      --router_mode small \
      --strict_teacher_paths \
      --samples_out "$sample" \
      --out_dir "$shard_dir" \
      > "$task_out/logs/collect_shard_${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  cat "$task_out"/samples/shard_*.jsonl > "$all"
  echo "MERGED $task samples $(wc -l < "$all") rows" | tee -a "$OUT/pipeline.log"
}

train_task() {
  local task=$1
  local task_lc
  task_lc=$(echo "$task" | tr '[:upper:]' '[:lower:]')
  local task_out="$OUT/$task_lc"
  local samples="$task_out/samples/val_seen_all_samples.jsonl"
  local model="$task_out/models/cv_l010_sup075_rank025"
  if [[ -s "$model/hetero_router.pt" ]]; then
    echo "SKIP train $task; checkpoint exists" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  echo "START train $task CV router $(date)" | tee -a "$OUT/pipeline.log"
  "$PY" "$TRAINER" \
    --samples "$samples" \
    --out_dir "$model" \
    --epochs 120 \
    --lr 8e-4 \
    --cost_lambda 0.10 \
    --supervised_coef 0.75 \
    --rank_coef 0.25 \
    --utility_reg_coef 0.05 \
    --budgets 0.20,0.30,0.40,0.50,0.60 \
    > "$task_out/logs/train_cv_l010_sup075_rank025.log" 2>&1
  echo "DONE train $task $(date)" | tee -a "$OUT/pipeline.log"
}

eval_sharded() {
  local task=$1
  local mode=$2
  local key=$3
  local max_steps=$4
  local total=$5
  local task_lc
  task_lc=$(echo "$task" | tr '[:upper:]' '[:lower:]')
  local task_out="$OUT/$task_lc"
  local tjson
  tjson=$(teacher_json "$task" val_unseen)
  local eval_dir="$task_out/eval/${mode}_${key:-nokey}"
  local model="$task_out/models/cv_l010_sup075_rank025/hetero_router.pt"
  mkdir -p "$eval_dir"/logs
  if [[ -s "$eval_dir/summary_full.json" ]]; then
    echo "SKIP eval $task $mode $key; summary exists" | tee -a "$OUT/pipeline.log"
    return 0
  fi
  local spans=("0 $((total/4))" "$((total/4)) $((total/2))" "$((total/2)) $((3*total/4))" "$((3*total/4)) $total")
  local pids=()
  for i in 0 1 2 3; do
    local start end gpu shard_dir
    read -r start end <<<"${spans[$i]}"
    gpu=$((4+i))
    shard_dir="$eval_dir/shard_${i}"
    echo "START eval $task $mode $key shard=$i gpu=$gpu range=$start:$end $(date)" | tee -a "$OUT/pipeline.log"
    cmd=(
      "$PY" "$SCRIPT"
      --task "$task"
      --split val_unseen
      --sample_seed -1
      --start_index "$start"
      --end_index "$end"
      --max_episodes 0
      --max_steps "$max_steps"
      --continuation_horizon "$max_steps"
      --teacher_json "$tjson"
      --gpu "$gpu"
      --router_mode "$mode"
      --strict_teacher_paths
      --out_dir "$shard_dir"
    )
    if [[ "$mode" == "trained" ]]; then
      cmd+=(--router_checkpoint "$model" --budget_key "$key")
    fi
    if [[ "$mode" == "random" ]]; then
      local budget
      budget="0.${key#b}"
      cmd+=(--budget "$budget")
    fi
    "${cmd[@]}" > "$eval_dir/logs/eval_${mode}_${key:-nokey}_shard_${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  "$PY" "$ROOT/edgecloud_experiments/continuation_router/summarize_object_cv_outputs.py" \
    --input_glob "$eval_dir/shard_*/*/$mode/cv_object_*.jsonl" \
    --out "$eval_dir/summary_full.json" \
    --name "${task}_${mode}_${key:-nokey}"
  echo "DONE eval $task $mode $key $(date)" | tee -a "$OUT/pipeline.log"
}

collect_task REVERIE 15 1423
collect_task SOON 20 1130
train_task REVERIE
train_task SOON

# Full val-unseen sizes from load_episodes(): REVERIE=3521, SOON=3390.
for task in REVERIE SOON; do
  if [[ "$task" == "REVERIE" ]]; then
    steps=15
    total=3521
  else
    steps=20
    total=3390
  fi
  eval_sharded "$task" oracle "" "$steps" "$total"
  for key in b30 b40 b50 b60; do
    eval_sharded "$task" trained "$key" "$steps" "$total"
  done
done

echo "FINISH object CV pipeline $(date)" | tee -a "$OUT/pipeline.log"
