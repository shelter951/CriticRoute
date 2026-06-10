#!/usr/bin/env bash
set -euo pipefail

# Extra threshold sweep for continuation-verified object-router checkpoints.
# This script is intentionally isolated from run_cv_object_reverie_soon_v1.sh:
# it never overwrites existing checkpoints or summaries and should be launched
# only after the main REVERIE/SOON CV pipeline releases GPUs 4--7.

ROOT=${ROOT:-${PROJECT_ROOT:-.}}
PY=${PY:-${PYTHON:-python}}
SCRIPT="$ROOT/edgecloud_experiments/continuation_router/eval_cv_object_edgecloud_nav.py"
SUMMARIZE="$ROOT/edgecloud_experiments/continuation_router/summarize_object_cv_outputs.py"
OUT=${OUT:-$ROOT/build/continuation_router/object_cv_threshold_sweep_v1}

mkdir -p "$OUT"

eval_threshold() {
  local task=$1
  local label=$2
  local threshold=$3
  local max_steps=$4
  local total=$5
  local model=$6
  local teacher_json=$7

  local eval_dir="$OUT/${task,,}/eval/${label}"
  mkdir -p "$eval_dir/logs"
  local spans=("0 $((total/4))" "$((total/4)) $((total/2))" "$((total/2)) $((3*total/4))" "$((3*total/4)) $total")
  local pids=()

  for i in 0 1 2 3; do
    local start end gpu shard_dir
    read -r start end <<<"${spans[$i]}"
    gpu=$((4+i))
    shard_dir="$eval_dir/shard_${i}"
    echo "START threshold sweep $task $label threshold=$threshold shard=$i gpu=$gpu range=$start:$end $(date)" | tee -a "$OUT/pipeline.log"
    "$PY" "$SCRIPT" \
      --task "$task" \
      --split val_unseen \
      --sample_seed -1 \
      --start_index "$start" \
      --end_index "$end" \
      --max_episodes 0 \
      --max_steps "$max_steps" \
      --continuation_horizon "$max_steps" \
      --teacher_json "$teacher_json" \
      --gpu "$gpu" \
      --router_mode trained \
      --strict_teacher_paths \
      --out_dir "$shard_dir" \
      --router_checkpoint "$model" \
      --budget_key b40 \
      --trained_threshold "$threshold" \
      > "$eval_dir/logs/eval_${label}_shard_${i}.log" 2>&1 &
    pids+=($!)
  done

  wait "${pids[@]}"
  "$PY" "$SUMMARIZE" \
    --input_glob "$eval_dir/shard_*/*/trained/cv_object_*.jsonl" \
    --out "$eval_dir/summary_full.json" \
    --name "${task}_${label}"
  echo "DONE threshold sweep $task $label $(date)" | tee -a "$OUT/pipeline.log"
}

REVERIE_MODEL="$ROOT/build/continuation_router/object_cv_reverie_soon_v1/reverie/models/cv_l010_sup075_rank025/hetero_router.pt"
REVERIE_TEACHER="$ROOT/build/object_router/cloud_navillm_full_v1/reverie/REVERIE_val_unseen.json"
SOON_MODEL="$ROOT/build/continuation_router/object_cv_reverie_soon_v1/soon/models/cv_l010_sup075_rank025/hetero_router.pt"
SOON_TEACHER="$ROOT/build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json"

# Lower thresholds increase cloud-call rate. Values are extrapolated from the
# b50/b60 calibration thresholds and are meant to diagnose whether object-CV is
# under-calling on val_unseen, not to change the online information boundary.
eval_threshold REVERIE t032 0.032 15 3521 "$REVERIE_MODEL" "$REVERIE_TEACHER"
eval_threshold REVERIE t024 0.024 15 3521 "$REVERIE_MODEL" "$REVERIE_TEACHER"
eval_threshold SOON t038 0.038 20 3390 "$SOON_MODEL" "$SOON_TEACHER"
eval_threshold SOON t030 0.030 20 3390 "$SOON_MODEL" "$SOON_TEACHER"

echo "FINISH threshold sweep $(date)" | tee -a "$OUT/pipeline.log"
