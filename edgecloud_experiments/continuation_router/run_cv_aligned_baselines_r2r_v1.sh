#!/usr/bin/env bash
set -euo pipefail

# Aligned continuation-label baselines for CriticRoute-Path.
#
# Goal:
#   Compare the group-relative path router against standard learners that use
#   the same continuation-verified samples, the same telemetry features, and
#   the same budget calibration. This isolates whether the sequence objective
#   adds value beyond the label-mining pipeline.
#
# GPU rule:
#   Only use GPUs 4-7 on the new server.

ROOT=${PROJECT_ROOT:-.}
PY=${PY:-${PYTHON:-python}}
TRAIN_SCRIPT=$ROOT/edgecloud_experiments/continuation_router/train_cv_group_router.py
EVAL_SCRIPT=$ROOT/edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py
SUM_SCRIPT=$ROOT/edgecloud_experiments/continuation_router/summarize_cv_outputs.py
export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
TEACHER_VAL_UNSEEN=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json

ORIGINAL=$ROOT/build/continuation_router/cv_train_r2r_2000_v1/samples_train_2000_cv.jsonl
ROUTED=$ROOT/build/continuation_router/cv_train_routed_r2r_2000_v3/samples_train_2000_routed_b40_cv.jsonl
OUT=$ROOT/build/continuation_router/cv_aligned_baselines_r2r_v1
mkdir -p "$OUT/logs"

train_variant() {
  local name="$1"; shift
  local dir="$OUT/models/$name"
  if [[ -f "$dir/hetero_router.pt" ]]; then
    echo "SKIP train $name; checkpoint exists"
    return
  fi
  mkdir -p "$dir"
  echo "START train $name $(date)"
  "$PY" "$TRAIN_SCRIPT" \
    --samples "$ORIGINAL" "$ROUTED" \
    --out_dir "$dir" \
    --epochs 100 \
    --hidden 128 \
    --dropout 0.10 \
    --episodes_per_batch 64 \
    --rollouts_per_episode 8 \
    --target_budget 0.40 \
    --cost_lambda 0.10 \
    --budget_penalty 0.20 \
    --budgets 0.30,0.40,0.50 \
    "$@" \
    > "$OUT/logs/train_${name}.log" 2>&1
  echo "DONE train $name $(date)"
}

eval_variant_budget() {
  local name="$1"
  local key="$2"
  local gpu="$3"
  local model="$OUT/models/$name/hetero_router.pt"
  local eval_dir="$OUT/eval/${name}_${key}"
  if [[ -f "$eval_dir/result_summary.json" ]]; then
    echo "SKIP eval $name $key; summary exists"
    return
  fi
  mkdir -p "$eval_dir"
  echo "START eval $name $key gpu=$gpu $(date)"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" "$EVAL_SCRIPT" \
    --split val_unseen \
    --sample_seed -1 \
    --max_episodes 0 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER_VAL_UNSEEN" \
    --router_mode trained \
    --router_ckpt "$model" \
    --budget_key "$key" \
    --out_dir "$eval_dir" \
    > "$OUT/logs/eval_${name}_${key}.log" 2>&1 &
}

finalize_variant_budget() {
  local name="$1"
  local key="$2"
  local eval_dir="$OUT/eval/${name}_${key}"
  local result
  result=$(ls "$eval_dir"/cv_trained_val_unseen_*.jsonl 2>/dev/null | tail -1 || true)
  if [[ -z "$result" ]]; then
    echo "WAIT result $name $key"
    return 1
  fi
  "$PY" "$SUM_SCRIPT" \
    --kind results \
    --inputs "$result" \
    --out "$eval_dir/result_summary.json" \
    > "$eval_dir/summary_stdout.json"
}

echo "START aligned CV baselines $(date)"

# No group-relative policy objective: standard learners on the same CV labels.
train_variant bce_only --policy_coef 0.0 --entropy_coef 0.0 --supervised_coef 1.0 --rank_coef 0.0 --utility_reg_coef 0.0
train_variant rank_only --policy_coef 0.0 --entropy_coef 0.0 --supervised_coef 0.0 --rank_coef 1.0 --utility_reg_coef 0.0
train_variant utility_reg --policy_coef 0.0 --entropy_coef 0.0 --supervised_coef 0.0 --rank_coef 0.0 --utility_reg_coef 1.0
train_variant bce_rank_reg --policy_coef 0.0 --entropy_coef 0.0 --supervised_coef 0.75 --rank_coef 0.25 --utility_reg_coef 0.05

# Keep GPU use bounded. Full evals are launched by budget for the most useful b40/b50 points.
for name in bce_only rank_only utility_reg bce_rank_reg; do
  eval_variant_budget "$name" b40 4
  eval_variant_budget "$name" b50 5
  wait
  finalize_variant_budget "$name" b40 || true
  finalize_variant_budget "$name" b50 || true
done

echo "FINISH aligned CV baselines $(date)"
