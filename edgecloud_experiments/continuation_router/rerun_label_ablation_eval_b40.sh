#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-${PROJECT_ROOT:-.}}
OUT=$ROOT/build/continuation_router/label_ablation_r2r_v1
PY=${PY:-${PYTHON:-python}}
EVAL=$ROOT/edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py
TEACHER=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"

launch_eval() {
  local mode=$1
  local gpu=$2
  local dir="$OUT/eval/${mode}_b40_rerun"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" "$EVAL" \
    --split val_unseen \
    --sample_seed -1 \
    --max_episodes 0 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER" \
    --router_mode trained \
    --router_ckpt "$OUT/models/$mode/hetero_router.pt" \
    --budget_key b40 \
    --out_dir "$dir" \
    > "$OUT/logs/eval_${mode}_b40_rerun.log" 2>&1 &
  echo $! > "$OUT/logs/eval_${mode}_b40_rerun.pid"
}

launch_eval disagreement 4
launch_eval one_step 5
launch_eval success_only 6
launch_eval cv 7

echo "label ablation b40 rerun launched at $(date)"
