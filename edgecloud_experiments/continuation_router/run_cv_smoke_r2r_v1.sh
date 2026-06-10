#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}

export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
PY=${PYTHON:-python}
ROOT=build/continuation_router/cv_smoke_r2r_v1
TEACHER_JSON=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_seen_decisions_v1/R2R_val_seen.json
mkdir -p "$ROOT"

"$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
  --split val_seen \
  --max_episodes 24 \
  --sample_seed 20260512 \
  --gpu 4 \
  --teacher_json "$TEACHER_JSON" \
  --router_mode oracle \
  --continuation_horizon 15 \
  --out_dir "$ROOT/oracle_cv" \
  --samples_out "$ROOT/oracle_cv_samples.jsonl"

"$PY" edgecloud_experiments/continuation_router/train_cv_group_router.py \
  --samples "$ROOT/oracle_cv_samples.jsonl" \
  --out_dir "$ROOT/router_cv" \
  --epochs 30 \
  --target_budget 0.40 \
  --cost_lambda 0.30

"$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
  --split val_unseen \
  --max_episodes 40 \
  --sample_seed 20260512 \
  --gpu 4 \
  --teacher_json ${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json \
  --router_mode trained \
  --router_ckpt "$ROOT/router_cv/hetero_router.pt" \
  --budget_key b40 \
  --out_dir "$ROOT/eval_trained_b40"
