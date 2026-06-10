#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
cd "$ROOT"

TRAIN_ROOT=build/continuation_router/cv_router_r2r_2000_v2
SUMMARY=$TRAIN_ROOT/training_variant_summary.json
TEACHER_JSON=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
OUT_ROOT=build/continuation_router/cv_eval_val_unseen_v2

echo "[watch] waiting for v2 variant training summary: $SUMMARY"
while [[ ! -s "$SUMMARY" ]]; do
  sleep 60
done

BEST=$("$PY" - <<'PY'
import json
rows = json.load(open("build/continuation_router/cv_router_r2r_2000_v2/training_variant_summary.json"))
def score(r):
    return float(r.get("b40_utility_per_100") or 0.0) + 0.25 * float(r.get("b40_critical_precision") or 0.0)
best = max(rows, key=score)
print(best["variant"])
PY
)

CKPT=$TRAIN_ROOT/$BEST/hetero_router.pt
if [[ ! -f "$CKPT" ]]; then
  echo "Missing v2 checkpoint: $CKPT" >&2
  exit 2
fi
echo "[select] best v2 variant: $BEST"

echo "[watch] waiting for v1 full val-unseen eval to finish"
while pgrep -af "eval_cv_edgecloud_r2r.py.*cv_eval_val_unseen_v1" >/dev/null; do
  sleep 120
done

mkdir -p "$OUT_ROOT"
launch_eval() {
  local budget=$1
  local gpu=$2
  local out_dir="$OUT_ROOT/${BEST}_${budget}"
  mkdir -p "$out_dir"
  echo "[launch] $BEST $budget on gpu $gpu -> $out_dir"
  nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
    --split val_unseen \
    --max_episodes 0 \
    --sample_seed -1 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER_JSON" \
    --router_mode trained \
    --router_ckpt "$CKPT" \
    --budget_key "$budget" \
    --out_dir "$out_dir" \
    > "$out_dir/nohup.log" 2>&1 &
}

launch_eval b30 4
launch_eval b40 5
launch_eval b50 6

cat > "$OUT_ROOT/README.txt" <<EOF
Continuation-verified v2 full val-unseen evaluation.
Selected variant: $BEST
Checkpoint: $CKPT
Launched: $(date '+%F %T')
EOF

nohup bash edgecloud_experiments/continuation_router/finalize_cv_eval_v2.sh \
  > build/continuation_router/finalize_cv_eval_v2.log 2>&1 &

echo "[done] v2 eval launched"
