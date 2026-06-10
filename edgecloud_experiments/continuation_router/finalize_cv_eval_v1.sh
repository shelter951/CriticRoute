#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}

PY=${PYTHON:-python}
ROOT=build/continuation_router/cv_eval_val_unseen_v1
REPORT=edgecloud_experiments/reports/CONTINUATION_VERIFIED_ROUTER_RESULTS_20260512_ZH.md

while pgrep -f 'continuation_router/eval_cv_edgecloud_r2r.py.*cv_eval_val_unseen_v1' >/dev/null; do
  sleep 60
done

mkdir -p "$(dirname "$REPORT")"
{
  echo "# Continuation-Verified Router Full Eval Results"
  echo
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  echo "## Result summaries"
  echo
  for d in "$ROOT"/cv_group_b30 "$ROOT"/cv_group_b40 "$ROOT"/cv_group_b50 "$ROOT"/cv_oracle; do
    name=$(basename "$d")
    result=$(ls "$d"/cv_*_val_unseen_*.jsonl 2>/dev/null | tail -1 || true)
    echo "### $name"
    if [ -n "$result" ]; then
      "$PY" edgecloud_experiments/continuation_router/summarize_cv_outputs.py \
        --kind results \
        --inputs "$result" \
        --out "$d/result_summary.json"
    else
      echo "No result JSONL found."
    fi
    echo
  done
  echo "## Training sample summary"
  cat build/continuation_router/cv_train_r2r_2000_v1/sample_summary.json
  echo
  echo "## Router training summary"
  cat build/continuation_router/cv_router_r2r_2000_v1/l030_btarget40/training_summary.json
} > "$REPORT"

echo "WROTE $REPORT"

