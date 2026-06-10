#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
cd "$ROOT"

SAMPLES=build/continuation_router/cv_train_r2r_2000_v1/samples_train_2000_cv.jsonl
BASE_OUT=build/continuation_router/cv_router_r2r_2000_v2

if [[ ! -f "$SAMPLES" ]]; then
  echo "Missing samples: $SAMPLES" >&2
  exit 2
fi

run_variant() {
  local name=$1
  local cost_lambda=$2
  local supervised_coef=$3
  local rank_coef=$4
  local utility_reg_coef=$5
  local out_dir="$BASE_OUT/$name"
  mkdir -p "$out_dir"
  echo "[train] $name cost=$cost_lambda sup=$supervised_coef rank=$rank_coef ureg=$utility_reg_coef"
  "$PY" edgecloud_experiments/continuation_router/train_cv_group_router.py \
    --samples "$SAMPLES" \
    --out_dir "$out_dir" \
    --epochs 90 \
    --lr 6e-4 \
    --hidden 128 \
    --dropout 0.10 \
    --episodes_per_batch 64 \
    --rollouts_per_episode 8 \
    --cost_lambda "$cost_lambda" \
    --target_budget 0.40 \
    --budget_penalty 0.20 \
    --entropy_coef 0.004 \
    --supervised_coef "$supervised_coef" \
    --rank_coef "$rank_coef" \
    --utility_reg_coef "$utility_reg_coef" \
    --budgets 0.10,0.20,0.30,0.40,0.50 \
    > "$out_dir/train.log" 2>&1
  tail -n 1 "$out_dir/train.log"
}

# Utility values are small after continuation verification, so v2 sweeps lower
# cloud-call prices and adds a stabilizing label/ranking objective.
run_variant l005_sup075_rank025 0.05 0.75 0.25 0.05
run_variant l010_sup075_rank025 0.10 0.75 0.25 0.05
run_variant l003_sup050_rank050 0.03 0.50 0.50 0.05

"$PY" - <<'PY'
import glob
import json
from pathlib import Path

rows = []
for path in sorted(glob.glob("build/continuation_router/cv_router_r2r_2000_v2/*/training_summary.json")):
    data = json.load(open(path))
    keep = {
        "variant": Path(path).parent.name,
        "best_epoch": data.get("best_epoch"),
        "cost_lambda": data.get("cost_lambda"),
        "supervised_coef": data.get("supervised_coef"),
        "rank_coef": data.get("rank_coef"),
        "utility_reg_coef": data.get("utility_reg_coef"),
        "positive_rate": data.get("positive_rate"),
    }
    for b in ["b20", "b30", "b40", "b50"]:
        m = data.get("budget_metrics", {}).get(b, {})
        keep[f"{b}_threshold"] = m.get("threshold")
        keep[f"{b}_call_rate"] = m.get("call_rate")
        keep[f"{b}_critical_precision"] = m.get("critical_precision")
        keep[f"{b}_utility_per_100"] = m.get("utility_sum_per_100_steps")
    rows.append(keep)
out = Path("build/continuation_router/cv_router_r2r_2000_v2/training_variant_summary.json")
out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY

echo "[done] variant summaries: $BASE_OUT/training_variant_summary.json"
