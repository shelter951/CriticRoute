#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
cd "$ROOT"

TEACHER_TRAIN=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_train_decisions_v2/R2R_train.json
TEACHER_VAL_UNSEEN=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
V2_VARIANT=l010_sup075_rank025
V2_CKPT=build/continuation_router/cv_router_r2r_2000_v2/${V2_VARIANT}/hetero_router.pt
V2_EVAL_ROOT=build/continuation_router/cv_eval_val_unseen_v2
V3_DATA=build/continuation_router/cv_train_routed_r2r_2000_v3
V3_ROUTER=build/continuation_router/cv_router_r2r_2000_v3/routed_b40_l010_sup075_rank025
V3_EVAL=build/continuation_router/cv_eval_val_unseen_v3
ORIGINAL_SAMPLES=build/continuation_router/cv_train_r2r_2000_v1/samples_train_2000_cv.jsonl

echo "[watch] waiting for v2 result summaries"
while true; do
  if [[ -s "$V2_EVAL_ROOT/${V2_VARIANT}_b40/result_summary.json" && -s "$V2_EVAL_ROOT/${V2_VARIANT}_b50/result_summary.json" ]]; then
    break
  fi
  sleep 180
done

DECISION=$("$PY" - <<'PY'
import json
from pathlib import Path
root = Path("build/continuation_router/cv_eval_val_unseen_v2")
variant = "l010_sup075_rank025"
rows = []
for b in ["b30", "b40", "b50"]:
    p = root / f"{variant}_{b}" / "result_summary.json"
    if p.exists():
        d = json.load(open(p))
        rows.append((b, float(d.get("sr", 0.0)), float(d.get("spl", 0.0)), float(d.get("cloud", 0.0))))
best = max(rows, key=lambda x: x[1]) if rows else ("none", 0.0, 0.0, 0.0)
print(json.dumps({"best": best, "need_v3": best[1] < 66.5}, ensure_ascii=False))
PY
)
echo "[decision] $DECISION"
NEED_V3=$("$PY" - <<PY
import json
print("1" if json.loads('''$DECISION''')["need_v3"] else "0")
PY
)
if [[ "$NEED_V3" != "1" ]]; then
  echo "[stop] v2 already reaches the old-mainline target; no v3 aggregation needed."
  exit 0
fi

echo "[v3] collecting routed-state aggregation samples with v2 b40"
rm -rf "$V3_DATA"
mkdir -p "$V3_DATA"

launch_collect() {
  local shard=$1
  local start=$2
  local end=$3
  local gpu=$4
  local out="$V3_DATA/shard_${shard}"
  mkdir -p "$out"
  nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
    --split train \
    --start_index "$start" \
    --end_index "$end" \
    --max_episodes 0 \
    --sample_seed -1 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER_TRAIN" \
    --router_mode trained \
    --router_ckpt "$V2_CKPT" \
    --budget_key b40 \
    --continuation_horizon 15 \
    --samples_out "$out/samples.jsonl" \
    --out_dir "$out" \
    > "$out/nohup.log" 2>&1 &
  echo "$! shard=$shard gpu=$gpu range=[$start,$end)"
}

launch_collect 0 0 500 4
launch_collect 1 500 1000 5
launch_collect 2 1000 1500 6
launch_collect 3 1500 2000 7
wait

echo "[v3] merging routed samples"
"$PY" - <<'PY'
from pathlib import Path
import json
import numpy as np
from collections import Counter
root = Path("build/continuation_router/cv_train_routed_r2r_2000_v3")
out = root / "samples_train_2000_routed_b40_cv.jsonl"
rows = []
with out.open("w", encoding="utf-8") as fw:
    for p in sorted(root.glob("shard_*/samples.jsonl")):
        with p.open() as f:
            for line in f:
                if line.strip():
                    fw.write(line)
                    rows.append(json.loads(line))
summary = {
    "n": len(rows),
    "episodes": len({r.get("instr_id") for r in rows}),
    "positive_rate": float(np.mean([float(r.get("label", 0.0)) for r in rows])) if rows else 0.0,
    "mean_cv_utility": float(np.mean([float(r.get("cv_utility", 0.0) or 0.0) for r in rows])) if rows else 0.0,
    "label_counts": dict(Counter(r.get("label_reason", "unknown") for r in rows).most_common()),
}
(root / "sample_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY

echo "[v3] training router on original + routed-state samples"
mkdir -p "$V3_ROUTER"
"$PY" edgecloud_experiments/continuation_router/train_cv_group_router.py \
  --samples "$ORIGINAL_SAMPLES" "$V3_DATA/samples_train_2000_routed_b40_cv.jsonl" \
  --out_dir "$V3_ROUTER" \
  --epochs 90 \
  --lr 6e-4 \
  --hidden 128 \
  --dropout 0.10 \
  --episodes_per_batch 64 \
  --rollouts_per_episode 8 \
  --cost_lambda 0.10 \
  --target_budget 0.40 \
  --budget_penalty 0.20 \
  --entropy_coef 0.004 \
  --supervised_coef 0.75 \
  --rank_coef 0.25 \
  --utility_reg_coef 0.05 \
  --budgets 0.10,0.20,0.30,0.40,0.50 \
  > "$V3_ROUTER/train.log" 2>&1

echo "[v3] launching full val-unseen b30/b40/b50"
mkdir -p "$V3_EVAL"
launch_eval() {
  local budget=$1
  local gpu=$2
  local out="$V3_EVAL/routed_v3_${budget}"
  mkdir -p "$out"
  nohup "$PY" edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py \
    --split val_unseen \
    --max_episodes 0 \
    --sample_seed -1 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER_VAL_UNSEEN" \
    --router_mode trained \
    --router_ckpt "$V3_ROUTER/hetero_router.pt" \
    --budget_key "$budget" \
    --out_dir "$out" \
    > "$out/nohup.log" 2>&1 &
  echo "$! $budget gpu=$gpu"
}
launch_eval b30 4
launch_eval b40 5
launch_eval b50 6

nohup bash edgecloud_experiments/continuation_router/finalize_cv_eval_v3.sh \
  > build/continuation_router/finalize_cv_eval_v3.log 2>&1 &

echo "[done] v3 aggregation/eval pipeline launched"
