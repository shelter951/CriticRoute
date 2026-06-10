#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-${PROJECT_ROOT:-.}}
PY=${PY:-${PYTHON:-python}}
OUT=$ROOT/build/continuation_router/label_ablation_r2r_v1
ORIGINAL=$ROOT/build/continuation_router/cv_train_r2r_2000_v1/samples_train_2000_cv.jsonl
ROUTED=$ROOT/build/continuation_router/cv_train_routed_r2r_2000_v3/samples_train_2000_routed_b40_cv.jsonl
TRAIN=$ROOT/edgecloud_experiments/continuation_router/train_cv_group_router.py
EVAL=$ROOT/edgecloud_experiments/continuation_router/eval_cv_edgecloud_r2r.py
TEACHER=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json

export PYTHONPATH="${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:$ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT" "$OUT/logs" "$OUT/samples" "$OUT/models" "$OUT/eval"

"$PY" - <<'PY'
import json
import pathlib

root = pathlib.Path("${PROJECT_ROOT:-.}/build/continuation_router/label_ablation_r2r_v1")
inputs = [
    pathlib.Path("${PROJECT_ROOT:-.}/build/continuation_router/cv_train_r2r_2000_v1/samples_train_2000_cv.jsonl"),
    pathlib.Path("${PROJECT_ROOT:-.}/build/continuation_router/cv_train_routed_r2r_2000_v3/samples_train_2000_routed_b40_cv.jsonl"),
]
rows = []
for path in inputs:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))


def same_action(row):
    return str(row.get("qwen_action")) == str(row.get("cloud_action"))


def transform(mode, row):
    out = dict(row)
    if mode == "disagreement":
        y = 0.0 if same_action(row) or row.get("cloud_action") is None else 1.0
        reason = "same_action" if y == 0.0 else "action_disagreement"
    elif mode == "one_step":
        y = float(row.get("one_step_label", 0.0) or 0.0)
        reason = row.get("one_step_reason", "safe")
    elif mode == "success_only":
        edge_success = float((row.get("cv_edge") or {}).get("success", 0.0) or 0.0)
        cloud_success = float((row.get("cv_cloud") or {}).get("success", 0.0) or 0.0)
        y = 1.0 if cloud_success > edge_success else 0.0
        reason = "success_flip_cont" if y else "no_success_flip"
    elif mode == "cv":
        y = float(row.get("label", 0.0) or 0.0)
        reason = row.get("label_reason", "")
    else:
        raise ValueError(mode)
    out["label"] = y
    out["label_reason"] = reason
    out["cv_utility"] = float(y)
    return out


summary = {}
for mode in ["disagreement", "one_step", "success_only", "cv"]:
    out_path = root / "samples" / f"samples_{mode}.jsonl"
    pos = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            item = transform(mode, row)
            pos += float(item["label"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary[mode] = {
        "rows": len(rows),
        "positive": int(pos),
        "positive_rate": pos / len(rows) if rows else 0.0,
    }

(root / "label_ablation_sample_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

train_mode() {
  local mode=$1
  mkdir -p "$OUT/models/$mode"
  "$PY" "$TRAIN" \
    --samples "$OUT/samples/samples_${mode}.jsonl" \
    --out_dir "$OUT/models/$mode" \
    --epochs 80 \
    --hidden 128 \
    --dropout 0.10 \
    --episodes_per_batch 64 \
    --target_budget 0.40 \
    --cost_lambda 0.10 \
    --budget_penalty 0.20 \
    --budgets 0.40,0.50 \
    --policy_coef 0.0 \
    --entropy_coef 0.0 \
    --supervised_coef 1.0 \
    --rank_coef 0.0 \
    --utility_reg_coef 0.0 \
    > "$OUT/logs/train_${mode}.log" 2>&1
}

for mode in disagreement one_step success_only cv; do
  train_mode "$mode"
done

launch_eval() {
  local mode=$1
  local gpu=$2
  local key=$3
  mkdir -p "$OUT/eval/${mode}_${key}"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" "$EVAL" \
    --split val_unseen \
    --sample_seed -1 \
    --max_episodes 0 \
    --gpu "$gpu" \
    --teacher_json "$TEACHER" \
    --router_mode trained \
    --router_ckpt "$OUT/models/$mode/router.pt" \
    --budget_key "$key" \
    --out_dir "$OUT/eval/${mode}_${key}" \
    > "$OUT/logs/eval_${mode}_${key}.log" 2>&1 &
  echo $! > "$OUT/logs/eval_${mode}_${key}.pid"
}

launch_eval disagreement 4 b40
launch_eval one_step 5 b40
launch_eval success_only 6 b40
launch_eval cv 7 b40

echo "label ablation b40 eval launched at $(date)"
