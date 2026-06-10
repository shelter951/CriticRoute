#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=${ROOT}/build/object_router/soon_edgecloud_300_v2_remap
SOURCE_TEACHER=${ROOT}/build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json
TEACHER=${OUT}/SOON_val_unseen_teacher_remapped.json
SCRIPT=${ROOT}/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/pipeline.log") 2>&1

cd "${ROOT}"
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

echo "START SOON edge-cloud remap v2 $(date)"

# Do not compete with the active REVERIE learned-router pipeline.  The wait is
# deliberately scoped to that output directory so this script does not wait on
# itself once SOON starts.
while pgrep -af eval_object_edgecloud_nav.py | grep -q reverie_router_train_300_v1; do
  echo "WAIT_REVERIE_ROUTER_PIPELINE $(date)"
  sleep 120
done
while pgrep -af run_reverie_router_train_eval_300_v1.sh >/dev/null; do
  echo "WAIT_REVERIE_RUN_SCRIPT $(date)"
  sleep 120
done
while pgrep -af run_reverie_small_comparable_300_v1.sh >/dev/null; do
  echo "WAIT_REVERIE_SMALL_BASELINE $(date)"
  sleep 120
done
while pgrep -af eval_object_edgecloud_nav.py | grep -q reverie_small_comparable_300_v1; do
  echo "WAIT_REVERIE_SMALL_EVAL $(date)"
  sleep 120
done

if [[ ! -s "${TEACHER}" ]]; then
  echo "BUILD_REMAP_TEACHER $(date)"
  "${PY}" - <<'PY'
import json
from pathlib import Path

from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import load_episodes

root = Path("${PROJECT_ROOT:-.}")
source = root / "build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json"
target = root / "build/object_router/soon_edgecloud_300_v2_remap/SOON_val_unseen_teacher_remapped.json"
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
print({"episodes": len(episodes), "teacher": len(teacher), "out": str(target)})
PY
fi

run_method() {
  local name="$1"
  local gpu="$2"
  shift 2
  local dir="${OUT}/${name}"
  mkdir -p "${dir}"
  echo "START ${name} gpu=${gpu} $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" \
    --task SOON \
    --split val_unseen \
    --max_episodes 300 \
    --max_steps 20 \
    --sample_seed 20260507 \
    --teacher_json "${TEACHER}" \
    --gpu "${gpu}" \
    --out_dir "${dir}" \
    "$@" \
    > "${dir}/run.log" 2>&1 &
  echo "$!" > "${dir}/pid"
  echo "PID ${name} $(cat "${dir}/pid")"
}

run_method cloud 4 --router_mode cloud
run_method random_b40 5 --router_mode random --budget 0.40
run_method heuristic_t045 6 --router_mode heuristic --threshold 0.45
run_method oracle 7 --router_mode oracle

wait

echo "DONE_ALL $(date)"
find "${OUT}" -name 'summary_*.json' -print -exec cat {} \;
