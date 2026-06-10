#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=${ROOT}/build/object_router/reverie_small_comparable_300_v1
TEACHER=${ROOT}/build/object_router/cloud_navillm_full_v1/reverie/REVERIE_val_unseen.json
SCRIPT=${ROOT}/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/pipeline.log") 2>&1

cd "${ROOT}"
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

echo "START REVERIE comparable small baseline $(date)"
while pgrep -af eval_object_edgecloud_nav.py | grep -q reverie_router_train_300_v1; do
  echo "WAIT_REVERIE_ROUTER_PIPELINE $(date)"
  sleep 120
done
while pgrep -af run_reverie_router_train_eval_300_v1.sh >/dev/null; do
  echo "WAIT_REVERIE_RUN_SCRIPT $(date)"
  sleep 120
done

CUDA_VISIBLE_DEVICES=4 "${PY}" "${SCRIPT}" \
  --task REVERIE \
  --split val_unseen \
  --max_episodes 300 \
  --max_steps 15 \
  --sample_seed 20260507 \
  --teacher_json "${TEACHER}" \
  --gpu 4 \
  --router_mode small \
  --out_dir "${OUT}" \
  > "${OUT}/run.log" 2>&1

echo "DONE REVERIE comparable small baseline $(date)"
find "${OUT}" -name 'summary_*.json' -print -exec cat {} \;
