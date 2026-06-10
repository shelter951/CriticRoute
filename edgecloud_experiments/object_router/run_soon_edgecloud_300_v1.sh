#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=${ROOT}/build/object_router/soon_edgecloud_300_v1
TEACHER=${ROOT}/build/object_router/cloud_navillm_full_v1/soon/SOON_val_unseen.json
SCRIPT=${ROOT}/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/pipeline.log") 2>&1

cd "${ROOT}"
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

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
