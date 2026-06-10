#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=${ROOT}/build/object_router/soon_router_clean_split_v2/eval_controls
TEACHER=${ROOT}/build/object_router/soon_router_clean_split_v2/SOON_val_unseen_teacher_remapped.json
SCRIPT=${ROOT}/edgecloud_experiments/object_router/eval_object_edgecloud_nav.py

cd "${ROOT}"
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/pipeline.log") 2>&1

echo "START SOON clean-split controls $(date)"

if [[ ! -s "${TEACHER}" ]]; then
  echo "Missing remapped teacher cache: ${TEACHER}" >&2
  echo "Run edgecloud_experiments/object_router/run_soon_router_clean_split_v1.sh first." >&2
  exit 2
fi

run_method() {
  local name="$1"
  local gpu="$2"
  shift 2
  local dir="${OUT}/${name}"
  mkdir -p "${dir}"
  if find "${dir}" -name 'summary_*.json' -type f | grep -q .; then
    echo "SKIP ${name}; summary exists"
    return 0
  fi
  echo "START ${name} gpu=${gpu} $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" \
    --task SOON \
    --split val_unseen \
    --max_episodes 300 \
    --max_steps 20 \
    --sample_seed 20260507 \
    --teacher_json "${TEACHER}" \
    --gpu "${gpu}" \
    --strict_teacher_paths \
    --out_dir "${dir}" \
    "$@" \
    > "${dir}/run.log" 2>&1 &
  local pid="$!"
  echo "${pid}" > "${dir}/pid"
  BATCH_PIDS+=("${pid}")
  echo "PID ${name} ${pid}"
}

wait_batch() {
  local pid
  local rc=0
  for pid in "${BATCH_PIDS[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  BATCH_PIDS=()
  return "${rc}"
}

run_batch_1() {
  BATCH_PIDS=()
  run_method small 4 --router_mode small
  run_method cloud 5 --router_mode cloud
  run_method random_b20 6 --router_mode random --budget 0.20
  run_method random_b40 7 --router_mode random --budget 0.40
  wait_batch
}

run_batch_2() {
  BATCH_PIDS=()
  run_method random_b50 4 --router_mode random --budget 0.50
  run_method heuristic_t035 5 --router_mode heuristic --threshold 0.35
  run_method heuristic_t045 6 --router_mode heuristic --threshold 0.45
  run_method heuristic_t055 7 --router_mode heuristic --threshold 0.55
  wait_batch
}

run_batch_3() {
  BATCH_PIDS=()
  run_method oracle 4 --router_mode oracle
  wait_batch
}

run_batch_1
run_batch_2
run_batch_3

echo "DONE SOON clean-split controls $(date)"
find "${OUT}" -name 'summary_*.json' -print -exec cat {} \;
