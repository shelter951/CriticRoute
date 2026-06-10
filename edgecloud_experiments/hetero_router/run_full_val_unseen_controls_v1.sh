#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
SUM_SCRIPT=edgecloud_experiments/hetero_router/summarize_hetero_results.py
ROOT=build/hetero_router/eval_val_unseen_full_controls_v1
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json

mkdir -p "${ROOT}"
exec > >(tee -a "${ROOT}/pipeline.log") 2>&1

run_method() {
  local name="$1"
  shift
  local method_dir="${ROOT}/${name}"
  if [[ -f "${method_dir}/summary_full.json" ]]; then
    echo "SKIP ${name}; summary_full exists"
    return 0
  fi
  if [[ -e "${method_dir}" ]]; then
    echo "Refusing to reuse incomplete method_dir=${method_dir}; move it aside first." >&2
    exit 2
  fi
  mkdir -p "${method_dir}"
  echo "START ${name} $(date)" | tee "${method_dir}/status.log"

  local ranges=("0 588 0" "588 1176 1" "1176 1764 2" "1764 0 3")
  local pids=()
  for i in 0 1 2 3; do
    read -r start end gpu <<< "${ranges[$i]}"
    local shard_dir="${method_dir}/shard_${i}"
    mkdir -p "${shard_dir}"
    nohup "${PY}" "${EVAL_SCRIPT}" \
      --split val_unseen \
      --sample_seed 20260504 \
      --start_index "${start}" \
      --end_index "${end}" \
      --max_episodes 0 \
      --teacher_json "${TEACHER_VAL}" \
      --out_dir "${shard_dir}" \
      --gpu "${gpu}" \
      --max_steps 15 \
      "$@" \
      > "${shard_dir}/run.log" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]}" > "${shard_dir}/pid"
    echo "${name} shard_${i} pid=${pids[-1]} gpu=${gpu} range=${start}:${end}" | tee -a "${method_dir}/status.log"
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "MERGE ${name} $(date)" | tee -a "${method_dir}/status.log"
  mapfile -t jsonls < <(find "${method_dir}" -mindepth 2 -maxdepth 2 -name 'hetero_*.jsonl' | sort)
  if [[ "${#jsonls[@]}" -lt 4 ]]; then
    echo "Expected 4 shard jsonl files for ${name}, found ${#jsonls[@]}" >&2
    exit 3
  fi
  cat "${jsonls[@]}" > "${method_dir}/merged_results.jsonl"
  "${PY}" "${SUM_SCRIPT}" \
    --inputs "${method_dir}/merged_results.jsonl" \
    --name "${name}" \
    --out "${method_dir}/summary_full.json" \
    | tee "${method_dir}/summary_full.stdout"
  echo "DONE ${name} $(date)" | tee -a "${method_dir}/status.log"
}

echo "START R2R full controls $(date)"
run_method random_b20 --router_mode random --budget 0.20
run_method random_b50 --router_mode random --budget 0.50
run_method heuristic_t035 --router_mode heuristic --threshold 0.35
run_method heuristic_t045 --router_mode heuristic --threshold 0.45
run_method heuristic_t055 --router_mode heuristic --threshold 0.55
echo "DONE R2R full controls $(date)"
find "${ROOT}" -maxdepth 2 -name 'summary_full.json' -print | sort
