#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
SUM_SCRIPT=edgecloud_experiments/hetero_router/summarize_hetero_results.py
ROOT=build/hetero_router/eval_val_unseen_full_key_v1
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
BINARY_CKPT=build/hetero_router/train_r2r_2000_v1/router_binary/hetero_router.pt
CRITICAL_CKPT=build/hetero_router/train_r2r_2000_v1/router_critical/hetero_router.pt

mkdir -p "${ROOT}"

run_method() {
  local name="$1"
  shift
  local method_dir="${ROOT}/${name}"
  mkdir -p "${method_dir}"
  echo "START ${name} $(date)" | tee "${method_dir}/status.log"

  local ranges=("0 588 4" "588 1176 5" "1176 1764 6" "1764 0 7")
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
    echo "$!" > "${shard_dir}/pid"
    echo "${name} shard_${i} pid=$(cat "${shard_dir}/pid") gpu=${gpu} range=${start}:${end}" | tee -a "${method_dir}/status.log"
  done

  wait
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

run_method critical_b30 --router_mode trained --router_ckpt "${CRITICAL_CKPT}" --budget_key b30
run_method critical_b40 --router_mode trained --router_ckpt "${CRITICAL_CKPT}" --budget_key b40
run_method binary_b50 --router_mode trained --router_ckpt "${BINARY_CKPT}" --budget_key b50
run_method random_b40 --router_mode random --budget 0.40
run_method small --router_mode small

echo "Full key-method eval finished."
find "${ROOT}" -maxdepth 2 -name 'summary_full.json' -print | sort

