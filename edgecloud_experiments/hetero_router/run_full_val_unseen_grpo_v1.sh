#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
SUM_SCRIPT=edgecloud_experiments/hetero_router/summarize_hetero_results.py
ROOT=${ROOT:-build/hetero_router/eval_val_unseen_full_grpo_v1}
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
GRPO_TRAIN=build/hetero_router/grpo_router_r2r_2000_v1
GRPO_L035_CKPT=${GRPO_TRAIN}/l035_btarget30/hetero_router.pt

mkdir -p "${ROOT}"
exec > >(tee -a "${ROOT}/pipeline.log") 2>&1

run_method_2gpu() {
  local name="$1"
  local budget_key="$2"
  local method_dir="${ROOT}/${name}"
  if [[ -f "${method_dir}/summary_full.json" ]]; then
    echo "SKIP ${name}: summary_full exists"
    return 0
  fi
  if [[ -e "${method_dir}" ]]; then
    echo "Refusing to reuse incomplete method_dir=${method_dir}; move it aside first." >&2
    exit 2
  fi
  mkdir -p "${method_dir}"
  echo "START ${name} budget=${budget_key} $(date)" | tee "${method_dir}/status.log"

  local ranges=("0 1176 6" "1176 0 7")
  local pids=()
  for i in 0 1; do
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
      --router_mode trained \
      --router_ckpt "${GRPO_L035_CKPT}" \
      --budget_key "${budget_key}" \
      > "${shard_dir}/run.log" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]}" > "${shard_dir}/pid"
    echo "${name} shard_${i} pid=${pids[-1]} gpu=${gpu} range=${start}:${end}" | tee -a "${method_dir}/status.log"
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "MERGE ${name} $(date)" | tee -a "${method_dir}/status.log"
  local jsonls=()
  for i in 0 1; do
    local shard_dir="${method_dir}/shard_${i}"
    mapfile -t shard_jsonls < <(find "${shard_dir}" -maxdepth 1 -name 'hetero_*.jsonl' | sort)
    if [[ "${#shard_jsonls[@]}" -ne 1 ]]; then
      echo "Expected exactly 1 jsonl in ${shard_dir}, found ${#shard_jsonls[@]}" >&2
      printf '%s\n' "${shard_jsonls[@]}" >&2
      exit 3
    fi
    jsonls+=("${shard_jsonls[0]}")
  done
  cat "${jsonls[@]}" > "${method_dir}/merged_results.jsonl"
  "${PY}" "${SUM_SCRIPT}" \
    --inputs "${method_dir}/merged_results.jsonl" \
    --name "${name}" \
    --out "${method_dir}/summary_full.json" \
    | tee "${method_dir}/summary_full.stdout"
  echo "DONE ${name} $(date)" | tee -a "${method_dir}/status.log"
}

echo "START_GRPO_FULL $(date)"
run_method_2gpu grpo_l035_b40 b40
run_method_2gpu grpo_l035_b30 b30
echo "DONE_GRPO_FULL $(date)"
find "${ROOT}" -maxdepth 2 -name 'summary_full.json' -print | sort
