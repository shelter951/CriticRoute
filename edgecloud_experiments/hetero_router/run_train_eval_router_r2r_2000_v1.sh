#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
COLLECT_ROOT=build/hetero_router/train_r2r_2000_v1
MERGED="${COLLECT_ROOT}/samples_train_2000.jsonl"
ROUTER_DIR="${COLLECT_ROOT}/router_binary"
CRITICAL_ROUTER_DIR="${COLLECT_ROOT}/router_critical"
EVAL_ROOT=build/hetero_router/eval_val_unseen_300_train2000_v1
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
TRAIN_SCRIPT=edgecloud_experiments/hetero_router/train_budget_router.py

for shard in 0 1 2 3; do
  sample_file="${COLLECT_ROOT}/shard_${shard}/samples.jsonl"
  summary_count=$(find "${COLLECT_ROOT}/shard_${shard}" -maxdepth 1 -name 'summary_small_train_*.json' | wc -l)
  if [[ ! -s "${sample_file}" || "${summary_count}" -lt 1 ]]; then
    echo "Shard ${shard} is not complete: sample_file=${sample_file}, summary_count=${summary_count}" >&2
    exit 2
  fi
done

cat "${COLLECT_ROOT}"/shard_*/samples.jsonl > "${MERGED}"
wc -l "${MERGED}" | tee "${COLLECT_ROOT}/merged_samples_count.txt"

train_router() {
  local target_mode="$1"
  local router_dir="$2"
  "${PY}" "${TRAIN_SCRIPT}" \
    --samples "${MERGED}" \
    --out_dir "${router_dir}" \
    --epochs 120 \
    --hidden 128 \
    --dropout 0.15 \
    --batch_size 512 \
    --budgets 0.10,0.20,0.30,0.40,0.50 \
    --target_mode "${target_mode}" \
    2>&1 | tee "${router_dir}_train.log"
}

train_router binary "${ROUTER_DIR}"
train_router critical "${CRITICAL_ROUTER_DIR}"

mkdir -p "${EVAL_ROOT}"

launch_eval() {
  local name="$1"
  local gpu="$2"
  shift 2
  local out_dir="${EVAL_ROOT}/${name}"
  mkdir -p "${out_dir}"
  nohup "${PY}" "${EVAL_SCRIPT}" \
    --split val_unseen \
    --sample_seed 20260504 \
    --max_episodes 300 \
    --teacher_json "${TEACHER_VAL}" \
    --out_dir "${out_dir}" \
    --gpu "${gpu}" \
    --max_steps 15 \
    "$@" \
    > "${out_dir}/run.log" 2>&1 &
  echo "$!" > "${out_dir}/pid"
  echo "${name} pid=$(cat "${out_dir}/pid") gpu=${gpu}"
}

launch_eval trained_b20 4 --router_mode trained --router_ckpt "${ROUTER_DIR}/hetero_router.pt" --budget_key b20
launch_eval trained_b30 5 --router_mode trained --router_ckpt "${ROUTER_DIR}/hetero_router.pt" --budget_key b30
launch_eval trained_b40 6 --router_mode trained --router_ckpt "${ROUTER_DIR}/hetero_router.pt" --budget_key b40
launch_eval trained_b50 7 --router_mode trained --router_ckpt "${ROUTER_DIR}/hetero_router.pt" --budget_key b50

wait

launch_eval critical_b20 4 --router_mode trained --router_ckpt "${CRITICAL_ROUTER_DIR}/hetero_router.pt" --budget_key b20
launch_eval critical_b30 5 --router_mode trained --router_ckpt "${CRITICAL_ROUTER_DIR}/hetero_router.pt" --budget_key b30
launch_eval critical_b40 6 --router_mode trained --router_ckpt "${CRITICAL_ROUTER_DIR}/hetero_router.pt" --budget_key b40
launch_eval critical_b50 7 --router_mode trained --router_ckpt "${CRITICAL_ROUTER_DIR}/hetero_router.pt" --budget_key b50

wait

launch_eval random_b40 4 --router_mode random --budget 0.40
wait

echo "All train2000 router eval jobs finished."
find "${EVAL_ROOT}" -maxdepth 2 -name 'summary_*.json' -print | sort
