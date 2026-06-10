#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
TRAIN_SCRIPT=edgecloud_experiments/hetero_router/train_reward_router.py
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
SAMPLES=build/hetero_router/train_r2r_2000_v1/samples_train_2000.jsonl
ROOT=build/hetero_router/reward_router_r2r_2000_v1
EVAL_ROOT=build/hetero_router/eval_val_unseen_300_reward_v1
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json

mkdir -p "${ROOT}" "${EVAL_ROOT}"

train_one() {
  local lambda="$1"
  local tag="lambda_${lambda//./}"
  local out_dir="${ROOT}/${tag}"
  mkdir -p "${out_dir}"
  "${PY}" "${TRAIN_SCRIPT}" \
    --samples "${SAMPLES}" \
    --out_dir "${out_dir}" \
    --epochs 140 \
    --hidden 128 \
    --dropout 0.10 \
    --batch_size 512 \
    --cost_lambda "${lambda}" \
    --budgets 0.10,0.20,0.30,0.40,0.50 \
    2>&1 | tee "${out_dir}/train.log"
}

if [[ ! -s "${SAMPLES}" ]]; then
  echo "Missing samples: ${SAMPLES}" >&2
  exit 2
fi

train_one 0.25
train_one 0.35

launch_eval() {
  local name="$1"
  local gpu="$2"
  local ckpt="$3"
  local budget_key="$4"
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
    --router_mode trained \
    --router_ckpt "${ckpt}" \
    --budget_key "${budget_key}" \
    > "${out_dir}/run.log" 2>&1 &
  echo "$!" > "${out_dir}/pid"
  echo "${name} pid=$(cat "${out_dir}/pid") gpu=${gpu}"
}

launch_eval reward_l025_b20 4 "${ROOT}/lambda_025/hetero_router.pt" b20
launch_eval reward_l025_b30 5 "${ROOT}/lambda_025/hetero_router.pt" b30
launch_eval reward_l035_b20 6 "${ROOT}/lambda_035/hetero_router.pt" b20
launch_eval reward_l035_b30 7 "${ROOT}/lambda_035/hetero_router.pt" b30
wait

launch_eval reward_l035_b40 4 "${ROOT}/lambda_035/hetero_router.pt" b40
launch_eval reward_l025_b40 5 "${ROOT}/lambda_025/hetero_router.pt" b40
wait

echo "Reward-router eval finished."
find "${EVAL_ROOT}" -maxdepth 2 -name 'summary_*.json' -print | sort

