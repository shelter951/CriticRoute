#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT:-.}
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build_osmesa}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

PY=${PYTHON:-python}
TRAIN_SCRIPT=edgecloud_experiments/hetero_router/train_reward_router.py
EVAL_SCRIPT=edgecloud_experiments/hetero_router/eval_hetero_edgecloud_r2r.py
SAMPLES=build/hetero_router/train_r2r_2000_v1/samples_train_2000.jsonl
ROOT=build/hetero_router/r2r_strong_module_ablation_v1
EVAL_ROOT=build/hetero_router/eval_r2r_strong_module_ablation_300_v1
TEACHER_VAL=${OFFICIAL_ROOT:-../official_clean}/build/official428_teacher_r2r_val_unseen_v1/R2R_val_unseen.json
FULL_CKPT=build/hetero_router/r2r_feature_reward_ablation_v1/feat_full/hetero_router.pt

mkdir -p "${ROOT}" "${EVAL_ROOT}"

CONF="qwen_entropy,qwen_margin,qwen_max_prob,qwen_selected_prob,qwen_stop_prob,qwen_invalid"
GEOM="cand_count_norm,cand_dist_min_norm,cand_dist_mean_norm,cand_dist_max_norm,cand_angle_abs_min_norm,cand_angle_abs_mean_norm,qwen_chosen_angle_norm,qwen_chosen_dist_norm"
ROUTE="current_revisit_count_norm,path_len_norm,step_norm,qwen_backtracks,qwen_is_stop"
TASK="instruction_len_norm"
FULL_COMPONENTS="naverr,success_flip,bad_stop,loop_break"

train_one() {
  local name="$1"
  local features="$2"
  local components="$3"
  local out_dir="${ROOT}/${name}"
  mkdir -p "${out_dir}"
  local args=("${PY}" "${TRAIN_SCRIPT}" --samples "${SAMPLES}" --out_dir "${out_dir}" --epochs 120 --hidden 128 --dropout 0.10 --batch_size 512 --cost_lambda 0.25 --budgets 0.40 --utility_components "${components}")
  if [[ -n "${features}" ]]; then
    args+=(--feature_names "${features}")
  fi
  echo "== train ${name} =="
  "${args[@]}" 2>&1 | tee "${out_dir}/train.log"
}

# Stronger feature/module ablations.  These remove whole decision subsystems,
# not a single weakly correlated scalar.
train_one no_confidence "${GEOM},${ROUTE},${TASK}" "${FULL_COMPONENTS}"
train_one geometry_only "${GEOM}" "${FULL_COMPONENTS}"
train_one route_only "${ROUTE},${TASK}" "${FULL_COMPONENTS}"

# Reward subsystem ablations that remove event-level criticality signals.
train_one reward_no_success "" "naverr,bad_stop,loop_break"
train_one reward_no_badstop "" "naverr,success_flip,loop_break"

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

echo "== wave 1: remove criticality inputs and budget calibration =="
launch_eval no_confidence_b40 4 --router_mode trained --router_ckpt "${ROOT}/no_confidence/hetero_router.pt" --budget_key b40
launch_eval geometry_only_b40 5 --router_mode trained --router_ckpt "${ROOT}/geometry_only/hetero_router.pt" --budget_key b40
launch_eval route_only_b40 6 --router_mode trained --router_ckpt "${ROOT}/route_only/hetero_router.pt" --budget_key b40
launch_eval no_budget_calibration_t05 7 --router_mode trained --router_ckpt "${FULL_CKPT}" --threshold 0.5
wait

echo "== wave 2: heuristic/random controls and reward-event removals =="
launch_eval heuristic_t015 4 --router_mode heuristic --threshold 0.15
launch_eval random_b30 5 --router_mode random --budget 0.30
launch_eval reward_no_success_b40 6 --router_mode trained --router_ckpt "${ROOT}/reward_no_success/hetero_router.pt" --budget_key b40
launch_eval reward_no_badstop_b40 7 --router_mode trained --router_ckpt "${ROOT}/reward_no_badstop/hetero_router.pt" --budget_key b40
wait

echo "== summaries =="
find "${EVAL_ROOT}" -maxdepth 2 -name 'summary_*.json' -print | sort
