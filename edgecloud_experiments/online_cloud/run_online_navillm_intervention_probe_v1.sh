#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT=${PROJECT_ROOT:-.}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-../official_clean}
PY_MAIN=${PYTHON:-python}
PY_OFFICIAL=${PYTHON:-python}
RUN_ROOT=${MAIN_ROOT}/build/online_cloud/navillm_r2r_intervention_probe_v1
ROUTER_JSON=${MAIN_ROOT}/build/hetero_router/eval_val_unseen_300_reward_v1/reward_l035_b30/hetero_trained_val_unseen_max300_1777996627.jsonl
SUBSET_JSON=${OFFICIAL_ROOT}/data/R2R/R2R_online_intervention_probe_v1.json
META_JSON=${RUN_ROOT}/probe_meta.json
CFG=${OFFICIAL_ROOT}/configs/multi_online_intervention_probe_v1.yaml
OUT_DIR=${RUN_ROOT}/official_eval

mkdir -p "${RUN_ROOT}" "${OUT_DIR}"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1

echo "START_ONLINE_NAVILLM_INTERVENTION_PROBE $(date)"
echo "main_root=${MAIN_ROOT}"
echo "official_root=${OFFICIAL_ROOT}"

if pgrep -af 'run_eval_grpo_and_continue_full_v2.sh|eval_hetero_edgecloud_r2r.py|qwen25vl_r2r_worker.py' >/dev/null; then
  echo "WAITING_FOR_HETERO_ROUTER_PIPELINE $(date)"
fi
while pgrep -af 'run_eval_grpo_and_continue_full_v2.sh|eval_hetero_edgecloud_r2r.py|qwen25vl_r2r_worker.py' >/dev/null; do
  sleep 120
done

cd "${MAIN_ROOT}"
"${PY_MAIN}" edgecloud_experiments/online_cloud/make_online_intervention_probe.py \
  --router_results "${ROUTER_JSON}" \
  --r2r_file "${MAIN_ROOT}/data/R2R/R2R_val_unseen_enc.json" \
  --out_json "${SUBSET_JSON}" \
  --meta_json "${META_JSON}" \
  --max_items 300 \
  --prefer_off_path

cd "${OFFICIAL_ROOT}"
"${PY_MAIN}" - <<'PY'
from pathlib import Path
src = Path("configs/multi.yaml")
dst = Path("configs/multi_online_intervention_probe_v1.yaml")
text = src.read_text()
old = '"val_unseen": "R2R_val_unseen_enc.json"'
new = '"val_unseen": "R2R_online_intervention_probe_v1.json"'
if old not in text:
    raise RuntimeError(f"Cannot find {old} in {src}")
dst.write_text(text.replace(old, new, 1))
print(f"WROTE {dst}")
PY

export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

"${PY_OFFICIAL}" train.py \
  --stage multi --mode test --data_dir data --cfg_file "${CFG}" \
  --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
  --resume_from_checkpoint navillm_roomtour3d_video_action_instruction.pt \
  --test_datasets R2R --batch_size 1 --val_batch_size 1 \
  --output_dir "${OUT_DIR}" \
  --validation_split val_unseen --save_pred_results --save_detail_results

cd "${MAIN_ROOT}"
"${PY_MAIN}" edgecloud_experiments/online_cloud/summarize_online_intervention_probe.py \
  --pred_json "${OUT_DIR}/R2R_val_unseen.json" \
  --meta_json "${META_JSON}" \
  --out_json "${RUN_ROOT}/online_probe_summary.json"

echo "DONE_ONLINE_NAVILLM_INTERVENTION_PROBE $(date)"
