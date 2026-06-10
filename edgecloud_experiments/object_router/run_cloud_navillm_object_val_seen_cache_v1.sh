#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_ROOT:-.}
PY=${PYTHON:-python}
OUT=${ROOT}/build/object_router/cloud_navillm_clean_split_v1

mkdir -p "${OUT}/logs"
cd "${ROOT}"

export PYTHONPATH=${MATTERSIM_PYTHONPATH:-/path/to/Matterport3DSimulator/build}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NAVILLM_VIS_CONFIG=${PROJECT_ROOT:-.}/local_bert_large_config
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NAVILLM_RAW_PRED_ONLY=1

run_cache() {
  local dataset=$1
  local split=$2
  local gpu=$3
  local out_dir=${OUT}/${dataset,,}_${split}
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/${dataset}_${split}_raw.json" ]]; then
    echo "SKIP ${dataset} ${split}: raw prediction JSON exists at ${out_dir}/${dataset}_${split}_raw.json"
    return 0
  fi
  echo "START ${dataset} ${split} gpu=${gpu} $(date)" | tee -a "${OUT}/pipeline.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" train.py \
    --stage multi --mode test --data_dir data --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B --precision amp_bf16 \
    --resume_from_checkpoint navillm_roomtour3d_video_action_instruction.pt \
    --test_datasets "${dataset}" \
    --batch_size 1 --val_batch_size 1 --num_workers 0 \
    --output_dir "${out_dir}" \
    --validation_split "${split}" --save_pred_results --save_detail_results \
    > "${OUT}/logs/${dataset}_${split}.log" 2>&1
  echo "DONE ${dataset} ${split} $(date)" | tee -a "${OUT}/pipeline.log"
}

run_cache REVERIE val_seen 4 &
p_reverie=$!
run_cache SOON val_seen 5 &
p_soon=$!

wait "${p_reverie}" "${p_soon}"
echo "FINISH object val_seen cloud cache $(date)" | tee -a "${OUT}/pipeline.log"
