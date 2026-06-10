#!/bin/bash
# 多卡并行评估脚本（Linux）

# 使用方法：
# bash run_multi_gpu_eval.sh --task CVDN --split val_unseen --teacher_ckpt ... --student_ckpt ...

# 设置CUDA设备（根据你的服务器调整）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU数量
NUM_GPUS=4

# Master端口（如果多进程同时运行，使用不同端口）
MASTER_PORT=29500

# 获取脚本参数
ARGS="$@"

# 运行多卡评估
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    edgecloud_experiments/eval_edgecloud.py \
    $ARGS \
    --multi_gpu

