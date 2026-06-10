# 从 Stage1 切换到 Stage2 训练指南

## 概述

当 Stage1 训练达到平台期（loss 不再明显下降，acc 稳定在 0.20-0.22 左右）时，应该切换到 Stage2 继续训练。

## Stage1 到 Stage2 的切换流程

### 1. 确认 Stage1 训练已完成

检查你的训练日志，确认：
- Loss 已经从初始的 ~3.0 下降到 ~2.6-2.7，且最近几个 epoch 变化很小
- Accuracy 已经达到 0.20-0.22 左右，且不再明显提升
- 训练已经运行了足够的 epoch（建议至少 7-10 个 epoch）

### 2. 找到 Stage1 的最佳 checkpoint

Stage1 训练会在 `output_dir/checkpoints/` 目录下保存：
- `navqwen3_stage1_best.pt` - 最佳模型（推荐使用）
- `navqwen3_stage1_epoch_N.pt` - 每个 epoch 的 checkpoint

**推荐使用 `navqwen3_stage1_best.pt`**，这是验证集上表现最好的模型。

### 3. 启动 Stage2 训练

使用以下命令从 Stage1 的 checkpoint 加载并开始 Stage2 训练：

```bash
screen -S distill_stage2

cd ~/edgecloud-vln-routing-supplement/distill_code

export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

torchrun \
    --nnodes=1 \
    --nproc_per_node=6 \
    --master_port=29501 \
    train_distill.py \
    --stage multi \
    --cfg_file ../configs/multi.yaml \
    --pretrained_model_name_or_path ../data/models/Qwen3-1.7B \
    --distill_jsonl_paths ../build/distill_collection/distill_logs/distill_cvdn_train.jsonl \
    --distill_stage stage2 \
    --resume_from_checkpoint ../build/distill_training/checkpoints/navqwen3_stage1_best.pt \
    --num_unfreeze_layers 6 \
    --batch_size 2 \
    --gradient_accumulation_step 2 \
    --lr 1e-5 \
    --lr_head 3e-4 \
    --lr_llm 1e-5 \
    --num_epochs 10 \
    --output_dir ../build/distill_training_stage2 \
    --precision amp_bf16 \
    --num_workers 2 \
    --weight_decay 0.01 \
    --lr_scheduler cosine \
    --num_warmup_steps 500

# Ctrl+A, D 来 detach
```

### 4. 关键参数说明

#### `--resume_from_checkpoint`
- **必须指定**：Stage1 最佳 checkpoint 的路径
- 代码会自动加载模型权重、optimizer 状态和 lr_scheduler 状态（如果存在）

#### `--distill_stage stage2`
- 切换到 Stage2 训练模式
- 会自动解冻 LLM 的最后 N 层（默认 6 层）

#### `--num_unfreeze_layers 6`
- 解冻 LLM 的最后 6 层
- 可以根据需要调整（建议 4-8 层）

#### `--lr_head 3e-4`
- 分类头的学习率（比 Stage1 稍大，继续细调）

#### `--lr_llm 1e-5`
- 解冻的 LLM 层的学习率（较小，避免破坏预训练能力）

#### `--output_dir ../build/distill_training_stage2`
- **使用新的输出目录**，避免与 Stage1 混淆

### 5. 预期效果

Stage2 训练开始后，你应该看到：

1. **Loss 进一步下降**
   - 从 Stage1 的 ~2.6-2.7 继续下降到 ~2.3-2.5
   - 前几个 epoch 下降会比较明显

2. **Accuracy 提升**
   - 从 Stage1 的 ~0.20-0.22 提升到 ~0.25-0.30+
   - 说明解冻的 LLM 层在学习导航相关的知识

3. **训练曲线**
   - Loss 曲线会有一个明显的"台阶"（Stage1 → Stage2 的过渡）
   - 然后继续下降

### 6. 监控训练

#### 查看实时日志
```bash
screen -r distill_stage2
```

#### 查看训练历史
训练过程中会保存 `training_history.json`，可以用 `plot_training_curves.py` 绘制曲线：

```bash
cd ~/edgecloud-vln-routing-supplement/distill_code
python plot_training_curves.py \
    --history_file ../build/distill_training_stage2/training_history.json \
    --output_dir ../build/distill_training_stage2/plots
```

#### Wandb 监控
如果启用了 Wandb，可以在网页上实时查看训练曲线。

### 7. 训练完成后评估

Stage2 训练完成后，建议在导航任务上评估最终效果：

```bash
# 使用 Stage2 最佳模型进行导航评估
CUDA_VISIBLE_DEVICES=0 python train.py \
    --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path ../build/distill_training_stage2/checkpoints/navqwen3_stage2_best.pt \
    --precision amp_bf16 \
    --test_datasets CVDN \
    --batch_size 4 \
    --output_dir build/eval_stage2 \
    --validation_split val_seen
```

### 8. 三方对比

建议对比以下三个模型：
1. **Student-init**：未蒸馏的 Qwen3-1.7B（基线）
2. **Student-stage1**：Stage1 训练后的模型
3. **Student-stage2**：Stage2 训练后的模型
4. **Teacher**：原始 RoomTour3D-NaviLLM（Vicuna-7B）

理想情况下：
- Student-init：SR ≈ 0.05（接近随机）
- Student-stage1：SR ≈ 0.10-0.15
- Student-stage2：SR ≈ 0.20-0.30（达到 Teacher 的 50-70%）

## 常见问题

### Q: Stage2 训练时 loss 反而上升了？
A: 这是正常的。因为解冻了更多层，模型需要重新适应。通常 1-2 个 epoch 后会开始下降。

### Q: 可以跳过 Stage1 直接训练 Stage2 吗？
A: 不推荐。Stage1 的作用是：
- 校准分类头和视觉投影层
- 让模型初步学习 teacher 的分布
- 为 Stage2 提供一个好的起点

### Q: Stage2 应该训练多少个 epoch？
A: 建议 5-10 个 epoch。如果 loss 和 acc 已经稳定，可以提前停止。

### Q: 如何选择 `num_unfreeze_layers`？
A: 
- 保守：4-6 层（适合数据量较小的情况）
- 标准：6-8 层（推荐）
- 激进：8-12 层（适合数据量大的情况）

### Q: Stage2 训练后还需要 Stage3 吗？
A: 通常不需要。Stage2 已经解冻了关键层，如果效果已经满足需求，可以停止。如果还想进一步提升，可以考虑 Stage3（解冻所有层），但需要更小的学习率和更长的训练时间。

## 总结

从 Stage1 切换到 Stage2 的关键步骤：
1. ✅ 确认 Stage1 已达到平台期
2. ✅ 找到 Stage1 最佳 checkpoint
3. ✅ 使用 `--resume_from_checkpoint` 加载 checkpoint
4. ✅ 设置 `--distill_stage stage2`
5. ✅ 调整学习率（head 稍大，LLM 较小）
6. ✅ 使用新的输出目录
7. ✅ 监控训练曲线，确认 loss 和 acc 继续提升

祝你训练顺利！🎉





