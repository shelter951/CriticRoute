# 知识蒸馏模块

本目录包含知识蒸馏相关的所有代码，与项目原有代码完全分离。

## 📁 目录结构

```
distill_code/
├── models/                    # 蒸馏模型
│   ├── __init__.py
│   ├── modified_qwen.py      # Qwen3模型适配
│   └── nav_qwen3.py          # NavQwen3学生模型
├── datasets/                  # 蒸馏数据集
│   ├── __init__.py
│   └── distill_dataset.py    # 蒸馏数据加载器
├── train_distill.py           # 蒸馏训练脚本
└── README.md                  # 本文件
```

## 🚀 使用方法

### 1. 数据收集（使用原有代码）

首先使用修改后的`train.py`收集教师模型数据：

```bash
python train.py \
    --mode test \
    --enable_distill_log \
    --distill_output_dir ./distill_logs \
    --validation_split train \
    --test_datasets CVDN R2R \
    ...
```

### 2. 蒸馏训练（使用本模块）

```bash
python distill_code/train_distill.py \
    --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Qwen3-1.7B \
    --distill_jsonl_paths distill_logs/distill_cvdn_train.jsonl \
    --distill_stage stage1 \
    --batch_size 4 \
    --lr 1e-5 \
    --num_epochs 10 \
    --output_dir build/distill_training
```

## 📝 文件说明

### models/

- **modified_qwen.py**: Qwen3-1.7B模型的适配类，类似`models/modified_lm.py`中的`ModifiedLlamaForCausalLM`
- **nav_qwen3.py**: 基于Qwen3-1.7B的导航模型，学生模型的核心实现

### datasets/

- **distill_dataset.py**: 从JSONL文件加载教师模型数据，处理视觉特征加载和数据预处理

### train_distill.py

- 蒸馏训练主脚本，包含：
  - 分阶段训练逻辑（stage1/stage2/stage3）
  - 不同组件的学习率配置
  - 分布式训练支持
  - wandb日志集成

## 🔗 依赖关系

本模块依赖项目原有代码：
- `models/nav_model.py` - 参考架构
- `models/image_embedding.py` - 视觉嵌入
- `models/ops.py` - 工具函数
- `tasks/feature_db.py` - 特征数据库
- `tools/optims.py` - 优化器工具
- `configs/multi.yaml` - 配置文件

## 📖 更多文档

- `DISTILL_README.md` - 完整的模块说明
- `DISTILL_FILES.md` - 文件清单和修改说明
- `DISTILL_USAGE.md` - 数据收集使用说明
- `蒸馏思路.txt` - 蒸馏策略设计

