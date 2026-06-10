# 运行说明

## ⚠️ 重要提示

**请从项目根目录运行训练脚本，而不是从 `distill_code/` 目录运行！**

## ✅ 正确的运行方式

```bash
# 在项目根目录下运行
cd ~/edgecloud-vln-routing-supplement
python distill_code/train_distill.py \
    --cfg_file configs/multi.yaml \
    --pretrained_model_name_or_path data/models/Qwen3-1.7B \
    --distill_jsonl_paths build/distill_collection/distill_logs/distill_cvdn_train.jsonl \
    --distill_stage stage1 \
    --batch_size 4 \
    --lr 1e-5 \
    --num_epochs 10 \
    --output_dir build/distill_training
```

## ❌ 错误的运行方式

```bash
# 不要这样做！
cd ~/edgecloud-vln-routing-supplement/distill_code
python train_distill.py ...
```

## 🔧 如果必须从 distill_code 目录运行

如果确实需要从 `distill_code/` 目录运行，可以设置 PYTHONPATH：

```bash
cd ~/edgecloud-vln-routing-supplement/distill_code
PYTHONPATH=.. python train_distill.py ...
```

或者：

```bash
cd ~/edgecloud-vln-routing-supplement/distill_code
python -m train_distill ...
```

但**强烈建议从项目根目录运行**，这样可以避免路径问题。

