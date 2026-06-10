# 安装和依赖说明

## Transformers 版本要求

本代码支持多种 transformers 版本，但推荐使用较新版本以支持 Qwen2 模型。

### 推荐版本

```bash
# 推荐：transformers >= 4.37.0（支持 Qwen2）
pip install transformers>=4.37.0

# 或者使用最新版本
pip install transformers --upgrade
```

### 最低版本

如果无法升级 transformers，代码会自动降级使用 `AutoModelForCausalLM`，但可能无法完全利用 Qwen2 的特性。

```bash
# 最低版本：transformers >= 4.20.0
pip install transformers>=4.20.0
```

## 检查 transformers 版本

```bash
python -c "import transformers; print(transformers.__version__)"
```

## 支持的模型类型

代码会自动检测并适配：

1. **Qwen2ForCausalLM** (transformers >= 4.37.0) - 推荐
2. **QwenForCausalLM** (transformers >= 4.30.0) - 备选
3. **AutoModelForCausalLM** (所有版本) - 降级方案

## 如果遇到导入错误

如果遇到 `ImportError: cannot import name 'Qwen2ForCausalLM'`：

### 方案1：升级 transformers（推荐）

```bash
pip install transformers --upgrade
```

### 方案2：使用代码的自动降级

代码已经实现了自动降级，如果无法导入 Qwen2，会自动使用 `AutoModelForCausalLM`。这应该可以正常工作，但可能无法完全利用 Qwen2 的特性。

### 方案3：手动指定使用 AutoModel

如果自动降级仍有问题，可以修改 `distill_code/models/modified_qwen.py`，将：

```python
QwenBaseModel = AutoModelForCausalLM
```

直接设置为 `AutoModelForCausalLM`。

## 其他依赖

确保安装所有必要的依赖：

```bash
pip install torch torchvision
pip install transformers
pip install wandb  # 可选，用于日志记录
pip install tqdm
pip install pyyaml
pip install easydict
```

