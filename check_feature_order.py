"""
检查训练时和运行时的特征顺序是否一致
"""
import json
from pathlib import Path

# 1. 检查训练数据中的特征顺序（sorted）
train_jsonl = Path("build/router_data/router_train_cvdn.jsonl")
if train_jsonl.exists():
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        train_keys = sorted(sample['features'].keys())
    print("=" * 60)
    print("训练时实际使用的feature_keys顺序（sorted）:")
    print("=" * 60)
    for i, key in enumerate(train_keys):
        print(f"  [{i:2d}] {key}")
    print(f"\n总共 {len(train_keys)} 个特征")
else:
    print(f"❌ 训练数据文件不存在: {train_jsonl}")
    train_keys = None

# 2. 运行时hardcoded的特征顺序
runtime_keys = [
    'entropy', 'margin', 'top1_prob', 'top2_prob', 'top3_prob',
    'num_cands', 'step_ratio', 'is_stop_top1',
    'dist_before', 'dist_change',
    'top1_logit', 'top2_logit', 'top3_logit',
]

print("\n" + "=" * 60)
print("运行时hardcoded的feature_keys顺序:")
print("=" * 60)
for i, key in enumerate(runtime_keys):
    print(f"  [{i:2d}] {key}")
print(f"\n总共 {len(runtime_keys)} 个特征")

# 3. 对比
if train_keys:
    print("\n" + "=" * 60)
    print("对比结果:")
    print("=" * 60)
    
    if train_keys == runtime_keys:
        print("✅ 顺序完全一致！")
    else:
        print("❌ 顺序不一致！")
        print("\n差异详情:")
        print("-" * 60)
        for i, (train_key, runtime_key) in enumerate(zip(train_keys, runtime_keys)):
            if train_key != runtime_key:
                print(f"  [{i:2d}] 训练时: {train_key:20s}  |  运行时: {runtime_key:20s}  ❌")
            else:
                print(f"  [{i:2d}] {train_key:20s}  ✅")
        
        # 检查是否有缺失的特征
        train_set = set(train_keys)
        runtime_set = set(runtime_keys)
        missing_in_runtime = train_set - runtime_set
        extra_in_runtime = runtime_set - train_set
        
        if missing_in_runtime:
            print(f"\n⚠️  运行时缺少的特征: {missing_in_runtime}")
        if extra_in_runtime:
            print(f"⚠️  运行时多余的特征: {extra_in_runtime}")





