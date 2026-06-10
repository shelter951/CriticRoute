"""
检查checkpoint文件的详细信息
"""
import torch
from pathlib import Path

checkpoint_dir = Path("build/router_data/checkpoints")

# 查找所有checkpoint文件
checkpoint_files = list(checkpoint_dir.glob("*.pt"))
print(f"找到 {len(checkpoint_files)} 个checkpoint文件:")
for f in checkpoint_files:
    print(f"  - {f.name}")

# 检查每个checkpoint
for ckpt_file in checkpoint_files:
    print("\n" + "=" * 60)
    print(f"检查: {ckpt_file.name}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        
        # 检查所有key
        print(f"\nCheckpoint中的所有key:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # 检查feature_keys
        if 'feature_keys' in checkpoint:
            print(f"\n✅ checkpoint中有feature_keys，共{len(checkpoint['feature_keys'])}个：")
            for i, key in enumerate(checkpoint['feature_keys']):
                print(f"  [{i:2d}] {key}")
        else:
            print(f"\n❌ checkpoint中没有feature_keys!")
        
        # 检查input_dim
        if 'input_dim' in checkpoint:
            print(f"\n✅ checkpoint中有input_dim: {checkpoint['input_dim']}")
        else:
            print(f"\n❌ checkpoint中没有input_dim!")
        
        # 检查模型权重
        if 'model_state_dict' in checkpoint:
            first_layer_key = 'net.0.weight'
            if first_layer_key in checkpoint['model_state_dict']:
                weight_shape = checkpoint['model_state_dict'][first_layer_key].shape
                print(f"\n模型第一层权重形状: {weight_shape}")
                print(f"推断的input_dim: {weight_shape[1]}")
                
                if 'feature_keys' in checkpoint:
                    if weight_shape[1] != len(checkpoint['feature_keys']):
                        print(f"⚠️  维度不匹配！权重期望{weight_shape[1]}维，feature_keys有{len(checkpoint['feature_keys'])}个")
                    else:
                        print(f"✅ 维度匹配")
        
        # 检查epoch
        if 'epoch' in checkpoint:
            print(f"\n训练epoch: {checkpoint['epoch']}")
        
        # 检查metrics
        if 'metrics' in checkpoint:
            print(f"\nMetrics: {checkpoint['metrics']}")
            
    except Exception as e:
        print(f"\n❌ 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()





