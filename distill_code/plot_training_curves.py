"""
绘制训练曲线
从 training_history.json 生成 loss 和 accuracy 曲线图
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history(history_file):
    """加载训练历史"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history_file, output_dir=None):
    """绘制训练曲线"""
    history = load_training_history(history_file)
    
    epochs = history['epochs']
    loss = history['loss']
    acc = history['acc']
    lr = history['lr']
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss 曲线
    axes[0].plot(epochs, loss, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Accuracy 曲线
    axes[1].plot(epochs, acc, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Learning Rate 曲线
    axes[2].plot(epochs, lr, 'r-', linewidth=2, marker='^', markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    if output_dir is None:
        output_dir = Path(history_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为PNG
    png_path = output_dir / 'training_curves.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {png_path}")
    
    # 保存为PDF（矢量图）
    pdf_path = output_dir / 'training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved training curves (PDF) to: {pdf_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("Training Statistics:")
    print("="*50)
    print(f"Total epochs: {len(epochs)}")
    print(f"Initial loss: {loss[0]:.4f}")
    print(f"Final loss: {loss[-1]:.4f}")
    print(f"Loss reduction: {loss[0] - loss[-1]:.4f} ({100*(loss[0]-loss[-1])/loss[0]:.2f}%)")
    print(f"Initial acc: {acc[0]:.4f}")
    print(f"Final acc: {acc[-1]:.4f}")
    print(f"Acc improvement: {acc[-1] - acc[0]:.4f} ({100*(acc[-1]-acc[0])/acc[0] if acc[0] > 0 else 0:.2f}%)")
    print(f"Best acc: {max(acc):.4f} (epoch {epochs[acc.index(max(acc))]})")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves from history JSON')
    parser.add_argument('--history_file', type=str, required=True,
                        help='Path to training_history.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as history_file)')
    
    args = parser.parse_args()
    plot_training_curves(args.history_file, args.output_dir)

