"""
结果分析和可视化脚本

功能：
1. 汇总不同模式/路由/τ的结果
2. 生成对比表格
3. 绘制SR/SPL vs 调用率曲线
4. 生成固定预算/固定性能对比
"""
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合Linux服务器
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """加载所有结果文件"""
    all_results = {}
    
    for result_file in results_dir.glob("results_*.json"):
        # 解析文件名: results_{mode}_{router_type}_{student_type}.json
        parts = result_file.stem.split('_')
        if len(parts) >= 4:
            mode = parts[1]
            router_type = parts[2]
            student_type = parts[3]
            key = f"{mode}_{router_type}_{student_type}"
        else:
            key = result_file.stem
        
        with open(result_file, 'r', encoding='utf-8') as f:
            all_results[key] = json.load(f)
    
    return all_results


def create_comparison_table(all_results: Dict[str, Dict], output_file: Path):
    """创建对比表格"""
    rows = []
    
    for key, results in all_results.items():
        mode, router_type, student_type = key.split('_', 2)
        
        for tau, summary in results.items():
            if isinstance(summary, dict) and 'sr' in summary:
                rows.append({
                    'Mode': mode,
                    'Router': router_type,
                    'Student': student_type,
                    'τ': float(tau),
                    'SR': summary['sr'],
                    'SPL': summary['spl'],
                    'Nav Error': summary['nav_error'],
                    'Call Rate': summary['teacher_call_rate'],
                    'Calls/Episode': summary['teacher_calls_per_episode'],
                    'T_episode (s)': summary['t_episode_avg'],
                    'T_teacher_ratio': summary['teacher_time_ratio'],
                })
    
    df = pd.DataFrame(rows)
    
    # 保存为CSV
    df.to_csv(output_file.with_suffix('.csv'), index=False, float_format='%.4f')
    
    # 保存为Markdown表格
    md_file = output_file.with_suffix('.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Edge-Cloud Navigation Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
    
    print(f"Comparison table saved to {output_file}")
    return df


def plot_tradeoff_curves(all_results: Dict[str, Dict], output_dir: Path):
    """绘制SR/SPL vs 调用率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 收集数据
    plot_data = {}
    for key, results in all_results.items():
        mode, router_type, student_type = key.split('_', 2)
        label = f"{mode}_{router_type}_{student_type}"
        
        taus = []
        srs = []
        spls = []
        call_rates = []
        
        for tau_str, summary in sorted(results.items(), key=lambda x: float(x[0])):
            if isinstance(summary, dict) and 'sr' in summary:
                taus.append(float(tau_str))
                srs.append(summary['sr'])
                spls.append(summary['spl'])
                call_rates.append(summary['teacher_call_rate'])
        
        if len(call_rates) > 0:
            plot_data[label] = {
                'call_rates': call_rates,
                'srs': srs,
                'spls': spls,
                'mode': mode,
                'router': router_type,
            }
    
    # 绘制SR vs Call Rate
    ax1 = axes[0]
    for label, data in plot_data.items():
        if data['mode'] == 'edgecloud':
            ax1.plot(data['call_rates'], data['srs'], 'o-', label=label, linewidth=2, markersize=6)
        elif data['mode'] == 'teacher_only':
            ax1.scatter([1.0], data['srs'], s=200, marker='*', label='Teacher-only', zorder=5)
        elif data['mode'] == 'student_only':
            ax1.scatter([0.0], data['srs'], s=200, marker='s', label='Student-only', zorder=5)
    
    ax1.set_xlabel('Teacher Call Rate', fontsize=12)
    ax1.set_ylabel('Success Rate (SR)', fontsize=12)
    ax1.set_title('SR vs Teacher Call Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 绘制SPL vs Call Rate
    ax2 = axes[1]
    for label, data in plot_data.items():
        if data['mode'] == 'edgecloud':
            ax2.plot(data['call_rates'], data['spls'], 'o-', label=label, linewidth=2, markersize=6)
        elif data['mode'] == 'teacher_only':
            ax2.scatter([1.0], data['spls'], s=200, marker='*', label='Teacher-only', zorder=5)
        elif data['mode'] == 'student_only':
            ax2.scatter([0.0], data['spls'], s=200, marker='s', label='Student-only', zorder=5)
    
    ax2.set_xlabel('Teacher Call Rate', fontsize=12)
    ax2.set_ylabel('SPL', fontsize=12)
    ax2.set_title('SPL vs Teacher Call Rate', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'tradeoff_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Trade-off curves saved to {output_file}")
    plt.close()


def find_fixed_budget_comparison(all_results: Dict[str, Dict], target_call_rate: float = 0.5, tolerance: float = 0.05):
    """固定预算对比：找到所有方法在目标调用率下的性能"""
    comparison = {}
    
    for key, results in all_results.items():
        mode, router_type, student_type = key.split('_', 2)
        
        # 找到最接近目标调用率的τ
        best_tau = None
        best_summary = None
        min_diff = float('inf')
        
        for tau_str, summary in results.items():
            if isinstance(summary, dict) and 'teacher_call_rate' in summary:
                call_rate = summary['teacher_call_rate']
                diff = abs(call_rate - target_call_rate)
                if diff < min_diff and diff <= tolerance:
                    min_diff = diff
                    best_tau = float(tau_str)
                    best_summary = summary
        
        if best_summary:
            comparison[key] = {
                'tau': best_tau,
                'call_rate': best_summary['teacher_call_rate'],
                'sr': best_summary['sr'],
                'spl': best_summary['spl'],
                'nav_error': best_summary['nav_error'],
            }
    
    return comparison


def find_fixed_performance_comparison(all_results: Dict[str, Dict], target_sr_ratio: float = 0.9):
    """固定性能对比：找到所有方法达到目标SR所需的最小调用率"""
    # 先找到Teacher-only的SR作为基准
    teacher_sr = None
    for key, results in all_results.items():
        if 'teacher_only' in key:
            for summary in results.values():
                if isinstance(summary, dict) and 'sr' in summary:
                    teacher_sr = summary['sr']
                    break
        if teacher_sr:
            break
    
    if teacher_sr is None:
        print("Warning: Teacher-only results not found, cannot compute fixed performance comparison")
        return {}
    
    target_sr = teacher_sr * target_sr_ratio
    comparison = {}
    
    for key, results in all_results.items():
        mode, router_type, student_type = key.split('_', 2)
        
        if mode == 'teacher_only':
            continue
        
        # 找到达到目标SR的最小调用率
        best_tau = None
        best_summary = None
        min_call_rate = float('inf')
        
        for tau_str, summary in sorted(results.items(), key=lambda x: float(x[0]), reverse=True):
            if isinstance(summary, dict) and 'sr' in summary:
                if summary['sr'] >= target_sr:
                    call_rate = summary['teacher_call_rate']
                    if call_rate < min_call_rate:
                        min_call_rate = call_rate
                        best_tau = float(tau_str)
                        best_summary = summary
        
        if best_summary:
            comparison[key] = {
                'tau': best_tau,
                'call_rate': best_summary['teacher_call_rate'],
                'sr': best_summary['sr'],
                'spl': best_summary['spl'],
                'nav_error': best_summary['nav_error'],
            }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze edge-cloud navigation results")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} result sets")
    
    # 创建对比表格
    print("\nCreating comparison table...")
    table_file = output_dir / 'comparison_table'
    df = create_comparison_table(all_results, table_file)
    
    # 绘制trade-off曲线
    print("\nPlotting trade-off curves...")
    plot_tradeoff_curves(all_results, output_dir)
    
    # 固定预算对比
    print("\nFixed budget comparison (Call Rate ≈ 50%):")
    fixed_budget = find_fixed_budget_comparison(all_results, target_call_rate=0.5)
    if fixed_budget:
        budget_file = output_dir / 'fixed_budget_comparison.json'
        with open(budget_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_budget, f, indent=2, ensure_ascii=False)
        print(f"Saved to {budget_file}")
    
    # 固定性能对比
    print("\nFixed performance comparison (SR ≥ 90% of Teacher):")
    fixed_perf = find_fixed_performance_comparison(all_results, target_sr_ratio=0.9)
    if fixed_perf:
        perf_file = output_dir / 'fixed_performance_comparison.json'
        with open(perf_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_perf, f, indent=2, ensure_ascii=False)
        print(f"Saved to {perf_file}")
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()





