#!/usr/bin/env python3
"""
可视化工具
==========
生成分析结果的可视化图表
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


def create_summary_plot(json_file: str, output_file: str = None):
    """
    创建汇总可视化图表
    
    Args:
        json_file: 分析结果JSON文件路径
        output_file: 输出图像路径，None则自动命名
    """
    # 读取数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    name = data.get('name', 'Unknown')
    ra = data.get('ra', 0)
    dec = data.get('dec', 0)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{name}\n(RA={ra:.4f}°, DEC={dec:.4f}°)', 
                 fontsize=16, fontweight='bold')
    
    # ===== 面板1: 基本信息和消光 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    ext = data.get('extinction', {})
    info_lines = [
        f"Target: {name}",
        f"",
        f"Coordinates:",
        f"  RA = {ra:.6f}°",
        f"  DEC = {dec:.6f}°",
        f"",
    ]
    
    if ext.get('success'):
        info_lines.extend([
            f"Extinction:",
            f"  A_V = {ext['A_V']:.3f} mag",
            f"  E(B-V) = {ext['E_B_V']:.3f} mag",
            f"",
            f"Distance Estimate:",
            f"  {estimate_distance(ext['A_V'])}",
        ])
    
    info_text = '\n'.join(info_lines)
    ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax1.set_title('Basic Information', fontsize=12, fontweight='bold')
    
    # ===== 面板2: 数据获取状态 =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    status_data = []
    colors = []
    
    # 消光状态
    if ext.get('success'):
        status_data.append(('Extinction', '✓'))
        colors.append('green')
    else:
        status_data.append(('Extinction', '✗'))
        colors.append('red')
    
    # ZTF状态
    ztf = data.get('ztf', {})
    if ztf.get('success'):
        status_data.append(('ZTF', f"✓ ({ztf.get('n_points', 0)})"))
        colors.append('green')
    else:
        status_data.append(('ZTF', '✗'))
        colors.append('red')
    
    # 测光状态
    phot = data.get('photometry', {})
    if phot.get('success'):
        cats = phot.get('catalogs', {})
        found = sum(1 for v in cats.values() if str(v) not in ['0', 'Not found', '-1'])
        status_data.append(('Photometry', f"✓ ({found})"))
        colors.append('green' if found > 0 else 'orange')
    else:
        status_data.append(('Photometry', '✗'))
        colors.append('red')
    
    # 绘制状态
    y_pos = np.arange(len(status_data))
    for i, ((name, status), color) in enumerate(zip(status_data, colors)):
        ax2.barh(i, 1, color=color, alpha=0.6, edgecolor='black')
        ax2.text(0.5, i, f'{name}: {status}', 
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, len(status_data) - 0.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('Data Availability', fontsize=12, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.6, label='Available'),
        mpatches.Patch(color='red', alpha=0.6, label='Not Available'),
        mpatches.Patch(color='orange', alpha=0.6, label='Partial')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # ===== 面板3: 测光数据分布 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    if phot.get('success'):
        cats = phot.get('catalogs', {})
        cat_names = list(cats.keys())
        cat_values = [1 if str(v) not in ['0', 'Not found', '-1'] else 0 for v in cats.values()]
        
        colors_bar = ['green' if v == 1 else 'red' for v in cat_values]
        bars = ax3.barh(cat_names, cat_values, color=colors_bar, alpha=0.6, edgecolor='black')
        ax3.set_xlim(0, 1.5)
        ax3.set_xlabel('Available')
        ax3.set_title('Photometry Catalogs', fontsize=12, fontweight='bold')
        
        # 添加标签
        for i, (bar, val) in enumerate(zip(bars, cat_values)):
            label = 'Found' if val == 1 else 'Not found'
            ax3.text(val + 0.1, i, label, va='center', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Photometry Data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Photometry Catalogs', fontsize=12, fontweight='bold')
        ax3.axis('off')
    
    # ===== 面板4: AI分析摘要 =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    analysis = data.get('analysis', 'No analysis available.')
    # 截断分析文本以适应图表
    if len(analysis) > 500:
        analysis = analysis[:497] + '...'
    
    ax4.text(0.05, 0.95, 'AI Analysis:', transform=ax4.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top')
    ax4.text(0.05, 0.88, analysis, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.set_title('Analysis Summary', fontsize=12, fontweight='bold')
    
    # 保存图像
    if output_file is None:
        output_file = json_file.replace('_analysis.json', '_summary.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 图表已保存: {output_file}")
    return output_file


def estimate_distance(a_v: float) -> str:
    """根据消光估计距离"""
    if a_v < 0.1:
        return "< 100 pc (local bubble)"
    elif a_v < 0.5:
        return "~100-500 pc"
    elif a_v < 1.0:
        return "~500 pc - 1 kpc"
    elif a_v < 2.0:
        return "~1-2 kpc"
    else:
        return "> 2 kpc or Galactic plane"


def visualize_all(output_dir: str = "./output"):
    """可视化所有结果"""
    import glob
    
    json_files = glob.glob(os.path.join(output_dir, "*_analysis.json"))
    
    if not json_files:
        print(f"✗ 在 {output_dir} 中没有找到分析结果")
        return
    
    print(f"找到 {len(json_files)} 个分析结果")
    
    for json_file in sorted(json_files):
        name = os.path.basename(json_file).replace('_analysis.json', '')
        print(f"\n处理: {name}")
        try:
            create_summary_plot(json_file)
        except Exception as e:
            print(f"  ✗ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='可视化分析结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/visualize.py --file output/MyStar_analysis.json
  python src/visualize.py --name MyStar
  python src/visualize.py --all
        """
    )
    
    parser.add_argument('--file', type=str, help='JSON文件路径')
    parser.add_argument('--name', type=str, help='结果名称')
    parser.add_argument('--all', action='store_true', help='可视化所有结果')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.file:
        create_summary_plot(args.file)
    elif args.name:
        json_file = os.path.join(args.output_dir, f"{args.name}_analysis.json")
        if os.path.exists(json_file):
            create_summary_plot(json_file)
        else:
            print(f"✗ 文件不存在: {json_file}")
            return 1
    elif args.all:
        visualize_all(args.output_dir)
    else:
        # 默认可视化所有
        visualize_all(args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
