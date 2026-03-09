#!/usr/bin/env python3
"""
结果查看工具
============
查看分析结果、数据和可视化
"""

import os
import sys
import json
import argparse
from datetime import datetime


def list_results(output_dir: str = "./output"):
    """列出所有分析结果"""
    if not os.path.exists(output_dir):
        print(f"✗ 输出目录不存在: {output_dir}")
        return []
    
    results = []
    for f in os.listdir(output_dir):
        if f.endswith('_analysis.json'):
            results.append(f)
    
    return sorted(results)


def view_result(name: str, output_dir: str = "./output"):
    """查看单个分析结果"""
    
    # 尝试查找文件
    json_file = os.path.join(output_dir, f"{name}_analysis.json")
    
    # 如果没有找到，尝试自动补全
    if not os.path.exists(json_file):
        # 列出所有可用结果
        available = list_results(output_dir)
        if available:
            print(f"✗ 未找到: {name}")
            print(f"\n可用结果:")
            for i, r in enumerate(available, 1):
                print(f"  {i}. {r.replace('_analysis.json', '')}")
        else:
            print(f"✗ 没有找到任何分析结果")
        return
    
    # 读取JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打印结果
    print("\n" + "="*70)
    print(f"📊 分析结果: {data.get('name', 'Unknown')}")
    print("="*70)
    
    print(f"\n【基本信息】")
    print(f"  坐标: RA={data.get('ra', 'N/A')}°, DEC={data.get('dec', 'N/A')}°")
    print(f"  时间: {data.get('timestamp', 'N/A')}")
    
    # 消光
    ext = data.get('extinction', {})
    print(f"\n【银河消光】")
    if ext.get('success'):
        print(f"  ✓ A_V = {ext['A_V']} mag")
        print(f"  ✓ E(B-V) = {ext['E_B_V']} mag")
        print(f"  → 距离估计: {estimate_distance(ext['A_V'])}")
    else:
        print(f"  ✗ {ext.get('error', '查询失败')}")
    
    # ZTF
    ztf = data.get('ztf', {})
    print(f"\n【ZTF光变曲线】")
    if ztf.get('success'):
        print(f"  ✓ 数据点数: {ztf['n_points']}")
        if ztf.get('mean_mag'):
            print(f"  ✓ 平均星等: {ztf['mean_mag']:.2f}")
        if ztf.get('time_span_days'):
            print(f"  ✓ 时间跨度: {ztf['time_span_days']:.1f} 天")
        if ztf.get('bands'):
            print(f"  ✓ 波段: {', '.join(ztf['bands'])}")
    else:
        print(f"  ✗ {ztf.get('error', '查询失败')}")
    
    # 测光
    phot = data.get('photometry', {})
    print(f"\n【多波段测光】")
    if phot.get('success'):
        cats = phot.get('catalogs', {})
        total = sum(1 for v in cats.values() if str(v) not in ['0', 'Not found', '-1'])
        print(f"  ✓ 匹配目录: {total} 个")
        for cat, count in cats.items():
            try:
                count_val = int(count) if count not in ['Not found', 'Found', 'Error'] else (1 if count == 'Found' else 0)
                status = "✓" if count_val > 0 else "✗"
            except:
                status = "✓" if count in ['Found'] else "✗"
                count_val = count
            print(f"    {status} {cat}: {count_val}")
    else:
        print(f"  ✗ {phot.get('error', '查询失败')}")
    
    # AI分析
    print(f"\n【AI分析】")
    print(data.get('analysis', '无分析结果'))
    
    # 文件信息
    print(f"\n【文件信息】")
    print(f"  JSON: {json_file}")
    
    # 检查其他文件
    base_name = name
    img_file = os.path.join(output_dir, f"{base_name}_summary.png")
    if os.path.exists(img_file):
        print(f"  图像: {img_file}")
    
    # 数据目录
    data_dir = os.path.join(output_dir, f"{data.get('ra', 0):.6f}_{data.get('dec', 0):.6f}")
    if os.path.exists(data_dir):
        print(f"  数据目录: {data_dir}")
        for f in os.listdir(data_dir):
            print(f"    - {f}")
    
    print("\n" + "="*70)


def estimate_distance(a_v: float) -> str:
    """根据消光估计距离"""
    # 简化的估计：A_V ≈ 1 mag / kpc (银道面附近)
    if a_v < 0.1:
        return "< 100 pc (本地泡)"
    elif a_v < 0.5:
        return "~100-500 pc"
    elif a_v < 1.0:
        return "~500 pc - 1 kpc"
    elif a_v < 2.0:
        return "~1-2 kpc"
    else:
        return "> 2 kpc 或银道面方向"


def view_all(output_dir: str = "./output"):
    """查看所有结果"""
    results = list_results(output_dir)
    
    if not results:
        print("✗ 没有找到任何分析结果")
        print(f"\n提示: 运行分析程序生成结果")
        print(f"  python src/astro_agent.py --ra 123.456 --dec 67.890")
        return
    
    print("="*70)
    print(f"📁 分析结果列表 ({len(results)} 个)")
    print("="*70)
    
    for i, result_file in enumerate(results, 1):
        name = result_file.replace('_analysis.json', '')
        json_file = os.path.join(output_dir, result_file)
        
        # 读取基本信息
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            ext = data.get('extinction', {})
            ztf = data.get('ztf', {})
            
            status = []
            if ext.get('success'):
                status.append(f"A_V={ext['A_V']}")
            if ztf.get('success'):
                status.append(f"ZTF={ztf['n_points']}pts")
            
            print(f"\n{i}. {name}")
            print(f"   坐标: RA={data.get('ra', 'N/A')}, DEC={data.get('dec', 'N/A')}")
            print(f"   状态: {', '.join(status) if status else '基础信息'}")
            print(f"   查看: python src/view_results.py --name {name}")
        except:
            print(f"\n{i}. {name} (读取失败)")
    
    print("\n" + "="*70)


def export_summary(output_dir: str = "./output", out_file: str = "summary.csv"):
    """导出所有结果摘要到CSV"""
    import csv
    
    results = list_results(output_dir)
    if not results:
        print("✗ 没有找到任何分析结果")
        return
    
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'RA', 'DEC', 'A_V', 'E_B_V', 'ZTF_Points', 'Timestamp'])
        
        for result_file in results:
            json_file = os.path.join(output_dir, result_file)
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                ext = data.get('extinction', {})
                ztf = data.get('ztf', {})
                
                writer.writerow([
                    data.get('name', ''),
                    data.get('ra', ''),
                    data.get('dec', ''),
                    ext.get('A_V', '') if ext.get('success') else '',
                    ext.get('E_B_V', '') if ext.get('success') else '',
                    ztf.get('n_points', '') if ztf.get('success') else '',
                    data.get('timestamp', '')
                ])
            except:
                pass
    
    print(f"✓ 摘要已导出: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description='查看分析结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/view_results.py --list              # 列出所有结果
  python src/view_results.py --name AM_Her       # 查看特定结果
  python src/view_results.py --all               # 查看所有结果详情
  python src/view_results.py --export            # 导出CSV摘要
        """
    )
    
    parser.add_argument('--list', action='store_true', help='列出所有结果')
    parser.add_argument('--name', type=str, help='查看特定结果')
    parser.add_argument('--all', action='store_true', help='查看所有结果')
    parser.add_argument('--export', action='store_true', help='导出CSV摘要')
    parser.add_argument('--output-dir', type=str, default='./output', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.list:
        results = list_results(args.output_dir)
        if results:
            print("可用结果:")
            for r in results:
                print(f"  - {r.replace('_analysis.json', '')}")
        else:
            print("没有找到任何结果")
    elif args.name:
        view_result(args.name, args.output_dir)
    elif args.all:
        view_all(args.output_dir)
    elif args.export:
        export_summary(args.output_dir)
    else:
        # 默认行为：列出所有结果
        view_all(args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
