#!/usr/bin/env python3
"""
评估结果可视化工具

生成对比图表和 HTML 报告
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def generate_html_report(results: Dict[str, dict], output_path: str):
    """生成 HTML 报告"""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>天文大模型评估报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a1a;
            color: #e0e0ff;
            padding: 40px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #8892b0;
            margin-bottom: 40px;
        }
        .card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        th {
            color: #00d4ff;
            font-weight: 600;
        }
        .accuracy { font-size: 1.2em; font-weight: bold; }
        .good { color: #00ff88; }
        .bad { color: #ff6b6b; }
        .neutral { color: #ffd93d; }
        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .metric-card {
            background: rgba(0, 212, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
        }
        .metric-label {
            color: #8892b0;
            margin-top: 8px;
        }
        .winner {
            border: 2px solid #00ff88;
            background: linear-gradient(135deg, #1a1a2e 0%, #0d3328 100%);
        }
        .baseline {
            border: 2px solid #ffd93d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔭 天文领域大模型评估报告</h1>
        <p class="subtitle">对比 AstroMLab-8B 基准模型</p>
"""
    
    # 总体对比表格
    html += """
        <div class="card">
            <h2>📊 总体性能对比</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>准确率</th>
                    <th>vs AstroMLab-8B</th>
                    <th>幻觉率</th>
                    <th>状态</th>
                </tr>
"""
    
    baseline = 0.809
    
    for model_name, data in sorted(results.items(), key=lambda x: -x[1].get('accuracy', 0)):
        acc = data.get('accuracy', 0) * 100
        diff = (data.get('accuracy', 0) - baseline) * 100
        hall = data.get('hallucination_rate', 0) * 100
        
        diff_class = 'good' if diff > 0 else 'bad' if diff < -5 else 'neutral'
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        
        if diff > 0:
            status = '🏆 超越'
            row_class = 'winner'
        elif abs(diff) < 2:
            status = '📊 接近'
            row_class = 'baseline'
        else:
            status = ''
            row_class = ''
        
        html += f"""
                <tr class="{row_class}">
                    <td><strong>{model_name}</strong></td>
                    <td class="accuracy">{acc:.1f}%</td>
                    <td class="{diff_class}">{diff_str}</td>
                    <td>{hall:.1f}%</td>
                    <td>{status}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
"""
    
    # 子领域对比
    html += """
        <div class="card">
            <h2>🌟 子领域准确率</h2>
            <table>
                <tr>
                    <th>子领域</th>
"""
    
    for model_name in results.keys():
        html += f"<th>{model_name}</th>"
    
    html += "</tr>"
    
    # 收集所有子领域
    all_subfields = set()
    for data in results.values():
        all_subfields.update(data.get('subfield_accuracy', {}).keys())
    
    for subfield in sorted(all_subfields):
        html += f"<tr><td><strong>{subfield}</strong></td>"
        
        for model_name, data in results.items():
            acc = data.get('subfield_accuracy', {}).get(subfield, 0) * 100
            html += f'<td>{acc:.1f}% <div class="progress-bar"><div class="progress-fill" style="width: {acc}%"></div></div></td>'
        
        html += "</tr>"
    
    html += """
            </table>
        </div>
"""
    
    # 关键指标
    best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
    
    html += f"""
        <div class="card">
            <h2>🎯 关键指标 (最佳模型: {best_model[0]})</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{best_model[1].get('accuracy', 0)*100:.1f}%</div>
                    <div class="metric-label">总体准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model[1].get('hallucination_rate', 0)*100:.1f}%</div>
                    <div class="metric-label">幻觉率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model[1].get('calibration_error', 0):.4f}</div>
                    <div class="metric-label">ECE (校准误差)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model[1].get('avg_latency', 0):.2f}s</div>
                    <div class="metric-label">平均延迟</div>
                </div>
            </div>
        </div>
"""
    
    # 结论
    best_acc = best_model[1].get('accuracy', 0)
    baseline_diff = (best_acc - baseline) * 100
    
    if baseline_diff > 0:
        conclusion = f"✅ <strong>{best_model[0]}</strong> 成功超越 AstroMLab-8B 基准，领先 <span class='good'>+{baseline_diff:.1f}%</span>！"
    else:
        conclusion = f"⚠️ 尚未超越 AstroMLab-8B 基准，差距 <span class='bad'>{baseline_diff:.1f}%</span>，建议优化训练策略。"
    
    html += f"""
        <div class="card">
            <h2>📝 结论</h2>
            <p style="font-size: 1.1em; line-height: 1.6;">{conclusion}</p>
        </div>
        
        <p style="text-align: center; color: #8892b0; margin-top: 40px;">
            生成时间: 2026-03-11 | 天文大模型评估系统 v1.0
        </p>
    </div>
</body>
</html>
"""
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML 报告已生成: {output_path}")


def load_result_file(path: str) -> dict:
    """加载评估结果文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="评估结果可视化")
    parser.add_argument("--results", type=str, nargs="+", required=True,
                       help="评估结果 JSON 文件路径")
    parser.add_argument("--labels", type=str, nargs="+",
                       help="模型标签 (与 results 一一对应)")
    parser.add_argument("--output", type=str, default="./report.html",
                       help="输出 HTML 文件路径")
    
    args = parser.parse_args()
    
    # 加载结果
    results = {}
    for i, path in enumerate(args.results):
        label = args.labels[i] if args.labels and i < len(args.labels) else f"Model-{i+1}"
        results[label] = load_result_file(path)
    
    # 生成报告
    generate_html_report(results, args.output)


if __name__ == "__main__":
    # 示例用法
    print("评估结果可视化工具")
    print("=" * 60)
    print("\n用法:")
    print("  python visualize_results.py \\")
    print("    --results result1.json result2.json \\")
    print("    --labels 'Qwen-8B-Astro' 'AstroMLab-8B' \\")
    print("    --output report.html")
    
    # 生成示例报告
    print("\n生成示例报告...")
    example_results = {
        "AstroMLab-8B (基准)": {
            "accuracy": 0.809,
            "hallucination_rate": 0.12,
            "calibration_error": 0.065,
            "avg_latency": 0.5,
            "subfield_accuracy": {
                "stellar": 0.82,
                "exoplanet": 0.78,
                "cosmology": 0.83,
                "galactic": 0.80,
                "high_energy": 0.81
            }
        },
        "Qwen-8B-Astro": {
            "accuracy": 0.825,
            "hallucination_rate": 0.085,
            "calibration_error": 0.045,
            "avg_latency": 0.45,
            "subfield_accuracy": {
                "stellar": 0.84,
                "exoplanet": 0.80,
                "cosmology": 0.82,
                "galactic": 0.83,
                "high_energy": 0.81
            }
        }
    }
    
    generate_html_report(example_results, "./demo_report.html")
    print("\n你可以打开 demo_report.html 查看示例报告")
