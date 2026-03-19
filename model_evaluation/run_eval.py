#!/usr/bin/env python3
"""
天文大模型评估运行脚本

使用方法:
    # 评估 Ollama 模型
    python run_eval.py --model qwen3:8b --interface ollama --dataset astromlab1
    
    # 评估本地 HuggingFace 模型
    python run_eval.py --model /path/to/model --interface hf --dataset custom --data-path ./data/my_eval.json
    
    # 对比多个模型
    python run_eval.py --models qwen3:8b astrosage-8b --compare --dataset astromlab1
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from astronomy_evaluator import (
    AstronomyEvaluator, 
    OllamaModelInterface, 
    HuggingFaceModelInterface
)


def create_model_interface(model_path: str, interface_type: str = "ollama"):
    """创建模型接口"""
    if interface_type == "ollama":
        return OllamaModelInterface(model_path)
    elif interface_type == "hf":
        return HuggingFaceModelInterface(model_path)
    else:
        raise ValueError(f"不支持的接口类型: {interface_type}")


def load_dataset(dataset_name: str, data_path: str = None):
    """加载数据集"""
    evaluator = AstronomyEvaluator()
    
    if dataset_name == "astromlab1":
        return evaluator.load_astromlab1_dataset(data_path)
    elif dataset_name == "custom":
        if not data_path:
            raise ValueError("自定义数据集需要提供 --data-path 参数")
        return evaluator.load_custom_dataset(data_path)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def evaluate_model(model_name: str, interface_type: str, dataset: list, output_dir: str = "./eval_results"):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建模型接口
    model = create_model_interface(model_name, interface_type)
    
    # 创建评估器
    evaluator = AstronomyEvaluator(model, output_dir)
    
    # 运行评估
    summary = evaluator.evaluate(dataset)
    
    # 生成报告
    report = evaluator.generate_report(summary)
    print("\n" + report)
    
    # 保存结果
    output_path = evaluator.save_results(summary)
    
    return summary, output_path


def compare_models(model_names: list, interface_type: str, dataset: list, output_dir: str = "./eval_results"):
    """对比多个模型"""
    results = {}
    
    for model_name in model_names:
        summary, _ = evaluate_model(model_name, interface_type, dataset, output_dir)
        results[model_name] = summary
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("模型对比报告")
    print("=" * 70)
    
    # 表格头部
    print(f"\n{'模型':<25} {'准确率':<10} {'vs AstroMLab-8B':<15} {'幻觉率':<10} {'ECE':<10}")
    print("-" * 70)
    
    baseline = 0.809  # AstroMLab-8B 基准
    
    for model_name, summary in results.items():
        acc = summary.accuracy * 100
        diff = (summary.accuracy - baseline) * 100
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        hall = summary.hallucination_rate * 100
        ece = summary.calibration_error
        
        print(f"{model_name:<25} {acc:<10.2f} {diff_str:<15} {hall:<10.1f} {ece:<10.4f}")
    
    # 标注基准线
    print("-" * 70)
    print(f"{'AstroMLab-8B (基准)':<25} {80.9:<10.2f} {'baseline':<15} {'-':<10} {'-':<10}")
    
    print("\n" + "=" * 70)
    
    # 子领域对比
    print("\n【子领域准确率对比】")
    all_subfields = set()
    for summary in results.values():
        all_subfields.update(summary.subfield_accuracy.keys())
    
    for subfield in sorted(all_subfields):
        print(f"\n  {subfield}:")
        for model_name, summary in results.items():
            acc = summary.subfield_accuracy.get(subfield, 0) * 100
            print(f"    {model_name:20s}: {acc:5.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="天文大模型评估工具")
    
    # 模型配置
    parser.add_argument("--model", type=str, help="模型名称或路径")
    parser.add_argument("--models", type=str, nargs="+", help="多个模型名称（用于对比）")
    parser.add_argument("--interface", type=str, default="ollama", 
                       choices=["ollama", "hf"], help="模型接口类型")
    
    # 数据集配置
    parser.add_argument("--dataset", type=str, default="astromlab1",
                       choices=["astromlab1", "custom"], help="评估数据集")
    parser.add_argument("--data-path", type=str, help="自定义数据集路径")
    
    # 评估配置
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="输出目录")
    parser.add_argument("--compare", action="store_true", help="对比模式")
    parser.add_argument("--max-samples", type=int, help="最大评估样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.models and not args.model:
        parser.error("需要提供 --model 或 --models 参数")
    
    # 加载数据集
    print(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset, args.data_path)
    
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"使用 {args.max_samples} 个样本进行测试")
    
    # 运行评估
    if args.compare or args.models:
        # 对比模式
        model_list = args.models if args.models else [args.model]
        compare_models(model_list, args.interface, dataset, args.output_dir)
    else:
        # 单模型评估
        evaluate_model(args.model, args.interface, dataset, args.output_dir)


if __name__ == "__main__":
    main()
