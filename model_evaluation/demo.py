#!/usr/bin/env python3
"""
天文大模型评估系统演示

演示如何：
1. 构建评估数据集
2. 创建模型接口
3. 运行评估
4. 生成对比报告
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from astronomy_evaluator import (
    AstronomyEvaluator,
    OllamaModelInterface,
    HuggingFaceModelInterface,
    EvalSummary
)
from build_dataset import DatasetBuilder


def demo_build_dataset():
    """演示: 构建评估数据集"""
    print("=" * 60)
    print("演示 1: 构建评估数据集")
    print("=" * 60)
    
    builder = DatasetBuilder()
    
    # 生成评估数据 (100 题用于演示)
    eval_data = builder.generate_eval_dataset(n_samples=100)
    
    # 保存
    builder.save_dataset(eval_data, "./data/demo_eval.json")
    
    # 显示样例
    print("\n样例问题:")
    for i, q in enumerate(eval_data[:3], 1):
        print(f"\n问题 {i}: {q['question']}")
        print(f"  选项: {q['options']}")
        print(f"  正确答案: {q['correct']}")
        print(f"  子领域: {q['subfield']}")
        print(f"  难度: {q['difficulty']}")
    
    return eval_data


def demo_hallucination_detection():
    """演示: 幻觉检测"""
    print("\n" + "=" * 60)
    print("演示 2: 幻觉检测")
    print("=" * 60)
    
    evaluator = AstronomyEvaluator()
    
    # 测试文本
    test_cases = [
        {
            "text": "AM CVn 是一个包含白矮星和中子星的双星系统。",
            "expected": True,  # 幻觉
            "reason": "AM CVn 不应该包含中子星"
        },
        {
            "text": "AM CVn 系统的周期通常在 5-65 分钟之间，由白矮星和氦星组成。",
            "expected": False,  # 正确
            "reason": "正确的 AM CVn 描述"
        },
        {
            "text": "激变变星的光变主要是由潮汐相互作用引起的。",
            "expected": True,  # 幻觉
            "reason": "CV 光变不是潮汐引起的"
        },
        {
            "text": "造父变星的周期-光度关系可以用来测量星系距离。",
            "expected": False,  # 正确
            "reason": "正确的天文学知识"
        }
    ]
    
    print("\n幻觉检测结果:")
    correct_detections = 0
    
    for case in test_cases:
        has_hallucination = evaluator._detect_hallucination(case["text"])
        is_correct = (has_hallucination == case["expected"])
        
        status = "✓" if is_correct else "✗"
        result = "幻觉" if has_hallucination else "正常"
        expected_str = "幻觉" if case["expected"] else "正常"
        
        print(f"\n  {status} 检测结果: {result} (期望: {expected_str})")
        print(f"    文本: {case['text'][:50]}...")
        print(f"    说明: {case['reason']}")
        
        if is_correct:
            correct_detections += 1
    
    print(f"\n检测准确率: {correct_detections}/{len(test_cases)} ({100*correct_detections/len(test_cases):.0f}%)")


def demo_evaluate_model():
    """演示: 评估模型 (使用模拟接口)"""
    print("\n" + "=" * 60)
    print("演示 3: 模型评估流程")
    print("=" * 60)
    
    # 创建一个模拟模型接口用于演示
    class MockModelInterface:
        def __init__(self):
            self.model_name = "demo-model"
        
        def generate(self, prompt: str, **kwargs) -> str:
            """模拟模型生成 - 70% 准确率"""
            import random
            
            # 简单规则: 对于包含特定关键词的问题返回固定答案
            if "造父变星" in prompt and "周期-光度" in prompt:
                return "答案是 B。造父变星的周期越长，光度越高。"
            elif "AM CVn" in prompt:
                if "组成" in prompt:
                    return "答案是 B。AM CVn 由白矮星和氦星组成。"
                elif "周期" in prompt:
                    return "答案是 B。AM CVn 周期范围是 5-65 分钟。"
                elif "光变" in prompt:
                    return "答案是 B。AM CVn 光变来自吸积盘不稳定性。"
            elif "哈勃定律" in prompt:
                return "答案是 B。哈勃定律描述星系退行速度与距离的关系。"
            
            # 随机回答
            options = ["A", "B", "C", "D"]
            weights = [0.15, 0.55, 0.15, 0.15]  # 偏向 B
            answer = random.choices(options, weights=weights)[0]
            
            return f"答案是 {answer}。"
    
    # 加载评估数据
    evaluator = AstronomyEvaluator()
    try:
        dataset = evaluator.load_custom_dataset("./data/demo_eval.json")
    except:
        print("未找到 demo_eval.json，使用样例数据")
        from build_dataset import DatasetBuilder
        builder = DatasetBuilder()
        dataset = builder.generate_eval_dataset(n_samples=20)
    
    # 创建模型接口
    model = MockModelInterface()
    evaluator = AstronomyEvaluator(model, output_dir="./demo_results")
    
    # 运行评估
    print(f"\n评估 {len(dataset)} 题...")
    summary = evaluator.evaluate(dataset)
    
    # 生成报告
    report = evaluator.generate_report(summary)
    print("\n" + report)
    
    # 保存结果
    evaluator.save_results(summary, "./demo_results/eval_result.json")
    
    return summary


def demo_compare_models():
    """演示: 对比多个模型"""
    print("\n" + "=" * 60)
    print("演示 4: 模型对比")
    print("=" * 60)
    
    # 模拟多个模型的评估结果
    results = {
        "AstroMLab-8B (基准)": {
            "accuracy": 0.809,
            "subfield_acc": {
                "stellar": 0.82,
                "exoplanet": 0.78,
                "cosmology": 0.83,
                "galactic": 0.80
            }
        },
        "Qwen-8B-Base": {
            "accuracy": 0.72,
            "subfield_acc": {
                "stellar": 0.74,
                "exoplanet": 0.70,
                "cosmology": 0.75,
                "galactic": 0.71
            }
        },
        "Qwen-8B-Astro (你的模型)": {
            "accuracy": 0.825,  # 超越基准!
            "subfield_acc": {
                "stellar": 0.84,
                "exoplanet": 0.80,
                "cosmology": 0.82,
                "galactic": 0.83
            }
        }
    }
    
    baseline = 0.809
    
    print("\n模型对比报告:")
    print(f"\n{'模型':<25} {'准确率':<10} {'vs AstroMLab-8B':<15} {'状态':<10}")
    print("-" * 65)
    
    for model_name, data in results.items():
        acc = data["accuracy"] * 100
        diff = (data["accuracy"] - baseline) * 100
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        
        if "你的模型" in model_name:
            status = "🏆 超越" if diff > 0 else "接近"
        elif "基准" in model_name:
            status = "📊 基准"
        else:
            status = ""
        
        print(f"{model_name:<25} {acc:<10.1f} {diff_str:<15} {status:<10}")
    
    print("\n子领域准确率对比:")
    subfields = ["stellar", "exoplanet", "cosmology", "galactic"]
    
    for subfield in subfields:
        print(f"\n  {subfield}:")
        for model_name, data in results.items():
            acc = data["subfield_acc"].get(subfield, 0) * 100
            marker = " ← 最佳" if acc >= max(r["subfield_acc"].get(subfield, 0) * 100 for r in results.values()) else ""
            print(f"    {model_name:25s}: {acc:5.1f}%{marker}")


def main():
    """运行所有演示"""
    print("=" * 60)
    print("天文领域大模型评估系统演示")
    print("=" * 60)
    
    # 演示 1: 构建数据集
    demo_build_dataset()
    
    # 演示 2: 幻觉检测
    demo_hallucination_detection()
    
    # 演示 3: 模型评估
    summary = demo_evaluate_model()
    
    # 演示 4: 模型对比
    demo_compare_models()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行完整评估: python run_eval.py --model your-model --dataset custom")
    print("  2. 构建训练数据: python build_dataset.py --build-training --n-samples 10000")
    print("  3. 开始微调: bash train_qwen_astro.sh")
    print("\n详细文档: ASTRONOMY_LLM_EVALUATION_GUIDE.md")
    print("快速开始: QUICKSTART.md")


if __name__ == "__main__":
    main()
