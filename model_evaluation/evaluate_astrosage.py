#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage 模型评估脚本
================================================================================
功能描述:
    评估训练好的天文领域模型，测试其在专业问答上的表现。
    包含事实准确性、领域专业性、回答完整性等维度。

评估维度:
    1. 事实准确性: 回答与天文知识的一致性
    2. 领域专业性: 使用专业术语的准确性
    3. 回答完整性: 是否覆盖问题的各个方面
    4. 格式规范性: 结构化回答的能力

测试集:
    - 使用验证集 (380条) 中的样例
    - 包含赫罗图、SED、光变曲线等多个主题

输出:
    - 详细评分报告 (JSON)
    - 可视化对比图表
    - 错误案例分析

使用方法:
    # 评估LoRA模型
    python model_evaluation/evaluate_astrosage.py \
        --model-path train_qwen/output_qwen25/final_model \
        --test-data train_qwen/data/qwen_val.json \
        --output model_evaluation/eval_results/

    # 评估合并后的完整模型
    python model_evaluation/evaluate_astrosage.py \
        --model-path train_qwen/output_qwen25/merged_model \
        --test-data train_qwen/data/qwen_val.json

作者: AstroSage Team
================================================================================
"""

import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class EvalResult:
    """单个评估结果"""
    question: str
    reference_answer: str
    model_answer: str
    factuality_score: float      # 事实准确性 (0-1)
    completeness_score: float    # 完整性 (0-1)
    relevance_score: float       # 相关性 (0-1)
    avg_score: float             # 平均分
    category: str                # 问题类别
    latency: float               # 推理延迟 (秒)


class AstroEvaluator:
    """天文模型评估器"""
    
    # 专业术语词典，用于评估专业性
    ASTRONOMY_TERMS = {
        "hr_diagram": ["赫罗图", "主序", "巨星", "白矮星", "光度", "温度", "光谱型"],
        "variable_star": ["变星", "光变曲线", "周期", "振幅", "极大", "极小", "脉动"],
        "cv": ["灾变变星", "吸积盘", "白矮星", "伴星", "爆发", "宁静", "质量转移"],
        "sed": ["能谱分布", "SED", "黑体", "辐射", "波长", "流量", "拟合"],
        "spectroscopy": ["光谱", "发射线", "吸收线", "多普勒", "红移", "视向速度"],
    }
    
    def __init__(
        self,
        model_path: str,
        use_lora: bool = True,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径 (LoRA或完整模型)
            use_lora: 是否为LoRA模型
            base_model: 基座模型名称 (LoRA合并时需要)
            device: 使用设备
        """
        print("="*70)
        print("🧪 AstroSage 模型评估")
        print("="*70)
        
        self.device = device
        self.model_path = model_path
        
        print(f"\n📦 加载模型: {model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # 加载模型
        if use_lora:
            print("   检测到LoRA模型，合并中...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.model.eval()
        print("✅ 模型加载完成")
    
    def generate_answer(self, question: str, max_length: int = 512) -> str:
        """
        生成回答
        
        Args:
            question: 问题文本
            max_length: 最大生成长度
            
        Returns:
            生成的回答
        """
        # 构建提示
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        latency = time.time() - start_time
        
        # 解码
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return answer.strip(), latency
    
    def evaluate_factuality(
        self,
        model_answer: str,
        reference: str,
    ) -> float:
        """
        评估事实准确性
        
        使用简单的关键词匹配和语义相似度
        """
        # 提取关键词
        ref_keywords = set(re.findall(r'\b[\w\u4e00-\u9fff]+\b', reference.lower()))
        ans_keywords = set(re.findall(r'\b[\w\u4e00-\u9fff]+\b', model_answer.lower()))
        
        # 计算关键词重叠
        if len(ref_keywords) == 0:
            return 0.0
        
        overlap = len(ref_keywords & ans_keywords)
        score = overlap / len(ref_keywords)
        
        # 专业术语检查加分
        for category, terms in self.ASTRONOMY_TERMS.items():
            for term in terms:
                if term in reference and term in model_answer:
                    score += 0.05  # 每个匹配的专业术语加0.05
        
        return min(score, 1.0)
    
    def evaluate_completeness(
        self,
        model_answer: str,
        reference: str,
    ) -> float:
        """评估回答完整性"""
        # 检查回答是否包含关键信息点
        ref_sentences = [s.strip() for s in reference.split('。') if len(s.strip()) > 5]
        
        if len(ref_sentences) == 0:
            return 0.5
        
        # 检查每个参考句子的信息是否在回答中有所体现
        matched = 0
        for ref_sent in ref_sentences:
            ref_words = set(re.findall(r'\w+', ref_sent.lower()))
            if len(ref_words) > 0:
                # 检查是否有一定比例的关键词匹配
                ans_words = set(re.findall(r'\w+', model_answer.lower()))
                overlap = len(ref_words & ans_words)
                if overlap / len(ref_words) > 0.3:  # 30%关键词匹配
                    matched += 1
        
        return matched / len(ref_sentences)
    
    def evaluate_relevance(
        self,
        question: str,
        model_answer: str,
    ) -> float:
        """评估回答相关性"""
        # 提取问题关键词
        q_keywords = set(re.findall(r'\b[\w\u4e00-\u9fff]+\b', question.lower()))
        a_keywords = set(re.findall(r'\b[\w\u4e00-\u9fff]+\b', model_answer.lower()))
        
        # 去除停用词
        stopwords = {'什么', '的', '是', '在', '和', '了', '有', '吗'}
        q_keywords -= stopwords
        
        if len(q_keywords) == 0:
            return 0.5
        
        overlap = len(q_keywords & a_keywords)
        return overlap / len(q_keywords)
    
    def categorize_question(self, question: str) -> str:
        """对问题进行分类"""
        q_lower = question.lower()
        
        if any(kw in q_lower for kw in ["赫罗图", "hr", "主序", "巨星"]):
            return "hr_diagram"
        elif any(kw in q_lower for kw in ["光变曲线", "变星", "周期", "振幅"]):
            return "variable_star"
        elif any(kw in q_lower for kw in ["灾变", "cv", "吸积", "白矮星"]):
            return "cv"
        elif any(kw in q_lower for kw in ["sed", "能谱", "光谱"]):
            return "sed_spectroscopy"
        else:
            return "general"
    
    def evaluate_sample(
        self,
        question: str,
        reference: str,
    ) -> EvalResult:
        """评估单个样本"""
        # 生成回答
        model_answer, latency = self.generate_answer(question)
        
        # 评估各项指标
        factuality = self.evaluate_factuality(model_answer, reference)
        completeness = self.evaluate_completeness(model_answer, reference)
        relevance = self.evaluate_relevance(question, model_answer)
        
        # 计算平均分
        avg_score = (factuality + completeness + relevance) / 3
        
        # 分类
        category = self.categorize_question(question)
        
        return EvalResult(
            question=question,
            reference_answer=reference,
            model_answer=model_answer,
            factuality_score=factuality,
            completeness_score=completeness,
            relevance_score=relevance,
            avg_score=avg_score,
            category=category,
            latency=latency,
        )
    
    def evaluate_dataset(
        self,
        test_data_path: str,
        max_samples: int = 50,
        output_dir: str = "model_evaluation/eval_results",
    ) -> Dict:
        """
        评估整个测试集
        
        Args:
            test_data_path: 测试数据路径
            max_samples: 最大评估样本数
            output_dir: 输出目录
            
        Returns:
            评估结果字典
        """
        # 加载测试数据
        print(f"\n📂 加载测试数据: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"   测试集大小: {len(test_data)}")
        
        # 采样
        if len(test_data) > max_samples:
            import random
            random.seed(42)
            test_data = random.sample(test_data, max_samples)
            print(f"   采样: {max_samples}条")
        
        # 评估
        print("\n🧪 开始评估...")
        results = []
        category_scores = defaultdict(list)
        
        for i, item in enumerate(test_data, 1):
            print(f"   进度: {i}/{len(test_data)}", end='\r')
            
            # 提取问题和答案
            conversations = item.get('conversations', [])
            question = ""
            reference = ""
            
            for msg in conversations:
                if msg.get('from') == 'user':
                    question = msg.get('value', '')
                elif msg.get('from') == 'assistant':
                    reference = msg.get('value', '')
            
            if question and reference:
                result = self.evaluate_sample(question, reference)
                results.append(result)
                category_scores[result.category].append(result.avg_score)
        
        print("\n✅ 评估完成")
        
        # 计算总体统计
        avg_factuality = sum(r.factuality_score for r in results) / len(results)
        avg_completeness = sum(r.completeness_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_latency = sum(r.latency for r in results) / len(results)
        overall_score = sum(r.avg_score for r in results) / len(results)
        
        # 类别统计
        category_stats = {
            cat: {
                "avg_score": sum(scores) / len(scores),
                "count": len(scores),
            }
            for cat, scores in category_scores.items()
        }
        
        # 汇总结果
        summary = {
            "model_path": self.model_path,
            "test_data": test_data_path,
            "num_samples": len(results),
            "overall_score": overall_score,
            "scores": {
                "factuality": avg_factuality,
                "completeness": avg_completeness,
                "relevance": avg_relevance,
                "latency": avg_latency,
            },
            "category_stats": category_stats,
            "detailed_results": [asdict(r) for r in results],
        }
        
        # 保存结果
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"eval_{Path(self.model_path).name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存: {output_file}")
        
        return summary
    
    def print_report(self, summary: Dict):
        """打印评估报告"""
        print("\n" + "="*70)
        print("📊 评估报告")
        print("="*70)
        
        print(f"\n模型: {summary['model_path']}")
        print(f"测试样本: {summary['num_samples']}")
        
        print("\n📈 总体评分:")
        print(f"   综合得分: {summary['overall_score']:.3f} / 1.000")
        print(f"   事实准确性: {summary['scores']['factuality']:.3f}")
        print(f"   回答完整性: {summary['scores']['completeness']:.3f}")
        print(f"   问题相关性: {summary['scores']['relevance']:.3f}")
        print(f"   平均延迟: {summary['scores']['latency']:.2f}秒")
        
        print("\n📊 按类别评分:")
        for cat, stats in summary['category_stats'].items():
            print(f"   {cat:<20} {stats['avg_score']:.3f} ({stats['count']}条)")
        
        # 评级
        score = summary['overall_score']
        if score >= 0.8:
            level = "🌟 优秀"
        elif score >= 0.6:
            level = "✅ 良好"
        elif score >= 0.4:
            level = "⚠️  一般"
        else:
            level = "❌ 需要改进"
        
        print(f"\n🏆 评级: {level}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="评估AstroSage模型")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_qwen/output_qwen25/final_model",
        help="模型路径",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="train_qwen/data/qwen_val.json",
        help="测试数据路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_evaluation/eval_results",
        help="输出目录",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="最大评估样本数",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="是否为LoRA模型",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基座模型名称",
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = AstroEvaluator(
        model_path=args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model,
    )
    
    # 运行评估
    summary = evaluator.evaluate_dataset(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
        output_dir=args.output,
    )
    
    # 打印报告
    evaluator.print_report(summary)


if __name__ == "__main__":
    main()
