#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage vs Ollama 模型对比评估
================================================================================
功能描述:
    对比训练好的AstroSage模型与Ollama中其他模型在天文问答上的表现。
    评估指标包括准确性、专业性、响应速度等。

对比维度:
    1. 天文专业知识准确性
    2. 专业术语使用
    3. 回答结构化程度
    4. 推理速度
    5. 幻觉率

测试问题集:
    - 覆盖赫罗图、变星、灾变变星、光谱分析等主题
    - 包含简单事实查询和复杂分析任务

支持的Ollama模型:
    - qwen2.5:7b (基座模型对比)
    - llama3.1:8b
    - deepseek-r1:7b
    - 以及其他已安装的模型

使用方法:
    # 1. 确保Ollama服务运行
    ollama serve
    
    # 2. 运行对比评估
    python model_evaluation/compare_with_ollama.py \
        --astrosage-model train_qwen/output_qwen25/merged_model \
        --ollama-models qwen2.5:7b,llama3.1:8b

输出:
    - 详细对比报告 (Markdown)
    - 评分对比表格
    - 示例问答对比

作者: AstroSage Team
================================================================================
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelResponse:
    """模型响应"""
    model_name: str
    question: str
    answer: str
    latency: float
    tokens_per_second: float


@dataclass
class ComparisonResult:
    """对比结果"""
    question: str
    category: str
    reference: str
    responses: List[ModelResponse]
    scores: Dict[str, Dict[str, float]]  # model_name -> metric -> score


class OllamaClient:
    """Ollama API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[ModelResponse]:
        """
        调用Ollama生成回答
        
        Args:
            model: 模型名称
            prompt: 提示文本
            system: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            ModelResponse对象
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,
            )
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "").strip()
                
                # 计算tokens/s
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 1) / 1e9  # 转换为秒
                tps = eval_count / eval_duration if eval_duration > 0 else 0
                
                return ModelResponse(
                    model_name=model,
                    question=prompt,
                    answer=answer,
                    latency=latency,
                    tokens_per_second=tps,
                )
            else:
                print(f"   ❌ {model} 请求失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ {model} 错误: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """获取已安装的模型列表"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return models
        except Exception as e:
            print(f"   ⚠️  获取模型列表失败: {e}")
        return []


class AstroSageClient:
    """AstroSage本地模型客户端"""
    
    def __init__(self, model_path: str):
        """
        初始化AstroSage模型
        
        Args:
            model_path: 模型路径
        """
        print(f"   加载 AstroSage 模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.model.eval()
        print("   ✅ AstroSage 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """生成回答"""
        # 构建完整提示
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        # 生成
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        latency = time.time() - start_time
        
        # 解码
        answer_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # 计算速度
        tps = len(answer_tokens) / latency if latency > 0 else 0
        
        return ModelResponse(
            model_name="AstroSage-Qwen2.5-7B",
            question=prompt,
            answer=answer.strip(),
            latency=latency,
            tokens_per_second=tps,
        )


class ModelComparator:
    """模型对比器"""
    
    # 天文领域测试问题集
    TEST_QUESTIONS = [
        {
            "question": "什么是灾变变星（CV）？简述其主要特征。",
            "category": "cv",
            "reference": "灾变变星（Cataclysmic Variable, CV）是一种由白矮星和红矮星组成的双星系统。主要特征包括：1) 白矮星通过吸积盘从伴星吸积物质；2) 经常表现出光度爆发；3) 轨道周期通常在1-10小时之间；4) 光谱中常出现发射线。",
        },
        {
            "question": "赫罗图上的主序带代表什么？太阳在主序带上的位置如何？",
            "category": "hr_diagram",
            "reference": "主序带是赫罗图上从左上到右下的对角带，代表恒星处于核心氢燃烧的稳定阶段。太阳位于主序带中部偏下，光谱型为G2V，表面温度约5778K，光度约1L☉。",
        },
        {
            "question": "解释光变曲线中的周期是什么意思，如何测量？",
            "category": "variable_star",
            "reference": "光变曲线的周期是变星亮度变化完成一个完整循环所需的时间。测量方法包括：1) 观测亮度极大值或极小值之间的时间间隔；2) 使用傅里叶分析找到主导频率；3) 相位分散最小化（PDM）方法。",
        },
        {
            "question": "什么是SED（能谱分布）？在天文研究中有什么用途？",
            "category": "sed",
            "reference": "SED（Spectral Energy Distribution）是天体在不同波长的电磁辐射能量分布。用途包括：1) 确定天体温度；2) 识别辐射机制（热辐射/非热辐射）；3) 测量尘埃消光；4) 估计天体的光度和距离。",
        },
        {
            "question": "白矮星的特征是什么？它们是如何形成的？",
            "category": "stellar_evolution",
            "reference": "白矮星是中小质量恒星演化的最终产物。特征包括：1) 密度极高（约10^6 g/cm³）；2) 半径约地球大小；3) 不再进行核聚变；4) 靠余热发光缓慢冷却。形成过程：红巨星抛射行星状星云后，核心坍缩形成。",
        },
        {
            "question": "解释双星系统中的质量转移过程。",
            "category": "binary",
            "reference": "质量转移发生在双星系统中一颗恒星充满其洛希瓣时。物质通过内拉格朗日点流向伴星。类型包括：1) 洛希瓣溢出转移；2) 星风吸积；3) 公共包层演化。这可能导致系统演化、轨道周期变化，甚至产生X射线双星。",
        },
        {
            "question": "什么是星际消光？如何校正？",
            "category": "observation",
            "reference": "星际消光是星际尘埃对星光的吸收和散射。特征：1) 蓝光比红光消光更强；2) 造成天体看起来更红（红化）。校正方法：使用消光定律（如Cardelli消光曲线），通过测量色余E(B-V)计算消光值A_V。",
        },
        {
            "question": "新星爆发的原因是什么？与超新星有什么区别？",
            "category": "cv",
            "reference": "新星爆发是白矮星表面吸积物质达到临界温度和密度后发生的热核失控。与超新星的区别：1) 新星不摧毁白矮星，可重复爆发；2) 能量低得多（10^38 vs 10^44 erg）；3) 光度上升慢、下降快；4) 抛射物质质量小得多。",
        },
    ]
    
    # 专业术语词典
    ASTRONOMY_TERMS = {
        "cv": ["灾变", "白矮星", "吸积", "吸积盘", "爆发", "质量转移", "洛希瓣"],
        "hr_diagram": ["赫罗图", "主序", "光度", "温度", "光谱型", "演化"],
        "variable_star": ["光变曲线", "周期", "振幅", "脉动", "极大", "极小"],
        "sed": ["SED", "能谱", "黑体", "辐射", "波长", "流量", "拟合"],
        "stellar_evolution": ["白矮星", "中子星", "黑洞", "演化", "核聚变"],
        "binary": ["双星", "质量转移", "洛希瓣", "轨道", "并合"],
        "observation": ["消光", "红化", "色余", "改正", "校准"],
    }
    
    def __init__(
        self,
        astrosage_path: str,
        ollama_models: List[str],
        ollama_url: str = "http://localhost:11434",
    ):
        """
        初始化对比器
        
        Args:
            astrosage_path: AstroSage模型路径
            ollama_models: Ollama模型名称列表
            ollama_url: Ollama服务地址
        """
        print("="*70)
        print("🔬 AstroSage vs Ollama 模型对比评估")
        print("="*70)
        
        self.ollama_models = ollama_models
        
        # 初始化客户端
        print("\n📦 初始化模型...")
        
        # AstroSage
        self.astrosage = AstroSageClient(astrosage_path)
        
        # Ollama
        self.ollama = OllamaClient(ollama_url)
        
        # 检查Ollama模型
        print("\n📋 检查Ollama模型...")
        available_models = self.ollama.list_models()
        print(f"   可用模型: {available_models}")
        
        self.available_ollama = []
        for model in ollama_models:
            if model in available_models:
                self.available_ollama.append(model)
                print(f"   ✅ {model}")
            else:
                print(f"   ⚠️  {model} (未安装)")
    
    def evaluate_answer(
        self,
        answer: str,
        reference: str,
        category: str,
    ) -> Dict[str, float]:
        """
        评估回答质量
        
        Returns:
            评分字典
        """
        import re
        
        scores = {}
        
        # 1. 事实准确性 - 关键词匹配
        ref_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', reference))
        ans_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', answer))
        
        if len(ref_keywords) > 0:
            overlap = len(ref_keywords & ans_keywords)
            scores["factuality"] = overlap / len(ref_keywords)
        else:
            scores["factuality"] = 0.5
        
        # 2. 专业性 - 术语使用
        terms = self.ASTRONOMY_TERMS.get(category, [])
        term_count = sum(1 for term in terms if term in answer)
        scores["professionalism"] = min(term_count / max(len(terms) * 0.3, 1), 1.0)
        
        # 3. 完整性 - 长度与结构
        if len(answer) > 50:
            scores["completeness"] = min(len(answer) / 200, 1.0)
        else:
            scores["completeness"] = 0.3
        
        # 4. 幻觉检测 - 不一致指示词
        hallucination_indicators = [
            "我不确定", "可能", "也许", "好像", "我不太清楚",
            "i'm not sure", "maybe", "perhaps", "i guess",
        ]
        has_hallucination = any(ind in answer.lower() for ind in hallucination_indicators)
        scores["confidence"] = 0.0 if has_hallucination else 1.0
        
        # 总分
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def run_comparison(self) -> List[ComparisonResult]:
        """运行对比评估"""
        print("\n🧪 开始对比评估...")
        print(f"   测试问题数: {len(self.TEST_QUESTIONS)}")
        print(f"   对比模型: AstroSage + {len(self.available_ollama)}个Ollama模型")
        
        results = []
        
        for i, test_case in enumerate(self.TEST_QUESTIONS, 1):
            print(f"\n📌 问题 {i}/{len(self.TEST_QUESTIONS)}: {test_case['category']}")
            print(f"   Q: {test_case['question'][:50]}...")
            
            responses = []
            scores = {}
            
            # 测试AstroSage
            print("   🤖 AstroSage 推理中...", end=' ')
            resp = self.astrosage.generate(test_case['question'])
            responses.append(resp)
            scores["AstroSage-Qwen2.5-7B"] = self.evaluate_answer(
                resp.answer, test_case['reference'], test_case['category']
            )
            print(f"✅ ({resp.latency:.2f}s)")
            
            # 测试Ollama模型
            for model_name in self.available_ollama:
                print(f"   🤖 {model_name} 推理中...", end=' ')
                resp = self.ollama.generate(
                    model=model_name,
                    prompt=test_case['question'],
                    system="你是一个天文专家，请用专业术语回答问题。",
                )
                
                if resp:
                    responses.append(resp)
                    scores[model_name] = self.evaluate_answer(
                        resp.answer, test_case['reference'], test_case['category']
                    )
                    print(f"✅ ({resp.latency:.2f}s)")
                else:
                    print("❌ 失败")
            
            results.append(ComparisonResult(
                question=test_case['question'],
                category=test_case['category'],
                reference=test_case['reference'],
                responses=responses,
                scores=scores,
            ))
        
        print("\n✅ 对比评估完成")
        return results
    
    def generate_report(self, results: List[ComparisonResult], output_path: str):
        """生成对比报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 计算统计
        model_scores = defaultdict(lambda: defaultdict(list))
        model_latency = defaultdict(list)
        
        for result in results:
            for model_name, scores in result.scores.items():
                for metric, score in scores.items():
                    model_scores[model_name][metric].append(score)
            
            for resp in result.responses:
                model_latency[resp.model_name].append(resp.latency)
        
        # 生成Markdown报告
        report_lines = [
            "# AstroSage vs Ollama 模型对比评估报告",
            "",
            f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**测试问题数**: {len(results)}",
            "",
            "## 📊 总体评分对比",
            "",
            "| 模型 | 综合得分 | 事实准确性 | 专业性 | 完整性 | 置信度 | 平均延迟 |",
            "|------|----------|------------|--------|--------|--------|----------|",
        ]
        
        # 按综合得分排序
        avg_scores = {}
        for model_name in model_scores:
            avg_scores[model_name] = sum(model_scores[model_name]["overall"]) / len(model_scores[model_name]["overall"])
        
        sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        for model_name, overall in sorted_models:
            scores = model_scores[model_name]
            avg_latency = sum(model_latency[model_name]) / len(model_latency[model_name])
            
            row = f"| {model_name} | {overall:.3f} | "
            row += f"{sum(scores['factuality'])/len(scores['factuality']):.3f} | "
            row += f"{sum(scores['professionalism'])/len(scores['professionalism']):.3f} | "
            row += f"{sum(scores['completeness'])/len(scores['completeness']):.3f} | "
            row += f"{sum(scores['confidence'])/len(scores['confidence']):.3f} | "
            row += f"{avg_latency:.2f}s |"
            
            report_lines.append(row)
        
        # 添加详细对比
        report_lines.extend([
            "",
            "## 📝 详细问答对比",
            "",
        ])
        
        for i, result in enumerate(results, 1):
            report_lines.extend([
                f"### 问题 {i}: {result.category}",
                "",
                f"**Q**: {result.question}",
                "",
                f"**参考答案**: {result.reference}",
                "",
            ])
            
            for resp in result.responses:
                score = result.scores.get(resp.model_name, {})
                overall = score.get("overall", 0)
                
                report_lines.extend([
                    f"#### {resp.model_name} (得分: {overall:.3f})",
                    "",
                    f"{resp.answer}",
                    "",
                    f"*延迟: {resp.latency:.2f}s, 速度: {resp.tokens_per_second:.1f} tokens/s*",
                    "",
                ])
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n💾 报告已保存: {output_path}")
        
        # 打印摘要
        print("\n" + "="*70)
        print("📊 评估摘要")
        print("="*70)
        
        for model_name, overall in sorted_models:
            emoji = "🥇" if overall > 0.7 else "🥈" if overall > 0.5 else "🥉"
            print(f"{emoji} {model_name:<30} 综合得分: {overall:.3f}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="对比AstroSage与Ollama模型")
    parser.add_argument(
        "--astrosage-model",
        type=str,
        default="train_qwen/output_qwen25/merged_model",
        help="AstroSage模型路径",
    )
    parser.add_argument(
        "--ollama-models",
        type=str,
        default="qwen2.5:7b,llama3.1:8b",
        help="Ollama模型名称，逗号分隔",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama服务地址",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_evaluation/comparison_report.md",
        help="输出报告路径",
    )
    
    args = parser.parse_args()
    
    # 解析模型列表
    ollama_models = [m.strip() for m in args.ollama_models.split(",")]
    
    # 创建对比器
    comparator = ModelComparator(
        astrosage_path=args.astrosage_model,
        ollama_models=ollama_models,
        ollama_url=args.ollama_url,
    )
    
    # 运行对比
    results = comparator.run_comparison()
    
    # 生成报告
    comparator.generate_report(results, args.output)


if __name__ == "__main__":
    main()
