#!/usr/bin/env python3
"""
天文领域大模型评估器
Astronomy LLM Evaluator

用于评估和对比天文领域大模型的性能
支持 AstroMLab-1 基准测试和自定义评估
"""

import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures
from pathlib import Path

import numpy as np
from tqdm import tqdm


@dataclass
class EvalResult:
    """单个问题的评估结果"""
    question_id: str
    question: str
    options: Dict[str, str]
    correct_answer: str
    model_answer: str
    is_correct: bool
    confidence: float = 0.0
    reasoning: str = ""
    latency: float = 0.0
    subfield: str = ""
    difficulty: str = "medium"


@dataclass
class EvalSummary:
    """评估汇总结果"""
    model_name: str
    total_questions: int
    correct_count: int
    accuracy: float
    
    # 子领域准确率
    subfield_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # 难度分层准确率
    difficulty_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # 幻觉检测结果
    hallucination_detected: int = 0
    hallucination_rate: float = 0.0
    
    # 校准指标
    calibration_error: float = 0.0
    
    # 性能指标
    avg_latency: float = 0.0
    
    # 详细结果
    results: List[EvalResult] = field(default_factory=list)


class AstronomyEvaluator:
    """
    天文领域大模型评估器
    
    功能:
    1. 加载和执行 AstroMLab-1 基准测试
    2. 自定义天文领域评估
    3. 幻觉检测
    4. 置信度校准分析
    5. 生成对比报告
    """
    
    # 关键天文概念 - 用于幻觉检测
    HALLUCINATION_CHECKS = {
        "am_cvn_neutron_star": {
            "pattern": r"中子星|neutron star",
            "context": r"AM CVn|AM\s+CVn",
            "should_exist": False,
            "description": "AM CVn 系统不应该包含中子星"
        },
        "pl_relation_formula": {
            "pattern": r"ΔF/F\s*=|\\frac\{\\Delta F\}\{F\}",
            "context": r"AM CVn|周期.*光度|P-L",
            "should_exist": False,
            "description": "不应该虚构 AM CVn 的周期-光度公式"
        },
        "tidal_variability": {
            "pattern": r"潮汐.*(导致|引起|造成)|tidal.*(cause|lead)",
            "context": r"光变|variability|CV|cataclysmic",
            "should_exist": False,
            "description": "CV/AM CVn 的光变不应该归因于潮汐"
        },
        "am_cvn_period": {
            "pattern": r"(period|周期).*(5-65\s*min|5.*65.*分钟)",
            "context": r"AM CVn",
            "should_exist": True,
            "description": "AM CVn 周期应该为 5-65 分钟"
        },
        "wd_composition": {
            "pattern": r"(氦星|helium\s*star|He\s*star).*(供| donor)|双白矮",
            "context": r"AM CVn",
            "should_exist": True,
            "description": "AM CVn 应该是白矮星 + 氦星组成"
        }
    }
    
    # 天文子领域分类关键词
    SUBFIELD_KEYWORDS = {
        "stellar": ["恒星", "star", "主序", "main sequence", "巨星", "giant", "白矮星", "white dwarf"],
        "exoplanet": ["系外行星", "exoplanet", "行星", "planet", "凌日", "transit", "径向速度", "radial velocity"],
        "cosmology": ["宇宙学", "cosmology", "宇宙", "universe", "大爆炸", "big bang", "暗能量", "dark energy"],
        "galactic": ["星系", "galaxy", "银河系", "milky way", "星云", "nebula"],
        "high_energy": ["高能", "high energy", "X射线", "gamma", "黑洞", "black hole", "中子星", "neutron star"],
        "instrumentation": ["仪器", "instrument", "望远镜", "telescope", "探测器", "detector", "光谱仪"],
        "solar": ["太阳", "sun", "日冕", "corona", "太阳风", "solar wind"],
        "mechanics": ["天体力学", "celestial mechanics", "轨道", "orbit", "开普勒", "Kepler"]
    }
    
    def __init__(self, model_interface=None, output_dir: str = "./eval_results"):
        """
        初始化评估器
        
        Args:
            model_interface: 模型接口，需要实现 generate() 方法
            output_dir: 结果输出目录
        """
        self.model = model_interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_astromlab1_dataset(self, dataset_path: Optional[str] = None) -> List[Dict]:
        """
        加载 AstroMLab-1 基准数据集
        
        数据集格式:
        [
            {
                "id": "q_001",
                "question": "什么是造父变星的周期-光度关系？",
                "options": {
                    "A": "周期越长，光度越低",
                    "B": "周期越长，光度越高",
                    "C": "周期和光度无关",
                    "D": "周期和光度呈随机关系"
                },
                "correct_answer": "B",
                "subfield": "stellar",
                "difficulty": "medium"
            },
            ...
        ]
        """
        if dataset_path is None:
            # 默认路径
            dataset_path = self.output_dir.parent / "data" / "astromlab1_benchmark.json"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 加载 AstroMLab-1 数据集: {len(data)} 题")
        return data
    
    def load_custom_dataset(self, dataset_path: str) -> List[Dict]:
        """加载自定义评估数据集"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 自动推断子领域
        for item in data:
            if 'subfield' not in item or not item['subfield']:
                item['subfield'] = self._infer_subfield(item['question'])
        
        print(f"✓ 加载自定义数据集: {len(data)} 题")
        return data
    
    def _infer_subfield(self, question: str) -> str:
        """根据问题内容推断子领域"""
        question_lower = question.lower()
        scores = {}
        
        for subfield, keywords in self.SUBFIELD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in question_lower)
            if score > 0:
                scores[subfield] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def evaluate_single(self, question: Dict) -> EvalResult:
        """
        评估单个问题
        
        Args:
            question: 包含问题、选项、正确答案的字典
            
        Returns:
            EvalResult: 评估结果
        """
        start_time = time.time()
        
        # 构建提示词
        prompt = self._build_prompt(question)
        
        # 调用模型生成答案
        response = self.model.generate(prompt)
        
        latency = time.time() - start_time
        
        # 解析答案
        model_answer, confidence, reasoning = self._parse_answer(response, question.get('options', {}))
        
        # 判断正确性
        is_correct = self._check_answer(model_answer, question.get('correct_answer', ''))
        
        # 检测幻觉
        has_hallucination = self._detect_hallucination(question['question'] + response)
        
        return EvalResult(
            question_id=question.get('id', 'unknown'),
            question=question['question'],
            options=question.get('options', {}),
            correct_answer=question.get('correct_answer', ''),
            model_answer=model_answer,
            is_correct=is_correct,
            confidence=confidence,
            reasoning=reasoning,
            latency=latency,
            subfield=question.get('subfield', 'general'),
            difficulty=question.get('difficulty', 'medium')
        )
    
    def _build_prompt(self, question: Dict) -> str:
        """构建评估提示词"""
        prompt = f"""请回答以下天文学问题：

问题：{question['question']}

"""
        if 'options' in question and question['options']:
            prompt += "选项：\n"
            for key, value in question['options'].items():
                prompt += f"{key}. {value}\n"
            prompt += "\n请只回答选项字母（如 A、B、C、D），并简要说明理由。"
        else:
            prompt += "\n请给出详细答案和推理过程。"
        
        return prompt
    
    def _parse_answer(self, response: str, options: Dict[str, str]) -> Tuple[str, float, str]:
        """
        解析模型回答
        
        Returns:
            (答案, 置信度, 推理过程)
        """
        response = response.strip()
        
        # 尝试提取选项字母
        answer = ""
        confidence = 0.5
        reasoning = response
        
        if options:
            # 匹配 "答案是 A" 或 "选 B" 或 "A." 等模式
            patterns = [
                r'(?:答案|选|选择|answer|option)[是为:：\s]*([A-D])',
                r'(?:^|\s)([A-D])(?:\s*[.、)）]|\s+选项)',
                r'(?:选项|choice)\s*([A-D])',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    answer = match.group(1).upper()
                    break
            
            # 如果没有匹配到，检查是否直接出现选项字母
            if not answer:
                for opt in ['A', 'B', 'C', 'D']:
                    if response.startswith(opt) and opt in options:
                        answer = opt
                        break
        
        # 尝试提取置信度
        conf_patterns = [
            r'置信度[:：\s]*([0-9.]+)',
            r'confidence[:\s]*([0-9.]+)',
            r'([0-9.]+)\s*%\s*(?:置信|confidence)',
        ]
        for pattern in conf_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    if confidence > 1:
                        confidence /= 100
                    break
                except:
                    pass
        
        return answer, confidence, reasoning
    
    def _check_answer(self, model_answer: str, correct_answer: str) -> bool:
        """检查答案是否正确"""
        if not model_answer or not correct_answer:
            return False
        
        model_answer = model_answer.strip().upper()
        correct_answer = correct_answer.strip().upper()
        
        return model_answer == correct_answer
    
    def _detect_hallucination(self, text: str) -> bool:
        """
        检测天文领域幻觉
        
        Returns:
            是否检测到幻觉
        """
        text_lower = text.lower()
        has_hallucination = False
        
        for check_name, check_info in self.HALLUCINATION_CHECKS.items():
            pattern = check_info['pattern']
            context = check_info['context']
            should_exist = check_info['should_exist']
            
            # 检查上下文是否匹配
            if re.search(context, text, re.IGNORECASE):
                # 检查关键模式是否存在
                pattern_exists = re.search(pattern, text, re.IGNORECASE)
                
                if should_exist and not pattern_exists:
                    # 应该存在但不存在（漏检）
                    pass  # 暂时不处理这种情况
                elif not should_exist and pattern_exists:
                    # 不应该存在但存在（幻觉）
                    has_hallucination = True
                    break
        
        return has_hallucination
    
    def evaluate(self, dataset: List[Dict], max_workers: int = 1) -> EvalSummary:
        """
        评估整个数据集
        
        Args:
            dataset: 问题列表
            max_workers: 并行工作数 (1 表示串行)
            
        Returns:
            EvalSummary: 评估汇总
        """
        print(f"\n开始评估: {len(dataset)} 题")
        print("=" * 60)
        
        results = []
        
        if max_workers > 1:
            # 并行评估
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.evaluate_single, q): q for q in dataset}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset)):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"评估失败: {e}")
        else:
            # 串行评估
            for question in tqdm(dataset):
                try:
                    result = self.evaluate_single(question)
                    results.append(result)
                except Exception as e:
                    print(f"评估失败 (ID={question.get('id')}): {e}")
        
        # 计算汇总统计
        summary = self._compute_summary(results)
        
        print(f"\n评估完成!")
        print(f"总体准确率: {summary.accuracy*100:.2f}%")
        
        return summary
    
    def _compute_summary(self, results: List[EvalResult]) -> EvalSummary:
        """计算评估汇总"""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0
        
        # 子领域统计
        subfield_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            subfield_stats[r.subfield]['total'] += 1
            if r.is_correct:
                subfield_stats[r.subfield]['correct'] += 1
        
        subfield_accuracy = {
            sf: stats['correct'] / stats['total'] 
            for sf, stats in subfield_stats.items()
        }
        
        # 难度分层统计
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            difficulty_stats[r.difficulty]['total'] += 1
            if r.is_correct:
                difficulty_stats[r.difficulty]['correct'] += 1
        
        difficulty_accuracy = {
            d: stats['correct'] / stats['total']
            for d, stats in difficulty_stats.items()
        }
        
        # 幻觉统计
        hallucination_count = sum(1 for r in results if self._detect_hallucination(r.reasoning))
        
        # 性能统计
        avg_latency = np.mean([r.latency for r in results]) if results else 0
        
        # 计算校准误差
        calibration_error = self._compute_calibration_error(results)
        
        return EvalSummary(
            model_name=getattr(self.model, 'model_name', 'unknown'),
            total_questions=total,
            correct_count=correct,
            accuracy=accuracy,
            subfield_accuracy=dict(subfield_accuracy),
            difficulty_accuracy=dict(difficulty_accuracy),
            hallucination_detected=hallucination_count,
            hallucination_rate=hallucination_count / total if total > 0 else 0,
            calibration_error=calibration_error,
            avg_latency=avg_latency,
            results=results
        )
    
    def _compute_calibration_error(self, results: List[EvalResult], n_bins: int = 10) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        """
        # 按置信度分桶
        bins = defaultdict(lambda: {'correct': 0, 'total': 0, 'conf_sum': 0})
        
        for r in results:
            if r.confidence > 0:
                bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
                bins[bin_idx]['total'] += 1
                bins[bin_idx]['conf_sum'] += r.confidence
                if r.is_correct:
                    bins[bin_idx]['correct'] += 1
        
        # 计算 ECE
        total = len(results)
        if total == 0:
            return 0.0
        
        ece = 0.0
        for bin_idx, stats in bins.items():
            if stats['total'] > 0:
                bin_acc = stats['correct'] / stats['total']
                bin_conf = stats['conf_sum'] / stats['total']
                ece += abs(bin_acc - bin_conf) * stats['total'] / total
        
        return ece
    
    def save_results(self, summary: EvalSummary, output_path: Optional[str] = None):
        """保存评估结果"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"eval_{summary.model_name}_{timestamp}.json"
        
        # 转换为可序列化的字典
        data = {
            'model_name': summary.model_name,
            'total_questions': summary.total_questions,
            'correct_count': summary.correct_count,
            'accuracy': summary.accuracy,
            'subfield_accuracy': summary.subfield_accuracy,
            'difficulty_accuracy': summary.difficulty_accuracy,
            'hallucination_detected': summary.hallucination_detected,
            'hallucination_rate': summary.hallucination_rate,
            'calibration_error': summary.calibration_error,
            'avg_latency': summary.avg_latency,
            'detailed_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'correct_answer': r.correct_answer,
                    'model_answer': r.model_answer,
                    'is_correct': r.is_correct,
                    'confidence': r.confidence,
                    'subfield': r.subfield,
                    'difficulty': r.difficulty
                }
                for r in summary.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存: {output_path}")
        return output_path
    
    def generate_report(self, summary: EvalSummary, comparison_model: Optional[str] = None) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 70)
        report.append(f"天文领域大模型评估报告 - {summary.model_name}")
        report.append("=" * 70)
        report.append("")
        
        # 总体结果
        report.append("【总体性能】")
        report.append(f"  总题数: {summary.total_questions}")
        report.append(f"  正确数: {summary.correct_count}")
        report.append(f"  准确率: {summary.accuracy*100:.2f}%")
        report.append(f"  平均延迟: {summary.avg_latency:.2f}s")
        report.append("")
        
        # 与基准对比
        report.append("【基准对比】")
        baseline = 0.809  # AstroMLab-8B
        diff = (summary.accuracy - baseline) * 100
        if diff > 0:
            report.append(f"  vs AstroMLab-8B (80.9%): +{diff:.2f}% ↑")
        else:
            report.append(f"  vs AstroMLab-8B (80.9%): {diff:.2f}% ↓")
        report.append("")
        
        # 子领域表现
        report.append("【子领域准确率】")
        for subfield, acc in sorted(summary.subfield_accuracy.items(), key=lambda x: -x[1]):
            bar = "█" * int(acc * 20)
            report.append(f"  {subfield:15s}: {acc*100:5.1f}% {bar}")
        report.append("")
        
        # 难度分层
        report.append("【难度分层准确率】")
        for diff_level in ['easy', 'medium', 'hard']:
            if diff_level in summary.difficulty_accuracy:
                acc = summary.difficulty_accuracy[diff_level]
                report.append(f"  {diff_level:10s}: {acc*100:.1f}%")
        report.append("")
        
        # 幻觉检测
        report.append("【幻觉检测】")
        report.append(f"  检测到幻觉: {summary.hallucination_detected}/{summary.total_questions}")
        report.append(f"  幻觉率: {summary.hallucination_rate*100:.2f}%")
        report.append("")
        
        # 校准度
        report.append("【置信度校准】")
        report.append(f"  期望校准误差 (ECE): {summary.calibration_error:.4f}")
        report.append("")
        
        # 错误样例
        report.append("【错误样例】")
        errors = [r for r in summary.results if not r.is_correct][:5]
        for i, r in enumerate(errors, 1):
            report.append(f"\n  错误 {i} (ID={r.question_id}):")
            report.append(f"  Q: {r.question[:100]}...")
            report.append(f"  正确答案: {r.correct_answer}")
            report.append(f"  模型答案: {r.model_answer}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


class ModelInterface:
    """
    模型接口基类
    实际使用时需要继承此类并实现 generate 方法
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外参数 (temperature, max_tokens 等)
            
        Returns:
            生成的文本
        """
        raise NotImplementedError("子类必须实现 generate 方法")


class OllamaModelInterface(ModelInterface):
    """Ollama 模型接口"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        
        # 测试连接
        import requests
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama 连接成功: {model_name}")
        except Exception as e:
            print(f"⚠ Ollama 连接失败: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        import requests
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"[错误: HTTP {response.status_code}]"
        except Exception as e:
            return f"[错误: {str(e)}]"


class HuggingFaceModelInterface(ModelInterface):
    """HuggingFace 本地模型接口"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path.split('/')[-1])
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"正在加载模型: {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            print(f"✓ 模型加载完成")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("天文领域大模型评估器")
    print("=" * 60)
    
    # 示例: 使用 Ollama 接口
    # model = OllamaModelInterface("qwen3:8b")
    # evaluator = AstronomyEvaluator(model)
    
    # 示例: 测试幻觉检测
    test_texts = [
        "AM CVn 是一个包含白矮星和中子星的双星系统",  # 幻觉
        "AM CVn 系统的周期通常在 5-65 分钟之间",  # 正确
        "AM CVn 的光变是由潮汐相互作用引起的",  # 幻觉
    ]
    
    evaluator = AstronomyEvaluator()
    
    print("\n幻觉检测测试:")
    for text in test_texts:
        has_hall = evaluator._detect_hallucination(text)
        status = "✗ 幻觉" if has_hall else "✓ 正常"
        print(f"  {status}: {text[:40]}...")
    
    print("\n评估器初始化完成!")
    print("请使用以下方式运行评估:")
    print("  python model_evaluation/run_eval.py --model your-model --dataset astromlab1")
