#!/usr/bin/env python3
"""
可解释性分析系统 (Explainable Analysis System)
==============================================
生成推理过程的自然语言解释，便于天文学家审校

核心功能:
- 推理链追踪
- 知识引用溯源
- 置信度可视化
- 反事实解释
- 人机交互验证

作者: AstroSage AI
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ReasoningStepType(Enum):
    """推理步骤类型"""
    DATA_EXTRACTION = "数据提取"
    KNOWLEDGE_RETRIEVAL = "知识检索"
    FEATURE_ANALYSIS = "特征分析"
    PATTERN_MATCHING = "模式匹配"
    INFERENCE = "逻辑推理"
    CONCLUSION = "结论生成"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    step_type: ReasoningStepType
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    knowledge_refs: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'step_type': self.step_type.value,
            'description': self.description,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'knowledge_refs': self.knowledge_refs,
            'timestamp': self.timestamp
        }


@dataclass
class AnalysisExplanation:
    """分析解释"""
    target_name: str
    final_conclusion: str
    confidence_overall: float
    reasoning_chain: List[ReasoningStep]
    knowledge_sources: List[Dict]
    uncertainties: List[str]
    alternative_hypotheses: List[str]
    verification_suggestions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'target_name': self.target_name,
            'final_conclusion': self.final_conclusion,
            'confidence_overall': self.confidence_overall,
            'reasoning_chain': [s.to_dict() for s in self.reasoning_chain],
            'knowledge_sources': self.knowledge_sources,
            'uncertainties': self.uncertainties,
            'alternative_hypotheses': self.alternative_hypotheses,
            'verification_suggestions': self.verification_suggestions
        }


class ReasoningTracer:
    """
    推理追踪器
    
    记录分析过程的每一步
    """
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.steps: List[ReasoningStep] = []
        self.current_step = 0
        self.knowledge_sources = []
    
    def add_step(self,
                 step_type: ReasoningStepType,
                 description: str,
                 input_data: Dict,
                 output_data: Dict,
                 confidence: float = 1.0,
                 evidence: List[str] = None,
                 knowledge_refs: List[str] = None):
        """添加推理步骤"""
        self.current_step += 1
        
        step = ReasoningStep(
            step_number=self.current_step,
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            evidence=evidence or [],
            knowledge_refs=knowledge_refs or []
        )
        
        self.steps.append(step)
        
        # 记录知识来源
        if knowledge_refs:
            for ref in knowledge_refs:
                if ref not in [s['id'] for s in self.knowledge_sources]:
                    self.knowledge_sources.append({
                        'id': ref,
                        'used_in_step': self.current_step
                    })
        
        return step
    
    def get_chain_summary(self) -> str:
        """获取推理链摘要"""
        lines = [f"\n{'='*60}", f"推理链追踪: {self.target_name}", f"{'='*60}\n"]
        
        for step in self.steps:
            lines.append(f"步骤 {step.step_number}: [{step.step_type.value}]")
            lines.append(f"  描述: {step.description}")
            lines.append(f"  置信度: {step.confidence:.2f}")
            if step.knowledge_refs:
                lines.append(f"  知识引用: {', '.join(step.knowledge_refs)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def visualize_confidence(self) -> str:
        """可视化置信度"""
        lines = [f"\n{'='*60}", "置信度可视化", f"{'='*60}\n"]
        
        for step in self.steps:
            bar_length = int(step.confidence * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"步骤{step.step_number:2d} [{step.step_type.value:12s}] |{bar}| {step.confidence:.2%}")
        
        # 整体置信度
        overall = np.mean([s.confidence for s in self.steps])
        bar_length = int(overall * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"{'─'*60}")
        lines.append(f"整体置信度 |{bar}| {overall:.2%}")
        
        return "\n".join(lines)


class ExplanationGenerator:
    """
    解释生成器
    
    将推理过程转换为自然语言解释
    """
    
    def __init__(self):
        self.uncertainty_indicators = [
            '可能', '也许', '不确定', '推测', '假设',
            'might', 'possibly', 'uncertain', 'speculated'
        ]
    
    def generate_explanation(self, tracer: ReasoningTracer,
                           final_conclusion: str,
                           confidence: float) -> AnalysisExplanation:
        """生成完整解释"""
        
        # 识别不确定性
        uncertainties = self._identify_uncertainties(tracer)
        
        # 生成替代假设
        alternatives = self._generate_alternatives(tracer, final_conclusion)
        
        # 生成验证建议
        verifications = self._generate_verifications(tracer, uncertainties)
        
        return AnalysisExplanation(
            target_name=tracer.target_name,
            final_conclusion=final_conclusion,
            confidence_overall=confidence,
            reasoning_chain=tracer.steps,
            knowledge_sources=tracer.knowledge_sources,
            uncertainties=uncertainties,
            alternative_hypotheses=alternatives,
            verification_suggestions=verifications
        )
    
    def _identify_uncertainties(self, tracer: ReasoningTracer) -> List[str]:
        """识别推理过程中的不确定性"""
        uncertainties = []
        
        for step in tracer.steps:
            # 低置信度步骤
            if step.confidence < 0.7:
                uncertainties.append(
                    f"步骤{step.step_number} [{step.step_type.value}] 置信度较低 ({step.confidence:.2f}): {step.description}"
                )
            
            # 缺少证据的步骤
            if not step.evidence:
                uncertainties.append(
                    f"步骤{step.step_number} 缺少直接证据支持"
                )
        
        return uncertainties
    
    def _generate_alternatives(self, tracer: ReasoningTracer,
                               conclusion: str) -> List[str]:
        """生成替代假设"""
        alternatives = []
        
        # 基于不同类型的分析生成替代假设
        for step in tracer.steps:
            if step.step_type == ReasoningStepType.PATTERN_MATCHING:
                if 'CataclyV*' in conclusion or 'CV' in conclusion:
                    alternatives.extend([
                        "可能是低质量X射线双星 (LMXB) 而非CV",
                        "可能是AM CVn型星而非标准CV",
                        "可能是后共有包层双星 (PCEB) 前身"
                    ])
                elif 'period' in str(step.output_data):
                    period = step.output_data.get('period', 0)
                    if period < 0.1:
                        alternatives.append(f"周期 {period:.4f} 天也可能对应于脉冲白矮星 (ZZ Ceti)")
        
        return alternatives
    
    def _generate_verifications(self, tracer: ReasoningTracer,
                               uncertainties: List[str]) -> List[str]:
        """生成验证建议"""
        suggestions = []
        
        for step in tracer.steps:
            if step.step_type == ReasoningStepType.DATA_EXTRACTION:
                suggestions.append(
                    "验证原始数据质量: 检查是否有系统误差或异常值"
                )
            
            elif step.step_type == ReasoningStepType.PERIOD_ANALYSIS:
                suggestions.append(
                    "使用独立的数据集重复周期分析 (如使用ZTF数据验证TESS结果)"
                )
                suggestions.append(
                    "检查周期是否稳定，或存在周期变化 (dP/dt)"
                )
            
            elif step.step_type == ReasoningStepType.KNOWLEDGE_RETRIEVAL:
                suggestions.append(
                    "人工审核知识库匹配结果，确认天体分类的准确性"
                )
        
        # 添加通用建议
        suggestions.extend([
            "获取多波段测光数据 (U, B, V, R, I) 以确定颜色特征",
            "获取时序光谱观测以确认发射线特征",
            "检查历史巡天档案 (如DASCH) 寻找爆发记录"
        ])
        
        return suggestions[:8]  # 限制数量
    
    def generate_natural_language_report(self, explanation: AnalysisExplanation,
                                        detail_level: str = 'detailed') -> str:
        """
        生成自然语言报告
        
        Args:
            explanation: 分析解释对象
            detail_level: 'brief', 'detailed', 'comprehensive'
            
        Returns:
            自然语言报告文本
        """
        lines = []
        
        # 标题
        lines.append(f"\n{'='*70}")
        lines.append(f"🔍 可解释性分析报告: {explanation.target_name}")
        lines.append(f"{'='*70}\n")
        
        # 1. 结论摘要
        lines.append("📋 结论摘要")
        lines.append("-" * 70)
        lines.append(f"主要结论: {explanation.final_conclusion}")
        lines.append(f"整体置信度: {explanation.confidence_overall:.1%}")
        lines.append("")
        
        if detail_level == 'brief':
            return "\n".join(lines)
        
        # 2. 推理过程
        lines.append("🧠 推理过程")
        lines.append("-" * 70)
        
        for step in explanation.reasoning_chain:
            lines.append(f"\n步骤 {step.step_number}: {step.step_type.value}")
            lines.append(f"  操作: {step.description}")
            lines.append(f"  置信度: {'█' * int(step.confidence*10)}{'░' * (10-int(step.confidence*10))} {step.confidence:.1%}")
            
            if step.knowledge_refs and detail_level == 'comprehensive':
                lines.append(f"  引用知识: {', '.join(step.knowledge_refs)}")
        
        lines.append("")
        
        # 3. 知识来源
        lines.append("📚 知识来源")
        lines.append("-" * 70)
        for source in explanation.knowledge_sources[:5]:
            lines.append(f"  • {source['id']} (步骤{source['used_in_step']})")
        lines.append("")
        
        if detail_level == 'detailed':
            return "\n".join(lines)
        
        # 4. 不确定性 (comprehensive)
        if explanation.uncertainties:
            lines.append("⚠️ 不确定性")
            lines.append("-" * 70)
            for unc in explanation.uncertainties:
                lines.append(f"  • {unc}")
            lines.append("")
        
        # 5. 替代假设
        if explanation.alternative_hypotheses:
            lines.append("💭 替代假设")
            lines.append("-" * 70)
            for i, alt in enumerate(explanation.alternative_hypotheses, 1):
                lines.append(f"  {i}. {alt}")
            lines.append("")
        
        # 6. 验证建议
        if explanation.verification_suggestions:
            lines.append("✅ 验证建议")
            lines.append("-" * 70)
            for i, sug in enumerate(explanation.verification_suggestions, 1):
                lines.append(f"  {i}. {sug}")
            lines.append("")
        
        lines.append(f"{'='*70}\n")
        
        return "\n".join(lines)


class CounterfactualExplainer:
    """
    反事实解释器
    
    解释"如果条件不同，结论会如何变化"
    """
    
    def __init__(self):
        pass
    
    def generate_counterfactuals(self, 
                                target_data: Dict,
                                current_conclusion: str,
                                tracer: ReasoningTracer) -> List[Dict]:
        """
        生成反事实场景
        
        Returns:
            反事实场景列表
        """
        counterfactuals = []
        
        # 场景1: 如果周期不同
        if 'period' in str(target_data):
            period = target_data.get('periods', {}).get('TESS', {}).get('period', 0)
            
            if period < 0.1:  # 当前是短周期
                counterfactuals.append({
                    'scenario': '如果周期是2-3小时（周期空缺范围）',
                    'current_value': f"{period*24:.2f} 小时",
                    'hypothetical_value': "2.5 小时",
                    'new_conclusion': '这将是罕见的位于周期空缺的CV系统，具有重要的演化研究价值',
                    'likelihood': '低 - 周期空缺中的系统稀少'
                })
            
            if period > 0.1:  # 当前是较长周期
                counterfactuals.append({
                    'scenario': '如果周期小于1小时',
                    'current_value': f"{period*24:.2f} 小时",
                    'hypothetical_value': "0.5 小时",
                    'new_conclusion': '这将强烈暗示AM CVn型星而非标准CV',
                    'likelihood': '中等 - 需要光谱确认氦线'
                })
        
        # 场景2: 如果光度不同
        if 'magnitude' in str(target_data):
            counterfactuals.append({
                'scenario': '如果目标比当前暗3个星等',
                'current_value': '当前亮度',
                'hypothetical_value': '暗3星等',
                'new_conclusion': '可能是距离更远或处于极低态的CV',
                'likelihood': '高 - CV常见在高低态间变化'
            })
        
        # 场景3: 如果没有周期性
        counterfactuals.append({
            'scenario': '如果没有检测到周期性',
            'current_value': '检测到周期',
            'hypothetical_value': '无周期',
            'new_conclusion': '可能是非周期性变星、新星遗迹或误分类',
            'likelihood': '中等 - 需要考虑数据质量问题'
        })
        
        return counterfactuals
    
    def explain_counterfactuals(self, counterfactuals: List[Dict]) -> str:
        """生成反事实解释文本"""
        lines = [f"\n{'='*70}", "🔮 反事实分析 (What-If Scenarios)", f"{'='*70}\n"]
        
        for i, cf in enumerate(counterfactuals, 1):
            lines.append(f"场景 {i}: {cf['scenario']}")
            lines.append(f"  当前值: {cf['current_value']}")
            lines.append(f"  假设值: {cf['hypothetical_value']}")
            lines.append(f"  可能结论: {cf['new_conclusion']}")
            lines.append(f"  可能性: {cf['likelihood']}")
            lines.append("")
        
        return "\n".join(lines)


class ExplainableAstroAnalyzer:
    """
    可解释天文分析器
    
    整合所有可解释性功能的高级分析器
    """
    
    def __init__(self):
        self.tracer = None
        self.explainer = ExplanationGenerator()
        self.counterfactual = CounterfactualExplainer()
    
    def analyze_with_explanation(self, target_data: Dict, 
                                 target_name: str,
                                 analysis_func: Callable) -> Tuple[Any, AnalysisExplanation]:
        """
        执行分析并生成解释
        
        Args:
            target_data: 观测数据
            target_name: 目标名称
            analysis_func: 分析函数 (接收tracer作为参数)
            
        Returns:
            (分析结果, 分析解释)
        """
        # 初始化追踪器
        self.tracer = ReasoningTracer(target_name)
        
        # 执行分析 (分析函数内部应使用tracer记录步骤)
        result = analysis_func(target_data, self.tracer)
        
        # 生成解释
        explanation = self.explainer.generate_explanation(
            self.tracer,
            final_conclusion=str(result),
            confidence=0.85  # 可由分析函数提供
        )
        
        # 添加反事实分析
        counterfactuals = self.counterfactual.generate_counterfactuals(
            target_data, str(result), self.tracer
        )
        
        return result, explanation, counterfactuals
    
    def print_full_report(self, explanation: AnalysisExplanation,
                         counterfactuals: List[Dict] = None):
        """打印完整报告"""
        # 自然语言报告
        report = self.explainer.generate_natural_language_report(
            explanation, detail_level='comprehensive'
        )
        print(report)
        
        # 反事实分析
        if counterfactuals:
            cf_report = self.counterfactual.explain_counterfactuals(counterfactuals)
            print(cf_report)
        
        # 可视化
        print(self.tracer.visualize_confidence())


# ==================== 测试 ====================

def demo_explainable_analysis():
    """演示可解释分析"""
    print("="*70)
    print("可解释性分析系统演示")
    print("="*70)
    
    # 创建分析器
    analyzer = ExplainableAstroAnalyzer()
    
    # 模拟分析函数
    def mock_analysis(data: Dict, tracer: ReasoningTracer):
        """模拟分析过程"""
        # 步骤1: 数据提取
        tracer.add_step(
            step_type=ReasoningStepType.DATA_EXTRACTION,
            description="从SIMBAD提取天体分类信息",
            input_data={'simbad': data.get('simbad')},
            output_data={'otype': 'CataclyV*'},
            confidence=0.95,
            knowledge_refs=['SIMBAD']
        )
        
        # 步骤2: 周期分析
        period = data.get('periods', {}).get('TESS', {}).get('period', 0)
        tracer.add_step(
            step_type=ReasoningStepType.FEATURE_ANALYSIS,
            description=f"TESS光变曲线周期分析，发现周期 {period:.6f} 天",
            input_data={'lightcurve': 'TESS data'},
            output_data={'period': period, 'period_hours': period * 24},
            confidence=0.85,
            evidence=['PDM theta_min=0.665'],
            knowledge_refs=['period_analysis_methods']
        )
        
        # 步骤3: 知识检索
        tracer.add_step(
            step_type=ReasoningStepType.KNOWLEDGE_RETRIEVAL,
            description="检索激变变星相关知识",
            input_data={'query': 'cataclysmic variable short period'},
            output_data={'matched_knowledge': 3},
            confidence=0.90,
            knowledge_refs=['cataclysmic_variable', 'polar']
        )
        
        # 步骤4: 推理
        tracer.add_step(
            step_type=ReasoningStepType.INFERENCE,
            description="基于短周期特征推断可能为Polar型激变变星",
            input_data={'period_hours': period * 24, 'otype': 'CataclyV*'},
            output_data={'subtype': 'Polar candidate'},
            confidence=0.75,
            knowledge_refs=['polar_classification']
        )
        
        # 步骤5: 结论
        conclusion = f"目标 {data.get('name')} 是一个候选Polar型激变变星，轨道周期 {period*24:.2f} 小时"
        tracer.add_step(
            step_type=ReasoningStepType.CONCLUSION,
            description="生成最终分析结论",
            input_data={'inferences': ['Polar candidate']},
            output_data={'conclusion': conclusion},
            confidence=0.80
        )
        
        return conclusion
    
    # 测试数据
    test_data = {
        'name': 'EV_UMa',
        'simbad': {'otype': 'CataclyV*', 'main_id': 'V* EV UMa'},
        'periods': {'TESS': {'period': 0.05534}}
    }
    
    # 执行分析
    result, explanation, counterfactuals = analyzer.analyze_with_explanation(
        test_data, 'EV_UMa', mock_analysis
    )
    
    # 打印报告
    analyzer.print_full_report(explanation, counterfactuals)
    
    print("\n✅ 演示完成!")


if __name__ == "__main__":
    demo_explainable_analysis()
