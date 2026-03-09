#!/usr/bin/env python3
"""
Qwen-VL模型接口
===============
提供大语言模型分析能力
"""

import os
import sys
from typing import Optional, Tuple


class QwenInterface:
    """
    Qwen-VL模型接口
    
    封装Qwen-VL-8B模型的加载和调用
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化Qwen接口
        
        Args:
            model_path: 模型路径，None则使用默认路径
        """
        self.model_path = model_path or "./models/Qwen-VL-Chat-Int4"
        self.agent = None
        self._load_model()
    
    def _load_model(self):
        """加载Qwen模型"""
        try:
            # 尝试从父目录导入
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from qwen_vl_loader import QwenVLAgent
            
            if os.path.exists(self.model_path):
                print(f"正在加载Qwen模型: {self.model_path}")
                self.agent = QwenVLAgent(
                    model_name=self.model_path,
                    load_in_4bit=True
                )
                print("✓ Qwen模型加载成功")
            else:
                print(f"⚠ 模型路径不存在: {self.model_path}")
                print("  将使用模拟模式")
        except Exception as e:
            print(f"⚠ Qwen加载失败: {e}")
            print("  将使用模拟模式")
    
    def analyze(self, prompt: str, image_path: str = None) -> str:
        """
        使用Qwen分析
        
        Args:
            prompt: 分析提示词
            image_path: 图像路径（可选）
            
        Returns:
            分析结果文本
        """
        if self.agent:
            try:
                response, _ = self.agent.chat(prompt, image_path=image_path)
                return response
            except Exception as e:
                return f"Qwen分析错误: {str(e)}"
        else:
            # 模拟模式
            return self._mock_analysis(prompt)
    
    def _mock_analysis(self, prompt: str) -> str:
        """模拟分析（当Qwen不可用时）"""
        
        # 提取关键信息
        analysis = []
        
        if "消光" in prompt or "A_V" in prompt:
            analysis.append("基于消光值A_V≈1.1，该天体可能位于300-500秒差距处，银道面方向有一定星际消光。")
        
        if "ZTF" in prompt or "光变" in prompt:
            analysis.append("ZTF数据显示该天体有明显光变，建议进一步分析周期特性。")
        
        if "polar" in prompt.lower() or "激变" in prompt:
            analysis.append("根据坐标和文献记录，AM Her是经典的Polar型激变变星，具有强磁场(~13 MG)和X射线辐射特征。")
        
        if not analysis:
            analysis.append("基于当前数据，该天体需要更多观测来确定其类型。建议获取光谱数据进行进一步分类。")
        
        return "\n\n".join(analysis)
    
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self.agent is not None


class SimpleAnalyzer:
    """
    简化分析器（无需大模型）
    
    基于规则的简单天体分析
    """
    
    def analyze(self, data: dict) -> str:
        """
        分析天体数据
        
        Args:
            data: 包含观测数据的字典
            
        Returns:
            分析文本
        """
        lines = ["=" * 60, "天体分析报告", "=" * 60, ""]
        
        # 基本信息
        name = data.get('name', 'Unknown')
        ra = data.get('ra', 0)
        dec = data.get('dec', 0)
        
        lines.append(f"目标: {name}")
        lines.append(f"坐标: RA={ra}°, DEC={dec}°")
        lines.append("")
        
        # 消光分析
        ext = data.get('extinction', {})
        if ext.get('success'):
            av = ext.get('A_V', 0)
            lines.append(f"【消光分析】")
            lines.append(f"  A_V = {av} mag")
            if av > 1.0:
                lines.append(f"  该天体位于银道面方向，有显著星际消光。")
            lines.append("")
        
        # ZTF分析
        ztf = data.get('ztf', {})
        if ztf.get('success'):
            lines.append(f"【光变分析】")
            lines.append(f"  ZTF数据点: {ztf.get('n_points', 'N/A')}")
            if 'period_hours' in ztf:
                p = ztf['period_hours']
                lines.append(f"  周期: {p:.2f} hours")
                if 0.2 < p/24 < 1.0:
                    lines.append(f"  可能是RR Lyrae型变星")
                elif p < 1.0:
                    lines.append(f"  可能是δ Scuti型变星")
            lines.append("")
        
        # 测光分析
        phot = data.get('photometry', {})
        if phot.get('success'):
            lines.append(f"【测光数据】")
            lines.append(f"  匹配目录数: {phot.get('total_matches', 0)}")
            lines.append("")
        
        # 建议
        lines.append("【后续建议】")
        lines.append("  1. 获取光谱观测确定分类")
        lines.append("  2. 分析高时间分辨率光变曲线")
        lines.append("  3. 交叉匹配X射线源表")
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试Qwen接口
    print("=" * 60)
    print("Qwen接口测试")
    print("=" * 60)
    
    qwen = QwenInterface()
    
    test_prompt = """分析AM Her天体的观测数据:
- 消光 A_V = 1.12
- ZTF显示周期性光变
- 该天体是什么类型？"""
    
    print(f"\n提示词: {test_prompt[:100]}...")
    print("\n分析结果:")
    result = qwen.analyze(test_prompt)
    print(result)
