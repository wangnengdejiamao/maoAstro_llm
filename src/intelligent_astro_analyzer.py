#!/usr/bin/env python3
"""
智能天文分析器 (Intelligent Astro Analyzer)
===========================================
整合增强版RAG知识库 + AI大模型的智能天体分析系统

特性:
- 自动检索相关天文知识
- 多模态分析 (文本 + 图像)
- 结构化输出
- 科学级分析质量

作者: AstroSage AI
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """分析结果数据结构"""
    target_name: str
    object_classification: str
    physical_properties: Dict[str, Any]
    scientific_significance: str
    follow_up_recommendations: List[str]
    raw_analysis: str
    knowledge_references: List[str]


class IntelligentAstroAnalyzer:
    """
    智能天文分析器
    
    整合RAG知识检索和AI大模型，提供专业级天体分析
    """
    
    def __init__(self, 
                 ollama_model: str = "astrosage-local:latest",
                 use_rag: bool = True,
                 use_vision: bool = True):
        """
        初始化智能分析器
        
        Args:
            ollama_model: Ollama模型名称
            use_rag: 是否使用RAG知识检索
            use_vision: 是否使用视觉分析
        """
        self.ollama_model = ollama_model
        self.use_rag = use_rag
        self.use_vision = use_vision
        
        # 初始化RAG系统
        if use_rag:
            try:
                from src.enhanced_rag_system import get_enhanced_rag
                self.rag = get_enhanced_rag(use_vector_store=True)
                print("✓ 增强版RAG系统已加载")
            except Exception as e:
                print(f"⚠ RAG系统加载失败: {e}")
                self.rag = None
        else:
            self.rag = None
        
        # 初始化Ollama接口
        try:
            from src.ollama_qwen_interface import OllamaQwenInterface
            self.ollama = OllamaQwenInterface(model_name=ollama_model)
            print(f"✓ Ollama接口已初始化 (模型: {ollama_model})")
        except Exception as e:
            print(f"✗ Ollama接口初始化失败: {e}")
            self.ollama = None
    
    def analyze_target(self, 
                       target_data: Dict[str, Any],
                       target_name: str = None) -> AnalysisResult:
        """
        执行完整的天体智能分析
        
        Args:
            target_data: 包含所有观测数据的字典
            target_name: 目标名称
            
        Returns:
            AnalysisResult 分析结果对象
        """
        name = target_name or target_data.get('name', 'Unknown')
        print(f"\n{'='*70}")
        print(f"🔬 智能分析: {name}")
        print(f"{'='*70}")
        
        # 步骤1: 提取关键信息
        print("\n📋 步骤1: 提取关键信息...")
        key_info = self._extract_key_info(target_data)
        self._print_key_info(key_info)
        
        # 步骤2: RAG知识检索
        knowledge_context = ""
        knowledge_refs = []
        if self.use_rag and self.rag:
            print("\n📚 步骤2: 检索相关知识...")
            knowledge_context, knowledge_refs = self._retrieve_knowledge(key_info)
            if knowledge_context:
                print(f"   ✓ 检索到 {len(knowledge_refs)} 个相关知识条目")
            else:
                print("   ℹ 未检索到特定知识")
        
        # 步骤3: 构建增强提示词
        print("\n📝 步骤3: 构建分析提示词...")
        prompt = self._build_enhanced_prompt(key_info, knowledge_context)
        
        # 步骤4: AI分析
        print("\n🤖 步骤4: 执行AI分析...")
        analysis_text = self._execute_ai_analysis(prompt, key_info)
        
        # 步骤5: 解析结构化结果
        print("\n🔍 步骤5: 解析分析结果...")
        result = self._parse_analysis_result(
            analysis_text, name, key_info, knowledge_refs
        )
        
        # 步骤6: 打印最终报告
        print("\n" + "="*70)
        print(f"📊 {name} - 智能分析报告")
        print("="*70)
        self._print_analysis_report(result)
        
        return result
    
    def _extract_key_info(self, data: Dict) -> Dict[str, Any]:
        """提取关键观测信息"""
        info = {
            'name': data.get('name', 'Unknown'),
            'ra': data.get('ra'),
            'dec': data.get('dec'),
            'object_type': None,
            'period': None,
            'magnitude': {},
            'distance': None,
            'extinction': {},
            'spectra': [],
            'sed_points': 0,
            'lightcurves': [],
            'figures': {}
        }
        
        # SIMBAD信息
        simbad = data.get('simbad', {})
        if simbad and simbad.get('matched'):
            info['simbad_id'] = simbad.get('main_id')
            info['object_type'] = simbad.get('otype')
            info['sp_type'] = simbad.get('sp_type', 'N/A')
            info['distance'] = simbad.get('distance')
            info['parallax'] = simbad.get('parallax')
            info['pm_ra'] = simbad.get('pm_ra')
            info['pm_dec'] = simbad.get('pm_dec')
        
        # Gaia信息
        gaia = data.get('gaia', {})
        if gaia and gaia.get('success'):
            gaia_data = gaia.get('data')
            if gaia_data and hasattr(gaia_data, 'phot_g_mean_mag'):
                info['magnitude']['G'] = float(gaia_data['phot_g_mean_mag'])
                if hasattr(gaia_data, 'parallax') and gaia_data['parallax']:
                    info['distance'] = 1000.0 / float(gaia_data['parallax'])
        
        # 消光信息
        extinction = data.get('extinction', {})
        if extinction and extinction.get('success'):
            info['extinction'] = {
                'A_V': extinction.get('A_V'),
                'E_B_V': extinction.get('E_B_V')
            }
        
        # 周期信息
        periods = data.get('periods', {})
        if periods:
            for source, p_data in periods.items():
                if isinstance(p_data, dict) and 'period' in p_data:
                    info['period'] = p_data['period']
                    info['period_source'] = source
                    break
        
        # 光谱信息
        spectrum = data.get('spectrum', {})
        if spectrum and spectrum.get('success'):
            info['spectra'] = spectrum.get('spectra', [])
        
        # SED信息
        sed = data.get('sed', {})
        if sed and sed.get('success'):
            info['sed_points'] = sed.get('n_points', 0)
        
        # 光变曲线
        lightcurves = data.get('lightcurves', {})
        info['lightcurves'] = list(lightcurves.keys())
        
        # 图表路径
        info['figures'] = data.get('figures', {})
        
        return info
    
    def _print_key_info(self, info: Dict):
        """打印关键信息摘要"""
        print(f"   名称: {info['name']}")
        print(f"   坐标: RA={info['ra']}, DEC={info['dec']}")
        if info.get('object_type'):
            print(f"   SIMBAD类型: {info['object_type']}")
        if info.get('period'):
            print(f"   光变周期: {info['period']:.6f} 天 ({info['period']*24:.4f} 小时)")
        if info.get('distance'):
            print(f"   距离: {info['distance']:.1f} pc")
        if info['extinction'].get('A_V'):
            print(f"   消光: A_V = {info['extinction']['A_V']:.3f}")
        print(f"   可用数据: {', '.join(info['lightcurves'])}")
    
    def _retrieve_knowledge(self, info: Dict) -> tuple:
        """检索相关知识"""
        contexts = []
        refs = []
        
        # 根据天体类型检索
        obj_type = info.get('object_type', '')
        if obj_type:
            result = self.rag.search_by_object_type(obj_type)
            if result and "未找到" not in result:
                contexts.append(f"【天体类型: {obj_type}】\n{result}")
                refs.append(f"天体类型: {obj_type}")
        
        # 根据周期特征检索
        period = info.get('period')
        if period:
            if period < 0.1:  # 短周期
                result = self.rag.search("ultra-compact binary period", top_k=2)
                if result and "未找到" not in result:
                    contexts.append(result)
                    refs.append("短周期双星")
            result = self.rag.search("period analysis methods", top_k=2)
            if result and "未找到" not in result:
                contexts.append(result)
                refs.append("周期分析方法")
        
        # 检索CV相关知识
        if obj_type and 'cataclysmic' in obj_type.lower():
            result = self.rag.search("cataclysmic variable", top_k=3)
            if result and "未找到" not in result:
                contexts.append(result)
                refs.append("激变变星")
        
        # 检索消光相关知识
        if info.get('extinction', {}).get('A_V'):
            result = self.rag.search("interstellar extinction", top_k=1)
            if result and "未找到" not in result:
                contexts.append(result)
                refs.append("星际消光")
        
        return "\n\n".join(contexts), refs
    
    def _build_enhanced_prompt(self, info: Dict, knowledge_context: str) -> str:
        """构建增强版分析提示词"""
        
        prompt = f"""你是一位资深天体物理学家。请基于以下观测数据和背景知识，对天体进行全面分析。

## 目标基本信息
- 名称: {info['name']}
- 坐标: RA={info['ra']}, DEC={info['dec']}
"""
        
        # 添加SIMBAD信息
        if info.get('simbad_id'):
            prompt += f"""
## SIMBAD数据库信息
- 主标识符: {info['simbad_id']}
- 天体类型: {info.get('object_type', 'N/A')}
- 光谱型: {info.get('sp_type', 'N/A')}
"""
        
        # 添加物理参数
        prompt += "\n## 物理参数\n"
        if info.get('distance'):
            prompt += f"- 距离: {info['distance']:.1f} pc\n"
        if info.get('parallax'):
            prompt += f"- 视差: {info['parallax']:.3f} mas\n"
        if info.get('period'):
            period_days = info['period']
            period_hours = period_days * 24
            prompt += f"- 光变周期: {period_days:.6f} 天 = {period_hours:.4f} 小时\n"
        if info['extinction'].get('A_V'):
            prompt += f"- 消光: A_V = {info['extinction']['A_V']:.3f}, E(B-V) = {info['extinction'].get('E_B_V', 'N/A')}\n"
        if info.get('pm_ra') and info.get('pm_dec'):
            prompt += f"- 自行: μ_ra = {info['pm_ra']:.2f}, μ_dec = {info['pm_dec']:.2f} mas/yr\n"
        
        # 添加测光信息
        if info['magnitude']:
            prompt += "\n## 测光数据\n"
            for band, mag in info['magnitude'].items():
                prompt += f"- {band} = {mag:.2f}\n"
        
        # 添加观测数据摘要
        prompt += f"""
## 可用观测数据
- 光谱: {len(info['spectra'])} 个
- SED数据点: {info['sed_points']} 个
- 光变曲线: {', '.join(info['lightcurves']) if info['lightcurves'] else '无'}
"""
        
        # 添加RAG检索到的知识
        if knowledge_context:
            prompt += f"""
## 背景知识 (来自天文知识库)
{knowledge_context}
"""
        
        # 添加分析要求
        prompt += """
## 分析要求

请提供以下分析 (用中文回答):

### 1. 天体类型判断
- 基于SIMBAD分类和观测特征，判断天体类型
- 说明分类依据和置信度
- 如果是双星系统，说明双星类型和物理机制

### 2. 物理参数分析
- 距离、消光、光度等参数的可靠性评估
- 周期特征分析 (如果是周期性变星)
- 与其他已知天体的参数对比

### 3. 科学价值和研究意义
- 该天体在天体物理中的重要性
- 可能的研究方向
- 对理解相关物理过程的贡献

### 4. 后续观测建议
- 推荐的观测设备和波段
- 优先观测项目
- 与现有观测的协同

### 5. 特别注意 (关键!)
- 基于背景知识，指出可能的常见错误理解
- 需要谨慎解释的观测特征
- 建议的验证方法

请用专业但清晰的术语回答，确保分析基于观测数据而非臆测。
"""
        
        return prompt
    
    def _execute_ai_analysis(self, prompt: str, info: Dict) -> str:
        """执行AI分析"""
        if not self.ollama:
            return "[AI分析不可用: Ollama接口未初始化]"
        
        system_prompt = """你是一位资深天体物理学家，擅长变星、双星系统和致密天体研究。

分析原则:
1. 基于数据说话，不编造信息
2. 注意区分相似天体类型的差异
3. 对不确定的结论给出置信度评估
4. 引用背景知识时确保准确性
5. 特别注意避免常见错误 (如混淆AM CVn和中子星系统)
"""
        
        try:
            print("   正在调用AI模型...")
            response = self.ollama.analyze_text(
                prompt, 
                system_prompt=system_prompt,
                max_retries=2
            )
            
            if response and not response.startswith('['):
                print("   ✓ AI分析完成")
                return response
            else:
                print(f"   ⚠ AI返回异常: {response[:100] if response else '空响应'}")
                return response or "[AI分析失败]"
                
        except Exception as e:
            print(f"   ✗ AI分析异常: {e}")
            return f"[AI分析异常: {str(e)}]"
    
    def _parse_analysis_result(self, 
                               analysis_text: str,
                               name: str,
                               info: Dict,
                               refs: List[str]) -> AnalysisResult:
        """解析AI分析结果为结构化数据"""
        
        # 提取天体分类
        classification = info.get('object_type', 'Unknown')
        
        # 物理参数字典
        physical_props = {
            'distance_pc': info.get('distance'),
            'period_days': info.get('period'),
            'period_hours': info.get('period', 0) * 24 if info.get('period') else None,
            'extinction_av': info['extinction'].get('A_V'),
            'magnitudes': info['magnitude'],
            'coordinates': {'ra': info['ra'], 'dec': info['dec']}
        }
        
        # 提取科学意义部分
        significance = ""
        if "科学价值" in analysis_text or "研究意义" in analysis_text:
            lines = analysis_text.split('\n')
            in_section = False
            for line in lines:
                if '科学价值' in line or '研究意义' in line:
                    in_section = True
                    continue
                if in_section:
                    if line.strip() and (line.startswith('###') or line.startswith('##')):
                        break
                    significance += line + '\n'
        
        if not significance:
            significance = "详见完整分析报告"
        
        # 提取后续建议
        recommendations = []
        if "观测建议" in analysis_text or "后续" in analysis_text:
            lines = analysis_text.split('\n')
            in_section = False
            for line in lines:
                if '观测建议' in line or '后续' in line:
                    in_section = True
                    continue
                if in_section:
                    if line.strip() and (line.startswith('###') or line.startswith('##')):
                        break
                    if line.strip().startswith('-') or line.strip().startswith('*'):
                        recommendations.append(line.strip()[1:].strip())
        
        return AnalysisResult(
            target_name=name,
            object_classification=classification,
            physical_properties=physical_props,
            scientific_significance=significance.strip(),
            follow_up_recommendations=recommendations,
            raw_analysis=analysis_text,
            knowledge_references=refs
        )
    
    def _print_analysis_report(self, result: AnalysisResult):
        """打印分析报告到控制台"""
        
        print("\n" + "-"*70)
        print("🌟 天体类型判断")
        print("-"*70)
        print(f"SIMBAD分类: {result.object_classification}")
        print(f"知识库参考: {', '.join(result.knowledge_references) if result.knowledge_references else '无'}")
        
        print("\n" + "-"*70)
        print("📐 物理参数摘要")
        print("-"*70)
        props = result.physical_properties
        if props.get('distance_pc'):
            print(f"• 距离: {props['distance_pc']:.1f} pc")
        if props.get('period_hours'):
            print(f"• 周期: {props['period_hours']:.4f} 小时 ({props['period_days']:.6f} 天)")
        if props.get('extinction_av'):
            print(f"• 消光: A_V = {props['extinction_av']:.3f}")
        if props.get('magnitudes'):
            mags = props['magnitudes']
            if 'G' in mags:
                print(f"• Gaia G = {mags['G']:.2f}")
        
        print("\n" + "-"*70)
        print("🔬 科学意义")
        print("-"*70)
        print(result.scientific_significance[:500] + "..." if len(result.scientific_significance) > 500 else result.scientific_significance)
        
        print("\n" + "-"*70)
        print("📡 后续观测建议")
        print("-"*70)
        if result.follow_up_recommendations:
            for i, rec in enumerate(result.follow_up_recommendations[:5], 1):
                print(f"{i}. {rec}")
        else:
            print("详见完整分析报告")
        
        print("\n" + "-"*70)
        print("📄 完整AI分析")
        print("-"*70)
        print(result.raw_analysis)
        print("="*70)


# ==================== 便捷函数 ====================

def analyze_with_ai(target_data: Dict, 
                    model: str = "astrosage-local:latest",
                    use_rag: bool = True) -> AnalysisResult:
    """
    使用AI分析天体数据 (便捷函数)
    
    Args:
        target_data: 观测数据字典
        model: Ollama模型名称
        use_rag: 是否使用RAG
        
    Returns:
        AnalysisResult
    """
    analyzer = IntelligentAstroAnalyzer(
        ollama_model=model,
        use_rag=use_rag
    )
    return analyzer.analyze_target(target_data)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("智能天文分析器测试")
    print("=" * 70)
    
    # 测试数据
    test_data = {
        'name': 'Test_Target',
        'ra': 196.9744,
        'dec': 53.8585,
        'simbad': {
            'matched': True,
            'main_id': 'V* EV UMa',
            'otype': 'CataclyV*',
            'sp_type': 'N/A',
            'distance': 656.4,
            'parallax': 1.5234
        },
        'periods': {
            'TESS': {'period': 0.05534, 'theta_min': 0.665}
        },
        'extinction': {
            'success': True,
            'A_V': 0.060,
            'E_B_V': 0.020
        },
        'lightcurves': {'TESS': {'n_points': 13683}},
        'figures': {}
    }
    
    # 执行分析
    analyzer = IntelligentAstroAnalyzer(use_rag=True)
    result = analyzer.analyze_target(test_data)
    
    print("\n测试完成!")
