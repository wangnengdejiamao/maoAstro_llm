#!/usr/bin/env python3
"""
高级RAG分析器 (Advanced RAG Analyzer)
=====================================
整合本地知识库 + 文献索引的智能分析系统

特性:
- 双层检索: 领域知识 + 学术文献
- 引用溯源: 可追踪的知识来源
- 自适应检索: 根据天体类型调整检索策略
- 支持50万+文献的大规模索引

作者: AstroSage AI
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class KnowledgeFragment:
    """知识片段"""
    content: str
    source: str  # 'domain_kb' 或 'literature'
    source_id: str
    relevance_score: float
    metadata: Dict = field(default_factory=dict)


class AdvancedRAGAnalyzer:
    """
    高级RAG分析器
    
    整合多种知识源，提供全面的天体分析
    """
    
    def __init__(self,
                 use_domain_kb: bool = True,
                 use_literature: bool = True,
                 literature_index_type: str = "simple"):
        """
        初始化高级RAG分析器
        
        Args:
            use_domain_kb: 是否使用领域知识库
            use_literature: 是否使用文献索引
            literature_index_type: 文献索引类型 ('simple' 或 'chroma')
        """
        self.use_domain_kb = use_domain_kb
        self.use_literature = use_literature
        self.literature_index_type = literature_index_type
        
        # 初始化知识库
        self.domain_kb = None
        self.literature_index = None
        
        if use_domain_kb:
            self._init_domain_kb()
        
        if use_literature:
            self._init_literature_index()
    
    def _init_domain_kb(self):
        """初始化领域知识库"""
        try:
            from src.enhanced_rag_system import get_enhanced_rag
            self.domain_kb = get_enhanced_rag(use_vector_store=True)
            print("✓ 领域知识库已加载")
        except Exception as e:
            print(f"⚠ 领域知识库加载失败: {e}")
    
    def _init_literature_index(self):
        """初始化文献索引"""
        try:
            from src.literature_indexer import get_literature_index
            self.literature_index = get_literature_index(
                index_type=self.literature_index_type
            )
            print(f"✓ 文献索引已加载 ({self.literature_index_type})")
        except Exception as e:
            print(f"⚠ 文献索引加载失败: {e}")
    
    def retrieve_knowledge(self, 
                          query: str,
                          object_type: Optional[str] = None,
                          top_k_domain: int = 3,
                          top_k_literature: int = 3) -> Tuple[str, List[KnowledgeFragment]]:
        """
        检索相关知识
        
        Args:
            query: 查询文本
            object_type: 天体类型 (用于优化检索)
            top_k_domain: 领域知识检索数量
            top_k_literature: 文献检索数量
            
        Returns:
            (知识文本, 知识片段列表)
        """
        fragments = []
        
        # 1. 检索领域知识
        if self.use_domain_kb and self.domain_kb:
            domain_results = self._search_domain_kb(query, object_type, top_k_domain)
            for result in domain_results:
                fragments.append(result)
        
        # 2. 检索文献
        if self.use_literature and self.literature_index:
            lit_results = self._search_literature(query, top_k_literature)
            for result in lit_results:
                fragments.append(result)
        
        # 3. 按相关度排序
        fragments.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 4. 构建知识文本
        knowledge_text = self._format_knowledge(fragments)
        
        return knowledge_text, fragments
    
    def _search_domain_kb(self, query: str, 
                          object_type: Optional[str],
                          top_k: int) -> List[KnowledgeFragment]:
        """搜索领域知识库"""
        fragments = []
        
        # 优先根据天体类型检索
        if object_type:
            result = self.domain_kb.search_by_object_type(object_type)
            if result and "未找到" not in result:
                fragments.append(KnowledgeFragment(
                    content=result,
                    source="domain_kb",
                    source_id=f"type:{object_type}",
                    relevance_score=10.0,
                    metadata={'type': 'object_type'}
                ))
        
        # 通用搜索
        results = self.domain_kb.search(query, top_k=top_k)
        if results and "未找到" not in results:
            # 解析搜索结果
            sections = results.split('【')[1:] if '【' in results else [results]
            for i, section in enumerate(sections[:top_k]):
                fragments.append(KnowledgeFragment(
                    content='【' + section if not section.startswith('【') else section,
                    source="domain_kb",
                    source_id=f"search:{i}",
                    relevance_score=5.0 - i,
                    metadata={'type': 'search'}
                ))
        
        return fragments
    
    def _search_literature(self, query: str, top_k: int) -> List[KnowledgeFragment]:
        """搜索文献索引"""
        fragments = []
        
        results = self.literature_index.search(query, top_k=top_k)
        for i, result in enumerate(results):
            if hasattr(result, 'title'):  # LiteratureEntry对象
                content = f"""标题: {result.title}
作者: {', '.join(result.authors[:3])}
期刊: {result.journal} ({result.year})
摘要: {result.abstract[:300]}..."""
                source_id = result.bibcode or result.id
                metadata = {
                    'year': result.year,
                    'journal': result.journal,
                    'citations': result.citations
                }
            else:  # ChromaDB结果 (字典)
                meta = result.get('metadata', {})
                content = f"""标题: {meta.get('title', 'N/A')}
作者: {meta.get('authors', 'N/A')}
期刊: {meta.get('journal', 'N/A')} ({meta.get('year', 'N/A')})
摘要: {result.get('document', '')[:300]}..."""
                source_id = result.get('id', 'unknown')
                metadata = {
                    'year': meta.get('year'),
                    'journal': meta.get('journal'),
                    'citations': meta.get('citations')
                }
            
            fragments.append(KnowledgeFragment(
                content=content,
                source="literature",
                source_id=source_id,
                relevance_score=3.0 - i * 0.5,
                metadata=metadata
            ))
        
        return fragments
    
    def _format_knowledge(self, fragments: List[KnowledgeFragment]) -> str:
        """格式化知识文本"""
        if not fragments:
            return ""
        
        sections = []
        
        # 分组
        domain_fragments = [f for f in fragments if f.source == "domain_kb"]
        lit_fragments = [f for f in fragments if f.source == "literature"]
        
        # 领域知识部分
        if domain_fragments:
            sections.append("## 领域专业知识")
            for i, f in enumerate(domain_fragments[:3], 1):
                sections.append(f"\n### 知识片段 {i} (来源: {f.source_id})\n{f.content}")
        
        # 文献部分
        if lit_fragments:
            sections.append("\n## 相关学术文献")
            for i, f in enumerate(lit_fragments[:3], 1):
                sections.append(f"\n### 文献 {i} [{f.source_id}]\n{f.content}")
        
        return "\n".join(sections)
    
    def analyze_with_rag(self,
                        target_data: Dict[str, Any],
                        target_name: str = None) -> Dict[str, Any]:
        """
        使用RAG增强进行分析
        
        Args:
            target_data: 观测数据
            target_name: 目标名称
            
        Returns:
            包含分析和知识来源的字典
        """
        name = target_name or target_data.get('name', 'Unknown')
        
        print(f"\n{'='*70}")
        print(f"🔬 高级RAG分析: {name}")
        print(f"{'='*70}")
        
        # 提取关键信息
        key_info = self._extract_analysis_info(target_data)
        
        # 构建查询
        queries = self._build_queries(key_info)
        
        # 检索知识
        print("\n📚 检索相关知识...")
        all_fragments = []
        all_knowledge = []
        
        for query in queries:
            print(f"   查询: {query[:60]}...")
            knowledge, fragments = self.retrieve_knowledge(
                query,
                object_type=key_info.get('object_type'),
                top_k_domain=2,
                top_k_literature=2
            )
            if knowledge:
                all_knowledge.append(knowledge)
                all_fragments.extend(fragments)
        
        # 去重
        seen_sources = set()
        unique_fragments = []
        for f in all_fragments:
            if f.source_id not in seen_sources:
                seen_sources.add(f.source_id)
                unique_fragments.append(f)
        
        combined_knowledge = "\n\n".join(all_knowledge)
        
        print(f"   ✓ 检索到 {len(unique_fragments)} 个独特知识片段")
        print(f"      - 领域知识: {len([f for f in unique_fragments if f.source == 'domain_kb'])}")
        print(f"      - 学术文献: {len([f for f in unique_fragments if f.source == 'literature'])}")
        
        return {
            'target_name': name,
            'key_info': key_info,
            'knowledge': combined_knowledge,
            'fragments': unique_fragments,
            'knowledge_sources': [f.source_id for f in unique_fragments]
        }
    
    def _extract_analysis_info(self, data: Dict) -> Dict:
        """提取分析所需的关键信息"""
        info = {
            'name': data.get('name', 'Unknown'),
            'ra': data.get('ra'),
            'dec': data.get('dec'),
            'object_type': None,
            'period': None,
            'period_hours': None,
            'magnitude': None,
            'distance': None,
            'extinction': None
        }
        
        # SIMBAD信息
        simbad = data.get('simbad', {})
        if simbad:
            info['object_type'] = simbad.get('otype')
            info['distance'] = simbad.get('distance')
            info['simbad_id'] = simbad.get('main_id')
        
        # 周期
        periods = data.get('periods', {})
        if periods:
            for source, p_data in periods.items():
                if isinstance(p_data, dict) and 'period' in p_data:
                    info['period'] = p_data['period']
                    info['period_hours'] = p_data['period'] * 24
                    break
        
        # 消光
        extinction = data.get('extinction', {})
        if extinction and extinction.get('success'):
            info['extinction'] = extinction.get('A_V')
        
        return info
    
    def _build_queries(self, info: Dict) -> List[str]:
        """构建检索查询"""
        queries = []
        
        obj_type = info.get('object_type', '')
        period_hours = info.get('period_hours')
        
        # 基于天体类型的查询
        if obj_type:
            queries.append(f"{obj_type} physical properties")
            queries.append(f"{obj_type} classification")
        
        # 基于周期的查询
        if period_hours:
            if period_hours < 0.1:
                queries.append("ultra-compact binary AM CVn period")
            elif period_hours < 2:
                queries.append("short period cataclysmic variable")
            elif period_hours < 12:
                queries.append("cataclysmic variable orbital period")
        
        # 通用查询
        queries.append("cataclysmic variable accretion physics")
        
        return queries
    
    def generate_enhanced_prompt(self, rag_result: Dict) -> str:
        """
        生成增强版分析提示词
        
        Args:
            rag_result: RAG检索结果
            
        Returns:
            完整的提示词文本
        """
        info = rag_result['key_info']
        knowledge = rag_result['knowledge']
        
        prompt = f"""你是一位资深天体物理学家。请基于以下观测数据和背景知识进行全面分析。

## 目标基本信息
- 名称: {info['name']}
- 坐标: RA={info['ra']}, DEC={info['dec']}
"""
        
        # 添加物理参数
        if info.get('object_type'):
            prompt += f"- SIMBAD分类: {info['object_type']}\n"
        if info.get('period_hours'):
            prompt += f"- 轨道周期: {info['period_hours']:.4f} 小时\n"
        if info.get('distance'):
            prompt += f"- 距离: {info['distance']:.1f} pc\n"
        if info.get('extinction'):
            prompt += f"- 消光: A_V = {info['extinction']:.3f}\n"
        
        # 添加知识背景
        prompt += f"""
## 背景知识 (来自专业领域知识库和学术文献)
{knowledge[:8000] if len(knowledge) > 8000 else knowledge}
"""
        
        # 添加分析要求
        prompt += """
## 分析要求

请提供以下分析 (用中文):

### 1. 天体类型判断与分类依据
- 基于SIMBAD分类和周期特征
- 与其他相似类型的区分要点
- 分类置信度评估

### 2. 物理参数解读
- 周期/距离的物理意义
- 参数的可靠性和误差来源
- 与同类天体的参数对比

### 3. 形成与演化
- 可能的形成通道
- 演化状态和最终命运
- 与其他系统的关联

### 4. 科学价值与研究前景
- 在当前天体物理中的重要性
- 可以回答的关键科学问题
- 推荐的后续研究方向

### 5. 需要特别注意的问题
- 常见误解和错误
- 数据解释的注意事项
- 建议的验证观测

请确保分析基于提供的数据和知识，避免臆测。引用背景知识时请注明来源。
"""
        
        return prompt


# ==================== 便捷函数 ====================

def create_advanced_analyzer(use_literature: bool = True) -> AdvancedRAGAnalyzer:
    """创建高级RAG分析器"""
    return AdvancedRAGAnalyzer(
        use_domain_kb=True,
        use_literature=use_literature,
        literature_index_type="simple"
    )


def quick_rag_analysis(target_data: Dict, target_name: str = None) -> Dict:
    """快速RAG分析"""
    analyzer = create_advanced_analyzer(use_literature=False)
    return analyzer.analyze_with_rag(target_data, target_name)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("高级RAG分析器测试")
    print("=" * 70)
    
    # 创建测试数据
    test_data = {
        'name': 'EV_UMa',
        'ra': 196.9744,
        'dec': 53.8585,
        'simbad': {
            'matched': True,
            'main_id': 'V* EV UMa',
            'otype': 'CataclyV*',
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
        }
    }
    
    # 创建分析器
    print("\n初始化高级RAG分析器...")
    analyzer = create_advanced_analyzer(use_literature=False)
    
    # 执行分析
    print("\n执行RAG增强分析...")
    result = analyzer.analyze_with_rag(test_data)
    
    print("\n" + "=" * 70)
    print("生成的增强提示词 (前2000字符):")
    print("=" * 70)
    prompt = analyzer.generate_enhanced_prompt(result)
    print(prompt[:2000])
    print("\n...")
    
    print("\n测试完成!")
