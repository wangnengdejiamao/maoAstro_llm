#!/usr/bin/env python3
"""
Kimi PDF 数据提取器
使用 Kimi API 从 PDF 中提取结构化信息：
1. 论文摘要和关键发现
2. 天体源参数表
3. 观测数据
4. 用于 RAG 的知识块
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from kimi_interface import KimiInterface, KimiConfig


@dataclass
class AstroSource:
    """天体源信息"""
    name: str = ""                    # 源名称
    ra: str = ""                      # 赤经
    dec: str = ""                     # 赤纬
    coordinates: str = ""             # 坐标字符串
    period: Optional[float] = None    # 周期（天）
    period_error: Optional[float] = None
    amplitude: Optional[float] = None # 振幅（星等）
    magnitude: Optional[float] = None # 视星等
    band: str = ""                    # 波段
    source_type: str = ""             # 源类型（CV, WD, DWD 等）
    subtype: str = ""                 # 子类型（AM CVn, Polar 等）
    has_outburst: bool = False        # 是否有爆发
    has_xray: bool = False            # 是否有 X 射线
    has_uv: bool = False              # 是否有 UV 数据
    has_lightcurve: bool = False      # 是否有光变曲线
    lightcurve_shape: str = ""        # 光变曲线形状描述
    sed_type: str = ""                # SED 类型
    hr_position: str = ""             # 赫罗图位置
    temperature: Optional[float] = None  # 温度（K）
    distance: Optional[float] = None     # 距离（pc）
    references: List[str] = None      # 参考文献
    notes: str = ""                   # 备注
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KnowledgeChunk:
    """RAG 知识块"""
    chunk_id: str
    source_pdf: str
    content: str
    chunk_type: str  # abstract, method, result, observation, source_data
    entities: List[str]  # 包含的实体
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class KimiPDFExtractor:
    """
    使用 Kimi API 提取 PDF 结构化信息
    """
    
    def __init__(self, api_key: Optional[str] = None):
        config = KimiConfig(api_key=api_key) if api_key else None
        self.kimi = KimiInterface(config)
        
        # 系统提示词 - 天体物理专家
        self.system_prompt_scientist = """你是天体物理学专家，专门研究白矮星、双星系统和激变变星。
请从论文中提取准确的天文数据，包括：
- 天体源的位置、周期、光度等参数
- 观测结果和发现
- 物理机制和解释

要求：
1. 使用标准天文术语
2. 保留数值和单位
3. 区分观测事实和理论推断
4. 注意误差范围"""

    def extract_paper_summary(self, pdf_text: str, max_length: int = 8000) -> Dict:
        """
        提取论文摘要和关键信息
        
        Args:
            pdf_text: PDF 文本内容（通常截断）
            max_length: 最大处理长度
            
        Returns:
            结构化摘要信息
        """
        # 截断文本
        text = pdf_text[:max_length]
        
        prompt = f"""请从以下天体物理论文中提取关键信息，以 JSON 格式返回：

论文内容：
{text}

请提取以下内容（JSON 格式）：
{{
    "title": "论文标题",
    "authors": ["作者1", "作者2"],
    "year": 2024,
    "abstract_summary": "摘要总结（3-5句话）",
    "key_findings": ["关键发现1", "关键发现2", "关键发现3"],
    "observed_sources": ["观测源1", "观测源2"],
    "methods": "观测或分析方法",
    "keywords": ["关键词1", "关键词2"],
    "category": "论文类别（white_dwarf, binary_star, cv, supernova 等）"
}}

只返回 JSON，不要有其他内容。"""
        
        response = self.kimi.generate(prompt, self.system_prompt_scientist, temperature=0.2)
        
        # 解析 JSON
        try:
            # 提取 JSON 部分
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON 解析失败: {e}")
            return {"raw_response": response}
    
    def extract_astro_sources(self, pdf_text: str) -> List[AstroSource]:
        """
        提取天体源信息
        
        Args:
            pdf_text: PDF 文本内容
            
        Returns:
            天体源列表
        """
        # 查找包含源信息的段落（通常包含 RA, Dec, Period 等）
        prompt = f"""请从以下论文中提取所有天体源的参数信息。

论文内容：
{pdf_text[:10000]}

请提取每个源的以下信息（JSON 数组格式）：
[
    {{
        "name": "源名称",
        "ra": "赤经（如 12:34:56.78）",
        "dec": "赤纬（如 +12:34:56.7）",
        "period": 0.123,  // 周期（天），如果不确定用 null
        "period_error": 0.001,
        "amplitude": 0.5,  // 振幅（星等）
        "magnitude": 15.2,
        "band": "g",  // 波段
        "source_type": "CV",  // 源类型：CV, WD, DWD, Nova 等
        "subtype": "AM CVn",  // 子类型
        "has_outburst": true,  // 是否有爆发
        "has_xray": false,  // 是否有 X 射线数据
        "has_uv": true,  // 是否有 UV 数据
        "lightcurve_shape": "sinusoidal with eclipse",  // 光变曲线形状描述
        "sed_type": "accretion disk dominated",  // SED 类型
        "hr_position": "blue, below main sequence",  // 赫罗图位置
        "temperature": 20000,  // 温度（K）
        "distance": 500,  // 距离（pc）
        "notes": "备注信息"
    }}
]

如果论文中没有具体源的数据，返回空数组 []。
只返回 JSON 数组，不要有其他内容。"""
        
        response = self.kimi.generate(prompt, self.system_prompt_scientist, temperature=0.2)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                data = [data]
            
            sources = []
            for item in data:
                try:
                    source = AstroSource(**item)
                    sources.append(source)
                except Exception as e:
                    print(f"解析源数据失败: {e}")
            
            return sources
            
        except Exception as e:
            print(f"天体源提取失败: {e}")
            return []
    
    def extract_lightcurve_info(self, text: str) -> Dict:
        """
        提取光变曲线信息
        
        Returns:
            光变曲线特征描述
        """
        prompt = f"""从以下文本中提取光变曲线的特征信息：

{text[:5000]}

请提取（JSON 格式）：
{{
    "variability_type": "变星类型（eclipsing, pulsating, eruptive 等）",
    "period_days": 0.123,
    "period_hours": 2.95,
    "amplitude_magnitude": 0.5,
    "lightcurve_shape": "光变曲线形状描述",
    "shape_classification": "形状分类（sinusoidal, double-humped, sawtooth, flat-bottomed 等）",
    "eclipse_depth": 0.3,  // 食深（星等）
    "eclipse_duration": 0.1,  // 食持续时间（周期分数）
    "outburst_recurrence": 30,  // 爆发周期（天）
    "quiescent_magnitude": 18.5,
    "outburst_magnitude": 12.0,
    "color_index": "g-r = 0.5",
    "special_features": ["特征1", "特征2"]  // 特殊特征
}}

只返回 JSON。"""
        
        response = self.kimi.generate(prompt, self.system_prompt_scientist, temperature=0.2)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            return {"raw": response}
    
    def create_knowledge_chunks(self, pdf_text: str, pdf_name: str) -> List[KnowledgeChunk]:
        """
        创建 RAG 知识块
        
        将论文分割成多个知识块，用于 RAG 检索
        """
        chunks = []
        
        # 1. 提取摘要块
        summary = self.extract_paper_summary(pdf_text)
        if summary:
            chunk = KnowledgeChunk(
                chunk_id=f"{pdf_name}_summary",
                source_pdf=pdf_name,
                content=json.dumps(summary, ensure_ascii=False),
                chunk_type="summary",
                entities=summary.get("keywords", []) + summary.get("observed_sources", [])
            )
            chunks.append(chunk)
        
        # 2. 提取天体源数据块
        sources = self.extract_astro_sources(pdf_text)
        for i, source in enumerate(sources):
            chunk = KnowledgeChunk(
                chunk_id=f"{pdf_name}_source_{i}",
                source_pdf=pdf_name,
                content=json.dumps(source.to_dict(), ensure_ascii=False),
                chunk_type="source_data",
                entities=[source.name, source.source_type, source.subtype]
            )
            chunks.append(chunk)
        
        # 3. 分块处理正文（滑动窗口）
        text_chunks = self._split_text(pdf_text, chunk_size=1500, overlap=300)
        for i, text_chunk in enumerate(text_chunks):
            # 提取实体
            entities = self._extract_entities(text_chunk)
            
            chunk = KnowledgeChunk(
                chunk_id=f"{pdf_name}_text_{i}",
                source_pdf=pdf_name,
                content=text_chunk,
                chunk_type="content",
                entities=entities
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """滑动窗口分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[str]:
        """简单实体提取（基于正则）"""
        entities = []
        
        # 源名称模式
        patterns = [
            r'\b[A-Z]{2,}\s+\d+\b',  # 如 WD 1145+017
            r'\bV\d+\s+[A-Za-z]+\b',  # 如 V404 Cyg
            r'\b[ATFGPS]\s*\d{4}[\-+]\d{2,4}\b',  # 暂现源名称
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))
    
    def generate_qa_from_paper(self, pdf_text: str, n_questions: int = 5) -> List[Dict]:
        """
        从论文生成问答对
        
        用于模型微调训练
        """
        prompt = f"""基于以下天体物理论文，生成 {n_questions} 个高质量的问答对。

论文内容：
{pdf_text[:8000]}

要求：
1. 问题要专业，不能直接从文本中找到原句
2. 答案要准确，包含定量数据
3. 涵盖论文的主要发现
4. 问题类型包括：概念解释、数据分析、物理解释

格式（JSON 数组）：
[
    {{
        "question": "问题",
        "answer": "答案",
        "type": "concept/analysis/calculation",
        "difficulty": "medium"
    }}
]

只返回 JSON 数组。"""
        
        response = self.kimi.generate(prompt, self.system_prompt_scientist, temperature=0.4)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            return []


class AstroSourceDatabase:
    """
    天体源数据库
    存储和管理提取的天体源信息
    """
    
    def __init__(self, db_path: str = "./astro_source_db.json"):
        self.db_path = Path(db_path)
        self.sources: Dict[str, AstroSource] = {}
        self.load()
    
    def load(self):
        """加载数据库"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sources = {k: AstroSource(**v) for k, v in data.items()}
            print(f"✓ 加载 {len(self.sources)} 个天体源")
        else:
            print("✓ 新建天体源数据库")
    
    def save(self):
        """保存数据库"""
        data = {k: v.to_dict() for k, v in self.sources.items()}
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存 {len(self.sources)} 个天体源")
    
    def add_source(self, source: AstroSource, merge: bool = True):
        """
        添加天体源
        
        Args:
            source: 天体源信息
            merge: 如果已存在是否合并信息
        """
        key = source.name.lower().replace(" ", "_") if source.name else f"source_{len(self.sources)}"
        
        if key in self.sources and merge:
            # 合并信息（非空字段覆盖）
            existing = self.sources[key]
            for field_name, value in source.to_dict().items():
                if value and value != "":
                    setattr(existing, field_name, value)
        else:
            self.sources[key] = source
    
    def search_by_type(self, source_type: str) -> List[AstroSource]:
        """按类型搜索"""
        return [s for s in self.sources.values() if s.source_type.lower() == source_type.lower()]
    
    def search_by_coordinate(self, ra: str, dec: str, radius: float = 1.0) -> List[AstroSource]:
        """按坐标搜索（简化版本）"""
        # 这里可以实现更复杂的坐标匹配
        results = []
        for source in self.sources.values():
            if source.ra and source.dec:
                # 简化匹配：字符串匹配
                if ra[:5] in source.ra and dec[:5] in source.dec:
                    results.append(source)
        return results
    
    def classify_source(self, source: AstroSource) -> Dict:
        """
        根据源的特征进行分类判断
        
        返回可能的分类和置信度
        """
        classification = {
            "primary_type": "unknown",
            "possible_types": [],
            "confidence": 0.0,
            "reasoning": []
        }
        
        # 基于周期的分类规则
        if source.period:
            p = source.period
            
            if p < 0.01:  # < 15 分钟
                classification["possible_types"].append(("AM CVn", 0.9))
                classification["reasoning"].append(f"周期极短 ({p*24*60:.1f} 分钟)，可能是 AM CVn")
            elif p < 0.1:  # < 2.4 小时
                classification["possible_types"].append(("极星 (Polar)", 0.7))
                classification["possible_types"].append(("中介极星 (IP)", 0.6))
                classification["reasoning"].append(f"周期短 ({p*24:.1f} 小时)，可能是磁性 CV")
            elif p < 1:  # < 1 天
                classification["possible_types"].append(("矮新星 (DN)", 0.6))
                classification["possible_types"].append(("经典新星 (CN)", 0.4))
                classification["reasoning"].append(f"周期 {p:.2f} 天，典型的 CV 周期")
            
            # 检查光变曲线形状
            if source.lightcurve_shape:
                shape = source.lightcurve_shape.lower()
                if "eclipse" in shape or "dip" in shape:
                    classification["possible_types"].append(("食双星", 0.8))
                    classification["reasoning"].append("光变曲线显示食特征")
                if "hump" in shape:
                    classification["possible_types"].append(("矮新星", 0.7))
                    classification["reasoning"].append("光变曲线显示 hump 特征")
        
        # 基于 X 射线和 UV
        if source.has_xray:
            classification["possible_types"].append(("吸积系统", 0.7))
            classification["reasoning"].append("有 X 射线发射，表明吸积过程")
        
        # 基于爆发
        if source.has_outburst:
            classification["possible_types"].append(("激变变星", 0.8))
            classification["reasoning"].append("有爆发历史")
        
        # 确定主要类型
        if classification["possible_types"]:
            classification["possible_types"].sort(key=lambda x: x[1], reverse=True)
            classification["primary_type"] = classification["possible_types"][0][0]
            classification["confidence"] = classification["possible_types"][0][1]
        
        return classification
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        stats = {
            "total_sources": len(self.sources),
            "by_type": {},
            "by_subtype": {},
            "with_period": sum(1 for s in self.sources.values() if s.period),
            "with_xray": sum(1 for s in self.sources.values() if s.has_xray),
            "with_outburst": sum(1 for s in self.sources.values() if s.has_outburst),
        }
        
        for source in self.sources.values():
            t = source.source_type or "unknown"
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
            
            st = source.subtype or "unknown"
            stats["by_subtype"][st] = stats["by_subtype"].get(st, 0) + 1
        
        return stats


def main():
    """测试"""
    print("Kimi PDF 数据提取器")
    print("=" * 60)
    
    # 初始化
    extractor = KimiPDFExtractor()
    db = AstroSourceDatabase()
    
    # 测试数据
    test_text = """
    We report the discovery of a new AM CVn system, ZTF J1901+5308, with an orbital period of 
    28.5 minutes. The object shows a double-humped light curve with an amplitude of 0.8 mag in 
    the g-band. Spectroscopy reveals helium emission lines with no hydrogen, consistent with 
    the AM CVn classification. The source has been detected in X-rays by XMM-Newton.
    
    Parameters:
    - RA: 19:01:23.45
    - Dec: +53:08:12.3
    - Period: 0.0198 days (28.5 min)
    - g-mag: 18.5
    - X-ray: detected
    """
    
    print("\n测试天体源提取...")
    sources = extractor.extract_astro_sources(test_text)
    print(f"提取到 {len(sources)} 个源")
    
    for source in sources:
        print(f"\n源名称: {source.name}")
        print(f"  坐标: {source.ra} {source.dec}")
        print(f"  周期: {source.period} 天")
        print(f"  类型: {source.source_type} - {source.subtype}")
        
        # 添加到数据库
        db.add_source(source)
    
    # 分类
    print("\n分类结果:")
    for source in sources:
        classification = db.classify_source(source)
        print(f"\n{source.name}:")
        print(f"  主要类型: {classification['primary_type']} (置信度: {classification['confidence']})")
        print(f"  推理: {classification['reasoning']}")
    
    # 保存
    db.save()


if __name__ == "__main__":
    main()
