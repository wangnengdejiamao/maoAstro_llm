#!/usr/bin/env python3
"""
实时知识更新系统 (Real-time Knowledge Update System)
====================================================
从新发表论文中自动提取知识，动态更新知识库

核心功能:
- 监控arXiv/ADS新论文
- 自动提取关键知识
- 增量更新向量索引
- 知识冲突检测与融合

作者: AstroSage AI
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import schedule


@dataclass
class KnowledgeExtracted:
    """提取的知识条目"""
    source: str  # 来源文献ID
    source_title: str
    knowledge_type: str  # 'fact', 'relation', 'parameter', 'method'
    content: str
    confidence: float  # 置信度 0-1
    entities: List[str]  # 相关实体
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeExtracted':
        return cls(**data)


class ArXivMonitor:
    """
    arXiv论文监控器
    
    自动获取天体物理领域新论文
    """
    
    def __init__(self, categories: List[str] = None):
        self.categories = categories or ['astro-ph.SR', 'astro-ph.HE', 'astro-ph.GA']
        self.last_check = None
        self.seen_papers: Set[str] = set()
        
        # 加载已处理记录
        self._load_state()
    
    def _load_state(self):
        """加载监控状态"""
        state_file = Path("./cache/arxiv_monitor_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.last_check = state.get('last_check')
                self.seen_papers = set(state.get('seen_papers', []))
    
    def _save_state(self):
        """保存监控状态"""
        state = {
            'last_check': datetime.now().isoformat(),
            'seen_papers': list(self.seen_papers)
        }
        Path("./cache").mkdir(exist_ok=True)
        with open("./cache/arxiv_monitor_state.json", 'w') as f:
            json.dump(state, f)
    
    def fetch_new_papers(self, max_results: int = 50) -> List[Dict]:
        """
        获取新论文
        
        Args:
            max_results: 最大返回数量
            
        Returns:
            论文列表
        """
        try:
            import feedparser
            import urllib.parse
            
            # 构建查询
            category_query = ' OR '.join([f'cat:{c}' for c in self.categories])
            
            # 如果有上次检查时间，只获取新论文
            if self.last_check:
                # 解析时间并添加条件
                pass
            
            base_url = 'http://export.arxiv.org/api/query'
            query_params = {
                'search_query': category_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
            
            print(f"🌐 查询arXiv: {url[:80]}...")
            feed = feedparser.parse(url)
            
            papers = []
            for entry in feed.entries:
                paper_id = entry.id.split('/abs/')[-1]
                
                # 跳过已处理的论文
                if paper_id in self.seen_papers:
                    continue
                
                paper = {
                    'id': paper_id,
                    'title': entry.title,
                    'authors': [a.name for a in entry.authors],
                    'abstract': entry.summary,
                    'published': entry.published,
                    'categories': [t.term for t in entry.tags],
                    'pdf_url': entry.links[1].href if len(entry.links) > 1 else None
                }
                papers.append(paper)
                self.seen_papers.add(paper_id)
            
            self._save_state()
            return papers
            
        except Exception as e:
            print(f"⚠ arXiv获取失败: {e}")
            return []


class KnowledgeExtractor:
    """
    知识提取器
    
    从论文中提取结构化知识
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.extraction_patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict:
        """初始化提取模式"""
        import re
        
        patterns = {
            # 物理参数模式
            'period': re.compile(r'period[s]?\s+of\s+([\d.]+)\s*(hour|hr|min|day)', re.I),
            'mass': re.compile(r'mass\s+of\s+([\d.]+)\s*M_?\s*sun', re.I),
            'distance': re.compile(r'distance\s+of\s+([\d.]+)\s*(pc|kpc|Mpc)', re.I),
            
            # 天体分类模式
            'object_type': re.compile(r'(cataclysmic variable|white dwarf|neutron star|black hole|AM CVn)', re.I),
            
            # 关系模式
            'correlation': re.compile(r'(correlat|relat|associat)\s+with', re.I),
        }
        
        return patterns
    
    def extract_from_paper(self, paper: Dict) -> List[KnowledgeExtracted]:
        """
        从单篇论文提取知识
        
        Args:
            paper: 论文字典
            
        Returns:
            知识条目列表
        """
        knowledge_list = []
        
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        
        # 1. 规则提取
        rule_based = self._extract_with_rules(title, abstract, paper)
        knowledge_list.extend(rule_based)
        
        # 2. LLM提取 (如果启用)
        if self.use_llm:
            llm_based = self._extract_with_llm(title, abstract, paper)
            knowledge_list.extend(llm_based)
        
        return knowledge_list
    
    def _extract_with_rules(self, title: str, abstract: str, 
                           paper: Dict) -> List[KnowledgeExtracted]:
        """基于规则的知识提取"""
        knowledge_list = []
        text = title + " " + abstract
        
        # 提取周期信息
        period_matches = self.extraction_patterns['period'].findall(text)
        for value, unit in period_matches:
            knowledge_list.append(KnowledgeExtracted(
                source=paper['id'],
                source_title=title,
                knowledge_type='parameter',
                content=f"Object has a period of {value} {unit}",
                confidence=0.7,
                entities=self._extract_entities(text),
                timestamp=datetime.now().isoformat()
            ))
        
        # 提取天体类型
        type_matches = self.extraction_patterns['object_type'].findall(text)
        for obj_type in set(type_matches):
            knowledge_list.append(KnowledgeExtracted(
                source=paper['id'],
                source_title=title,
                knowledge_type='fact',
                content=f"Object is classified as {obj_type}",
                confidence=0.8,
                entities=self._extract_entities(text),
                timestamp=datetime.now().isoformat()
            ))
        
        return knowledge_list
    
    def _extract_with_llm(self, title: str, abstract: str, 
                         paper: Dict) -> List[KnowledgeExtracted]:
        """使用LLM提取知识"""
        knowledge_list = []
        
        try:
            from src.ollama_qwen_interface import OllamaQwenInterface
            
            ollama = OllamaQwenInterface(model_name="qwen3:8b")
            
            prompt = f"""从以下天文论文摘要中提取关键知识点。

标题: {title}
摘要: {abstract}

请以JSON格式提取:
1. 新发现的事实
2. 测量的物理参数
3. 天体分类信息
4. 方法创新

格式:
{{
  "facts": ["事实1", "事实2"],
  "parameters": [{{"name": "周期", "value": "1.2", "unit": "小时"}}],
  "classifications": ["激变变星", "白矮星"],
  "methods": ["新方法1"]
}}

只返回JSON，不要有其他文字。"""
            
            response = ollama.analyze_text(prompt, max_retries=1)
            
            # 解析JSON
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    for fact in data.get('facts', []):
                        knowledge_list.append(KnowledgeExtracted(
                            source=paper['id'],
                            source_title=title,
                            knowledge_type='fact',
                            content=fact,
                            confidence=0.75,
                            entities=self._extract_entities(text),
                            timestamp=datetime.now().isoformat()
                        ))
                    
                    for param in data.get('parameters', []):
                        knowledge_list.append(KnowledgeExtracted(
                            source=paper['id'],
                            source_title=title,
                            knowledge_type='parameter',
                            content=f"{param.get('name')}: {param.get('value')} {param.get('unit', '')}",
                            confidence=0.8,
                            entities=self._extract_entities(text),
                            timestamp=datetime.now().isoformat()
                        ))
            except:
                pass
                
        except Exception as e:
            print(f"LLM提取失败: {e}")
        
        return knowledge_list
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取命名实体"""
        # 简化版实体提取
        import re
        
        # 匹配天体名称 (如: EV UMa, AM Her, SS Cyg)
        pattern = r'\b[A-Z]{1,2}\s+[A-Za-z]{1,3}\b'
        entities = re.findall(pattern, text)
        
        return list(set(entities))[:10]  # 去重并限制数量


class KnowledgeConflictResolver:
    """
    知识冲突解决器
    
    检测并解决知识库中的冲突
    """
    
    def __init__(self):
        self.similarity_threshold = 0.85
    
    def detect_conflicts(self, new_knowledge: KnowledgeExtracted,
                        existing_kb: List[KnowledgeExtracted]) -> List[KnowledgeExtracted]:
        """
        检测与现有知识的冲突
        
        Returns:
            冲突的知识条目列表
        """
        conflicts = []
        
        for existing in existing_kb:
            # 计算相似度
            sim = self._knowledge_similarity(new_knowledge, existing)
            
            if sim > self.similarity_threshold:
                # 检查是否矛盾
                if self._is_contradictory(new_knowledge, existing):
                    conflicts.append(existing)
        
        return conflicts
    
    def _knowledge_similarity(self, k1: KnowledgeExtracted, 
                              k2: KnowledgeExtracted) -> float:
        """计算两条知识的相似度"""
        # 基于实体重叠和内容相似度
        entity_overlap = len(set(k1.entities) & set(k2.entities))
        entity_sim = entity_overlap / max(len(k1.entities), len(k2.entities), 1)
        
        # 内容相似度 (简化版)
        content_sim = self._text_similarity(k1.content, k2.content)
        
        return 0.6 * entity_sim + 0.4 * content_sim
    
    def _text_similarity(self, t1: str, t2: str) -> float:
        """计算文本相似度"""
        words1 = set(t1.lower().split())
        words2 = set(t2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0
    
    def _is_contradictory(self, k1: KnowledgeExtracted, 
                          k2: KnowledgeExtracted) -> bool:
        """判断两条知识是否矛盾"""
        # 简化判断：如果是同一类型但数值不同
        if k1.knowledge_type == k2.knowledge_type == 'parameter':
            # 提取数值比较
            import re
            nums1 = re.findall(r'\d+\.?\d*', k1.content)
            nums2 = re.findall(r'\d+\.?\d*', k2.content)
            
            if nums1 and nums2:
                v1, v2 = float(nums1[0]), float(nums2[0])
                # 如果数值差异大于20%，认为是矛盾
                if abs(v1 - v2) / max(v1, v2, 1) > 0.2:
                    return True
        
        return False
    
    def resolve(self, new_knowledge: KnowledgeExtracted,
               conflicts: List[KnowledgeExtracted]) -> KnowledgeExtracted:
        """
        解决冲突
        
        策略:
        1. 优先使用高置信度知识
        2. 优先使用新论文 (科学进步)
        3. 保留冲突标记供人工审核
        """
        if not conflicts:
            return new_knowledge
        
        # 选择最佳知识
        all_candidates = [new_knowledge] + conflicts
        
        # 按置信度和时间排序
        def score(k):
            conf_score = k.confidence
            # 时间越新分数越高
            try:
                time_score = datetime.fromisoformat(k.timestamp).timestamp() / 1e9
            except:
                time_score = 0
            return conf_score + 0.1 * time_score
        
        best = max(all_candidates, key=score)
        
        # 添加冲突标记
        if best != new_knowledge:
            best.content += f" [Updated from newer source: {new_knowledge.source}]"
        
        return best


class RealtimeKnowledgeBase:
    """
    实时知识库
    
    动态更新的知识管理系统
    """
    
    def __init__(self, kb_dir: str = "./astro_knowledge/dynamic"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge: List[KnowledgeExtracted] = []
        self.conflict_resolver = KnowledgeConflictResolver()
        
        # 加载现有知识
        self._load_knowledge()
        
        # 初始化监控和提取
        self.monitor = ArXivMonitor()
        self.extractor = KnowledgeExtractor(use_llm=False)  # 使用规则提取，更快
        
        # 更新锁
        self.update_lock = threading.Lock()
    
    def _load_knowledge(self):
        """加载已有知识"""
        kb_file = self.kb_dir / "extracted_knowledge.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.knowledge = [KnowledgeExtracted.from_dict(k) for k in data]
            print(f"✓ 已加载 {len(self.knowledge)} 条提取的知识")
    
    def _save_knowledge(self):
        """保存知识"""
        kb_file = self.kb_dir / "extracted_knowledge.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump([k.to_dict() for k in self.knowledge], f, 
                     indent=2, ensure_ascii=False)
    
    def update_from_arxiv(self, max_papers: int = 10) -> int:
        """
        从arXiv更新知识
        
        Returns:
            新增知识数量
        """
        print("\n🔄 开始从arXiv更新知识...")
        
        # 获取新论文
        papers = self.monitor.fetch_new_papers(max_results=max_papers)
        print(f"📄 发现 {len(papers)} 篇新论文")
        
        new_knowledge_count = 0
        
        with self.update_lock:
            for paper in papers:
                print(f"  处理: {paper['title'][:60]}...")
                
                # 提取知识
                extracted = self.extractor.extract_from_paper(paper)
                print(f"    提取到 {len(extracted)} 条知识")
                
                for knowledge in extracted:
                    # 检测冲突
                    conflicts = self.conflict_resolver.detect_conflicts(
                        knowledge, self.knowledge
                    )
                    
                    if conflicts:
                        print(f"    发现 {len(conflicts)} 个冲突，正在解决...")
                        knowledge = self.conflict_resolver.resolve(knowledge, conflicts)
                    
                    # 添加到知识库
                    self.knowledge.append(knowledge)
                    new_knowledge_count += 1
            
            # 保存
            self._save_knowledge()
        
        print(f"✓ 更新完成，新增 {new_knowledge_count} 条知识")
        print(f"  知识库总计: {len(self.knowledge)} 条")
        
        return new_knowledge_count
    
    def get_knowledge_by_type(self, k_type: str) -> List[KnowledgeExtracted]:
        """按类型获取知识"""
        return [k for k in self.knowledge if k.knowledge_type == k_type]
    
    def get_knowledge_by_entity(self, entity: str) -> List[KnowledgeExtracted]:
        """按实体获取知识"""
        return [k for k in self.knowledge if entity in k.entities]
    
    def search(self, query: str, top_k: int = 5) -> List[KnowledgeExtracted]:
        """搜索知识"""
        # 简化搜索：基于关键词匹配
        query_words = set(query.lower().split())
        
        scored = []
        for k in self.knowledge:
            content_words = set(k.content.lower().split())
            score = len(query_words & content_words)
            if score > 0:
                scored.append((score, k))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in scored[:top_k]]
    
    def start_auto_update(self, interval_hours: int = 24):
        """
        启动自动更新
        
        Args:
            interval_hours: 更新间隔（小时）
        """
        print(f"⏰ 启动自动更新，间隔: {interval_hours}小时")
        
        schedule.every(interval_hours).hours.do(self.update_from_arxiv)
        
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        # 在后台线程运行
        thread = threading.Thread(target=run_schedule, daemon=True)
        thread.start()
        
        return thread


# ==================== 便捷函数 ====================

def create_realtime_kb() -> RealtimeKnowledgeBase:
    """创建实时知识库"""
    return RealtimeKnowledgeBase()


def demo_update():
    """演示更新流程"""
    print("="*70)
    print("实时知识更新系统演示")
    print("="*70)
    
    kb = create_realtime_kb()
    
    # 手动触发更新
    kb.update_from_arxiv(max_papers=5)
    
    # 搜索知识
    print("\n🔍 搜索 'period':")
    results = kb.search("period", top_k=3)
    for r in results:
        print(f"  • [{r.knowledge_type}] {r.content[:60]}...")
    
    print("\n完成!")


# ==================== 测试 ====================

if __name__ == "__main__":
    demo_update()
