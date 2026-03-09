#!/usr/bin/env python3
"""
天文文献索引器 (Astronomy Literature Indexer)
=============================================
支持大规模文献摘要的向量索引和检索

目标: 支持50万+文献摘要的索引和检索
依赖: 需要安装 chromadb 或 faiss-cpu/gpu

作者: AstroSage AI
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Iterator, Generator
from pathlib import Path
from dataclasses import dataclass
import pickle


@dataclass
class LiteratureEntry:
    """文献条目数据结构"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    journal: str
    keywords: List[str]
    bibcode: str = ""
    doi: str = ""
    citations: int = 0
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'year': self.year,
            'journal': self.journal,
            'keywords': self.keywords,
            'bibcode': self.bibcode,
            'doi': self.doi,
            'citations': self.citations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LiteratureEntry':
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            authors=data.get('authors', []),
            abstract=data.get('abstract', ''),
            year=data.get('year', 0),
            journal=data.get('journal', ''),
            keywords=data.get('keywords', []),
            bibcode=data.get('bibcode', ''),
            doi=data.get('doi', ''),
            citations=data.get('citations', 0)
        )


class LiteratureIndexBase:
    """文献索引基类"""
    
    def __init__(self, index_dir: str = "./cache/literature_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'total_entries': 0, 'indexed': 0}
    
    def add_entries(self, entries: List[LiteratureEntry]):
        """添加文献条目"""
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 10) -> List[LiteratureEntry]:
        """搜索文献"""
        raise NotImplementedError
    
    def save(self):
        """保存索引"""
        raise NotImplementedError
    
    def load(self):
        """加载索引"""
        raise NotImplementedError


class SimpleLiteratureIndex(LiteratureIndexBase):
    """
    简易文献索引 (基于关键词匹配)
    适用于小规模文献集 (数千篇)
    """
    
    def __init__(self, index_dir: str = "./cache/literature_index"):
        super().__init__(index_dir)
        self.entries: Dict[str, LiteratureEntry] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        self._load_if_exists()
    
    def _load_if_exists(self):
        """加载已有索引"""
        index_file = self.index_dir / "simple_index.pkl"
        if index_file.exists():
            with open(index_file, 'rb') as f:
                data = pickle.load(f)
                self.entries = {k: LiteratureEntry(**v) if isinstance(v, dict) else v 
                               for k, v in data.get('entries', {}).items()}
                self.keyword_index = data.get('keyword_index', {})
                self.stats = data.get('stats', self.stats)
            print(f"✓ 已加载简易索引: {len(self.entries)} 篇文献")
    
    def add_entries(self, entries: List[LiteratureEntry]):
        """添加文献条目"""
        for entry in entries:
            self.entries[entry.id] = entry
            self.stats['total_entries'] += 1
            
            # 索引关键词
            all_text = f"{entry.title} {entry.abstract} {' '.join(entry.keywords)}"
            words = self._extract_keywords(all_text)
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                if entry.id not in self.keyword_index[word]:
                    self.keyword_index[word].append(entry.id)
        
        self.stats['indexed'] = len(self.entries)
        self.save()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # 天文领域停用词
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 
                     'all', 'can', 'had', 'her', 'was', 'one', 'our', 
                     'out', 'day', 'get', 'has', 'him', 'his', 'how', 
                     'its', 'may', 'new', 'now', 'old', 'see', 'two', 
                     'way', 'who', 'boy', 'did', 'she', 'use', 'her'}
        
        # 只保留有意义的词
        return [w for w in words if len(w) > 3 and w not in stopwords]
    
    def search(self, query: str, top_k: int = 10) -> List[LiteratureEntry]:
        """关键词搜索"""
        query_words = self._extract_keywords(query)
        if not query_words:
            return []
        
        # 统计匹配
        entry_scores: Dict[str, float] = {}
        for word in query_words:
            if word in self.keyword_index:
                for entry_id in self.keyword_index[word]:
                    entry_scores[entry_id] = entry_scores.get(entry_id, 0) + 1
        
        # 排序
        sorted_entries = sorted(entry_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_k]
        
        return [self.entries[eid] for eid, _ in sorted_entries 
                if eid in self.entries]
    
    def save(self):
        """保存索引"""
        data = {
            'entries': {k: v.to_dict() for k, v in self.entries.items()},
            'keyword_index': self.keyword_index,
            'stats': self.stats
        }
        with open(self.index_dir / "simple_index.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """加载索引"""
        self._load_if_exists()


class ChromaLiteratureIndex(LiteratureIndexBase):
    """
    ChromaDB文献索引 (推荐用于大规模数据)
    支持50万+文献的语义检索
    
    安装: pip install chromadb
    """
    
    def __init__(self, index_dir: str = "./cache/chroma_index", 
                 collection_name: str = "astro_papers"):
        super().__init__(index_dir)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_chroma()
    
    def _init_chroma(self):
        """初始化ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=str(self.index_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 更新统计
            self.stats['indexed'] = self.collection.count()
            print(f"✓ ChromaDB索引已加载: {self.stats['indexed']} 篇文献")
            
        except ImportError:
            print("⚠ ChromaDB未安装，请运行: pip install chromadb")
            self.client = None
    
    def add_entries(self, entries: List[LiteratureEntry], 
                    embedding_function=None):
        """
        批量添加文献
        
        Args:
            entries: 文献条目列表
            embedding_function: 自定义嵌入函数，默认使用Chroma内置
        """
        if not self.collection:
            print("✗ ChromaDB未初始化")
            return
        
        # 分批处理 (避免内存溢出)
        batch_size = 1000
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            
            ids = [e.id for e in batch]
            documents = [f"{e.title}\n{e.abstract}" for e in batch]
            metadatas = [{
                'title': e.title,
                'year': e.year,
                'journal': e.journal,
                'authors': ', '.join(e.authors[:3]),
                'keywords': ', '.join(e.keywords),
                'bibcode': e.bibcode,
                'doi': e.doi,
                'citations': e.citations
            } for e in batch]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"  已索引 {i+len(batch)}/{len(entries)} 篇文献")
        
        self.stats['indexed'] = self.collection.count()
        print(f"✓ 索引完成，总计: {self.stats['indexed']} 篇文献")
    
    def search(self, query: str, top_k: int = 10,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            filter_dict: 过滤条件，如 {'year': {'$gte': 2020}}
            
        Returns:
            搜索结果列表，每项包含文献信息和距离
        """
        if not self.collection:
            print("✗ ChromaDB未初始化")
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict
        )
        
        # 格式化结果
        formatted = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'document': results['documents'][0][i] if results['documents'] else ""
                })
        
        return formatted
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        if not self.collection:
            return {}
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection_name,
            'index_directory': str(self.index_dir)
        }


class ADSAbstractFetcher:
    """
    ADS文献摘要获取器
    用于从NASA ADS获取文献数据
    
    需要ADS API Token: https://ui.adsabs.harvard.edu/user/settings/token
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get('ADS_API_TOKEN')
        self.base_url = "https://api.adsabs.harvard.edu/v1"
    
    def search(self, query: str, rows: int = 100) -> List[LiteratureEntry]:
        """
        搜索ADS数据库
        
        Args:
            query: ADS查询语法
            rows: 返回结果数
            
        Returns:
            LiteratureEntry列表
        """
        if not self.api_token:
            print("⚠ 未设置ADS API Token")
            return []
        
        import requests
        
        url = f"{self.base_url}/search/query"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        params = {
            'q': query,
            'fl': 'title,author,abstract,year,bibcode,doi,keyword,citation_count',
            'rows': rows
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            entries = []
            for doc in data.get('response', {}).get('docs', []):
                entry = LiteratureEntry(
                    id=doc.get('bibcode', hashlib.md5(doc.get('title', '').encode()).hexdigest()[:12]),
                    title=doc.get('title', [''])[0] if isinstance(doc.get('title'), list) else doc.get('title', ''),
                    authors=doc.get('author', [])[:5],
                    abstract=doc.get('abstract', ''),
                    year=doc.get('year', 0),
                    journal=doc.get('bibcode', '')[4:13] if doc.get('bibcode') else '',
                    keywords=doc.get('keyword', []),
                    bibcode=doc.get('bibcode', ''),
                    doi=doc.get('doi', [''])[0] if isinstance(doc.get('doi'), list) else doc.get('doi', ''),
                    citations=doc.get('citation_count', 0)
                )
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            print(f"✗ ADS查询失败: {e}")
            return []


def create_sample_literature_db(output_file: str = "./astro_knowledge/sample_papers.json"):
    """创建示例文献数据库"""
    
    sample_papers = [
        {
            'id': '2023ApJ...947..123C',
            'title': 'The Population of AM CVn Systems in the Galaxy',
            'authors': ['Carter, P. J.', 'Marsh, T. R.', 'Nelemans, G.'],
            'abstract': 'AM CVn systems are ultra-compact binary systems with orbital periods between 5 and 65 minutes. They are important gravitational wave sources for space-based detectors like LISA. We present a population synthesis study of these systems...',
            'year': 2023,
            'journal': 'ApJ',
            'keywords': ['binaries: close', 'white dwarfs', 'gravitational waves', 'stars: individual: AM CVn'],
            'bibcode': '2023ApJ...947..123C',
            'citations': 45
        },
        {
            'id': '2022MNRAS.512..456R',
            'title': 'Period Changes in Cataclysmic Variables: Evidence for Angular Momentum Loss',
            'authors': ['Robinson, E. L.', 'Barker, M.'],
            'abstract': 'We analyze period variations in 127 cataclysmic variables using long-term photometric data. Our results show systematic period changes consistent with angular momentum loss via magnetic braking and gravitational radiation...',
            'year': 2022,
            'journal': 'MNRAS',
            'keywords': ['cataclysmic variables', 'stars: evolution', 'binaries: eclipsing'],
            'bibcode': '2022MNRAS.512..456R',
            'citations': 78
        },
        {
            'id': '2024A&A...681A..45S',
            'title': 'TESS Observations of Polars: New Insights into Accretion Physics',
            'authors': ['Schmidt, G. D.', 'Smith, P. S.'],
            'abstract': 'The TESS mission provides high-precision, continuous light curves of magnetic cataclysmic variables (polars). We analyze 23 polars observed by TESS during sectors 1-60, revealing new details about accretion column physics...',
            'year': 2024,
            'journal': 'A&A',
            'keywords': ['accretion', 'stars: magnetic field', 'white dwarfs', 'stars: cataclysmic variables'],
            'bibcode': '2024A&A...681A..45S',
            'citations': 23
        },
        {
            'id': '2023ApJS..266...12W',
            'title': 'A Catalog of White Dwarf Binaries from Gaia DR3',
            'authors': ['Williams, K. A.', 'Bloemen, S.', 'Marsh, T. R.'],
            'abstract': 'We present a catalog of 2,847 white dwarf binaries identified from Gaia DR3 astrometry and photometry. The catalog includes double degenerates, white dwarf-main sequence binaries, and pre-cataclysmic variables...',
            'year': 2023,
            'journal': 'ApJS',
            'keywords': [' catalogs', 'astrometry', 'white dwarfs', 'binaries: spectroscopic'],
            'bibcode': '2023ApJS..266...12W',
            'citations': 156
        },
        {
            'id': '2021MNRAS.503..890P',
            'title': 'The Orbital Period Distribution of Cataclysmic Variables',
            'authors': ['Pala, A. F.', 'Schmidtobreick, L.', 'Tappert, C.'],
            'abstract': 'We investigate the orbital period distribution of 1,032 confirmed cataclysmic variables. The distribution shows the well-known period gap between 2-3 hours and a pile-up near the minimum period...',
            'year': 2021,
            'journal': 'MNRAS',
            'keywords': ['cataclysmic variables', 'stars: evolution', 'binaries: close'],
            'bibcode': '2021MNRAS.503..890P',
            'citations': 203
        }
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_papers, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 示例文献数据库已创建: {output_file}")
    print(f"  包含 {len(sample_papers)} 篇文献")
    
    return sample_papers


# ==================== 便捷函数 ====================

def get_literature_index(index_type: str = "simple", 
                         index_dir: str = "./cache/literature_index") -> LiteratureIndexBase:
    """
    获取文献索引实例
    
    Args:
        index_type: 'simple' 或 'chroma'
        index_dir: 索引目录
        
    Returns:
        LiteratureIndexBase实例
    """
    if index_type == "chroma":
        return ChromaLiteratureIndex(index_dir)
    else:
        return SimpleLiteratureIndex(index_dir)


def search_literature(query: str, top_k: int = 5, 
                      index_type: str = "simple") -> List:
    """便捷搜索函数"""
    index = get_literature_index(index_type)
    return index.search(query, top_k)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("天文文献索引器测试")
    print("=" * 70)
    
    # 创建示例数据库
    print("\n1. 创建示例文献数据库...")
    papers = create_sample_literature_db()
    
    # 测试简易索引
    print("\n2. 测试简易索引...")
    simple_index = SimpleLiteratureIndex()
    entries = [LiteratureEntry(**p) for p in papers]
    simple_index.add_entries(entries)
    
    results = simple_index.search("cataclysmic variables period", top_k=3)
    print(f"   搜索 'cataclysmic variables period': {len(results)} 个结果")
    for r in results:
        print(f"   - {r.title[:60]}...")
    
    # 测试Chroma索引 (如果可用)
    print("\n3. 测试Chroma索引...")
    chroma_index = ChromaLiteratureIndex()
    if chroma_index.collection:
        chroma_index.add_entries(entries)
        results = chroma_index.search("gravitational waves AM CVn", top_k=3)
        print(f"   搜索 'gravitational waves AM CVn': {len(results)} 个结果")
        for r in results:
            print(f"   - {r['metadata']['title'][:60]}...")
    else:
        print("   (跳过 - ChromaDB未安装)")
    
    print("\n测试完成!")
