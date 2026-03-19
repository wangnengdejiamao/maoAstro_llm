#!/usr/bin/env python3
"""
倒排索引模块 - 关键词检索
基于倒排索引实现快速关键词匹配
"""

import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import jieba


class KeywordIndex:
    """
    倒排索引类
    用于关键词精确匹配检索
    """
    
    # 天文领域停用词
    STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
        '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
        '这些', '那些', '这个', '那个', '之', '与', '及', '等',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'and', 'but', 'or', 'yet', 'so',
    }
    
    def __init__(self, index_dir: str = "rag_system/inverted_index_db"):
        """
        初始化倒排索引
        
        Args:
            index_dir: 索引存储目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 倒排索引: 词 -> 文档ID列表
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 文档存储: 文档ID -> 文档内容
        self.documents: Dict[str, Dict] = {}
        
        # 词频统计: 词 -> 出现次数
        self.term_freq: Dict[str, int] = defaultdict(int)
        
        # 文档数
        self.doc_count = 0
        
        # 尝试加载已有索引
        self._load_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词处理
        同时支持中文和英文
        """
        if not text:
            return []
        
        tokens = []
        
        # 中文分词
        chinese_tokens = jieba.lcut(text)
        tokens.extend(chinese_tokens)
        
        # 英文单词提取
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        tokens.extend(english_words)
        
        # 过滤停用词和单字
        filtered = []
        for token in tokens:
            token = token.strip().lower()
            if len(token) > 1 and token not in self.STOP_WORDS:
                filtered.append(token)
        
        return filtered
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """
        添加文档到索引
        
        Args:
            doc_id: 文档唯一ID
            text: 文档内容
            metadata: 文档元数据
        """
        # 分词
        tokens = self._tokenize(text)
        
        # 更新倒排索引
        for token in set(tokens):  # 使用set去重
            self.inverted_index[token].add(doc_id)
            self.term_freq[token] += 1
        
        # 存储文档
        self.documents[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "tokens": list(set(tokens)),
            "term_count": len(tokens),
        }
        
        self.doc_count += 1
    
    def add_documents(self, documents: List[Tuple[str, str, Dict]]):
        """
        批量添加文档
        
        Args:
            documents: 列表，每项为 (doc_id, text, metadata)
        """
        for doc_id, text, metadata in documents:
            self.add_document(doc_id, text, metadata)
        
        print(f"✅ 已索引 {len(documents)} 个文档")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_match: int = 1,
    ) -> List[Dict]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            min_match: 最少匹配词数
        
        Returns:
            检索结果列表
        """
        # 查询分词
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # 收集候选文档
        candidate_scores: Dict[str, float] = defaultdict(float)
        matched_terms: Dict[str, Set[str]] = defaultdict(set)
        
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_id in self.inverted_index[token]:
                    # TF-IDF 评分
                    tf = 1  # 简单起见，使用词频为1
                    idf = self._calculate_idf(token)
                    candidate_scores[doc_id] += tf * idf
                    matched_terms[doc_id].add(token)
        
        # 过滤最少匹配
        filtered_candidates = {
            doc_id: score
            for doc_id, score in candidate_scores.items()
            if len(matched_terms[doc_id]) >= min_match
        }
        
        # 排序并返回Top-K
        sorted_results = sorted(
            filtered_candidates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 格式化结果
        results = []
        for doc_id, score in sorted_results:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "document": doc["text"],
                "metadata": doc["metadata"],
                "score": round(score, 4),
                "matched_terms": list(matched_terms[doc_id]),
                "match_count": len(matched_terms[doc_id]),
                "retrieval_type": "keyword",
            })
        
        return results
    
    def _calculate_idf(self, term: str) -> float:
        """计算IDF值"""
        import math
        
        doc_with_term = len(self.inverted_index.get(term, set()))
        if doc_with_term == 0:
            return 0
        
        return math.log(self.doc_count / (doc_with_term + 1)) + 1
    
    def save(self):
        """保存索引到磁盘"""
        index_file = self.index_dir / "index.pkl"
        
        data = {
            "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
            "documents": self.documents,
            "term_freq": dict(self.term_freq),
            "doc_count": self.doc_count,
        }
        
        with open(index_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ 索引已保存: {index_file}")
    
    def _load_index(self):
        """从磁盘加载索引"""
        index_file = self.index_dir / "index.pkl"
        
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.inverted_index = defaultdict(
                    set,
                    {k: set(v) for k, v in data["inverted_index"].items()}
                )
                self.documents = data["documents"]
                self.term_freq = defaultdict(int, data["term_freq"])
                self.doc_count = data["doc_count"]
                
                print(f"✅ 已加载索引: {self.doc_count} 个文档")
            except Exception as e:
                print(f"⚠️ 加载索引失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_documents": self.doc_count,
            "total_terms": len(self.inverted_index),
            "avg_doc_length": sum(
                d["term_count"] for d in self.documents.values()
            ) / max(self.doc_count, 1),
        }
    
    def reset(self):
        """重置索引"""
        self.inverted_index = defaultdict(set)
        self.documents = {}
        self.term_freq = defaultdict(int)
        self.doc_count = 0
        
        index_file = self.index_dir / "index.pkl"
        if index_file.exists():
            index_file.unlink()
        
        print("✅ 索引已重置")


def build_keyword_index_from_qa(
    qa_file: str = "output/qa_hybrid/qa_dataset_full.json",
    output_dir: str = "rag_system/inverted_index_db",
) -> KeywordIndex:
    """
    从问答数据构建关键词索引
    """
    print("="*60)
    print("🔧 构建关键词索引")
    print("="*60)
    
    # 初始化索引
    index = KeywordIndex(index_dir=output_dir)
    
    # 加载问答数据
    print(f"📂 加载问答数据: {qa_file}")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"✅ 加载 {len(qa_pairs)} 条问答对")
    
    # 准备文档
    documents = []
    for i, qa in enumerate(qa_pairs):
        doc_id = f"qa_{i}"
        # 组合问答内容
        text = f"{qa['question']} {qa['answer']}"
        
        metadata = {
            "question": qa['question'],
            "answer": qa['answer'],
            "question_type": qa.get('question_type', 'general'),
            "source_file": qa.get('source_file', ''),
            "page_number": qa.get('page_number', 0),
        }
        
        documents.append((doc_id, text, metadata))
    
    # 批量添加
    index.add_documents(documents)
    
    # 保存索引
    index.save()
    
    print(f"\n✅ 关键词索引构建完成！")
    stats = index.get_stats()
    print(f"   文档数: {stats['total_documents']}")
    print(f"   词项数: {stats['total_terms']}")
    
    return index


if __name__ == "__main__":
    # 测试
    index = build_keyword_index_from_qa()
    
    # 测试检索
    print("\n🧪 测试关键词检索:")
    query = "赫罗图 白矮星"
    results = index.search(query, top_k=3)
    
    print(f"\n查询: {query}")
    for i, r in enumerate(results, 1):
        print(f"\n结果 {i} (匹配词: {r['match_count']}, 分数: {r['score']}):")
        print(f"  匹配词: {r['matched_terms']}")
        print(f"  来源: {r['metadata']['source_file']}")
        print(f"  内容: {r['document'][:150]}...")
