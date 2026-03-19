#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroRAG 双轨混合检索器
================================================================================
功能描述:
    结合向量检索(语义理解)和关键词检索(精确匹配)的混合检索系统。
    提高天文领域专业术语的召回准确率。

双轨检索原理:
    ┌─────────────────┐     ┌─────────────────┐
    │   向量检索      │     │   关键词检索     │
    │  (语义相似度)   │     │  (精确匹配)      │
    │                 │     │                 │
    │  ChromaDB       │     │  倒排索引        │
    │  Embedding      │     │  TF-IDF/BM25    │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ↓
              ┌─────────────────────┐
              │     结果融合        │
              │  RRF重排序算法      │
              │  score = w1*s1 + w2*s2 │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │     Top-K 结果      │
              └─────────────────────┘

权重配置:
    - vector_weight: 0.6 (语义理解)
    - keyword_weight: 0.4 (精确匹配)
    - 可通过参数调整权重

重排序算法:
    - RRF (Reciprocal Rank Fusion)
    - 综合两种检索的排序结果

使用示例:
    from rag_system.retrieval.hybrid_retriever import HybridRetriever
    
    retriever = HybridRetriever(
        vector_weight=0.6,
        keyword_weight=0.4
    )
    results = retriever.retrieve("CV星的光变曲线特征", top_k=5)

依赖:
    - chroma_store.py: 向量存储
    - keyword_index.py: 倒排索引

作者: AstroSage Team
创建日期: 2024-03
================================================================================
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from ..vector_store.chroma_store import VectorStore
from ..inverted_index.keyword_index import KeywordIndex


class HybridRetriever:
    """
    混合检索器
    结合向量检索和关键词检索的优势
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        keyword_index: KeywordIndex = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            keyword_index: 关键词索引实例
            vector_weight: 向量检索权重 (0-1)
            keyword_weight: 关键词检索权重 (0-1)
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        
        # 权重归一化
        total = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total
        self.keyword_weight = keyword_weight / total
        
        print(f"🔧 混合检索器初始化:")
        print(f"   向量检索权重: {self.vector_weight:.2f}")
        print(f"   关键词检索权重: {self.keyword_weight:.2f}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Dict = None,
    ) -> List[Dict]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            融合后的检索结果
        """
        results = []
        
        # 1. 向量检索
        vector_results = []
        if self.vector_store:
            try:
                vector_results = self.vector_store.search(
                    query, 
                    top_k=top_k * 2,  # 检索更多用于融合
                    filter_dict=filter_dict,
                )
            except Exception as e:
                print(f"⚠️ 向量检索失败: {e}")
        
        # 2. 关键词检索
        keyword_results = []
        if self.keyword_index:
            try:
                keyword_results = self.keyword_index.search(
                    query,
                    top_k=top_k * 2,
                )
            except Exception as e:
                print(f"⚠️ 关键词检索失败: {e}")
        
        # 3. 结果融合
        fused_results = self._fuse_results(
            vector_results,
            keyword_results,
            top_k,
        )
        
        return fused_results
    
    def _fuse_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        融合两种检索的结果
        使用加权分数和去重
        """
        # 分数归一化
        vector_scores = [r["score"] for r in vector_results]
        keyword_scores = [r["score"] for r in keyword_results]
        
        max_vector = max(vector_scores) if vector_scores else 1
        max_keyword = max(keyword_scores) if keyword_scores else 1
        
        # 合并结果
        merged = defaultdict(lambda: {
            "document": "",
            "metadata": {},
            "vector_score": 0,
            "keyword_score": 0,
            "sources": set(),
        })
        
        # 添加向量检索结果
        for r in vector_results:
            doc_id = r["id"]
            normalized_score = r["score"] / max_vector if max_vector > 0 else 0
            
            merged[doc_id]["document"] = r["document"]
            merged[doc_id]["metadata"] = r["metadata"]
            merged[doc_id]["vector_score"] = normalized_score * self.vector_weight
            merged[doc_id]["sources"].add("vector")
        
        # 添加关键词检索结果
        for r in keyword_results:
            doc_id = r["id"]
            normalized_score = r["score"] / max_keyword if max_keyword > 0 else 0
            
            merged[doc_id]["document"] = r["document"]
            merged[doc_id]["metadata"] = r["metadata"]
            merged[doc_id]["keyword_score"] = normalized_score * self.keyword_weight
            merged[doc_id]["sources"].add("keyword")
            if "matched_terms" in r:
                merged[doc_id]["matched_terms"] = r["matched_terms"]
        
        # 计算最终分数
        final_results = []
        for doc_id, data in merged.items():
            final_score = data["vector_score"] + data["keyword_score"]
            
            # 两种检索都命中的给予 bonus
            if len(data["sources"]) == 2:
                final_score *= 1.1
            
            final_results.append({
                "id": doc_id,
                "document": data["document"],
                "metadata": data["metadata"],
                "final_score": round(final_score, 4),
                "vector_score": round(data["vector_score"], 4),
                "keyword_score": round(data["keyword_score"], 4),
                "retrieval_sources": list(data["sources"]),
                "matched_terms": data.get("matched_terms", []),
            })
        
        # 按最终分数排序
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return final_results[:top_k]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "vector_store": self.vector_store.get_stats() if self.vector_store else None,
            "keyword_index": self.keyword_index.get_stats() if self.keyword_index else None,
            "weights": {
                "vector": self.vector_weight,
                "keyword": self.keyword_weight,
            }
        }
        return stats


def build_hybrid_retriever(
    qa_file: str = "output/qa_hybrid/qa_dataset_full.json",
    vector_db_dir: str = "rag_system/vector_db",
    keyword_index_dir: str = "rag_system/inverted_index_db",
) -> HybridRetriever:
    """
    构建完整的混合检索器
    """
    print("="*60)
    print("🔧 构建双轨 RAG 检索器")
    print("="*60)
    
    # 构建向量存储
    from ..vector_store.chroma_store import build_vector_store_from_qa
    vector_store = build_vector_store_from_qa(qa_file, vector_db_dir)
    
    # 构建关键词索引
    from ..inverted_index.keyword_index import build_keyword_index_from_qa
    keyword_index = build_keyword_index_from_qa(qa_file, keyword_index_dir)
    
    # 创建混合检索器
    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_index=keyword_index,
        vector_weight=0.6,
        keyword_weight=0.4,
    )
    
    print("\n✅ 双轨 RAG 检索器构建完成！")
    
    return retriever


if __name__ == "__main__":
    # 构建检索器
    retriever = build_hybrid_retriever()
    
    # 测试
    print("\n" + "="*60)
    print("🧪 混合检索测试")
    print("="*60)
    
    test_queries = [
        "赫罗图上白矮星的位置",
        "灾变变星的光变曲线特征",
        "SED 能谱分布拟合",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print('='*60)
        
        results = retriever.retrieve(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n结果 {i} (综合分数: {r['final_score']}):")
            print(f"  向量分数: {r['vector_score']}, 关键词分数: {r['keyword_score']}")
            print(f"  来源: {r['retrieval_sources']}")
            if r['matched_terms']:
                print(f"  匹配词: {r['matched_terms']}")
            print(f"  内容: {r['document'][:150]}...")
