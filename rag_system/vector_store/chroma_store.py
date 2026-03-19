#!/usr/bin/env python3
"""
向量存储模块 - 基于 ChromaDB
用于语义相似度检索
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ ChromaDB 未安装，运行: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers 未安装，运行: pip install sentence-transformers")


class VectorStore:
    """
    向量存储类
    使用 ChromaDB 存储文档向量，支持语义检索
    """
    
    # 默认使用的嵌入模型
    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
    
    def __init__(
        self,
        collection_name: str = "astro_papers",
        persist_directory: str = "rag_system/vector_db",
        embedding_model: str = None,
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 数据持久化目录
            embedding_model: 嵌入模型名称
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("请先安装 ChromaDB: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.Client(
            Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        # 初始化嵌入模型
        self.embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.embedding_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"📦 加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"✅ 嵌入模型加载完成")
        
        self._stats = {
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name,
        }
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        if self.embedding_model is None:
            raise RuntimeError("嵌入模型未加载")
        
        # 生成嵌入向量
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,  # 归一化
            show_progress_bar=False,
        )
        return embedding.tolist()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
    ) -> List[str]:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档内容列表
            metadatas: 文档元数据列表
            ids: 文档ID列表（可选，默认自动生成）
        
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        # 生成 IDs
        if ids is None:
            ids = []
            for i, doc in enumerate(documents):
                # 使用内容哈希作为ID
                doc_hash = hashlib.md5(doc.encode()).hexdigest()[:16]
                ids.append(f"doc_{doc_hash}_{i}")
        
        # 确保 metadatas
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 生成嵌入向量
        print(f"🔄 正在生成 {len(documents)} 个文档的嵌入向量...")
        embeddings = []
        for doc in documents:
            embedding = self._get_embedding(doc)
            embeddings.append(embedding)
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        # 持久化
        self.client.persist()
        
        print(f"✅ 成功添加 {len(documents)} 个文档")
        self._stats["total_documents"] = self.collection.count()
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Dict = None,
    ) -> List[Dict]:
        """
        语义检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件
        
        Returns:
            检索结果列表，每个结果包含:
            - id: 文档ID
            - document: 文档内容
            - metadata: 元数据
            - score: 相似度分数 (0-1，越高越相似)
        """
        # 生成查询向量
        query_embedding = self._get_embedding(query)
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )
        
        # 格式化结果
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # 余弦距离转换为相似度分数
                distance = results["distances"][0][i]
                similarity = 1 - distance  # 距离越小，相似度越高
                
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(similarity, 4),
                    "retrieval_type": "vector",
                })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """删除文档"""
        self.collection.delete(ids=ids)
        self.client.persist()
        self._stats["total_documents"] = self.collection.count()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self._stats.copy()
    
    def reset(self):
        """清空集合"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._stats["total_documents"] = 0
        print("✅ 向量存储已重置")


def build_vector_store_from_qa(
    qa_file: str = "output/qa_hybrid/qa_dataset_full.json",
    output_dir: str = "rag_system/vector_db",
) -> VectorStore:
    """
    从问答数据构建向量存储
    
    Args:
        qa_file: 问答数据文件
        output_dir: 输出目录
    
    Returns:
        VectorStore 实例
    """
    print("="*60)
    print("🔧 构建向量存储")
    print("="*60)
    
    # 初始化存储
    store = VectorStore(
        collection_name="astro_qa",
        persist_directory=output_dir,
    )
    
    # 加载问答数据
    print(f"📂 加载问答数据: {qa_file}")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"✅ 加载 {len(qa_pairs)} 条问答对")
    
    # 准备文档
    documents = []
    metadatas = []
    
    for qa in qa_pairs:
        # 将问答对组合为文档
        doc_text = f"问题: {qa['question']}\n答案: {qa['answer']}"
        
        metadata = {
            "question": qa['question'],
            "answer": qa['answer'],
            "question_type": qa.get('question_type', 'general'),
            "source_file": qa.get('source_file', ''),
            "page_number": qa.get('page_number', 0),
            "confidence": qa.get('confidence', 0),
            "generation_method": qa.get('generation_method', 'unknown'),
        }
        
        documents.append(doc_text)
        metadatas.append(metadata)
    
    # 分批添加（避免内存问题）
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        print(f"🔄 添加批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        store.add_documents(batch_docs, batch_meta)
    
    print(f"\n✅ 向量存储构建完成！")
    print(f"   总文档数: {store.get_stats()['total_documents']}")
    
    return store


if __name__ == "__main__":
    # 测试
    store = build_vector_store_from_qa()
    
    # 测试检索
    print("\n🧪 测试检索:")
    query = "赫罗图上白矮星的位置在哪里？"
    results = store.search(query, top_k=3)
    
    print(f"\n查询: {query}")
    for i, r in enumerate(results, 1):
        print(f"\n结果 {i} (相似度: {r['score']}):")
        print(f"  来源: {r['metadata']['source_file']}, 第{r['metadata']['page_number']}页")
        print(f"  内容: {r['document'][:200]}...")
