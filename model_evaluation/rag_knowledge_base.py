#!/usr/bin/env python3
"""
RAG 知识库构建器

功能：
1. 从 PDF 提取的知识块构建向量数据库
2. 支持语义检索
3. 与模型微调结合
4. 提供天体源查询接口
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class RAGDocument:
    """RAG 文档"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }


class SimpleEmbedding:
    """
    简单嵌入模型（使用 sentence-transformers 或 API）
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"✓ 加载嵌入模型: {self.model_name}")
            except ImportError:
                print("请安装 sentence-transformers: pip install sentence-transformers")
                raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        self._load_model()
        return self.model.encode(texts, show_progress_bar=True)
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """编码查询（可以添加指令前缀）"""
        # 对于 BGE 模型，查询需要添加指令
        if "bge" in self.model_name.lower():
            queries = [f"Represent this sentence for searching relevant passages: {q}" for q in queries]
        return self.encode(queries)


class RAGKnowledgeBase:
    """
    RAG 知识库
    
    基于向量检索的知识库系统
    """
    
    def __init__(self, 
                 kb_dir: str = "./rag_knowledge_base",
                 embedding_model: str = "BAAI/bge-small-en"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SimpleEmbedding(embedding_model)
        self.documents: List[RAGDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # 索引映射
        self.doc_id_to_idx: Dict[str, int] = {}
        
        # 加载已有知识库
        self.load()
    
    def add_documents(self, documents: List[RAGDocument], compute_embeddings: bool = True):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            compute_embeddings: 是否计算嵌入
        """
        start_idx = len(self.documents)
        
        for i, doc in enumerate(documents):
            self.doc_id_to_idx[doc.doc_id] = start_idx + i
            self.documents.append(doc)
        
        if compute_embeddings:
            self._compute_embeddings(documents)
        
        print(f"✓ 添加 {len(documents)} 个文档，总计 {len(self.documents)}")
    
    def _compute_embeddings(self, documents: List[RAGDocument]):
        """计算文档嵌入"""
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # 更新整体嵌入矩阵
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        print(f"✓ 计算 {len(documents)} 个嵌入")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[RAGDocument, float]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            
        Returns:
            (文档, 相似度) 列表
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # 编码查询
        query_embedding = self.embedding_model.encode_queries([query])[0]
        
        # 计算相似度（余弦相似度）
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append((doc, score))
        
        return results
    
    def search_by_source(self, source_name: str) -> List[RAGDocument]:
        """按源名称搜索"""
        results = []
        for doc in self.documents:
            if source_name.lower() in doc.content.lower():
                results.append(doc)
        return results
    
    def search_by_type(self, source_type: str) -> List[RAGDocument]:
        """按类型搜索"""
        results = []
        for doc in self.documents:
            doc_type = doc.metadata.get("source_type", "").lower()
            if source_type.lower() in doc_type:
                results.append(doc)
        return results
    
    def get_retrieval_context(self, query: str, top_k: int = 3) -> str:
        """
        获取检索上下文（用于 RAG 生成）
        
        Args:
            query: 查询
            top_k: 检索文档数
            
        Returns:
            拼接的上下文文本
        """
        results = self.search(query, top_k=top_k)
        
        contexts = []
        for doc, score in results:
            ctx = f"[相关度: {score:.3f}]\n{doc.content[:500]}\n"
            contexts.append(ctx)
        
        return "\n---\n".join(contexts)
    
    def save(self):
        """保存知识库"""
        # 保存文档
        docs_data = [doc.to_dict() for doc in self.documents]
        with open(self.kb_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # 保存嵌入
        if self.embeddings is not None:
            np.save(self.kb_dir / "embeddings.npy", self.embeddings)
        
        # 保存索引
        with open(self.kb_dir / "index.pkl", 'wb') as f:
            pickle.dump(self.doc_id_to_idx, f)
        
        print(f"✓ 知识库已保存: {self.kb_dir}")
    
    def load(self):
        """加载知识库"""
        # 加载文档
        docs_path = self.kb_dir / "documents.json"
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                self.documents = [
                    RAGDocument(
                        doc_id=d["doc_id"],
                        content=d["content"],
                        metadata=d["metadata"],
                        embedding=np.array(d["embedding"]) if d["embedding"] else None
                    )
                    for d in docs_data
                ]
            print(f"✓ 加载 {len(self.documents)} 个文档")
        
        # 加载嵌入
        emb_path = self.kb_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            print(f"✓ 加载嵌入矩阵: {self.embeddings.shape}")
        
        # 加载索引
        idx_path = self.kb_dir / "index.pkl"
        if idx_path.exists():
            with open(idx_path, 'rb') as f:
                self.doc_id_to_idx = pickle.load(f)
    
    def get_statistics(self) -> Dict:
        """获取知识库统计"""
        stats = {
            "total_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "by_type": {},
            "by_source": {}
        }
        
        for doc in self.documents:
            doc_type = doc.metadata.get("chunk_type", "unknown")
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            
            source = doc.metadata.get("source_pdf", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        return stats


class SourceQueryTool:
    """
    天体源查询工具
    结合 RAG 知识库和天体源数据库
    """
    
    def __init__(self, 
                 kb: RAGKnowledgeBase,
                 source_db_path: str = "./astro_source_db.json"):
        self.kb = kb
        
        from kimi_pdf_extractor import AstroSourceDatabase
        self.source_db = AstroSourceDatabase(source_db_path)
    
    def query_source(self, source_name: str) -> Dict:
        """
        查询特定天体源的完整信息
        
        Returns:
            包含所有已知信息的字典
        """
        result = {
            "source_name": source_name,
            "basic_info": {},
            "rag_context": "",
            "classification": {},
            "related_sources": []
        }
        
        # 1. 从数据库获取基本信息
        key = source_name.lower().replace(" ", "_")
        if key in self.source_db.sources:
            source = self.source_db.sources[key]
            result["basic_info"] = source.to_dict()
            
            # 分类
            result["classification"] = self.source_db.classify_source(source)
        
        # 2. 从 RAG 获取上下文
        rag_results = self.kb.search_by_source(source_name)
        if rag_results:
            contexts = [doc.content[:300] for doc in rag_results[:3]]
            result["rag_context"] = "\n".join(contexts)
        
        # 3. 查找相关源（同类型或同区域）
        if result["basic_info"].get("source_type"):
            related = self.source_db.search_by_type(result["basic_info"]["source_type"])
            result["related_sources"] = [s.name for s in related[:5] if s.name != source_name]
        
        return result
    
    def classify_unknown_source(self, 
                               period: Optional[float] = None,
                               lightcurve_shape: str = "",
                               has_xray: bool = False,
                               has_outburst: bool = False,
                               color_index: Optional[float] = None) -> Dict:
        """
        根据观测特征分类未知源
        
        Args:
            period: 周期（天）
            lightcurve_shape: 光变曲线形状描述
            has_xray: 是否有 X 射线
            has_outburst: 是否有爆发
            color_index: 颜色指数
            
        Returns:
            分类结果和置信度
        """
        from kimi_pdf_extractor import AstroSource
        
        # 创建临时源对象
        temp_source = AstroSource(
            name="unknown",
            period=period,
            lightcurve_shape=lightcurve_shape,
            has_xray=has_xray,
            has_outburst=has_outburst
        )
        
        # 使用数据库分类
        classification = self.source_db.classify_source(temp_source)
        
        # 从 RAG 检索相似源
        query = f"period {period} days light curve {lightcurve_shape}"
        if has_xray:
            query += " X-ray"
        if has_outburst:
            query += " outburst"
        
        similar_docs = self.kb.search(query, top_k=3)
        
        classification["similar_sources"] = [
            {
                "doc_id": doc.doc_id,
                "content_preview": doc.content[:200],
                "similarity": score
            }
            for doc, score in similar_docs
        ]
        
        return classification
    
    def generate_observation_suggestions(self, source_name: str) -> List[str]:
        """
        基于源的特征生成观测建议
        
        Args:
            source_name: 源名称
            
        Returns:
            观测建议列表
        """
        query_info = self.query_source(source_name)
        
        suggestions = []
        
        basic = query_info.get("basic_info", {})
        classification = query_info.get("classification", {})
        
        # 基于类型建议
        source_type = classification.get("primary_type", "unknown")
        
        if "AM CVn" in source_type:
            suggestions.extend([
                "建议进行光谱观测，确认氦线存在且无氢线",
                "测光观测验证短周期（< 1 小时）光变",
                "X射线观测（XMM-Newton/Chandra）检测吸积活动",
                "时序测光寻找超爆发（superoutburst）"
            ])
        elif "CV" in source_type or "dwarf nova" in source_type:
            suggestions.extend([
                "长期测光监测寻找爆发周期",
                "多波段测光（ugriz）确定颜色变化",
                "低分辨率光谱确认发射线特征",
                "X射线观测确定吸积状态"
            ])
        elif "Polar" in source_type or "AM Her" in source_type:
            suggestions.extend([
                "偏振测光检测圆偏振（强磁场特征）",
                "X射线观测（吸积柱辐射）",
                "时序测光确认轨道周期",
                "光谱观测检测强磁场塞曼分裂"
            ])
        
        # 基于已有数据建议
        if not basic.get("has_xray"):
            suggestions.append("建议进行 X射线观测确认吸积活动")
        
        if not basic.get("period"):
            suggestions.append("建议进行时序测光确定周期")
        
        return suggestions


def build_rag_from_pdfs(pdf_dir: str, 
                       output_dir: str = "./rag_knowledge_base",
                       api_key: Optional[str] = None) -> RAGKnowledgeBase:
    """
    从 PDF 目录构建 RAG 知识库
    
    完整流程：
    1. 扫描 PDF 文件
    2. 提取文本
    3. 使用 Kimi 提取结构化信息
    4. 创建知识块
    5. 构建向量数据库
    """
    from pdf_processor import PDFScanner, PDFTextExtractor, PDFDeduplicator
    from kimi_pdf_extractor import KimiPDFExtractor, AstroSourceDatabase
    
    print("=" * 60)
    print("构建 RAG 知识库")
    print("=" * 60)
    
    # 1. 扫描 PDF
    print("\n[1/5] 扫描 PDF 文件...")
    scanner = PDFScanner(pdf_dir)
    pdf_files = scanner.scan()
    
    if not pdf_files:
        print("未找到 PDF 文件")
        return None
    
    # 2. 去重
    print("\n[2/5] 去重处理...")
    deduplicator = PDFDeduplicator()
    unique_files, _ = deduplicator.deduplicate(pdf_files)
    
    # 3. 提取文本和结构化信息
    print("\n[3/5] 提取文本和数据...")
    text_extractor = PDFTextExtractor()
    kimi_extractor = KimiPDFExtractor(api_key=api_key)
    source_db = AstroSourceDatabase()
    
    all_chunks = []
    
    for i, pdf_info in enumerate(unique_files[:20], 1):  # 限制处理数量
        print(f"\n[{i}/20] {pdf_info.filename[:40]}")
        
        try:
            # 提取文本
            text = text_extractor.extract_text(pdf_info.filepath, max_pages=30)
            
            if len(text) < 100:
                print("  ⚠ 文本过短，跳过")
                continue
            
            # 使用 Kimi 提取信息
            summary = kimi_extractor.extract_paper_summary(text)
            sources = kimi_extractor.extract_astro_sources(text)
            chunks = kimi_extractor.create_knowledge_chunks(text, pdf_info.filename)
            
            # 保存天体源到数据库
            for source in sources:
                source_db.add_source(source)
                print(f"  ✓ 源: {source.name}")
            
            all_chunks.extend(chunks)
            print(f"  ✓ 提取 {len(chunks)} 个知识块")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    # 4. 构建 RAG 知识库
    print("\n[4/5] 构建向量数据库...")
    kb = RAGKnowledgeBase(output_dir)
    kb.add_documents(all_chunks)
    kb.save()
    
    # 5. 保存天体源数据库
    print("\n[5/5] 保存天体源数据库...")
    source_db.save()
    
    # 统计
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)
    print(f"PDF 文件: {len(unique_files)}")
    print(f"知识块: {len(all_chunks)}")
    print(f"天体源: {len(source_db.sources)}")
    
    stats = kb.get_statistics()
    print(f"\n知识库统计:")
    print(f"  总文档: {stats['total_documents']}")
    print(f"  嵌入维度: {stats['embedding_dim']}")
    
    return kb


def main():
    """测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 知识库工具")
    parser.add_argument("--build", action="store_true", help="构建知识库")
    parser.add_argument("--pdf-dir", type=str, default=r"F:\storage", help="PDF 目录")
    parser.add_argument("--query", type=str, help="查询")
    parser.add_argument("--source", type=str, help="查询特定源")
    
    args = parser.parse_args()
    
    if args.build:
        kb = build_rag_from_pdfs(args.pdf_dir)
    
    elif args.query:
        kb = RAGKnowledgeBase()
        results = kb.search(args.query, top_k=5)
        
        print(f"查询: {args.query}")
        print("=" * 60)
        for doc, score in results:
            print(f"\n[相似度: {score:.3f}] {doc.doc_id}")
            print(doc.content[:300])
    
    elif args.source:
        kb = RAGKnowledgeBase()
        tool = SourceQueryTool(kb)
        result = tool.query_source(args.source)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
