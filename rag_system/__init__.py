"""
AstroRAG - 天文领域双轨 RAG 系统
支持向量检索 + 关键词检索，提供引用溯源和去幻觉功能
"""

from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.rag_pipeline import RAGPipeline

__version__ = "1.0.0"
__all__ = ["HybridRetriever", "RAGPipeline"]
