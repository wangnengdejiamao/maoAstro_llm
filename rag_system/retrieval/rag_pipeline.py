#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroRAG Pipeline - 检索增强生成管道
================================================================================
功能描述:
    完整的RAG流程实现，包括检索、生成、幻觉检测和引用溯源。
    是AstroSage系统的核心推理引擎。

核心组件:
    1. HybridRetriever: 双轨检索(向量+关键词)
    2. HallucinationDetector: 幻觉检测器
    3. CitationExtractor: 引用提取器
    4. RAGPipeline: 主流程控制器

去幻觉机制:
    - 指示词检测: 识别不确定表达("可能"、"也许"等)
    - 数值验证: 验证回答中的数字是否在文档范围内
    - 置信度评分: 基于检索相关性计算

引用溯源:
    - 自动标注引用来源文档
    - 显示页码和段落信息
    - 支持多文档引用

使用示例:
    from rag_system.retrieval.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    response = pipeline.query("什么是灾变变星？")
    print(response.answer)
    print(response.citations)

依赖:
    - hybrid_retriever.py: 混合检索器
    - chroma_store.py: 向量存储
    - keyword_index.py: 关键词索引

作者: AstroSage Team
创建日期: 2024-03
================================================================================
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .hybrid_retriever import HybridRetriever


@dataclass
class RAGResponse:
    """RAG 响应数据结构"""
    answer: str
    retrieved_documents: List[Dict]
    citations: List[Dict]
    confidence: float
    hallucination_check: Dict


class HallucinationDetector:
    """
    幻觉检测器
    检测 LLM 回答是否与检索到的文档一致
    """
    
    def __init__(self):
        # 幻觉指示词
        self.hallucination_indicators = [
            "我不确定", "可能", "也许", "我猜", "好像",
            "我不清楚", "没有相关信息", "无法确定",
            "i'm not sure", "maybe", "perhaps", "i guess",
        ]
    
    def check(
        self,
        answer: str,
        retrieved_docs: List[Dict],
    ) -> Dict:
        """
        检测回答中的幻觉
        
        Returns:
            {
                "has_hallucination": bool,
                "confidence": float,  # 0-1，越高表示越可能是幻觉
                "indicators_found": List[str],
                "unsupported_claims": List[str],
            }
        """
        result = {
            "has_hallucination": False,
            "confidence": 0.0,
            "indicators_found": [],
            "unsupported_claims": [],
        }
        
        answer_lower = answer.lower()
        
        # 1. 检测指示词
        for indicator in self.hallucination_indicators:
            if indicator in answer_lower:
                result["indicators_found"].append(indicator)
                result["confidence"] += 0.2
        
        # 2. 检查关键信息是否在检索文档中
        # 提取回答中的数字、专有名词
        numbers = re.findall(r'\d+\.?\d*\s*[a-zA-Z]*', answer)
        
        # 合并所有检索文档文本
        all_doc_text = " ".join([doc["document"] for doc in retrieved_docs]).lower()
        
        # 检查回答中的数字是否能在文档中找到
        unsupported = []
        for num in numbers[:5]:  # 只检查前5个数字
            num_clean = num.lower().replace(" ", "")
            if num_clean and len(num_clean) > 1 and num_clean not in all_doc_text:
                unsupported.append(num)
        
        if unsupported:
            result["unsupported_claims"].extend(unsupported)
            result["confidence"] += min(0.3 * len(unsupported), 0.5)
        
        # 3. 判断是否有幻觉
        if result["confidence"] > 0.3:
            result["has_hallucination"] = True
        
        result["confidence"] = min(result["confidence"], 1.0)
        
        return result


class CitationGenerator:
    """
    引用生成器
    为回答生成规范的引用
    """
    
    def generate(
        self,
        answer: str,
        retrieved_docs: List[Dict],
    ) -> List[Dict]:
        """
        生成引用
        
        Returns:
            引用列表，每项包含:
            - index: 引用编号
            - source: 来源文件
            - page: 页码
            - content: 引用内容
            - relevance: 相关度
        """
        citations = []
        
        for i, doc in enumerate(retrieved_docs[:3], 1):  # 最多前3个
            metadata = doc.get("metadata", {})
            
            citation = {
                "index": i,
                "source": metadata.get("source_file", "未知来源"),
                "page": metadata.get("page_number", 0),
                "content": doc["document"][:200] + "..." if len(doc["document"]) > 200 else doc["document"],
                "relevance": doc.get("final_score", doc.get("score", 0)),
                "retrieval_type": doc.get("retrieval_sources", ["unknown"]),
            }
            citations.append(citation)
        
        return citations
    
    def format_citations_text(self, citations: List[Dict]) -> str:
        """格式化引用为文本"""
        if not citations:
            return ""
        
        text = "\n\n📚 参考来源:\n"
        for c in citations:
            text += f"[{c['index']}] {c['source']}"
            if c['page'] > 0:
                text += f", 第{c['page']}页"
            text += f" (相关度: {c['relevance']:.2f})\n"
        
        return text


class RAGPipeline:
    """
    RAG Pipeline
    完整的检索增强生成流程
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client=None,
        top_k: int = 5,
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            retriever: 混合检索器
            llm_client: LLM 客户端（可选，用于生成回答）
            top_k: 检索文档数量
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        
        self.hallucination_detector = HallucinationDetector()
        self.citation_generator = CitationGenerator()
        
        print("✅ RAG Pipeline 初始化完成")
    
    def query(
        self,
        question: str,
        use_llm: bool = True,
    ) -> RAGResponse:
        """
        执行 RAG 查询
        
        Args:
            question: 用户问题
            use_llm: 是否使用 LLM 生成回答
        
        Returns:
            RAGResponse 对象
        """
        print(f"\n🔍 处理查询: {question}")
        
        # 1. 检索相关文档
        print("📚 正在检索相关文档...")
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
        
        if not retrieved_docs:
            return RAGResponse(
                answer="未找到相关文档，无法回答该问题。",
                retrieved_documents=[],
                citations=[],
                confidence=0.0,
                hallucination_check={"has_hallucination": True, "confidence": 1.0},
            )
        
        print(f"✅ 检索到 {len(retrieved_docs)} 个相关文档")
        
        # 2. 构建上下文
        context = self._build_context(retrieved_docs)
        
        # 3. 生成回答
        if use_llm and self.llm_client:
            answer = self._generate_with_llm(question, context)
        else:
            # 不使用 LLM，直接拼接检索结果
            answer = self._generate_simple_answer(question, retrieved_docs)
        
        # 4. 幻觉检测
        hallucination_check = self.hallucination_detector.check(answer, retrieved_docs)
        
        # 5. 生成引用
        citations = self.citation_generator.generate(answer, retrieved_docs)
        
        # 6. 添加引用到回答
        if citations:
            citation_text = self.citation_generator.format_citations_text(citations)
            answer += citation_text
        
        # 7. 计算置信度
        confidence = self._calculate_confidence(
            retrieved_docs,
            hallucination_check,
        )
        
        return RAGResponse(
            answer=answer,
            retrieved_documents=retrieved_docs,
            citations=citations,
            confidence=confidence,
            hallucination_check=hallucination_check,
        )
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建上下文"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "未知")
            page = metadata.get("page_number", 0)
            
            context_parts.append(
                f"[文档 {i}] 来源: {source}, 第{page}页\n{doc['document']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(self, question: str, context: str) -> str:
        """使用 LLM 生成回答"""
        prompt = f"""基于以下参考文档回答问题。如果文档中没有相关信息，请明确说明。

参考文档：
{context}

问题：{question}

请基于上述文档提供准确、详细的回答。回答末尾请标注信息来源。"""
        
        try:
            # 这里调用 LLM
            if hasattr(self.llm_client, 'generate'):
                return self.llm_client.generate(prompt)
            else:
                return self._generate_simple_answer(question, [])
        except Exception as e:
            print(f"⚠️ LLM 生成失败: {e}")
            return self._generate_simple_answer(question, [])
    
    def _generate_simple_answer(
        self,
        question: str,
        retrieved_docs: List[Dict],
    ) -> str:
        """简单回答生成（不使用 LLM）"""
        if not retrieved_docs:
            return "未找到相关信息。"
        
        # 使用最相关的文档
        best_doc = retrieved_docs[0]
        metadata = best_doc.get("metadata", {})
        
        answer = f"根据检索结果：\n\n{best_doc['document'][:500]}"
        
        return answer
    
    def _calculate_confidence(
        self,
        retrieved_docs: List[Dict],
        hallucination_check: Dict,
    ) -> float:
        """计算整体置信度"""
        if not retrieved_docs:
            return 0.0
        
        # 基于检索分数
        avg_score = sum(d.get("final_score", 0) for d in retrieved_docs) / len(retrieved_docs)
        
        # 减去幻觉概率
        hallucination_penalty = hallucination_check.get("confidence", 0)
        
        confidence = avg_score * (1 - hallucination_penalty)
        
        return round(max(0, confidence), 4)
    
    def query_with_verification(
        self,
        question: str,
        verification_threshold: float = 0.5,
    ) -> RAGResponse:
        """
        带验证的查询
        如果检测到幻觉，尝试重新检索或降低置信度
        """
        response = self.query(question)
        
        # 如果置信度太低，标记为需要人工验证
        if response.confidence < verification_threshold:
            response.answer += (
                f"\n\n⚠️ 注意：该回答置信度较低 ({response.confidence:.2f})，"
                "建议人工核实。"
            )
        
        return response


# 与 Qwen 集成的示例
class QwenRAGClient:
    """
    Qwen + RAG 集成客户端
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        rag_pipeline: RAGPipeline = None,
    ):
        self.model_path = model_path
        self.rag_pipeline = rag_pipeline
        
        # 这里可以加载 Qwen 模型
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def chat(self, question: str) -> str:
        """
        带 RAG 的对话
        """
        if self.rag_pipeline:
            # 使用 RAG
            response = self.rag_pipeline.query(question)
            return response.answer
        else:
            # 直接使用 LLM
            return "直接调用 LLM 回答"


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("🧪 RAG Pipeline 测试")
    print("="*60)
    
    # 构建检索器
    retriever = HybridRetriever(
        vector_store=None,  # 实际使用时传入
        keyword_index=None,
    )
    
    # 创建 Pipeline
    pipeline = RAGPipeline(retriever=retriever)
    
    # 测试查询
    test_questions = [
        "什么是赫罗图？",
        "灾变变星的光变曲线有什么特征？",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        
        # 模拟响应
        response = RAGResponse(
            answer=f"这是关于'{question}'的回答...",
            retrieved_documents=[],
            citations=[
                {"index": 1, "source": "test.pdf", "page": 10, "relevance": 0.95},
            ],
            confidence=0.85,
            hallucination_check={"has_hallucination": False, "confidence": 0.1},
        )
        
        print(f"A: {response.answer}")
        print(f"\n置信度: {response.confidence}")
        print(f"幻觉检测: {response.hallucination_check}")
