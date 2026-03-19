#!/usr/bin/env python3
"""
RAG + Ollama 本地模型集成
使用 Ollama 运行本地模型，结合 RAG 检索
"""

import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass

from .rag_pipeline import RAGPipeline, HybridRetriever


@dataclass
class OllamaResponse:
    """Ollama 响应结构"""
    answer: str
    done: bool
    context: List[int] = None


class OllamaRAGClient:
    """
    Ollama RAG 客户端
    使用本地 Ollama 模型 + RAG 检索
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        model_name: str = "qwen3",
        ollama_url: str = "http://localhost:11434",
        system_prompt: str = None,
    ):
        """
        初始化
        
        Args:
            rag_pipeline: RAG Pipeline 实例
            model_name: Ollama 模型名称
            ollama_url: Ollama 服务地址
            system_prompt: 系统提示词
        """
        self.rag_pipeline = rag_pipeline
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')
        
        if system_prompt is None:
            self.system_prompt = """你是 AstroSage，一位专业的天文学专家助手。
你精通恒星演化、赫罗图、能谱分布、光变曲线、双星系统等天文知识。
请基于提供的参考信息回答问题，如果信息不足请明确说明。"""
        else:
            self.system_prompt = system_prompt
        
        print(f"✅ Ollama RAG 客户端初始化")
        print(f"   模型: {model_name}")
        print(f"   服务: {ollama_url}")
    
    def check_ollama(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"⚠️ 获取模型列表失败: {e}")
        return []
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        """
        调用 Ollama 生成回答
        """
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream,
        }
        
        try:
            if stream:
                # 流式输出
                response = requests.post(url, json=payload, stream=True, timeout=120)
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                            print(data["response"], end="", flush=True)
                        if data.get("done", False):
                            break
                return full_response
            else:
                # 非流式
                response = requests.post(url, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    return f"请求失败: {response.status_code}"
        except Exception as e:
            return f"错误: {e}"
    
    def chat_with_rag(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_citations: bool = True,
    ) -> Dict:
        """
        带 RAG 的对话
        
        Returns:
            {
                "question": str,
                "answer": str,
                "retrieved_docs": List[Dict],
                "citations": List[Dict],
                "confidence": float,
            }
        """
        print(f"\n🔍 问题: {question}")
        print("="*60)
        
        # 1. RAG 检索
        print("📚 检索相关知识...")
        retrieved_docs = self.rag_pipeline.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "未找到相关知识，无法回答。",
                "retrieved_docs": [],
                "citations": [],
                "confidence": 0.0,
            }
        
        print(f"✅ 检索到 {len(retrieved_docs)} 条相关知识")
        
        # 2. 构建增强的 prompt
        context = self._build_context(retrieved_docs)
        
        prompt = f"""基于以下参考信息回答问题：

参考信息：
{context}

问题：{question}

请基于上述参考信息提供准确、详细的回答。如果参考信息不足，请明确说明。"""
        
        # 3. 调用 Ollama 生成
        print("\n🤖 生成回答...")
        answer = self.generate(prompt, temperature=temperature)
        
        # 4. 幻觉检测
        hallucination_check = self.rag_pipeline.hallucination_detector.check(
            answer, retrieved_docs
        )
        
        # 5. 生成引用
        citations = self.rag_pipeline.citation_generator.generate(answer, retrieved_docs)
        
        # 6. 添加引用
        if include_citations and citations:
            citation_text = self.rag_pipeline.citation_generator.format_citations_text(citations)
            answer += citation_text
        
        # 7. 计算置信度
        confidence = self.rag_pipeline._calculate_confidence(
            retrieved_docs, hallucination_check
        )
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "citations": citations,
            "confidence": confidence,
            "hallucination_check": hallucination_check,
        }
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建上下文"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3], 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "未知")
            
            context_parts.append(
                f"[{i}] 来源: {source}\n{doc['document'][:400]}..."
            )
        
        return "\n\n".join(context_parts)
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "="*60)
        print("🌟 AstroSage - RAG + Ollama 模式")
        print("="*60)
        print(f"使用模型: {self.model_name}")
        print("输入 'quit' 退出，输入 'models' 查看可用模型\n")
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n再见！👋")
                    break
                
                if user_input.lower() == "models":
                    models = self.list_models()
                    print(f"\n可用模型: {', '.join(models)}")
                    continue
                
                # 执行 RAG 对话
                result = self.chat_with_rag(user_input)
                
                print("\n🤖 助手: ")
                print(result["answer"])
                print(f"\n[置信度: {result['confidence']:.2f}]")
                
            except KeyboardInterrupt:
                print("\n\n再见！👋")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def create_ollama_rag_client(
    model_name: str = "qwen3",
    use_vector_store: bool = False,
) -> OllamaRAGClient:
    """
    创建 Ollama RAG 客户端
    """
    from ..vector_store.chroma_store import VectorStore
    from ..inverted_index.keyword_index import KeywordIndex
    from .hybrid_retriever import HybridRetriever
    
    print("="*60)
    print("🔧 构建 RAG + Ollama 系统")
    print("="*60)
    
    # 创建检索器
    if use_vector_store:
        vector_store = VectorStore(
            collection_name="astro_qa",
            persist_directory="rag_system/vector_db",
        )
    else:
        vector_store = None
    
    keyword_index = KeywordIndex(
        index_dir="rag_system/inverted_index_db",
    )
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_index=keyword_index,
        vector_weight=0.6 if vector_store else 0,
        keyword_weight=0.4 if vector_store else 1.0,
    )
    
    # 创建 RAG Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=None,
        top_k=5,
    )
    
    # 创建 Ollama 客户端
    client = OllamaRAGClient(
        rag_pipeline=pipeline,
        model_name=model_name,
    )
    
    # 检查 Ollama
    if not client.check_ollama():
        print("\n⚠️ 警告: Ollama 服务未启动")
        print("请运行: ollama serve")
        print(f"可用模型: {client.list_models()}")
    else:
        print(f"✅ Ollama 服务正常")
        print(f"可用模型: {client.list_models()}")
    
    return client


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG + Ollama")
    parser.add_argument("--model", default="qwen3", help="Ollama 模型名称")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--no-vector", action="store_true", help="不使用向量检索")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = create_ollama_rag_client(
        model_name=args.model,
        use_vector_store=not args.no_vector,
    )
    
    # 启动交互式对话
    client.interactive_chat()
