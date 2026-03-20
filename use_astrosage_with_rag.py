#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
使用 AstroSage 8B (astrosage-local) + RAG 问答系统
================================================================================
由于 astrosage-local 已在 Ollama 中，直接使用 Ollama API + RAG 是最佳方案

使用方法:
    python use_astrosage_with_rag.py

特点:
    - 使用 Ollama 中的 astrosage-local 模型
    - 集成 RAG 检索增强
    - 无需下载或转换模型
    - 支持引用溯源

作者: AstroSage Team
================================================================================
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

import requests


@dataclass
class RAGResponse:
    """RAG响应"""
    answer: str
    sources: List[Dict]
    model_used: str
    latency: float


class AstroSageRAG:
    """AstroSage 8B + RAG 系统"""
    
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "astrosage-local"
    
    def __init__(self, data_dir: str = "output/qa_hybrid"):
        print("="*70)
        print("🚀 AstroSage 8B + RAG 系统启动")
        print("="*70)
        
        # 检查 Ollama
        if not self._check_ollama():
            print("\n❌ Ollama 未运行，请先启动:")
            print("   ollama serve")
            sys.exit(1)
        
        # 检查模型
        if not self._check_model():
            print(f"\n❌ 模型 {self.MODEL_NAME} 未在 Ollama 中")
            print("   可用模型:")
            self._list_models()
            sys.exit(1)
        
        # 初始化RAG
        print("\n📚 初始化 RAG 系统...")
        self.retriever = SimpleRAGRetriever(data_dir)
        
        print("\n✅ 系统启动完成!")
        self._print_help()
    
    def _check_ollama(self) -> bool:
        """检查 Ollama 是否运行"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_model(self) -> bool:
        """检查模型是否存在"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get("models", [])
            return any(m["name"].startswith(self.MODEL_NAME) for m in models)
        except:
            return False
    
    def _list_models(self):
        """列出可用模型"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get("models", [])
            for m in models:
                print(f"   - {m['name']}")
        except:
            pass
    
    def _print_help(self):
        """打印帮助信息"""
        print("""
📖 使用说明:

  直接输入问题    - 使用RAG增强回答
  /rag <问题>     - 显示检索到的参考资料
  /sources        - 查看知识库统计
  /help           - 显示帮助
  /quit           - 退出系统

💡 示例:
  > 什么是灾变变星?
  > /rag 赫罗图上的主序带特征
  > /sources
        """)
    
    def query_ollama(self, prompt: str, system: str = "") -> str:
        """
        查询 Ollama 模型
        
        Args:
            prompt: 提示文本
            system: 系统提示
            
        Returns:
            模型生成的回答
        """
        payload = {
            "model": self.MODEL_NAME,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.OLLAMA_URL,
                json=payload,
                timeout=120,
            )
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip(), latency
            else:
                return f"错误: HTTP {response.status_code}", 0
                
        except Exception as e:
            return f"错误: {e}", 0
    
    def generate_with_rag(self, question: str, show_sources: bool = False) -> RAGResponse:
        """
        使用 RAG 生成回答
        
        Args:
            question: 用户问题
            show_sources: 是否显示来源
            
        Returns:
            RAGResponse 对象
        """
        # 1. 检索相关文档
        retrieved = self.retriever.retrieve(question, top_k=3)
        
        # 2. 构建增强提示
        if retrieved:
            context = "\n\n".join([
                f"[参考资料 {i+1}]\n问题: {doc['question']}\n答案: {doc['answer'][:300]}"
                for i, doc in enumerate(retrieved)
            ])
            
            enhanced_prompt = f"""基于以下参考资料回答问题：

{context}

用户问题：{question}

请结合以上资料回答，如果资料不足请说明。回答要专业、准确。"""
        else:
            enhanced_prompt = question
            retrieved = []
        
        # 3. 系统提示
        system_msg = "你是 AstroSage，天文领域专家AI助手。请用专业、准确的中文回答。"
        
        # 4. 生成回答
        answer, latency = self.query_ollama(enhanced_prompt, system_msg)
        
        return RAGResponse(
            answer=answer,
            sources=retrieved,
            model_used=self.MODEL_NAME,
            latency=latency,
        )
    
    def chat(self):
        """交互式对话"""
        print("\n💬 开始对话 (输入 /quit 退出)\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("\n👋 再见!")
                    break
                
                if user_input == "/help":
                    self._print_help()
                    continue
                
                if user_input == "/sources":
                    self.show_sources()
                    continue
                
                # 显示检索详情
                if user_input.startswith("/rag "):
                    question = user_input[5:].strip()
                    print(f"\n🔍 检索相关内容...")
                    
                    response = self.generate_with_rag(question, show_sources=True)
                    
                    print(f"\n📚 找到 {len(response.sources)} 条相关资料:")
                    for i, src in enumerate(response.sources, 1):
                        print(f"   [{i}] {src['question'][:60]}...")
                    
                    print(f"\n🤖 AstroSage ({response.latency:.2f}s):")
                    print(response.answer)
                    print()
                    continue
                
                # 默认RAG模式
                question = user_input
                print(f"\n🔍 使用RAG检索...")
                
                response = self.generate_with_rag(question)
                
                print(f"\n🤖 AstroSage ({response.latency:.2f}s):")
                print(response.answer)
                
                if response.sources:
                    print(f"\n📚 参考了 {len(response.sources)} 条资料")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")
    
    def show_sources(self):
        """显示知识库信息"""
        print("\n" + "="*70)
        print("📚 知识库统计")
        print("="*70)
        
        stats = self.retriever.get_stats()
        print(f"\n总QA数: {stats['total']:,}")
        print(f"关键词数: {stats['keywords']:,}")
        
        print("\n✅ 知识库已加载，可以开始问答")


class SimpleRAGRetriever:
    """简单RAG检索器"""
    
    def __init__(self, data_dir: str = "output/qa_hybrid"):
        self.data_dir = Path(data_dir)
        self.qa_data = []
        self.keyword_index = {}
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        dataset_path = self.data_dir / "qa_dataset_full.json"
        
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
        
        # 加载或构建索引
        index_path = self.data_dir / "simple_index.json"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                self.keyword_index = json.load(f)
        else:
            self._build_index()
    
    def _build_index(self):
        """构建索引"""
        self.keyword_index = defaultdict(list)
        
        for i, qa in enumerate(self.qa_data):
            text = (qa.get('question', '') + " " + qa.get('answer', '')).lower()
            words = set(text.replace('，', ' ').replace('。', ' ').split())
            
            for word in words:
                if len(word) > 2:
                    self.keyword_index[word].append(i)
        
        self.keyword_index = dict(self.keyword_index)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关文档"""
        query_lower = query.lower()
        query_words = set(query_lower.replace('，', ' ').replace('。', ' ').split())
        
        matches = defaultdict(int)
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    matches[idx] += 1
        
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for idx, score in sorted_matches[:top_k]:
            if idx < len(self.qa_data):
                qa = self.qa_data[idx]
                results.append({
                    "question": qa.get('question', ''),
                    "answer": qa.get('answer', ''),
                    "source": qa.get('source', 'unknown'),
                    "score": score,
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total": len(self.qa_data),
            "keywords": len(self.keyword_index),
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AstroSage 8B + RAG")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/qa_hybrid",
        help="知识库数据目录",
    )
    
    args = parser.parse_args()
    
    # 启动系统
    system = AstroSageRAG(data_dir=args.data_dir)
    system.chat()


if __name__ == "__main__":
    main()
