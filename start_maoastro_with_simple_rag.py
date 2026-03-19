#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maoAstro LLM + 简单RAG (关键词检索)
无需ChromaDB和Whoosh，使用基础JSON索引
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
import torch
from pathlib import Path
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2.5 聊天模板
QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


class SimpleRAGRetriever:
    """简单RAG检索器 (基于关键词)"""
    
    def __init__(self, data_dir: str = "output/qa_hybrid"):
        self.data_dir = Path(data_dir)
        self.qa_data = []
        self.keyword_index = {}
        self._load_data()
    
    def _load_data(self):
        """加载QA数据和索引"""
        print("   加载知识库...")
        
        # 加载QA数据
        dataset_path = self.data_dir / "qa_dataset_full.json"
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            print(f"      加载了 {len(self.qa_data)} 条QA")
        
        # 加载或构建关键词索引
        index_path = self.data_dir / "simple_index.json"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                self.keyword_index = json.load(f)
            print(f"      加载了 {len(self.keyword_index)} 个关键词")
        else:
            print("      ⚠️  关键词索引不存在，实时构建中...")
            self._build_index()
    
    def _build_index(self):
        """实时构建简单索引"""
        self.keyword_index = defaultdict(list)
        
        for i, qa in enumerate(self.qa_data):
            question = qa.get('question', '').lower()
            answer = qa.get('answer', '').lower()
            text = question + " " + answer
            
            # 简单分词
            words = set(text.replace('，', ' ').replace('。', ' ').replace('？', ' ').split())
            for word in words:
                if len(word) > 2:
                    self.keyword_index[word].append(i)
        
        self.keyword_index = dict(self.keyword_index)
        print(f"      构建了 {len(self.keyword_index)} 个关键词")
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            相关QA列表
        """
        query_lower = query.lower()
        
        # 分词
        query_words = set(query_lower.replace('，', ' ').replace('。', ' ').split())
        
        # 统计匹配
        matches = defaultdict(int)
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    matches[idx] += 1
        
        # 排序并返回Top-K
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


class MaoAstroWithRAG:
    """maoAstro + 简单RAG"""
    
    def __init__(self, model_path: str = "train_qwen/output_qwen25/merged_model"):
        print("="*70)
        print("🚀 maoAstro LLM + RAG 启动")
        print("="*70)
        
        # 1. 加载模型
        print(f"\n📦 加载模型: {model_path}")
        
        if not Path(model_path).exists():
            print(f"❌ 模型路径不存在: {model_path}")
            sys.exit(1)
        
        print("   加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        
        print("   加载模型权重 (约2-3分钟)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        print("✅ 模型加载完成!")
        
        # 2. 初始化RAG
        print("\n📚 初始化简单RAG...")
        self.retriever = SimpleRAGRetriever()
        print("✅ RAG初始化完成!")
        
        self._print_help()
    
    def _print_help(self):
        print("\n" + "="*70)
        print("📖 使用说明:")
        print("   直接输入问题  - 使用RAG增强回答")
        print("   /rag <问题>   - 显示检索到的参考资料")
        print("   /direct <问题> - 直接模型推理")
        print("   /help         - 显示帮助")
        print("   /quit         - 退出")
        print("="*70 + "\n")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成回答"""
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_with_rag(self, question: str, show_context: bool = False) -> str:
        """使用RAG生成回答"""
        # 检索相关文档
        retrieved = self.retriever.retrieve(question, top_k=3)
        
        if not retrieved:
            # 没有检索到，直接生成
            system_msg = "你是maoAstro，天文领域专家。"
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return self.generate(prompt), []
        
        # 构建增强提示
        context = "\n\n".join([
            f"[资料{i+1}] Q: {doc['question']}\nA: {doc['answer'][:300]}"
            for i, doc in enumerate(retrieved)
        ])
        
        enhanced_prompt = f"""基于以下参考资料回答问题：

{context}

用户问题：{question}

请结合以上资料回答，如果资料不足请说明。"""
        
        system_msg = "你是maoAstro，天文领域专家。请基于提供的资料回答问题。"
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{enhanced_prompt}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        answer = self.generate(prompt)
        
        if show_context:
            return answer, retrieved
        return answer, retrieved
    
    def chat(self):
        """交互式对话"""
        print("💬 开始对话 (输入 /quit 退出)\n")
        
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
                
                # 显示检索结果模式
                if user_input.startswith("/rag "):
                    question = user_input[5:].strip()
                    print(f"\n🔍 检索相关问题...")
                    answer, sources = self.generate_with_rag(question, show_context=True)
                    
                    print(f"\n📚 检索到的资料:")
                    for i, doc in enumerate(sources, 1):
                        print(f"   [{i}] {doc['question'][:50]}... (匹配度: {doc['score']})")
                    
                    print(f"\n🤖 maoAstro:\n{answer}\n")
                    continue
                
                # 直接推理模式
                if user_input.startswith("/direct "):
                    question = user_input[8:].strip()
                    print(f"\n🤖 直接推理...")
                    
                    system_msg = "你是maoAstro，天文领域专家。"
                    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
                    prompt += "<|im_start|>assistant\n"
                    
                    answer = self.generate(prompt)
                    print(f"\n🤖 maoAstro:\n{answer}\n")
                    continue
                
                # 默认RAG模式
                question = user_input
                print(f"\n🔍 使用RAG检索...")
                answer, sources = self.generate_with_rag(question)
                
                print(f"\n🤖 maoAstro:\n{answer}")
                
                if sources:
                    print(f"\n📚 参考资料: {len(sources)} 条")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="启动 maoAstro + RAG")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_qwen/output_qwen25/merged_model",
        help="模型路径",
    )
    
    args = parser.parse_args()
    
    bot = MaoAstroWithRAG(model_path=args.model_path)
    bot.chat()


if __name__ == "__main__":
    main()
