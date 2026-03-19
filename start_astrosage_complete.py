#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage 完整启动脚本 (模型 + RAG)
================================================================================
功能描述:
    启动训练好的AstroSage模型，并集成RAG双轨检索系统进行问答。
    支持直接推理和RAG增强两种模式。

工作流程:
    1. 加载训练好的AstroSage模型 (merged_model)
    2. 初始化RAG系统 (向量存储 + 关键词索引)
    3. 交互式问答循环
    4. 根据问题类型选择:
       - 简单问题: 直接模型推理
       - 专业问题: RAG检索 + 模型生成

使用方式:
    python start_astrosage_complete.py

交互命令:
    - 直接输入问题: 进行问答
    - /rag <问题>: 强制使用RAG模式
    - /direct <问题>: 强制直接推理
    - /quit: 退出

作者: AstroSage Team
================================================================================
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入RAG系统
sys.path.insert(0, str(Path(__file__).parent))
from rag_system.retrieval.rag_pipeline import RAGPipeline
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.vector_store.chroma_store import VectorStore
from rag_system.inverted_index.keyword_index import KeywordIndex


class AstroSageComplete:
    """AstroSage完整系统 (模型 + RAG)"""
    
    def __init__(
        self,
        model_path: str = "train_qwen/output_qwen25/merged_model",
        rag_knowledge_base: str = "output/qa_hybrid",
        use_rag: bool = True,
    ):
        """
        初始化AstroSage完整系统
        
        Args:
            model_path: 训练好的模型路径
            rag_knowledge_base: RAG知识库路径
            use_rag: 是否启用RAG
        """
        print("="*70)
        print("🚀 AstroSage 完整系统启动")
        print("="*70)
        
        self.model_path = model_path
        self.use_rag = use_rag
        
        # 1. 加载模型
        self._load_model()
        
        # 2. 初始化RAG (如果启用)
        if use_rag:
            self._init_rag(rag_knowledge_base)
        else:
            self.rag_pipeline = None
            print("\n⚠️ RAG系统已禁用")
        
        print("\n" + "="*70)
        print("✅ 系统启动完成!")
        print("="*70)
        self._print_help()
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"\n📦 加载模型: {self.model_path}")
        
        if not Path(self.model_path).exists():
            print(f"❌ 模型路径不存在: {self.model_path}")
            print(f"   请确认模型已训练并合并")
            sys.exit(1)
        
        # 加载tokenizer
        print("   加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",  # Qwen使用left padding
        )
        
        # 确保pad_token设置正确
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"   设置 pad_token = eos_token")
        
        # 检查chat_template
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            print("   ⚠️ 使用默认Qwen聊天模板")
            # Qwen2.5默认模板
            self.tokenizer.chat_template = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        
        # 加载模型
        print("   加载模型权重...")
        print("   这可能需要 2-3 分钟...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        print("✅ 模型加载完成")
        print(f"   模型类型: {self.model.config.model_type}")
        print(f"   词汇表大小: {len(self.tokenizer)}")
    
    def _init_rag(self, knowledge_base_path: str):
        """初始化RAG系统"""
        print("\n📚 初始化RAG系统...")
        
        kb_path = Path(knowledge_base_path)
        
        if not kb_path.exists():
            print(f"⚠️  知识库不存在: {knowledge_base_path}")
            print(f"   RAG功能将不可用")
            self.rag_pipeline = None
            return
        
        try:
            # 初始化向量存储
            print("   初始化向量存储...")
            vector_store = VectorStore(
                collection_name="astro_qa",
                persist_directory=str(kb_path / "chroma_db"),
            )
            
            # 初始化关键词索引
            print("   初始化关键词索引...")
            keyword_index = KeywordIndex(
                index_dir=str(kb_path / "keyword_index"),
            )
            
            # 初始化混合检索器
            print("   初始化混合检索器...")
            retriever = HybridRetriever(
                vector_store=vector_store,
                keyword_index=keyword_index,
                vector_weight=0.6,
                keyword_weight=0.4,
            )
            
            # 初始化RAG管道
            print("   初始化RAG管道...")
            self.rag_pipeline = RAGPipeline(
                retriever=retriever,
                llm_client=None,  # 使用本地模型
            )
            
            print("✅ RAG系统初始化完成")
            
        except Exception as e:
            print(f"⚠️  RAG初始化失败: {e}")
            print(f"   将使用直接推理模式")
            self.rag_pipeline = None
    
    def _print_help(self):
        """打印帮助信息"""
        print("""
📖 使用说明:

  直接输入问题     - 自动选择模式进行回答
  /rag <问题>      - 强制使用RAG检索增强
  /direct <问题>   - 强制直接模型推理
  /help            - 显示帮助
  /quit            - 退出系统

💡 示例:
  > 什么是灾变变星?
  > /rag 赫罗图上的主序带特征
  > /direct 写一个Python函数
        """)
    
    def generate_direct(self, question: str, max_length: int = 512) -> str:
        """
        直接模型推理
        
        Args:
            question: 问题文本
            max_length: 最大生成长度
            
        Returns:
            生成的回答
        """
        # 清理输入
        question = str(question).strip()
        if not question:
            return "请输入有效的问题"
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 构建提示 (Qwen2.5 格式)
        messages = [
            {"role": "system", "content": "你是AstroSage，一个天文领域专家AI助手。请用专业、准确的中文回答天文问题。"},
            {"role": "user", "content": question},
        ]
        
        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,  # 模板已包含
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码 (只取生成的部分)
        input_length = inputs.input_ids.shape[1]
        answer = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        
        return answer.strip()
    
    def generate_with_rag(self, question: str) -> Dict:
        """
        RAG检索增强生成
        
        Args:
            question: 问题文本
            
        Returns:
            包含回答和引用的字典
        """
        if self.rag_pipeline is None:
            # RAG不可用，回退到直接推理
            answer = self.generate_direct(question)
            return {
                "answer": answer,
                "citations": [],
                "mode": "direct (RAG unavailable)",
            }
        
        # 1. 检索相关文档
        retrieved_docs = self.rag_pipeline.retriever.retrieve(question, top_k=3)
        
        # 2. 构建增强提示
        context = "\n\n".join([
            f"[文档 {i+1}] {doc.get('content', '')[:500]}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        enhanced_prompt = f"""基于以下参考资料回答问题:

{context}

问题: {question}

请基于以上资料回答，如果资料不足请说明。"""
        
        # 3. 生成回答
        answer = self.generate_direct(enhanced_prompt, max_length=512)
        
        # 4. 提取引用
        citations = [
            {
                "source": doc.get("source", "未知"),
                "page": doc.get("page", "未知"),
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "citations": citations,
            "mode": "RAG",
            "retrieved_count": len(retrieved_docs),
        }
    
    def should_use_rag(self, question: str) -> bool:
        """
        判断是否应该使用RAG
        
        根据问题的专业性判断是否使用RAG
        """
        # 天文专业关键词
        astro_keywords = [
            "灾变", "变星", "CV", "白矮星", "吸积", "吸积盘",
            "赫罗图", "主序", "巨星", "光度", "温度",
            "光变曲线", "周期", "振幅", "极大", "极小",
            "光谱", "SED", "能谱", "发射线", "吸收线",
            "双星", "质量转移", "洛希瓣", "轨道",
            "消光", "红化", "色余", "距离",
            "X射线", "UV", "红外", "射电",
        ]
        
        # 如果问题包含天文专业词汇，使用RAG
        question_lower = question.lower()
        return any(kw in question_lower for kw in astro_keywords)
    
    def chat(self, user_input: str) -> str:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入
            
        Returns:
            系统回复
        """
        user_input = user_input.strip()
        
        if not user_input:
            return "请输入问题或命令"
        
        # 处理命令
        if user_input == "/help":
            self._print_help()
            return ""
        
        if user_input == "/quit":
            return "__EXIT__"
        
        # 强制RAG模式
        if user_input.startswith("/rag "):
            question = user_input[5:].strip()
            print(f"\n🔍 使用RAG模式回答...")
            result = self.generate_with_rag(question)
            
            response = f"\n🤖 AstroSage:\n{result['answer']}\n"
            
            if result.get('citations'):
                response += "\n📚 参考资料:\n"
                for i, citation in enumerate(result['citations'], 1):
                    response += f"   [{i}] {citation['source']}"
                    if citation.get('page'):
                        response += f" (第{citation['page']}页)"
                    response += "\n"
            
            return response
        
        # 强制直接模式
        if user_input.startswith("/direct "):
            question = user_input[8:].strip()
            print(f"\n🤖 使用直接推理...")
            answer = self.generate_direct(question)
            return f"\n🤖 AstroSage:\n{answer}\n"
        
        # 自动选择模式
        question = user_input
        use_rag = self.should_use_rag(question) and self.rag_pipeline is not None
        
        if use_rag:
            print(f"\n🔍 检测到专业问题，使用RAG增强...")
            result = self.generate_with_rag(question)
            
            response = f"\n🤖 AstroSage:\n{result['answer']}\n"
            
            if result.get('citations'):
                response += "\n📚 参考资料:\n"
                for i, citation in enumerate(result['citations'], 1):
                    response += f"   [{i}] {citation['source']}"
                    if citation.get('page'):
                        response += f" (第{citation['page']}页)"
                    response += "\n"
            
            return response
        else:
            print(f"\n🤖 使用直接推理...")
            answer = self.generate_direct(question)
            return f"\n🤖 AstroSage:\n{answer}\n"
    
    def run_interactive(self):
        """运行交互式会话"""
        print("\n" + "="*70)
        print("💬 开始对话 (输入 /help 查看帮助, /quit 退出)")
        print("="*70 + "\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 You: ").strip()
                
                # 处理输入
                response = self.chat(user_input)
                
                # 检查退出
                if response == "__EXIT__":
                    print("\n👋 再见!")
                    break
                
                # 打印回复
                if response:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动AstroSage完整系统")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_qwen/output_qwen25/merged_model",
        help="模型路径",
    )
    parser.add_argument(
        "--rag-path",
        type=str,
        default="output/qa_hybrid",
        help="RAG知识库路径",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="禁用RAG",
    )
    
    args = parser.parse_args()
    
    # 创建系统
    system = AstroSageComplete(
        model_path=args.model_path,
        rag_knowledge_base=args.rag_path,
        use_rag=not args.no_rag,
    )
    
    # 运行交互式会话
    system.run_interactive()


if __name__ == "__main__":
    main()
