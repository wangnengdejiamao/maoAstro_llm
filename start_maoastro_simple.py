#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maoAstro LLM - 简化启动脚本 (绕过Jinja2和RAG问题)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2.5 默认模板 (手动定义，避免Jinja2问题)
QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful assistant.<|im_end|>
' }}{% endif %}{{ '<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>
' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


class MaoAstroLLM:
    """maoAstro LLM 简化版"""
    
    def __init__(self, model_path: str = "train_qwen/output_qwen25/merged_model"):
        print("="*70)
        print("🚀 maoAstro LLM 启动")
        print("="*70)
        
        print(f"\n📦 加载模型: {model_path}")
        
        if not Path(model_path).exists():
            print(f"❌ 模型路径不存在: {model_path}")
            sys.exit(1)
        
        # 加载 tokenizer
        print("   加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 手动设置模板（避免Jinja2问题）
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
        print(f"   类型: {self.model.config.model_type}")
        print(f"   词汇表: {len(self.tokenizer)}")
        self._print_help()
    
    def _print_help(self):
        print("\n" + "="*70)
        print("📖 使用说明:")
        print("   直接输入问题 - 获得回答")
        print("   /help        - 显示帮助")
        print("   /quit        - 退出")
        print("="*70 + "\n")
    
    def build_prompt(self, question: str) -> str:
        """手动构建prompt (避免apply_chat_template的Jinja2问题)"""
        system_msg = "你是 maoAstro LLM，一个天文领域专家AI助手。请用专业、准确的中文回答天文问题。"
        
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def generate(self, question: str, max_tokens: int = 512) -> str:
        """生成回答"""
        # 清理输入
        question = str(question).strip()
        if not question:
            return "请输入有效的问题"
        
        # 构建prompt
        prompt = self.build_prompt(question)
        
        # 编码 - 使用列表包装确保格式正确
        inputs = self.tokenizer(
            [prompt],  # 必须是列表
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,  # 我们已经在prompt中添加了特殊token
        )
        
        # 移动到设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 如果没有attention_mask，创建一个
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # 使用**解包传递所有参数
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码（只取新生成的部分）
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
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
                
                # 生成回答
                print("🤖 maoAstro: ", end="", flush=True)
                answer = self.generate(user_input)
                print(answer)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="启动 maoAstro LLM")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_qwen/output_qwen25/merged_model",
        help="模型路径",
    )
    
    args = parser.parse_args()
    
    # 启动
    bot = MaoAstroLLM(model_path=args.model_path)
    bot.chat()


if __name__ == "__main__":
    main()
