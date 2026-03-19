#!/usr/bin/env python3
"""
maoAstro-Llama31-8B 推理测试脚本
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, use_lora: bool = True):
    """加载模型"""
    print(f"📦 加载模型: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if use_lora:
        # 需要同时加载基础模型和 LoRA 权重
        # 这里假设已经合并或单独保存
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_tokens: int = 512) -> str:
    """生成回答"""
    # Llama-3.1 格式
    system_msg = "你是 maoAstro，天文领域专家。请用专业、准确的中文回答。"
    
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 编码
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()


def interactive_chat(model, tokenizer):
    """交互式对话"""
    print("\n💬 maoAstro-Llama31 交互模式 (输入 /quit 退出)\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/quit":
                print("\n👋 再见!")
                break
            
            print("🤖 maoAstro: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="maoAstro-Llama31 推理")
    parser.add_argument(
        "--model",
        type=str,
        default="train_qwen/maoAstro-Llama31-8B/final_model",
        help="模型路径",
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 maoAstro-Llama31-8B 推理测试")
    print("="*70)
    
    # 加载模型
    model, tokenizer = load_model(args.model)
    print("✅ 模型加载完成\n")
    
    # 测试问题
    test_questions = [
        "什么是灾变变星?",
        "赫罗图上的主序带代表什么?",
        "如何测量变星的光变周期?",
    ]
    
    print("🧪 运行测试问题:\n")
    for q in test_questions:
        print(f"Q: {q}")
        a = generate_response(model, tokenizer, q)
        print(f"A: {a[:200]}...\n")
    
    # 进入交互模式
    print("="*70)
    interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
