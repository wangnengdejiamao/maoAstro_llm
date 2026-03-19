#!/usr/bin/env python3
"""
快速测试训练好的模型
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "train_qwen/output_qwen25/merged_model"

# 天文测试问题
TEST_QUESTIONS = [
    "什么是灾变变星？",
    "赫罗图上的主序带代表什么？",
    "如何测量变星的光变周期？",
    "白矮星是如何形成的？",
]

print("="*60)
print("🧪 AstroSage 模型快速测试")
print("="*60)

# 加载模型
print(f"\n📦 加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("✅ 模型加载完成\n")

# 测试每个问题
for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {question}")
    print('='*60)
    
    # 构建提示
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 解码
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nA: {answer.strip()}")

print("\n" + "="*60)
print("✅ 测试完成!")
print("="*60)
