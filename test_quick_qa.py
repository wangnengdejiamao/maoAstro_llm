#!/usr/bin/env python3
"""
快速测试 AstroSage 问答功能
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "train_qwen/output_qwen25/merged_model"

print("="*60)
print("🧪 AstroSage 快速问答测试")
print("="*60)

# 加载模型
print(f"\n📦 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print("✅ 模型加载完成\n")

# 测试问题
test_questions = [
    "什么是灾变变星?",
    "赫罗图上的主序带是什么?",
]

for question in test_questions:
    print(f"Q: {question}")
    
    # 使用chat template
    messages = [
        {"role": "system", "content": "你是AstroSage，天文领域专家。"},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"A: {answer.strip()[:200]}...\n")
    print("-"*60)

print("\n✅ 测试完成!")
