#!/usr/bin/env python3
"""
人工评估 maoAstro LLM - 实际问答测试
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 天文专业测试问题（带参考答案）
TEST_CASES = [
    {
        "question": "什么是灾变变星（CV）？",
        "key_points": ["白矮星", "伴星", "双星", "吸积", "爆发"],
    },
    {
        "question": "赫罗图上的主序带代表什么？",
        "key_points": ["主序", "氢燃烧", "稳定", "对角", "太阳"],
    },
    {
        "question": "如何测量变星的光变周期？",
        "key_points": ["极大", "极小", "时间间隔", "傅里叶", "周期"],
    },
    {
        "question": "白矮星是如何形成的？",
        "key_points": ["中小质量", "演化", "行星状星云", "核心", "坍缩"],
    },
    {
        "question": "什么是吸积盘？",
        "key_points": ["物质", "旋转", "盘状", "引力", "双星"],
    },
]

print("="*70)
print("🧪 maoAstro LLM 人工评估")
print("="*70)

# 加载模型
MODEL_PATH = "train_qwen/output_qwen25/merged_model"
print(f"\n📦 加载模型: {MODEL_PATH}")

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

# 评估每个问题
results = []
for i, test in enumerate(TEST_CASES, 1):
    question = test["question"]
    key_points = test["key_points"]
    
    print(f"\n{'='*70}")
    print(f"Q{i}: {question}")
    print('='*70)
    
    # 构建prompt
    system_msg = "你是 maoAstro LLM，天文领域专家。请用专业、准确的中文回答。"
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    # 生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    import time
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start
    
    # 解码
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 检查关键知识点
    answer_lower = answer.lower()
    matched = sum(1 for kp in key_points if kp in answer_lower)
    coverage = matched / len(key_points)
    
    print(f"\n🤖 回答 ({latency:.2f}s):")
    print(answer[:500] + "..." if len(answer) > 500 else answer)
    
    print(f"\n📊 评估:")
    print(f"   关键知识点覆盖: {matched}/{len(key_points)} ({coverage*100:.0f}%)")
    print(f"   回答长度: {len(answer)} 字符")
    print(f"   推理时间: {latency:.2f} 秒")
    
    # 人工评分提示
    print(f"\n💡 请人工评分 (1-5):")
    print(f"   5=非常准确专业, 3=基本正确, 1=完全错误")
    
    results.append({
        "question": question,
        "answer": answer,
        "coverage": coverage,
        "latency": latency,
    })

# 总结
print("\n" + "="*70)
print("📈 评估总结")
print("="*70)

avg_coverage = sum(r["coverage"] for r in results) / len(results)
avg_latency = sum(r["latency"] for r in results) / len(results)

print(f"\n平均关键知识点覆盖: {avg_coverage*100:.1f}%")
print(f"平均推理时间: {avg_latency:.2f} 秒")

if avg_coverage >= 0.7:
    print("\n🌟 评级: 优秀 - 模型掌握良好")
elif avg_coverage >= 0.5:
    print("\n✅ 评级: 良好 - 基本掌握但有提升空间")
else:
    print("\n⚠️  评级: 需要改进 - 可能训练不足或模型有问题")

print("\n💡 建议:")
print("   1. 检查回答是否使用了正确的专业术语")
print("   2. 对比参考答案看是否有事实错误")
print("   3. 如果推理时间>3秒，检查GPU是否正常工作")
