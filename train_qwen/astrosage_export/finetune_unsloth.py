#!/usr/bin/env python3
"""
使用 Unsloth 微调 AstroSage 8B
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer

GGUF_MODEL = "train_qwen/astrosage_export/astrosage-8b-Q4_K_M.gguf"
OUTPUT_DIR = "train_qwen/astrosage_finetuned"

print("="*70)
print("🚀 Unsloth 微调 AstroSage 8B")
print("="*70)

# 加载模型
print("\n📦 加载 GGUF 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=GGUF_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 添加 LoRA
print("\n🔧 配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 加载数据
print("\n📂 加载训练数据...")
with open("train_qwen/data/qwen_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# 格式化数据
def format_data(item):
    conv = item.get("conversations", [])
    text = ""
    for msg in conv:
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role == "system":
            text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    return {"text": text + "<|start_header_id|>assistant<|end_header_id|>\n\n"}

dataset = [format_data(item) for item in train_data]
print(f"   样本数: {len(dataset)}")

# 训练参数
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=200,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    output_dir=OUTPUT_DIR,
    report_to="none",
)

# 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=args,
)

print("\n🚀 开始训练...")
trainer.train()

# 保存
print("\n💾 保存模型...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged", tokenizer)
print(f"✅ 完成! 保存在: {OUTPUT_DIR}/merged")
