#!/usr/bin/env python3
"""
使用 Transformers 继续训练 (基于 Llama-3.1-8B)
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 使用无需权限的替代模型
BASE_MODEL = "shenzhi-wang/Llama3.1-8B-Chinese-Chat"
OUTPUT_DIR = "train_qwen/astrosage_continued"

print("="*70)
print("🚀 AstroSage 风格继续训练")
print("="*70)
print(f"基础模型: {BASE_MODEL}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型 (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 数据格式化
def format_prompt(conv):
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
    return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"

class Dataset:
    def __init__(self, path, tokenizer, max_len=2048):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = format_prompt(self.data[idx].get("conversations", []))
        enc = self.tokenizer(text, max_length=self.max_len, padding="max_length", 
                            truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
        }

# 加载数据
train_dataset = Dataset("train_qwen/data/qwen_train.json", tokenizer)
val_dataset = Dataset("train_qwen/data/qwen_val.json", tokenizer)

# 训练
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100),
)

print("\n🚀 开始训练...")
trainer.train()

print("\n💾 保存模型...")
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print(f"✅ 完成! 保存在: {OUTPUT_DIR}/final_model")
