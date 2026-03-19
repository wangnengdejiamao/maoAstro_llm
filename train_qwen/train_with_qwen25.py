#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage Qwen2.5 LoRA 训练脚本
================================================================================
功能描述:
    使用LoRA(Low-Rank Adaptation)方法微调Qwen2.5-7B模型。
    在保持基座模型不变的情况下，通过少量可训练参数适配天文领域。

技术细节:
    - 基座模型: Qwen/Qwen2.5-7B-Instruct
    - 微调方法: LoRA (r=64, alpha=16)
    - 量化: 4-bit NF4 (节省显存)
    - 可训练参数: ~161M (占总参数2.08%)
    - 显存需求: ~12GB (4-bit量化后)

训练配置:
    - 学习率: 2e-4
    - Batch size: 1
    - 梯度累积: 8
    - Epochs: 3
    - Max length: 2048
    - 优化器: AdamW

输入数据:
    - train_qwen/data/qwen_train.json (3,413条)
    - train_qwen/data/qwen_val.json (380条)

输出模型:
    - LoRA权重: train_qwen/output_qwen25/final_model/
    - 检查点: train_qwen/output_qwen25/checkpoint-*/

使用方法:
    # 设置镜像(国内推荐)
    export HF_ENDPOINT=https://hf-mirror.com
    
    # 开始训练
    python train_qwen/train_with_qwen25.py
    
    # 监控训练
    tail -f train_qwen/training_qwen25_7b.log

依赖:
    - transformers >= 4.35
    - peft >= 0.7
    - bitsandbytes >= 0.41
    - torch >= 2.0

作者: AstroSage Team
创建日期: 2024-03
================================================================================
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_json_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_qwen_prompt(example: Dict) -> str:
    conversations = example.get('conversations', [])
    formatted = ""
    
    for msg in conversations:
        role = msg.get('from', '')
        value = msg.get('value', '')
        
        if role == 'system':
            formatted += f"<|im_start|>system\n{value}<|im_end|>\n"
        elif role == 'user':
            formatted += f"<|im_start|>user\n{value}<|im_end|>\n"
        elif role == 'assistant':
            formatted += f"<|im_start|>assistant\n{value}<|im_end|>\n"
    
    return formatted


class SimpleDataset:
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        text = format_qwen_prompt(example)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


def main():
    print("="*60)
    print("🚀 Qwen2.5 LoRA 训练")
    print("="*60)
    
    # 配置
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 官方模型
    train_data_path = "train_qwen/data/qwen_train.json"
    val_data_path = "train_qwen/data/qwen_val.json"
    output_dir = "train_qwen/output_qwen25"
    
    batch_size = 1
    gradient_accumulation = 8
    learning_rate = 2e-4
    num_epochs = 3
    max_seq_length = 2048
    lora_r = 64
    lora_alpha = 16
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"\n📂 加载训练数据: {train_data_path}")
    train_data = load_json_data(train_data_path)
    print(f"✅ 训练集: {len(train_data)} 条")
    
    print(f"\n📂 加载验证数据: {val_data_path}")
    val_data = load_json_data(val_data_path)
    print(f"✅ 验证集: {len(val_data)} 条")
    
    # 加载模型（从 HuggingFace）
    print(f"\n📦 加载模型: {model_name}")
    print("   首次运行需要下载约 14GB 模型文件...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit量化
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    print(f"✅ 模型加载完成")
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 配置 LoRA
    print(f"\n🔧 配置 LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 创建数据集
    train_dataset = SimpleDataset(train_data, tokenizer, max_seq_length)
    val_dataset = SimpleDataset(val_data, tokenizer, max_seq_length)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n🚀 开始训练...")
    trainer.train()
    
    # 保存
    final_output_dir = Path(output_dir) / "final_model"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\n✅ 训练完成！模型保存在: {final_output_dir}")


if __name__ == "__main__":
    main()
