#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Llama-3.1-8B-Instruct LoRA 微调脚本 (天文领域)
================================================================================
功能描述:
    使用天文QA数据对 Llama-3.1-8B-Instruct 进行 LoRA 微调
    创建 maoAstro-Llama31-8B 模型

特点:
    - 4-bit 量化训练 (节省显存)
    - LoRA r=64 (高效微调)
    - 使用现有的 3,413 条天文QA数据
    - 支持导出到 Ollama

硬件要求:
    - GPU: 12GB+ 显存 (4-bit量化后)
    - 内存: 32GB+
    - 磁盘: 20GB 可用空间

使用方法:
    # 基础用法
    python train_llama31_lora.py
    
    # 指定参数
    python train_llama31_lora.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --epochs 3 \
        --output maoAstro-Llama31-8B

训练时间: 约 4-8 小时 (取决于GPU)

作者: AstroSage Team
================================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


@dataclass
class TrainingConfig:
    """训练配置"""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    train_data: str = "train_qwen/data/qwen_train.json"
    val_data: str = "train_qwen/data/qwen_val.json"
    output_dir: str = "train_qwen/maoAstro-Llama31-8B"
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # 量化配置
    load_in_4bit: bool = True


def format_llama3_prompt(conversations: List[Dict]) -> str:
    """
    格式化为 Llama-3.1 对话格式
    
    Llama-3.1 格式:
    <|start_header_id|>system<|end_header_id|>
    
    {system_message}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    
    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    
    {assistant_message}<|eot_id|>
    """
    formatted = ""
    
    for msg in conversations:
        role = msg.get('from', '')
        content = msg.get('value', '')
        
        if role == 'system':
            formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == 'user':
            formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == 'assistant':
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # 添加生成提示
    if not formatted.endswith("assistant<|end_header_id|>\n\n"):
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return formatted


class AstronomyDataset:
    """天文QA数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        print(f"   加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"   样本数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        conversations = example.get('conversations', [])
        
        # 格式化为 Llama-3.1 格式
        text = format_llama3_prompt(conversations)
        
        # 编码
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


def train_maoastro_llama31(config: TrainingConfig):
    """训练 maoAstro-Llama31-8B"""
    print("="*70)
    print("🚀 maoAstro-Llama31-8B LoRA 训练")
    print("="*70)
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 tokenizer
    print(f"\n📦 加载 Tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载模型 (4-bit 量化)
    print("\n📦 加载模型 (4-bit 量化)...")
    print("   这可能需要 5-10 分钟 (下载约 8GB)...")
    
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    print(f"✅ 模型加载完成")
    print(f"   设备: {model.device}")
    
    # 3. 准备模型用于训练
    print("\n🔧 准备模型用于 LoRA 训练...")
    model = prepare_model_for_kbit_training(model)
    
    # 4. 配置 LoRA
    print(f"\n🔧 配置 LoRA:")
    print(f"   r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. 加载数据
    print("\n📂 加载训练数据...")
    train_dataset = AstronomyDataset(config.train_data, tokenizer, config.max_seq_length)
    val_dataset = AstronomyDataset(config.val_data, tokenizer, config.max_seq_length)
    
    # 6. 训练参数
    print("\n🎯 训练配置:")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max seq length: {config.max_seq_length}")
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
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
    
    # 7. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # 8. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 9. 开始训练
    print("\n" + "="*70)
    print("🚀 开始训练!")
    print("="*70)
    
    trainer.train()
    
    # 10. 保存模型
    print("\n💾 保存模型...")
    final_output = output_path / "final_model"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    # 保存配置
    config_dict = {
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "num_epochs": config.num_epochs,
        "training_samples": len(train_dataset),
    }
    
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"\n模型保存位置: {final_output}")
    print(f"\n下一步:")
    print(f"  1. 测试模型: python inference_llama31.py --model {final_output}")
    print(f"  2. 导出Ollama: python export_to_ollama.py --model {final_output}")
    print(f"  3. 合并权重: python merge_lora_llama31.py --model {final_output}")
    
    return final_output


def main():
    parser = argparse.ArgumentParser(description="训练 maoAstro-Llama31-8B")
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="基础模型",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_qwen/maoAstro-Llama31-8B",
        help="输出目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批次大小",
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output,
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
    )
    
    # 开始训练
    train_maoastro_llama31(config)


if __name__ == "__main__":
    main()
