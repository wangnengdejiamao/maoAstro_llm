#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
替代模型训练脚本 (无需申请权限)
================================================================================
由于 Llama-3.1 需要 HuggingFace 权限申请，改用以下替代方案:

可选模型 (无需权限，直接下载):
1. shenzhi-wang/Llama3.1-8B-Chinese-Chat (中文版 Llama3.1) ✅推荐
2. Qwen/Qwen2.5-7B-Instruct (之前训练过，可以继续优化)
3. 01-ai/Yi-1.5-9B-Chat (国产优秀模型)
4. google/gemma-2-9b-it (Google 开源模型)

使用方法:
    python train_alternative_model.py --model shenzhi-wang/Llama3.1-8B-Chinese-Chat

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


# 推荐的无需权限模型
RECOMMENDED_MODELS = {
    "llama3-chinese": {
        "name": "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
        "description": "Llama3.1 中文版，针对中文优化，无需权限",
        "size": "16GB",
        "format": "llama3",
    },
    "qwen2.5": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "阿里通义千问2.5，中文优秀",
        "size": "15GB",
        "format": "qwen2",
    },
    "yi-1.5": {
        "name": "01-ai/Yi-1.5-9B-Chat",
        "description": "零一万物 Yi-1.5，国产优秀模型",
        "size": "18GB",
        "format": "yi",
    },
    "gemma-2": {
        "name": "google/gemma-2-9b-it",
        "description": "Google Gemma 2，开源可商用",
        "size": "18GB",
        "format": "gemma",
    },
}


def print_model_options():
    """打印模型选项"""
    print("\n" + "="*70)
    print("📋 无需权限的可用模型")
    print("="*70)
    
    for key, info in RECOMMENDED_MODELS.items():
        print(f"\n🔹 {key}")
        print(f"   模型: {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   大小: {info['size']}")
    
    print("\n" + "="*70)


def get_model_info(model_key: str):
    """获取模型信息"""
    if model_key in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[model_key]
    
    # 如果直接输入完整模型名
    return {
        "name": model_key,
        "description": "自定义模型",
        "size": "未知",
        "format": "auto",
    }


def format_prompt(messages: List[Dict], format_type: str) -> str:
    """根据不同模型格式化为对话"""
    
    if format_type == "llama3":
        # Llama-3 格式
        formatted = ""
        for msg in messages:
            role = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))
            
            if role == 'system':
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'user':
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'assistant':
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        if not formatted.endswith("assistant<|end_header_id|>\n\n"):
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
    
    elif format_type == "qwen2":
        # Qwen-2 格式
        formatted = ""
        for msg in messages:
            role = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))
            
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        if not formatted.endswith("assistant\n"):
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    elif format_type == "yi":
        # Yi 格式
        formatted = ""
        for msg in messages:
            role = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))
            
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        if not formatted.endswith("assistant\n"):
            formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    elif format_type == "gemma":
        # Gemma 格式
        formatted = ""
        for msg in messages:
            role = msg.get('from', msg.get('role', ''))
            content = msg.get('value', msg.get('content', ''))
            
            if role == 'user':
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            elif role == 'system':
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        
        if not formatted.endswith("<end_of_turn>\n"):
            formatted += "<start_of_turn>model\n"
        
        return formatted
    
    else:
        # 自动检测或使用默认格式
        return format_prompt(messages, "qwen2")


class AstronomyDataset:
    """天文QA数据集"""
    
    def __init__(self, data_path: str, tokenizer, format_type: str = "auto", max_length: int = 2048):
        print(f"   加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.format_type = format_type
        self.max_length = max_length
        
        print(f"   样本数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        conversations = example.get('conversations', [])
        
        # 格式化为对应模型的对话格式
        text = format_prompt(conversations, self.format_type)
        
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


def train_model(
    model_name: str,
    output_dir: str,
    format_type: str,
    epochs: int = 3,
    lora_r: int = 64,
):
    """训练模型"""
    print("="*70)
    print(f"🚀 训练模型: {model_name}")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 tokenizer
    print(f"\n📦 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载模型 (4-bit 量化)
    print("\n📦 加载模型 (4-bit 量化)...")
    print("   这可能需要 10-20 分钟 (下载模型)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print(f"✅ 模型加载完成")
    
    # 3. 准备模型用于训练
    print("\n🔧 准备模型用于 LoRA 训练...")
    model = prepare_model_for_kbit_training(model)
    
    # 4. 配置 LoRA
    print(f"\n🔧 配置 LoRA (r={lora_r})...")
    
    # 根据模型类型调整 target_modules
    if "llama" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj"]
    elif "qwen" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]  # 保守配置
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r // 4,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. 加载数据
    print("\n📂 加载训练数据...")
    train_dataset = AstronomyDataset(
        "train_qwen/data/qwen_train.json",
        tokenizer,
        format_type,
    )
    val_dataset = AstronomyDataset(
        "train_qwen/data/qwen_val.json",
        tokenizer,
        format_type,
    )
    
    # 6. 训练参数
    print("\n🎯 训练配置:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: 2e-4")
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
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
        report_to="none",
    )
    
    # 7. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
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
    config = {
        "base_model": model_name,
        "format_type": format_type,
        "lora_r": lora_r,
        "epochs": epochs,
    }
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print("="*70)
    print(f"\n模型保存位置: {final_output}")
    
    return final_output


def main():
    parser = argparse.ArgumentParser(description="训练替代模型")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3-chinese",
        help="模型名称 (llama3-chinese/qwen2.5/yi-1.5/gemma-2 或完整模型名)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
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
        "--list",
        action="store_true",
        help="列出可用模型",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_model_options()
        return
    
    # 获取模型信息
    model_info = get_model_info(args.model)
    model_name = model_info["name"]
    format_type = model_info["format"]
    
    # 确定输出目录
    if args.output is None:
        output_dir = f"train_qwen/maoAstro-{args.model}"
    else:
        output_dir = args.output
    
    print("\n" + "="*70)
    print("🚀 模型训练配置")
    print("="*70)
    print(f"模型: {model_name}")
    print(f"格式: {format_type}")
    print(f"输出: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA r: {args.lora_r}")
    print("="*70)
    
    # 开始训练
    train_model(
        model_name=model_name,
        output_dir=output_dir,
        format_type=format_type,
        epochs=args.epochs,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
