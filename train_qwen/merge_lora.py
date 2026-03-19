#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage LoRA 权重合并脚本
================================================================================
功能描述:
    将训练好的LoRA权重与基座模型(Qwen2.5-7B)合并，生成完整的模型文件。
    合并后的模型可以直接用于Ollama部署或独立推理。

技术说明:
    - 基座模型: Qwen/Qwen2.5-7B-Instruct
    - LoRA权重: train_qwen/output_qwen25/final_model/
    - 输出模型: train_qwen/output_qwen25/merged_model/
    - 合并方法: PEFT merge_and_unload

显存需求:
    - 合并过程需要约 16GB 显存
    - 或使用 CPU 合并 (较慢但不需要显存)

使用方法:
    # GPU合并 (推荐)
    python train_qwen/merge_lora.py --device cuda
    
    # CPU合并
    python train_qwen/merge_lora.py --device cpu

输出文件:
    - merged_model/config.json
    - merged_model/model.safetensors (约14GB)
    - merged_model/tokenizer.json

作者: AstroSage Team
================================================================================
"""

import os
import sys
import torch
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_path: str = "train_qwen/output_qwen25/final_model",
    output_path: str = "train_qwen/output_qwen25/merged_model",
    device: str = "cuda",
):
    """
    合并LoRA权重到基座模型
    
    Args:
        base_model_name: 基座模型名称或路径
        lora_path: LoRA权重路径
        output_path: 合并后的模型输出路径
        device: 使用的设备 (cuda/cpu)
    """
    print("="*70)
    print("🔧 LoRA 权重合并")
    print("="*70)
    
    # 检查路径
    lora_path = Path(lora_path)
    output_path = Path(output_path)
    
    if not lora_path.exists():
        print(f"❌ LoRA路径不存在: {lora_path}")
        print(f"   请先完成模型训练")
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 基座模型: {base_model_name}")
    print(f"🔌 LoRA权重: {lora_path}")
    print(f"📁 输出目录: {output_path}")
    print(f"💻 使用设备: {device}")
    
    # 加载tokenizer
    print("\n⏳ 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
    )
    print("✅ Tokenizer加载完成")
    
    # 加载基座模型
    print("\n⏳ 加载基座模型...")
    print("   这可能需要几分钟...")
    
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    print("✅ 基座模型加载完成")
    
    # 加载LoRA权重
    print("\n⏳ 加载LoRA权重...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("✅ LoRA权重加载完成")
    
    # 合并权重
    print("\n🔧 合并权重...")
    print("   将LoRA权重合并到基座模型...")
    model = model.merge_and_unload()
    print("✅ 合并完成")
    
    # 保存合并后的模型
    print("\n💾 保存合并后的模型...")
    print(f"   保存到: {output_path}")
    print("   这可能需要几分钟 (模型约14GB)...")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ 模型保存完成!")
    
    # 显示输出文件
    print("\n📁 输出文件:")
    for f in sorted(output_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name:<30} {size_mb:>8.1f} MB")
    
    print("\n" + "="*70)
    print("🎉 合并完成!")
    print(f"   完整模型已保存到: {output_path}")
    print("\n📖 下一步:")
    print("   1. 部署到Ollama: ./start_local_rag.sh")
    print("   2. 直接推理测试: python train_qwen/inference.py")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基座模型名称",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="train_qwen/output_qwen25/final_model",
        help="LoRA权重路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="train_qwen/output_qwen25/merged_model",
        help="输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用的设备",
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
