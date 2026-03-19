#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Ollama 模型导出 + LoRA 微调脚本
================================================================================
功能描述:
    1. 从 Ollama 导出模型 (GGUF 格式)
    2. 转换为 HuggingFace 格式
    3. 使用 LoRA 进行微调
    4. 导出为 Ollama 可用格式

支持的模型:
    - astrosage-local (当前Ollama中的模型)
    - llama31-base
    - 其他 Ollama 安装的模型

使用方法:
    # 导出并微调 astrosage-local
    python export_and_finetune_ollama.py --model astrosage-local

依赖安装:
    pip install llama-cpp-python transformers peft datasets

作者: AstroSage Team
================================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict

# 配置
OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"
EXPORT_DIR = Path("models/exported")
HF_EXPORT_DIR = Path("models/hf_format")
FINETUNE_DIR = Path("train_qwen/ollama_finetuned")


def run_command(cmd: List[str], description: str = "") -> bool:
    """运行命令并显示输出"""
    if description:
        print(f"\n   {description}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 失败: {e}")
        print(f"   错误输出: {e.stderr[:500]}")
        return False


def export_from_ollama(model_name: str) -> Path:
    """
    从 Ollama 导出模型
    
    Ollama 模型存储格式较复杂，这里使用 llama.cpp 进行转换
    """
    print("="*70)
    print(f"📦 从 Ollama 导出模型: {model_name}")
    print("="*70)
    
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 方法1: 使用 ollama show 获取信息并手动导出
    print("\n1️⃣  获取模型信息...")
    
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True,
            text=True,
        )
        modelfile_content = result.stdout
        print("   Modelfile 内容:")
        print("   " + "\n   ".join(modelfile_content.split("\n")[:20]))
        
        # 解析 FROM 行获取 GGUF 文件
        gguf_path = None
        for line in modelfile_content.split("\n"):
            if line.strip().startswith("FROM"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    gguf_path = parts[1]
                    break
        
        if gguf_path and Path(gguf_path).exists():
            print(f"\n   找到 GGUF: {gguf_path}")
            return Path(gguf_path)
        
    except Exception as e:
        print(f"   ⚠️  无法获取 modelfile: {e}")
    
    # 方法2: 直接从 blobs 目录查找
    print("\n2️⃣  从 Ollama blobs 目录查找...")
    
    blobs_dir = OLLAMA_MODELS_DIR / "blobs"
    if not blobs_dir.exists():
        print(f"   ❌ Blobs 目录不存在: {blobs_dir}")
        return None
    
    # 查找最大的文件（通常是 GGUF）
    gguf_files = list(blobs_dir.glob("sha256-*"))
    if not gguf_files:
        print("   ❌ 未找到模型文件")
        return None
    
    # 按大小排序，取最大的几个
    gguf_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    print(f"   找到 {len(gguf_files)} 个 blob 文件")
    for i, f in enumerate(gguf_files[:5], 1):
        size_gb = f.stat().st_size / (1024**3)
        print(f"   [{i}] {f.name[:20]}... ({size_gb:.2f} GB)")
    
    # 选择最大的作为模型文件（通常是 GGUF）
    model_blob = gguf_files[0]
    size_gb = model_blob.stat().st_size / (1024**3)
    
    print(f"\n   选择最大的 blob 作为模型: {size_gb:.2f} GB")
    
    # 复制到导出目录
    export_path = EXPORT_DIR / f"{model_name.replace(':', '_')}.gguf"
    
    if export_path.exists():
        print(f"   模型已存在: {export_path}")
        return export_path
    
    print(f"\n   导出模型中...")
    print(f"   从: {model_blob}")
    print(f"   到: {export_path}")
    
    # 使用硬链接或复制
    try:
        os.link(model_blob, export_path)
        print("   ✅ 硬链接创建成功")
    except:
        import shutil
        print("   使用复制...")
        shutil.copy2(model_blob, export_path)
        print("   ✅ 复制完成")
    
    return export_path


def convert_gguf_to_hf(gguf_path: Path, model_name: str) -> Path:
    """
    将 GGUF 转换为 HuggingFace 格式
    
    使用 llama.cpp 的 convert.py 或 transformers
    """
    print("\n" + "="*70)
    print("🔄 GGUF 转 HuggingFace 格式")
    print("="*70)
    
    HF_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = HF_EXPORT_DIR / model_name.replace(":", "_")
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\n   HF 格式已存在: {output_dir}")
        return output_dir
    
    print(f"\n   源 GGUF: {gguf_path}")
    print(f"   目标目录: {output_dir}")
    
    # 方法: 使用 llama.cpp 的 convert-hf-to-gguf.py 反向转换较复杂
    # 更简单的方法是直接用 transformers 加载 GGUF
    
    print("\n   方法: 使用 ctransformers 或 llama-cpp-python 加载 GGUF")
    print("   由于 GGUF 转 HF 格式较复杂，建议使用以下方案:")
    print()
    print("   方案A: 直接下载 Llama-3.1-8B 基础模型")
    print("      从 HuggingFace 下载 meta-llama/Llama-3.1-8B")
    print()
    print("   方案B: 使用现有的 Qwen2.5-7B 继续训练")
    print("      你的 Qwen2.5 已经训练好了，可以直接使用")
    print()
    
    # 实际方案：直接下载 Llama-3.1-8B-Instruct 进行微调
    return None


def download_llama31_base(output_dir: Path) -> Path:
    """下载 Llama-3.1-8B 基础模型"""
    print("\n" + "="*70)
    print("📥 下载 Llama-3.1-8B-Instruct")
    print("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if any(output_dir.iterdir()):
        print(f"\n   模型已存在: {output_dir}")
        return output_dir
    
    print("\n   使用 huggingface-cli 下载...")
    print("   这可能需要 10-20 分钟 (约 16GB)")
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 使用 snapshot_download
    try:
        from huggingface_hub import snapshot_download
        
        print(f"\n   下载 {model_id}...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("   ✅ 下载完成")
        return output_dir
        
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        print("\n   备选方案: 使用镜像下载")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct")
        return None


def finetune_llama31(
    base_model_path: Path,
    output_name: str = "maoAstro-llama31-8b-lora",
) -> Path:
    """
    使用 LoRA 微调 Llama-3.1-8B
    """
    print("\n" + "="*70)
    print(f"🎯 LoRA 微调: {output_name}")
    print("="*70)
    
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FINETUNE_DIR / output_name
    
    print(f"\n📦 加载基础模型: {base_model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (4-bit 量化)
    print("   4-bit 量化加载...")
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    print("\n🔧 配置 LoRA...")
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
    
    # 加载训练数据
    print("\n📂 加载训练数据...")
    train_data_path = Path("train_qwen/data/qwen_train.json")
    
    if not train_data_path.exists():
        print(f"   ❌ 训练数据不存在: {train_data_path}")
        return None
    
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"   训练样本: {len(train_data)}")
    
    # 数据预处理 (Llama-3.1 格式)
    def format_llama_prompt(example):
        messages = example.get('conversations', [])
        formatted = ""
        
        for msg in messages:
            role = msg.get('from', '')
            content = msg.get('value', '')
            
            if role == 'system':
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'user':
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == 'assistant':
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return formatted
    
    # 简单数据集类
    class SimpleDataset:
        def __init__(self, data, tokenizer, max_length=2048):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            example = self.data[idx]
            text = format_llama_prompt(example)
            
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
    
    dataset = SimpleDataset(train_data, tokenizer)
    
    # 训练参数
    print("\n🚀 开始训练...")
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 保存
    print("\n💾 保存模型...")
    trainer.save_model(output_path / "final_model")
    tokenizer.save_pretrained(output_path / "final_model")
    
    print(f"✅ 训练完成: {output_path}")
    return output_path


def export_to_ollama(hf_model_path: Path, model_name: str = "maoAstro-llama31"):
    """导出为 Ollama 格式"""
    print("\n" + "="*70)
    print(f"📤 导出到 Ollama: {model_name}")
    print("="*70)
    
    # 创建 Modelfile
    modelfile_content = f"""FROM {hf_model_path}

SYSTEM """你是 {model_name}，一个经过天文领域专业训练的AI助手。
你基于 Llama-3.1-8B 模型，使用 3,413 条天文 QA 数据进行了微调。
请用专业、准确的中文回答天文问题。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""
    
    modelfile_path = hf_model_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\n   Modelfile 已创建: {modelfile_path}")
    print(f"\n   创建 Ollama 模型:")
    print(f"   ollama create {model_name} -f {modelfile_path}")
    
    # 自动执行
    result = run_command(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        "创建 Ollama 模型",
    )
    
    if result:
        print(f"\n   ✅ Ollama 模型创建成功!")
        print(f"   使用: ollama run {model_name}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Ollama 模型导出与微调")
    parser.add_argument(
        "--model",
        type=str,
        default="astrosage-local",
        help="Ollama 模型名称",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="跳过导出，直接下载 Llama-3.1",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="跳过微调，仅导出",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 Ollama 模型导出 + LoRA 微调")
    print("="*70)
    
    if not args.skip_export:
        # 步骤1: 从 Ollama 导出
        gguf_path = export_from_ollama(args.model)
        
        if gguf_path:
            print(f"\n✅ 导出成功: {gguf_path}")
        else:
            print("\n⚠️  导出失败，将直接下载 Llama-3.1")
    
    # 步骤2: 获取基础模型
    llama31_path = HF_EXPORT_DIR / "Llama-3.1-8B-Instruct"
    
    if not llama31_path.exists() or not any(llama31_path.iterdir()):
        llama31_path = download_llama31_base(llama31_path)
    
    if not llama31_path or not llama31_path.exists():
        print("\n❌ 无法获取基础模型")
        print("\n建议:")
        print("  1. 手动下载: huggingface-cli download meta-llama/Llama-3.1-8B-Instruct")
        print("  2. 或使用现有模型继续训练 Qwen2.5")
        return
    
    # 步骤3: 微调
    if not args.skip_finetune:
        finetuned_path = finetune_llama31(llama31_path)
        
        if finetuned_path:
            # 步骤4: 导出到 Ollama
            export_to_ollama(finetuned_path / "final_model")
    
    print("\n" + "="*70)
    print("✅ 流程结束")
    print("="*70)


if __name__ == "__main__":
    main()
