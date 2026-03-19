#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
导出 AstroSage 8B 用于 Transformers + LoRA 微调
================================================================================

由于 astrosage-local 是 GGUF 格式，要使用和 Qwen 相同的微调方式，需要：
1. 找到 GGUF 文件
2. 反量化/转换（使用 llama.cpp 工具）
3. 或用兼容的基础模型 + LoRA 叠加

推荐方案:
    方案A: 直接使用 Llama-3.1-8B-Instruct + 加载 astrosage 适配器
    方案B: 导出 GGUF 后用 llama.cpp 进行 LoRA 微调
    方案C: 使用 unsloth 直接微调 GGUF (推荐)

使用方法:
    python export_astrosage_for_finetune.py --method [auto|gguf|unsloth]

作者: AstroSage Team
================================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# 工作目录
WORK_DIR = Path("train_qwen/astrosage_export")
OLLAMA_DIR = Path.home() / ".ollama" / "models"


def find_astrosage_gguf() -> Optional[Path]:
    """找到 astrosage-local 的 GGUF 文件"""
    print("="*70)
    print("🔍 查找 astrosage-local GGUF 文件")
    print("="*70)
    
    blobs_dir = OLLAMA_DIR / "blobs"
    if not blobs_dir.exists():
        print(f"❌ Blobs 目录不存在")
        return None
    
    # 查找大于 4GB 的 blob 文件
    blobs = [f for f in blobs_dir.glob("sha256-*") if f.stat().st_size > 4 * 1024**3]
    blobs.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    print(f"\n找到 {len(blobs)} 个模型文件:")
    for i, blob in enumerate(blobs[:3], 1):
        size_gb = blob.stat().st_size / (1024**3)
        print(f"   [{i}] {blob.name} ({size_gb:.2f} GB)")
    
    if not blobs:
        return None
    
    # 选择最大的（通常是主模型）
    model_file = blobs[0]
    size_gb = model_file.stat().st_size / (1024**3)
    
    print(f"\n✅ 选择模型文件: {size_gb:.2f} GB")
    return model_file


def export_gguf():
    """导出 GGUF 文件"""
    print("\n" + "="*70)
    print("📦 导出 GGUF 文件")
    print("="*70)
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # 找到模型文件
    source_file = find_astrosage_gguf()
    if not source_file:
        print("❌ 无法找到模型文件")
        return None
    
    # 目标路径
    target_file = WORK_DIR / "astrosage-8b-Q4_K_M.gguf"
    
    if target_file.exists():
        print(f"\n   文件已存在: {target_file}")
        print(f"   大小: {target_file.stat().st_size / (1024**3):.2f} GB")
        return target_file
    
    # 创建硬链接（节省空间）
    print(f"\n   创建链接...")
    print(f"   源: {source_file}")
    print(f"   目标: {target_file}")
    
    try:
        os.link(source_file, target_file)
        print("   ✅ 硬链接创建成功")
    except:
        print("   复制文件中...")
        shutil.copy2(source_file, target_file)
        print("   ✅ 复制完成")
    
    return target_file


def create_unsloth_finetune_script():
    """创建使用 unsloth 微调的脚本"""
    print("\n" + "="*70)
    print("📝 创建 Unsloth 微调脚本")
    print("="*70)
    
    script_content = '''#!/usr/bin/env python3
"""
使用 Unsloth 微调 AstroSage 8B (GGUF 格式)
Unsloth 可以直接微调 GGUF，速度快且节省显存
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
import json

# 配置
GGUF_MODEL = "train_qwen/astrosage_export/astrosage-8b-Q4_K_M.gguf"
OUTPUT_DIR = "train_qwen/astrosage_unsloth_finetuned"
MAX_SEQ_LENGTH = 2048

print("="*70)
print("🚀 Unsloth 微调 AstroSage 8B")
print("="*70)

# 1. 加载模型
print("\\n📦 加载 GGUF 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=GGUF_MODEL,  # 直接加载 GGUF
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # 自动检测
    load_in_4bit=True,  # 4-bit 量化
)

print("✅ 模型加载完成")

# 2. 添加 LoRA
print("\\n🔧 配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. 加载训练数据
def load_dataset():
    with open("train_qwen/data/qwen_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 转换为对话格式
    formatted = []
    for item in data:
        conv = item.get("conversations", [])
        text = ""
        for msg in conv:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            
            if role == "system":
                text += f"<|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>"
            elif role == "user":
                text += f"<|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>"
            elif role == "assistant":
                text += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{content}<|eot_id|>"
        
        formatted.append({"text": text})
    
    return formatted

print("\\n📂 加载训练数据...")
dataset = load_dataset()
print(f"   样本数: {len(dataset)}")

# 4. 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=100,  # 可以根据需要调整
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=OUTPUT_DIR,
    report_to="none",
)

# 5. 创建 Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=training_args,
)

# 6. 开始训练
print("\\n" + "="*70)
print("🚀 开始训练!")
print("="*70)

trainer.train()

# 7. 保存模型
print("\\n💾 保存模型...")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged_model",
    tokenizer,
    save_method="merged_16bit",
)

print("\\n✅ 训练完成!")
print(f"模型保存在: {OUTPUT_DIR}/merged_model")
'''
    
    script_path = WORK_DIR / "finetune_with_unsloth.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ 脚本已创建: {script_path}")
    return script_path


def create_llamacpp_lora_script():
    """创建使用 llama.cpp 进行 LoRA 微调的脚本"""
    print("\n" + "="*70)
    print("📝 创建 llama.cpp LoRA 微调脚本")
    print("="*70)
    
    script_content = '''#!/bin/bash
# 使用 llama.cpp 微调 AstroSage 8B

cd train_qwen/astrosage_export

# 准备训练数据
echo "📚 准备训练数据..."
python3 << 'PYTHON'
import json

with open("../../train_qwen/data/qwen_train.json", "r") as f:
    data = json.load(f)

with open("training.txt", "w") as f:
    for item in data:
        conv = item.get("conversations", [])
        for msg in conv:
            role = msg.get("from", "")
            content = msg.get("value", "")
            if role == "user":
                f.write(f"### User:\\n{content}\\n\\n")
            elif role == "assistant":
                f.write(f"### Assistant:\\n{content}\\n\\n")
        f.write("---\\n\\n")

print(f"准备了 {len(data)} 条训练数据")
PYTHON

# 检查 llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "📥 下载 llama.cpp..."
    git clone https://github.com/ggerganov/llama.git llama.cpp
    cd llama.cpp && make finetune && cd ..
fi

# 训练 LoRA
echo "🚀 开始 LoRA 训练..."
./llama.cpp/finetune \\
    --model-base astrosage-8b-Q4_K_M.gguf \\
    --train-data training.txt \\
    --lora-out lora_weights.bin \\
    --threads 8 \\
    --batch 4 \\
    --ctx 2048 \\
    --epochs 3 \\
    --learning-rate 0.0001

# 合并权重
echo "🔧 合并权重..."
./llama.cpp/export_lora \\
    --model-base astrosage-8b-Q4_K_M.gguf \\
    --lora lora_weights.bin \\
    --output astrosage-8b-finetuned.gguf

echo "✅ 完成! 输出: astrosage-8b-finetuned.gguf"
'''
    
    script_path = WORK_DIR / "finetune_with_llamacpp.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    
    print(f"\n✅ 脚本已创建: {script_path}")
    return script_path


def create_transformers_finetune_script():
    """创建使用 Transformers 的微调脚本（使用 Llama-3.1-8B 作为基础）"""
    print("\n" + "="*70)
    print("📝 创建 Transformers 微调脚本")
    print("="*70)
    print("\n注意: 由于 GGUF 转 HF 格式复杂，这个脚本使用 Llama-3.1-8B 作为基础模型")
    print("      这与 astrosage-local 的基础相同，可以达到类似效果")
    
    script_content = '''#!/usr/bin/env python3
"""
使用 Transformers + PEFT 继续训练 AstroSage 风格模型
基于 Llama-3.1-8B-Instruct (与 astrosage-local 相同的基础)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json

# 配置 - 使用 Llama-3.1-8B，这与 astrosage-local 相同
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # 需要权限
# 或者使用无需权限的替代:
# BASE_MODEL = "shenzhi-wang/Llama3.1-8B-Chinese-Chat"
# BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

OUTPUT_DIR = "train_qwen/astrosage_continued"

print("="*70)
print("🚀 AstroSage 风格模型继续训练")
print("="*70)
print(f"基础模型: {BASE_MODEL}")
print("注意: 这个基础模型与 astrosage-local 相同")

# 1. 加载 tokenizer
print("\\n📦 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型 (4-bit)
print("\\n📦 加载模型...")
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

# 3. LoRA 配置
print("\\n🔧 配置 LoRA...")
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

# 4. 加载数据
def format_llama3_prompt(conv):
    text = ""
    for msg in conv:
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role == "system":
            text += f"<|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{content}<|eot_id|>"
    return text + "<|start_header_id|>assistant<|end_header_id|>\\n\\n"

class AstronomyDataset:
    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = format_llama3_prompt(self.data[idx].get("conversations", []))
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

print("\\n📂 加载训练数据...")
train_dataset = AstronomyDataset("train_qwen/data/qwen_train.json", tokenizer)
val_dataset = AstronomyDataset("train_qwen/data/qwen_val.json", tokenizer)

# 5. 训练
training_args = TrainingArguments(
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
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100),
)

print("\\n🚀 开始训练...")
trainer.train()

# 6. 保存
print("\\n💾 保存模型...")
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print(f"✅ 完成! 保存在: {OUTPUT_DIR}/final_model")
'''
    
    script_path = WORK_DIR / "finetune_with_transformers.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ 脚本已创建: {script_path}")
    return script_path


def print_recommendations():
    """打印推荐方案"""
    print("\n" + "="*70)
    print("📋 推荐方案")
    print("="*70)
    
    print("""
根据你的需求（像微调 Qwen 一样微调 astrosage-local），推荐以下方案:

🥇 方案1: 使用 Unsloth (推荐 ⭐)
   优点:
   - 可以直接微调 GGUF 文件
   - 速度快，显存占用低
   - 代码简洁，类似 transformers
   
   缺点:
   - 需要安装 unsloth (pip install unsloth)
   
   使用方法:
   1. pip install unsloth
   2. python train_qwen/astrosage_export/finetune_with_unsloth.py

🥈 方案2: 使用 llama.cpp
   优点:
   - 原生支持 GGUF
   - 无需额外依赖
   
   缺点:
   - 需要编译 llama.cpp
   - 命令行工具，不够灵活
   
   使用方法:
   1. bash train_qwen/astrosage_export/finetune_with_llamacpp.sh

🥉 方案3: 使用 Transformers (基于 Llama-3.1-8B)
   优点:
   - 和你之前微调 Qwen 的方式完全一样
   - 最熟悉的流程
   
   缺点:
   - 不是真正的 astrosage-local 继续微调
   - 是基于相同基础模型重新训练
   
   使用方法:
   1. 申请 Llama-3.1 权限或使用替代模型
   2. python train_qwen/astrosage_export/finetune_with_transformers.py

💡 建议选择方案1 (Unsloth)，它最符合你的需求！
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="导出 AstroSage 用于微调")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "unsloth", "llamacpp", "transformers"],
        help="选择微调方法",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AstroSage 8B 导出和微调工具")
    print("="*70)
    
    # 步骤1: 导出 GGUF
    gguf_file = export_gguf()
    if not gguf_file:
        print("\n❌ 导出失败")
        return
    
    # 步骤2: 创建各种微调脚本
    create_unsloth_finetune_script()
    create_llamacpp_lora_script()
    create_transformers_finetune_script()
    
    # 步骤3: 打印推荐
    print_recommendations()
    
    print("\n" + "="*70)
    print("✅ 导出完成!")
    print("="*70)
    print(f"\nGGUF 文件: {gguf_file}")
    print(f"微调脚本: {WORK_DIR}/")
    print("\n下一步:")
    print("  1. 选择上方推荐的方案")
    print("  2. 按照对应的方法进行微调")


if __name__ == "__main__":
    main()
'''
    
    print(f"\n✅ 已创建: {script_path}")


# 创建 requirements
def create_requirements():
    """创建依赖文件"""
    req_file = WORK_DIR / "requirements.txt"
    
    content = """# Unsloth 微调 (推荐)
unsloth>=0.3.0
xformers>=0.0.27
trl>=0.9.0
peft>=0.12.0
accelerate>=0.33.0
bitsandbytes>=0.43.0

# 基础依赖
transformers>=4.44.0
datasets>=2.21.0
sentencepiece>=0.2.0
protobuf>=3.20.0
"""
    
    with open(req_file, 'w') as f:
        f.write(content)
    
    print(f"\n✅ 依赖文件: {req_file}")


# 主流程
if __name__ == "__main__":
    print("="*70)
    print("🚀 AstroSage 8B 导出工具")
    print("="*70)
    
    # 1. 导出 GGUF
    gguf_path = export_astrosage_gguf()
    
    if not gguf_path:
        print("\n❌ 导出失败")
        sys.exit(1)
    
    # 2. 创建微调脚本
    print("\n" + "="*70)
    print("📝 创建微调脚本")
    print("="*70)
    
    create_unsloth_finetune_script()
    create_llamacpp_lora_script()
    create_transformers_finetune_script()
    create_requirements()
    
    # 3. 打印说明
    print_guide()
    
    print("\n" + "="*70)
    print("✅ 完成!")
    print("="*70)
    print(f"\n导出文件: {gguf_path}")
    print(f"工作目录: {WORK_DIR}/")
    print("\n🎯 推荐方案: 使用 Unsloth 进行微调")
    print("   pip install -r train_qwen/astrosage_export/requirements.txt")
    print("   python train_qwen/astrosage_export/finetune_with_unsloth.py")
