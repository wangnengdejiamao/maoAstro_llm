#!/usr/bin/env python3
"""
导出 AstroSage 8B 并创建微调脚本
"""

import os
import sys
import shutil
from pathlib import Path

WORK_DIR = Path("train_qwen/astrosage_export")
OLLAMA_DIR = Path.home() / ".ollama" / "models"


def export_gguf():
    """导出 GGUF 文件"""
    print("="*70)
    print("📦 导出 AstroSage 8B GGUF 文件")
    print("="*70)
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找 blob 文件
    blobs_dir = OLLAMA_DIR / "blobs"
    blobs = [f for f in blobs_dir.glob("sha256-*") if f.stat().st_size > 4 * 1024**3]
    blobs.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    if not blobs:
        print("❌ 未找到模型文件")
        return None
    
    model_file = blobs[0]
    target_file = WORK_DIR / "astrosage-8b-Q4_K_M.gguf"
    
    print(f"\n源文件: {model_file.name}")
    print(f"大小: {model_file.stat().st_size / (1024**3):.2f} GB")
    print(f"目标: {target_file}")
    
    if target_file.exists():
        print("\n✅ 文件已存在")
        return target_file
    
    # 创建硬链接
    try:
        os.link(model_file, target_file)
        print("\n✅ 硬链接创建成功")
    except:
        print("\n复制中...")
        shutil.copy2(model_file, target_file)
        print("✅ 复制完成")
    
    return target_file


def create_unsloth_script():
    """创建 Unsloth 微调脚本"""
    script = '''#!/usr/bin/env python3
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
print("\\n📦 加载 GGUF 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=GGUF_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 添加 LoRA
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
)

# 加载数据
print("\\n📂 加载训练数据...")
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
            text += f"<|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{content}<|eot_id|>"
    return {"text": text + "<|start_header_id|>assistant<|end_header_id|>\\n\\n"}

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

print("\\n🚀 开始训练...")
trainer.train()

# 保存
print("\\n💾 保存模型...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged", tokenizer)
print(f"✅ 完成! 保存在: {OUTPUT_DIR}/merged")
'''
    
    script_path = WORK_DIR / "finetune_unsloth.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    print(f"\n✅ Unsloth 脚本: {script_path}")
    return script_path


def create_transformers_script():
    """创建 Transformers 微调脚本 (使用 Llama-3.1-8B)"""
    script = '''#!/usr/bin/env python3
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
            text += f"<|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{content}<|eot_id|>"
    return text + "<|start_header_id|>assistant<|end_header_id|>\\n\\n"

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

print("\\n🚀 开始训练...")
trainer.train()

print("\\n💾 保存模型...")
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print(f"✅ 完成! 保存在: {OUTPUT_DIR}/final_model")
'''
    
    script_path = WORK_DIR / "finetune_transformers.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Transformers 脚本: {script_path}")
    return script_path


def print_guide():
    """打印使用指南"""
    print("\n" + "="*70)
    print("📋 使用指南")
    print("="*70)
    print("""
🥇 推荐方案: 使用 Unsloth (可直接微调 GGUF)

1. 安装依赖:
   pip install unsloth xformers trl peft accelerate bitsandbytes

2. 运行微调:
   python train_qwen/astrosage_export/finetune_unsloth.py

3. 导出到 Ollama:
   ollama create maoAstro-finetuned -f Modelfile

---

🥈 备选方案: 使用 Transformers (基于 Llama-3.1-8B)

1. 运行微调:
   python train_qwen/astrosage_export/finetune_transformers.py

2. 导出到 Ollama:
   python export_llama31_to_ollama.py \\
       --model train_qwen/astrosage_continued/final_model

---

⚠️ 注意:
   - Unsloth 方案可以直接微调 astrosage-local GGUF
   - Transformers 方案使用 Llama-3.1-8B 中文替代版
   - 两者都可以达到类似效果
""")


def main():
    print("\n" + "="*70)
    print("🚀 AstroSage 8B 导出工具")
    print("="*70)
    
    # 导出
    gguf = export_gguf()
    if not gguf:
        print("❌ 导出失败")
        return
    
    # 创建脚本
    print("\n" + "="*70)
    print("📝 创建微调脚本")
    print("="*70)
    create_unsloth_script()
    create_transformers_script()
    
    # 打印指南
    print_guide()
    
    print("\n" + "="*70)
    print("✅ 完成!")
    print("="*70)
    print(f"\n导出文件: {gguf}")
    print(f"脚本目录: {WORK_DIR}/")


if __name__ == "__main__":
    main()
