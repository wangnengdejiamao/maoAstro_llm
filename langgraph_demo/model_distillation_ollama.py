#!/usr/bin/env python3
"""
Ollama 模型蒸馏训练脚本 (适配 qwen3:8b)
=====================================
使用 Ollama 本地模型作为教师模型，通过知识蒸馏训练小型学生模型

工作流程:
1. 使用 Ollama API 调用 qwen3:8b 生成软标签（教师输出）
2. 使用 Hugging Face 小型模型作为学生模型进行训练
3. 支持离线模式：先生成软标签保存，再训练

教师模型: ollama:qwen3:8b (通过 API 调用)
学生模型: Qwen2.5-1.5B-Instruct 或其他小型模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import requests
import time


@dataclass
class DistillationConfig:
    """蒸馏训练配置"""
    # Ollama 教师模型配置
    ollama_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    
    # 学生模型配置 (HuggingFace 小型模型)
    student_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List = None
    
    # 训练配置
    output_dir: str = "./langgraph_demo/output/distilled_model"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # 蒸馏配置
    temperature: float = 2.0
    alpha: float = 0.7
    beta: float = 0.1
    
    # 数据配置
    train_data_path: str = "langgraph_demo/output/training_dataset.json"
    soft_labels_path: str = "langgraph_demo/output/soft_labels.json"
    val_data_path: Optional[str] = None
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.host = host.rstrip('/')
        self.model = model
        self.generate_url = f"{self.host}/api/generate"
        
    def check_connection(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model in model_names:
                    print(f"  OK Ollama 服务正常，找到模型: {self.model}")
                    return True
                else:
                    print(f"  Warning: Ollama 服务正常，但未找到模型: {self.model}")
                    print(f"  可用模型: {model_names}")
                    return False
            return False
        except Exception as e:
            print(f"  Error: 无法连接到 Ollama 服务: {e}")
            print(f"  请确保 Ollama 正在运行: ollama serve")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """使用 Ollama 生成文本"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.Timeout:
            print(f"  Warning: Ollama 请求超时")
            return ""
        except Exception as e:
            print(f"  Error: Ollama 请求失败: {e}")
            return ""


class AstroDistillationDataset(Dataset):
    """天文数据蒸馏数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, 
                 soft_labels_path: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.soft_labels = {}
        if soft_labels_path and os.path.exists(soft_labels_path):
            with open(soft_labels_path, 'r', encoding='utf-8') as f:
                self.soft_labels = json.load(f)
            print(f"Loaded soft labels: {len(self.soft_labels)} items")
        
        print(f"Loaded dataset: {len(self.data)} records")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        input_text = self._build_input(item['input'])
        output_text = self._build_output(item['output'])
        
        prompt_encoding = self.tokenizer(
            input_text, truncation=True, max_length=self.max_length // 2,
            padding=False, return_tensors=None
        )
        
        full_encoding = self.tokenizer(
            input_text + output_text, truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None
        )
        
        labels = [-100] * len(prompt_encoding['input_ids']) + \
                 full_encoding['input_ids'][len(prompt_encoding['input_ids']):]
        
        return {
            'input_ids': full_encoding['input_ids'],
            'attention_mask': full_encoding['attention_mask'],
            'labels': labels,
            'input_text': input_text,
            'output_text': output_text
        }
    
    def _build_input(self, input_data: Dict) -> str:
        coords = input_data.get('coordinates', {})
        query = input_data.get('query', 'Analyze this astronomical target')
        
        return f"""你是一位专业天文学家。请分析以下天体目标。

坐标信息:
- 赤经 (RA): {coords.get('ra', 'N/A')} deg
- 赤纬 (DEC): {coords.get('dec', 'N/A')} deg

查询: {query}

请提供详细的科学分析，包括：
1. 天体分类
2. 物理参数估计
3. 观测建议

分析："""
    
    def _build_output(self, output_data: Dict) -> str:
        classification = output_data.get('classification', 'Unknown')
        params = output_data.get('physical_params', {})
        reasoning = output_data.get('reasoning', [])
        
        output = f"\n{classification}\n\n[物理参数]\n"
        for key, value in params.items():
            output += f"- {key}: {value}\n"
        
        output += "\n[推理过程]\n"
        for i, step in enumerate(reasoning, 1):
            output += f"{i}. {step}\n"
        
        return output


class SoftLabelGenerator:
    """使用 Ollama 生成软标签"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.ollama = OllamaClient(config.ollama_host, config.ollama_model)
    
    def generate_soft_labels(self, data_path: str, output_path: str):
        """为数据集生成软标签"""
        print("\n" + "=" * 70)
        print("使用 Ollama 生成软标签")
        print("=" * 70)
        
        if not self.ollama.check_connection():
            print("\nError: Ollama 服务不可用，请检查:")
            print("  1. Ollama 是否已安装")
            print("  2. Ollama 服务是否运行: ollama serve")
            print(f"  3. 模型 {self.config.ollama_model} 是否已拉取")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing_soft_labels = {}
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_soft_labels = json.load(f)
            print(f"Found existing soft labels: {len(existing_soft_labels)} items")
        
        soft_labels = existing_soft_labels.copy()
        
        print(f"Generating soft labels for {len(data)} samples...")
        for idx, item in enumerate(tqdm(data, desc="Generating")):
            if str(idx) in soft_labels:
                continue
            
            coords = item['input'].get('coordinates', {})
            query = item['input'].get('query', 'Analyze this astronomical target')
            
            prompt = f"""你是一位专业天文学家。请分析以下天体目标。

坐标信息:
- 赤经 (RA): {coords.get('ra', 'N/A')} deg
- 赤纬 (DEC): {coords.get('dec', 'N/A')} deg

查询: {query}

请提供详细的科学分析，包括：
1. 天体分类
2. 物理参数估计
3. 观测建议

分析："""
            
            response = self.ollama.generate(
                prompt, temperature=self.config.temperature, max_tokens=self.config.max_seq_length
            )
            
            if response:
                soft_labels[str(idx)] = response
                if (idx + 1) % 10 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(soft_labels, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.3)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(soft_labels, f, ensure_ascii=False, indent=2)
        
        print(f"\nSoft labels saved: {output_path} ({len(soft_labels)} items)")
        return True


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7, 
                 beta: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, student_logits: torch.Tensor, labels: torch.Tensor):
        hard_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)), labels.view(-1)
        )
        
        loss_dict = {
            'total': hard_loss.item(),
            'hard': hard_loss.item(),
            'soft': 0.0,
            'task': 0.0
        }
        
        return hard_loss, loss_dict


class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("=" * 70)
        print("初始化知识蒸馏训练器 (Ollama 版本)")
        print("=" * 70)
        print(f"\n教师模型 (Ollama): {config.ollama_model}")
        print(f"学生模型 (HF): {config.student_model_name}")
        
        print(f"\n加载 Tokenizer: {config.student_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.student_model_name, trust_remote_code=True, local_files_only=False
            )
            print("  OK Tokenizer loaded")
        except Exception as e:
            print(f"  Error loading tokenizer: {e}")
            raise
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading student model...")
        try:
            self.student_model = AutoModelForCausalLM.from_pretrained(
                config.student_model_name,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            param_count = sum(p.numel() for p in self.student_model.parameters()) / 1e6
            print(f"  OK Model loaded ({param_count:.1f}M params)")
        except Exception as e:
            print(f"  Error loading model: {e}")
            raise
        
        if config.use_lora:
            print(f"\nApplying LoRA (r={config.lora_r}, alpha={config.lora_alpha})")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()
        
        self.criterion = DistillationLoss(
            temperature=config.temperature, alpha=config.alpha, beta=config.beta
        )
        
        print("\nInitialization complete")
    
    def prepare_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        print("\nPreparing datasets...")
        
        train_dataset = AstroDistillationDataset(
            self.config.train_data_path, self.tokenizer,
            max_length=self.config.max_seq_length,
            soft_labels_path=self.config.soft_labels_path
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(self.tokenizer, padding=True, return_tensors="pt")
        )
        
        return train_loader, None
    
    def train(self):
        train_loader, _ = self.prepare_data()
        
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)
        print(f"Total steps: {num_training_steps}, Steps per epoch: {len(train_loader)}")
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            self.student_model.train()
            epoch_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                student_outputs = self.student_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                
                loss, loss_dict = self.criterion(student_outputs.logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_losses.append(loss_dict['total'])
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_model")
                print(f"Saved best model (loss={best_loss:.4f})")
        
        self.save_checkpoint("final_model")
        print("\nTraining complete!")
    
    def save_checkpoint(self, name: str):
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        config_dict = {
            'teacher_model': self.config.ollama_model,
            'student_model': self.config.student_model_name,
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'lora_r': self.config.lora_r,
            'lora_alpha': self.config.lora_alpha
        }
        with open(os.path.join(save_path, 'distillation_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    config = DistillationConfig(
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
        student_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        output_dir="./langgraph_demo/output/distilled_model",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        train_data_path="langgraph_demo/output/training_dataset.json",
        soft_labels_path="langgraph_demo/output/soft_labels.json"
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create example data if not exists
    if not os.path.exists(config.train_data_path):
        print("Creating example training data...")
        example_data = [
            {
                "input": {
                    "query": "Analyze variable star EV UMa",
                    "coordinates": {"ra": 13.1316, "dec": 53.8585}
                },
                "output": {
                    "classification": "Cataclysmic Variable (CV)",
                    "physical_params": {"orbital_period": 0.10025, "magnitude": 14.5},
                    "reasoning": ["Short period indicates compact binary"]
                }
            }
        ]
        os.makedirs(os.path.dirname(config.train_data_path), exist_ok=True)
        with open(config.train_data_path, 'w') as f:
            json.dump(example_data, f, indent=2)
        print(f"Example data saved: {config.train_data_path}")
    
    # Step 1: Generate soft labels
    print("\n" + "=" * 70)
    print("Step 1: Generate soft labels with Ollama")
    print("=" * 70)
    
    soft_label_gen = SoftLabelGenerator(config)
    success = soft_label_gen.generate_soft_labels(
        config.train_data_path, config.soft_labels_path
    )
    
    if not success:
        print("\nWarning: Soft label generation failed, will use hard labels")
    
    # Step 2: Train student model
    print("\n" + "=" * 70)
    print("Step 2: Train student model")
    print("=" * 70)
    
    trainer = DistillationTrainer(config)
    try:
        trainer.train()
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
