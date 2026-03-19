#!/usr/bin/env python3
"""
Qwen 模型蒸馏训练脚本
=====================
使用 LangGraph 生成的训练数据，通过知识蒸馏训练天文领域专用模型

训练策略:
1. Hard Loss: 标准监督学习 (Token-level Cross Entropy)
2. Soft Loss: 知识蒸馏 (KL Divergence with Temperature)
3. Task Loss: 天文领域特定损失

模型说明:
---------
为什么使用 Qwen-VL-Chat-Int4?
- Qwen-VL 是视觉-语言多模态模型，支持图像理解
- Int4 量化版本大幅减小模型体积，降低显存需求
- 8B 参数量适合单卡训练 (需要约16GB显存)
- 支持本地部署，无需连接 HuggingFace

如果使用 HuggingFace 模型:
- 需要稳定的网络连接
- 自动下载 Qwen-7B-Chat (约14GB)
- 需要更多显存 (推荐24GB+)

作者: AI Assistant
日期: 2026-03-02
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
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


# ==================== 配置类 ====================

@dataclass
class DistillationConfig:
    """蒸馏训练配置"""
    # 模型配置
    # 使用本地模型路径 (相对于项目根目录)
    # 可选: 使用HuggingFace模型名称 (需要网络连接)
    teacher_model_name: str = "models/qwen/Qwen-VL-Chat-Int4"  # 教师模型 (本地)
    student_model_name: str = "models/qwen/Qwen-VL-Chat-Int4"  # 学生模型 (本地)
    use_local_model: bool = True  # 是否使用本地模型
    
    # LoRA 配置
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 训练配置
    output_dir: str = "./distilled_model"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # 蒸馏配置
    temperature: float = 2.0      # 蒸馏温度
    alpha: float = 0.7            # 软标签权重 (1-alpha 为硬标签权重)
    beta: float = 0.1             # 任务损失权重
    
    # 数据配置
    train_data_path: str = "langgraph_demo/output/training_dataset.json"
    val_data_path: Optional[str] = None
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]


# ==================== 数据集类 ====================

class AstroDistillationDataset(Dataset):
    """
    天文数据蒸馏数据集
    支持软标签（教师模型输出）和硬标签（真实标签）
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        teacher_tokenizer=None
    ):
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer or tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ 加载数据集: {len(self.data)} 条记录")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 构建输入文本
        input_text = self._build_input(item['input'])
        
        # 构建输出文本（硬标签）
        output_text = self._build_output(item['output'])
        
        # 编码
        prompt_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding=False,
            return_tensors=None
        )
        
        full_encoding = self.tokenizer(
            input_text + output_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # 创建标签（只计算输出部分的损失）
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
        """构建输入提示"""
        coords = input_data.get('coordinates', {})
        query = input_data.get('query', 'Analyze this astronomical target')
        
        prompt = f"""你是一位专业天文学家。请分析以下天体目标。

坐标信息:
- 赤经 (RA): {coords.get('ra', 'N/A')}°
- 赤纬 (DEC): {coords.get('dec', 'N/A')}°

查询: {query}

请提供详细的科学分析，包括：
1. 天体分类
2. 物理参数估计
3. 观测建议

分析："""
        return prompt
    
    def _build_output(self, output_data: Dict) -> str:
        """构建输出文本"""
        classification = output_data.get('classification', 'Unknown')
        params = output_data.get('physical_params', {})
        reasoning = output_data.get('reasoning', [])
        
        output = f"""\n{classification}

【物理参数】
"""
        for key, value in params.items():
            output += f"- {key}: {value}\n"
        
        output += "\n【推理过程】\n"
        for i, step in enumerate(reasoning, 1):
            output += f"{i}. {step}\n"
        
        return output


# ==================== 蒸馏损失函数 ====================

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    L = α * L_soft + (1-α) * L_hard + β * L_task
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,
        beta: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch, seq_len, vocab_size]
            labels: 硬标签 [batch, seq_len]
            teacher_logits: 教师模型输出 [batch, seq_len, vocab_size]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各组件损失字典
        """
        # Hard Loss: 标准交叉熵
        hard_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Soft Loss: KL 散度
        if teacher_logits is not None:
            soft_loss = self._kl_divergence_loss(
                student_logits, teacher_logits
            )
        else:
            soft_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Task Loss: 天文领域特定损失 (这里简化为辅助分类损失)
        task_loss = self._task_loss(student_logits, labels)
        
        # 总损失
        total_loss = (
            self.alpha * soft_loss +
            (1 - self.alpha) * hard_loss +
            self.beta * task_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'hard': hard_loss.item(),
            'soft': soft_loss.item() if torch.is_tensor(soft_loss) else 0.0,
            'task': task_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _kl_divergence_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """KL 散度损失"""
        student_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss
    
    def _task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """任务特定损失（简化版）"""
        # 可以添加天文特定的约束，如物理参数范围等
        return torch.tensor(0.0, device=logits.device)


# ==================== 蒸馏训练器 ====================

class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("=" * 70)
        print("初始化知识蒸馏训练器")
        print("=" * 70)
        
        # 检查本地模型路径
        student_path = config.student_model_name
        if config.use_local_model and not os.path.isabs(student_path):
            # 尝试从项目根目录解析路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.join(project_root, student_path)
            if os.path.exists(abs_path):
                student_path = abs_path
                print(f"\n使用本地模型: {student_path}")
            else:
                print(f"\n⚠️ 本地模型路径不存在: {abs_path}")
                print(f"   尝试使用HuggingFace模型: {config.student_model_name}")
        
        # 检查模型路径是否存在
        if not os.path.exists(student_path) and config.use_local_model:
            raise FileNotFoundError(
                f"\n❌ 模型路径不存在: {student_path}\n"
                f"请确保模型文件存在于该路径，或设置 use_local_model=False 使用HuggingFace模型\n"
                f"当前配置: use_local_model={config.use_local_model}"
            )
        
        # 加载 Tokenizer
        print(f"\n加载 Tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                student_path,
                trust_remote_code=True,
                local_files_only=config.use_local_model
            )
            print(f"  ✓ Tokenizer加载成功")
        except Exception as e:
            print(f"  ✗ Tokenizer加载失败: {e}")
            raise
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载学生模型
        print(f"加载学生模型...")
        try:
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_path,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=config.use_local_model
            )
            print(f"  ✓ 学生模型加载成功")
        except Exception as e:
            print(f"  ✗ 学生模型加载失败: {e}")
            print(f"\n可能的原因:")
            print(f"  1. 模型文件不完整或损坏")
            print(f"  2. 内存不足 (8B模型需要约16GB显存)")
            print(f"  3. CUDA驱动问题")
            raise
        
        # 应用 LoRA
        if config.use_lora:
            print(f"\n应用 LoRA 适配器:")
            print(f"  r={config.lora_r}, alpha={config.lora_alpha}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()
        
        # 加载教师模型（可选，用于在线蒸馏）
        self.teacher_model = None
        if os.environ.get('USE_TEACHER', '0') == '1':
            teacher_path = config.teacher_model_name
            if config.use_local_model and not os.path.isabs(teacher_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                abs_path = os.path.join(project_root, teacher_path)
                if os.path.exists(abs_path):
                    teacher_path = abs_path
            
            print(f"\n加载教师模型...")
            try:
                self.teacher_model = AutoModelForCausalLM.from_pretrained(
                    teacher_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=config.use_local_model
                )
                self.teacher_model.eval()
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                print(f"  ✓ 教师模型加载成功")
            except Exception as e:
                print(f"  ⚠ 教师模型加载失败: {e}")
                print(f"  将使用离线模式（软标签从训练数据加载）")
        
        # 初始化损失函数
        self.criterion = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta
        )
        
        print("\n✓ 初始化完成")
    
    def prepare_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """准备数据加载器"""
        print("\n准备数据集...")
        
        # 训练集
        train_dataset = AstroDistillationDataset(
            self.config.train_data_path,
            self.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
        )
        
        # 验证集
        val_loader = None
        if self.config.val_data_path:
            val_dataset = AstroDistillationDataset(
                self.config.val_data_path,
                self.tokenizer,
                max_length=self.config.max_seq_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(
                    self.tokenizer,
                    padding=True,
                    return_tensors="pt"
                )
            )
        
        return train_loader, val_loader
    
    def train(self):
        """执行训练"""
        train_loader, val_loader = self.prepare_data()
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)
        print(f"总训练步数: {num_training_steps}")
        print(f"每个 epoch: {len(train_loader)} 步")
        
        # 训练循环
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            self.student_model.train()
            epoch_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播（学生）
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits
                
                # 获取教师输出（如果可用）
                teacher_logits = None
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        teacher_logits = teacher_outputs.logits
                
                # 计算损失
                loss, loss_dict = self.criterion(
                    student_logits, labels, teacher_logits
                )
                
                # 反向传播
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # 记录
                epoch_losses.append(loss_dict['total'])
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'hard': f"{loss_dict['hard']:.4f}",
                    'soft': f"{loss_dict['soft']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Epoch 统计
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(f"best_model")
                print(f"✓ 保存最佳模型 (loss={best_loss:.4f})")
        
        # 保存最终模型
        self.save_checkpoint("final_model")
        print("\n✓ 训练完成")
    
    def save_checkpoint(self, name: str):
        """保存模型检查点"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置
        config_dict = {
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'lora_r': self.config.lora_r,
            'lora_alpha': self.config.lora_alpha
        }
        with open(os.path.join(save_path, 'distillation_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def generate_with_teacher(config: DistillationConfig, input_text: str) -> str:
    """使用教师模型生成软标签"""
    teacher_path = config.teacher_model_name
    
    # 处理本地模型路径
    if config.use_local_model and not os.path.isabs(teacher_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(project_root, teacher_path)
        if os.path.exists(abs_path):
            teacher_path = abs_path
    
    print(f"使用教师模型生成: {teacher_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            teacher_path,
            trust_remote_code=True,
            local_files_only=config.use_local_model
        )
        model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=config.use_local_model
        )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ==================== 主程序 ====================

def main():
    """主函数"""
    # 检查本地模型是否存在
    local_model_path = "models/qwen/Qwen-VL-Chat-Int4"
    use_local = os.path.exists(local_model_path)
    
    if use_local:
        print("✓ 检测到本地模型，使用本地模型进行训练")
        teacher_name = local_model_path
        student_name = local_model_path
    else:
        print("⚠️ 未检测到本地模型，需要下载HuggingFace模型")
        print("  需要网络连接，且下载时间较长")
        teacher_name = "Qwen/Qwen-14B-Chat"  # 需要网络
        student_name = "Qwen/Qwen-7B-Chat"   # 需要网络
    
    # 创建配置
    config = DistillationConfig(
        # 模型配置
        teacher_model_name=teacher_name,
        student_model_name=student_name,
        use_local_model=use_local,
        
        # LoRA 配置
        use_lora=True,
        lora_r=64,
        lora_alpha=16,
        
        # 训练配置
        output_dir="./langgraph_demo/output/distilled_model",
        num_epochs=3,
        batch_size=1,  # 根据显存调整，8B模型建议batch_size=1
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,  # 使用半精度节省显存
        
        # 蒸馏配置
        temperature=2.0,
        alpha=0.7,  # 软标签权重
        beta=0.1,
        
        # 数据路径
        train_data_path="langgraph_demo/output/training_dataset.json"
    )
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 检查训练数据
    if not os.path.exists(config.train_data_path):
        print("⚠️  训练数据不存在，请先运行 langgraph_demo.py 生成训练数据")
        print("   创建示例训练数据...")
        
        # 创建示例数据
        example_data = [
            {
                "input": {
                    "query": "Analyze variable star EV UMa",
                    "coordinates": {"ra": 13.1316, "dec": 53.8585}
                },
                "output": {
                    "classification": "Cataclysmic Variable (CV)",
                    "physical_params": {"orbital_period": 0.10025},
                    "reasoning": ["Short period indicates compact binary"]
                }
            }
        ]
        
        os.makedirs(os.path.dirname(config.train_data_path), exist_ok=True)
        with open(config.train_data_path, 'w') as f:
            json.dump(example_data, f, indent=2)
        print(f"✓ 示例数据已保存: {config.train_data_path}")
    
    # 初始化训练器
    trainer = DistillationTrainer(config)
    
    # 开始训练
    try:
        trainer.train()
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
