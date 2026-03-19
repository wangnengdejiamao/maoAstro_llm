# 天文领域 Qwen-8B 模型评估与微调 - 快速开始

## 🎯 目标

构建一个超越 **AstroMLab-8B (80.9%)** 的天文领域 Qwen-8B 微调模型。

## 📋 环境准备

### 1. 安装依赖

```bash
cd model_evaluation
pip install -r requirements.txt

# 如果使用 LLaMA-Factory 微调
pip install llmtuner
```

### 2. 准备数据目录

```bash
mkdir -p data models eval_results
```

## 🚀 快速评估现有模型

### 方式 1: 评估 Ollama 部署的模型

```bash
# 评估 Qwen3-8B
python run_eval.py \
    --model qwen3:8b \
    --interface ollama \
    --dataset custom \
    --data-path ./data/eval_dataset.json

# 对比多个模型
python run_eval.py \
    --models qwen3:8b llama3.1:8b gemma2:9b \
    --interface ollama \
    --dataset custom \
    --data-path ./data/eval_dataset.json \
    --compare
```

### 方式 2: 评估本地 HuggingFace 模型

```bash
# 评估本地模型
python run_eval.py \
    --model /path/to/your/model \
    --interface hf \
    --dataset custom \
    --data-path ./data/eval_dataset.json
```

## 📊 生成评估数据

```bash
# 生成 1000 题评估数据集
python build_dataset.py \
    --build-eval \
    --n-samples 1000 \
    --output ./data/eval_dataset.json
```

## 🔧 模型微调 (可选)

### 快速微调 (单卡训练)

```bash
# 运行完整训练流程
bash train_qwen_astro.sh
```

### 手动分步微调

#### Step 1: 准备数据

```bash
# 下载 arXiv 论文 (用于 CPT)
python build_dataset.py \
    --download-arxiv \
    --categories astro-ph.SR astro-ph.EP astro-ph.CO \
    --max-papers 5000 \
    --output ./data/arxiv_papers.json

# 生成训练数据 (用于 SFT)
python build_dataset.py \
    --build-training \
    --n-samples 10000 \
    --output ./data/astro_qa.json \
    --format sharegpt
```

#### Step 2: 持续预训练 (CPT)

```bash
llamafactory-cli train \
    --stage pt \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --do_train \
    --dataset arxiv_astro_cpt \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./models/qwen8b-astro-cpt \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16
```

#### Step 3: 监督微调 (SFT)

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path ./models/qwen8b-astro-cpt \
    --do_train \
    --dataset astro_qa_sft \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --output_dir ./models/qwen8b-astro-sft \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16
```

#### Step 4: 合并导出

```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path ./models/qwen8b-astro-sft \
    --finetuning_type lora \
    --template qwen \
    --export_dir ./models/qwen8b-astro-final \
    --export_size 2
```

## 📈 对比评估

```bash
# 对比你的模型和基准模型
python run_eval.py \
    --models qwen8b-astro-final qwen3:8b \
    --interface hf \
    --dataset custom \
    --data-path ./data/eval_dataset.json \
    --compare \
    --output-dir ./eval_results
```

## 🔍 关键评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **AstroMLab-1 准确率** | 天文多选题准确率 | **> 80.9%** |
| **子领域准确率** | 各天文子领域表现 | 均衡 > 75% |
| **幻觉率** | 关键概念错误率 | **< 10%** |
| **ECE** | 期望校准误差 | < 0.1 |

## 🎓 提升准确率的技巧

### 1. 数据质量
- ✅ 使用高质量 arXiv 论文 (2023-2025)
- ✅ 重点标注反幻觉数据
- ✅ 平衡各子领域样本

### 2. 训练策略
- ✅ 两阶段训练: CPT → SFT
- ✅ CPT 学习率 2e-5，SFT 学习率 5e-5
- ✅ LoRA rank: CPT 64, SFT 128

### 3. 推理优化
- ✅ 使用 Chain-of-Thought 提示
- ✅ 调整 temperature (0.3-0.7)
- ✅ 自洽性解码 (多次采样投票)

## 📚 参考

- [AstroMLab 官网](https://astromlab.org/)
- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen 模型](https://huggingface.co/Qwen)

## ❓ 常见问题

**Q: 需要多少显存？**
- LoRA 微调: ~20GB (A100 40GB 或 RTX 4090)
- 推理: ~10GB (8-bit)

**Q: 训练需要多久？**
- CPT (5000 论文): ~6-8 小时 (A100)
- SFT (10000 样本): ~4 小时 (A100)

**Q: 如何验证超越 AstroMLab-8B？**
```bash
python run_eval.py \
    --models your-model AstroSage-LLaMA-3.1-8B \
    --dataset astromlab1 \
    --compare
```
