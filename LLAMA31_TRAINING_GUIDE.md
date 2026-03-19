# maoAstro-Llama31-8B 训练与部署指南

> 使用 Llama-3.1-8B-Instruct 作为基础，训练你的天文领域大模型

---

## 🎯 方案概述

由于 Qwen2.5-7B 训练效果不理想，改用 **Llama-3.1-8B-Instruct** 进行训练：

**优势**:
- Llama-3.1 英文理解能力强，中文也不错
- 社区支持好，工具链完善
- 推理速度快
- 易于部署到 Ollama

---

## 📋 训练流程

```
下载 Llama-3.1-8B-Instruct (约 16GB)
    ↓
LoRA 微调训练 (使用 3,413 条天文QA)
    ↓
保存 LoRA 权重
    ↓
测试推理效果
    ↓
导出到 Ollama
```

---

## 🚀 快速开始

### 步骤1: 准备环境

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 安装依赖 (如果还没有)
pip install transformers peft datasets bitsandbytes accelerate -q
```

### 步骤2: 开始训练

```bash
# 基础训练 (推荐)
python train_llama31_lora.py

# 或自定义参数
python train_llama31_lora.py \
    --epochs 3 \
    --lora-r 64 \
    --output train_qwen/maoAstro-Llama31-8B
```

**训练参数**:
- Base Model: `meta-llama/Llama-3.1-8B-Instruct`
- LoRA rank: 64
- Epochs: 3
- Batch size: 1 (gradient accumulation=8)
- 显存需求: ~10GB (4-bit量化后)
- 训练时间: 4-8 小时 (取决于GPU)

### 步骤3: 测试模型

```bash
# 推理测试
python inference_llama31.py \
    --model train_qwen/maoAstro-Llama31-8B/final_model
```

### 步骤4: 导出到 Ollama

```bash
# 导出为 Ollama 格式
python export_llama31_to_ollama.py \
    --model train_qwen/maoAstro-Llama31-8B/final_model \
    --name maoAstro-llama31 \
    --test
```

### 步骤5: 使用模型

```bash
# 通过 Ollama 使用
ollama run maoAstro-llama31

# 或在 Python 中使用
python start_maoastro_with_simple_rag.py
```

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `train_llama31_lora.py` | 训练脚本 |
| `inference_llama31.py` | 推理测试 |
| `export_llama31_to_ollama.py` | 导出到 Ollama |
| `export_and_finetune_ollama.py` | 从 Ollama 导出再训练 (高级) |

---

## ⚙️ 训练配置

### 默认配置

```python
base_model = "meta-llama/Llama-3.1-8B-Instruct"
lora_r = 64
lora_alpha = 16
epochs = 3
learning_rate = 2e-4
batch_size = 1
max_seq_length = 2048
load_in_4bit = True  # 4-bit 量化
```

### 硬件要求

| 配置 | 要求 |
|------|------|
| GPU | 12GB+ 显存 (RTX 3080 Ti 可运行) |
| 内存 | 32GB+ |
| 磁盘 | 20GB 可用空间 |
| 时间 | 4-8 小时 |

---

## 🔧 故障排除

### 问题1: 下载模型失败

**症状**: `meta-llama/Llama-3.1-8B-Instruct` 下载超时

**解决**:
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 重新运行训练
python train_llama31_lora.py
```

### 问题2: CUDA Out of Memory

**症状**: 显存不足错误

**解决**:
```bash
# 减小 batch size 或序列长度
python train_llama31_lora.py \
    --batch-size 1 \
    --lora-r 32  # 减小 LoRA rank
```

### 问题3: 训练后效果不佳

**可能原因**:
1. 训练数据质量问题
2. 学习率不合适
3. 训练轮数不足

**解决**:
```bash
# 尝试不同学习率
python train_llama31_lora.py --lr 1e-4  # 或 5e-4

# 增加训练轮数
python train_llama31_lora.py --epochs 5
```

---

## 📝 与 Qwen2.5 对比

| 方面 | Qwen2.5-7B | Llama-3.1-8B |
|------|-----------|--------------|
| 中文 | ✅ 原生支持 | ⚠️ 需微调优化 |
| 英文 | ✅ 良好 | ✅ 优秀 |
| 推理速度 | 中等 | 快 |
| 社区支持 | 中等 | 丰富 |
| Ollama兼容 | 好 | 优秀 |
| 训练难度 | 低 | 中等 |

---

## 🎉 成功标志

训练成功后：

```bash
$ ollama run maoAstro-llama31

>>> 什么是灾变变星?
灾变变星(Cataclysmic Variable, CV)是一种由白矮星和伴星组成的双星系统...
[专业、准确的回答]
```

---

## 📚 相关文档

- `LLAMA31_TRAINING_GUIDE.md` - 本指南
- `TRAINING_GUIDE.md` - 通用训练指南
- `MODEL_EVALUATION_REPORT.md` - 模型评估报告

---

## 💡 建议

1. **先试运行推理测试**，确保环境正常
2. **训练时监控 GPU 温度和利用率**
3. **保存好训练日志**，便于排查问题
4. **训练完成后先测试再导出**

---

**开始训练吧！** 🚀

```bash
python train_llama31_lora.py
```
