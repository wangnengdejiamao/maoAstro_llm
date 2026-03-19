# ⚡ 本地模型快速开始

使用已有的 Ollama 或本地 HuggingFace 模型

---

## 🎯 一键启动

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
./start_local_rag.sh
```

然后按提示选择：
- `1` → Ollama + RAG (最快)
- `2` → 本地模型训练
- `3` → 检查系统状态

---

## 方案A: Ollama + RAG (30秒启动)

### 1. 确保 Ollama 运行

```bash
# 窗口1: 启动 Ollama
ollama serve

# 窗口2: 检查模型
ollama list
```

### 2. 启动 RAG

```bash
# 方式1: 交互式启动
./start_local_rag.sh
# 然后选择 1

# 方式2: 直接启动
python rag_system/retrieval/rag_with_ollama.py --model qwen3
```

### 3. 开始对话

```
👤 用户: 什么是赫罗图？

🔍 检索相关知识...
✅ 检索到 5 条相关知识

🤖 助手: [基于检索文档的回答]

📚 参考来源:
[1] xxx.pdf, 第10页 (相关度: 0.95)
```

---

## 方案B: 本地模型微调

### 使用已有模型

```bash
# 使用本地 Qwen-VL-Chat-Int4
python train_qwen/train_lora_local.py \
    --model-path models/qwen/Qwen-VL-Chat-Int4
```

### 训练参数

| 配置 | 值 |
|-----|-----|
| 数据 | 3,413 条高质量问答 |
| LoRA r | 64 |
| 学习率 | 2e-4 |
| 轮数 | 3 |
| 显存 | ~10-12GB |

---

## 📁 本地模型位置

```
models/
├── qwen/Qwen-VL-Chat-Int4/     # 多模态模型 (~9.7GB)
└── astrosage-llama-3.1-8b/      # 已微调模型
```

---

## 🔥 推荐流程

### 只想快速体验
```bash
ollama serve                    # 窗口1
./start_local_rag.sh            # 窗口2, 选1
```

### 想要最好效果
```bash
# 1. 训练本地模型 (2-4小时)
python train_qwen/train_lora_local.py

# 2. 运行推理
python train_qwen/inference.py \
    --model train_qwen/output_local/final_model \
    -i
```

---

## 🆘 常见问题

**Q: Ollama 服务未启动**
```bash
# 在新终端运行
ollama serve
```

**Q: 找不到模型**
```bash
# 检查可用模型
ollama list

# 拉取模型
ollama pull qwen3
```

**Q: 显存不足**
```bash
# 使用 Ollama 方案（不需要显存）
./start_local_rag.sh
# 选 1
```

---

**现在可以开始了！** 🚀

```bash
./start_local_rag.sh
```
