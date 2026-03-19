# 🏠 本地模型集成方案

使用已有的 Ollama/本地模型，无需重新下载

---

## 📋 方案选择

根据你的需求，提供两种集成方案：

### 方案1: Ollama + RAG (推荐 ⭐)
使用 Ollama 运行本地模型，结合 RAG 检索增强

**适用场景**:
- 已有 Ollama 模型 (qwen3, llama3.1 等)
- 想要快速体验 RAG 效果
- 不需要微调模型

### 方案2: 本地模型微调
使用本地 HuggingFace 格式模型进行 LoRA 微调

**适用场景**:
- 有 HuggingFace 格式的本地模型
- 需要针对天文数据微调
- 追求最佳效果

---

## 方案1: Ollama + RAG

### 步骤1: 确保 Ollama 运行

```bash
# 启动 Ollama 服务
ollama serve

# 检查可用模型
ollama list
```

### 步骤2: 启动 RAG + Ollama

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 使用 qwen3 模型
python rag_system/retrieval/rag_with_ollama.py --model qwen3

# 或使用其他模型
python rag_system/retrieval/rag_with_ollama.py --model llama3.1
```

### 功能特性

```
👤 用户: 什么是赫罗图？

🔍 检索相关知识...
✅ 检索到 5 条相关知识

🤖 助手: 
根据参考信息，赫罗图（Hertzsprung-Russell Diagram）是...

📚 参考来源:
[1] Abrahams et al. 2022.pdf, 第10页 (相关度: 0.95)
[2] Abril et al. 2020.pdf, 第23页 (相关度: 0.87)

[置信度: 0.92]
```

---

## 方案2: 本地模型微调

### 可用本地模型

检查到的本地模型：

| 模型 | 路径 | 格式 | 大小 |
|-----|------|------|------|
| Qwen-VL-Chat-Int4 | `models/qwen/Qwen-VL-Chat-Int4/` | HuggingFace | ~9.7GB |

### 开始训练

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 使用本地 Qwen 模型训练
python train_qwen/train_lora_local.py \
    --model-path models/qwen/Qwen-VL-Chat-Int4 \
    --train-data train_qwen/data/qwen_train.json \
    --val-data train_qwen/data/qwen_val.json \
    --output-dir train_qwen/output_local
```

### 训练参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| LoRA r | 64 | 低秩矩阵维度 |
| LoRA alpha | 16 | 缩放参数 |
| Learning rate | 2e-4 | 学习率 |
| Batch size | 1 | 批次大小 |
| Epochs | 3 | 训练轮数 |
| Quantization | 4-bit | 节省显存 |

### 显存需求

- **4-bit量化**: 约 10-12GB 显存
- **8GB显卡**: 可能需要调整 batch_size=1, max_length=1024

---

## 🔄 系统架构

```
用户提问
    ↓
┌─────────────────────────────────────────────┐
│              RAG Pipeline                    │
│  ┌──────────────┐    ┌──────────────┐      │
│  │  向量检索     │    │  关键词检索   │      │
│  │  (可选)      │    │              │      │
│  └──────┬───────┘    └──────┬───────┘      │
│         └─────────┬──────────┘              │
│                   ↓                         │
│           检索结果融合                       │
│                   ↓                         │
│           幻觉检测 + 引用生成                 │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              生成层                          │
├─────────────────────────────────────────────┤
│  方案A: Ollama API                           │
│    - 调用本地 Ollama 服务                    │
│    - 支持 qwen3, llama3.1 等                 │
├─────────────────────────────────────────────┤
│  方案B: 本地 HuggingFace 模型                │
│    - 微调后的模型                            │
│    - 更适合天文领域                          │
└─────────────────────────────────────────────┘
                    ↓
              最终回答 + 引用
```

---

## 🚀 快速启动

### 方式1: 一键启动 Ollama+RAG

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 创建启动脚本
cat > start_ollama_rag.sh << 'SCRIPT'
#!/bin/bash
echo "🚀 启动 AstroSage RAG + Ollama"
echo ""

# 检查 Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "正在启动 Ollama..."
    ollama serve &
    sleep 3
fi

echo "可用模型:"
ollama list

echo ""
echo "启动 RAG 客户端..."
python rag_system/retrieval/rag_with_ollama.py --model qwen3
SCRIPT

chmod +x start_ollama_rag.sh
./start_ollama_rag.sh
```

### 方式2: 训练本地模型

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 训练
python train_qwen/train_lora_local.py

# 推理（训练完成后）
python train_qwen/inference.py \
    --model train_qwen/output_local/final_model \
    -i
```

---

## 📊 对比

| 特性 | Ollama+RAG | 本地微调 |
|-----|-----------|---------|
| 启动速度 | ⚡ 快速 | 🐢 慢（需要训练） |
| 领域适配 | ⭐ 通过RAG实现 | ⭐⭐ 模型本身适配 |
| 引用溯源 | ✅ 支持 | ✅ 支持 |
| 幻觉检测 | ✅ 支持 | ✅ 支持 |
| 显存需求 | 💾 中等 | 💾💾 较高 |
| 效果 | 良好 | 最佳 |

**建议**: 
- 快速体验 → 使用 Ollama+RAG
- 追求效果 → 本地微调

---

## 🔧 故障排除

### Ollama 连接失败

```bash
# 检查 Ollama 是否运行
pgrep ollama

# 手动启动
ollama serve

# 测试 API
curl http://localhost:11434/api/tags
```

### 模型加载失败

```bash
# 检查模型文件
ls -lh models/qwen/Qwen-VL-Chat-Int4/

# 检查配置文件
ls models/qwen/Qwen-VL-Chat-Int4/config.json
```

### 显存不足

```bash
# 使用更小的 batch_size
python train_qwen/train_lora_local.py \
    --model-path models/qwen/Qwen-VL-Chat-Int4

# 修改脚本中的参数:
# batch_size = 1
# gradient_accumulation = 16
# max_seq_length = 1024
```

---

## 📝 示例对话

### Ollama+RAG 模式

```
👤 用户: 灾变变星的光变曲线有什么特征？

🔍 检索相关知识...
✅ 检索到 5 条相关知识

🤖 助手: 
根据参考文档，灾变变星(CV)的光变曲线具有以下特征：

1. **周期性变化**: 呈现明显的周期性变化，周期通常为几小时
2. **爆发事件**: 矮新星(DN)类型会周期性爆发，亮度增加2-5等
3. **轨道调制**: 由于轨道运动导致的光度变化

具体数据示例：
- 周期范围: 70分钟 - 8小时
- 爆发幅度: 2-6等
- 宁静期亮度: 19-20等

📚 参考来源:
[1] Abrahams et al. 2022 - Informing the Cataclysmic Variable Sequence.pdf, 第2页 (相关度: 0.94)
[2] Abril et al. 2020 - Disentangling cataclysmic variables.pdf, 第5页 (相关度: 0.88)

[置信度: 0.91]
```

---

## ✅ 检查清单

- [ ] Ollama 服务已启动
- [ ] 本地模型文件存在
- [ ] RAG 检索器已构建
- [ ] 运行 Ollama+RAG 测试
- [ ] (可选) 本地模型训练完成

---

**现在你可以开始使用本地模型了！** 🎉
