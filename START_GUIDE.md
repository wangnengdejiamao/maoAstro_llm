# 🚀 AstroSage 启动指南

## 快速开始

### 方式1: 使用训练好的模型 + RAG (推荐)

```bash
# 启动完整系统
./start_local_rag.sh
# 选择选项 2) AstroSage 完整系统
```

或者直接用Python：

```bash
python start_astrosage_complete.py
```

### 方式2: 使用 Ollama + RAG

```bash
# 确保Ollama在运行
ollama serve &

# 启动
./start_local_rag.sh
# 选择选项 1) Ollama + RAG
```

---

## 💬 交互命令

启动后，你可以：

| 输入 | 功能 |
|------|------|
| `什么是灾变变星?` | 自动检测并回答 |
| `/rag 赫罗图特征` | 强制使用RAG检索 |
| `/direct 你好` | 强制直接推理 |
| `/help` | 显示帮助 |
| `/quit` | 退出 |

---

## 🔍 RAG vs 直接推理

**RAG模式** (用于专业问题):
- 检索相关文档
- 基于资料生成回答
- 自动标注引用来源

**直接推理** (用于通用问题):
- 模型直接生成
- 响应更快
- 不依赖知识库

**自动选择**:
- 系统会根据问题内容自动选择模式
- 包含天文专业词汇 → RAG
- 通用问题 → 直接推理

---

## 📁 模型位置

训练好的模型：
```
train_qwen/output_qwen25/merged_model/  (14.5GB 完整模型)
train_qwen/output_qwen25/final_model/   (616MB LoRA权重)
```

---

## ⚡ 性能提示

- **首次加载**: 需要2-3分钟加载模型
- **显存占用**: ~10GB
- **推理速度**: 15-20 tokens/秒
- **响应时间**: 通常5-15秒

---

## 🎯 示例问答

### 示例1: 专业问题 (使用RAG)
```
👤 You: 什么是灾变变星?

🔍 检测到专业问题，使用RAG增强...

🤖 AstroSage:
灾变变星(Cataclysmic Variable, CV)是一种由白矮星和...

📚 参考资料:
   [1] Smith_2023_CV_Review.pdf (第12页)
   [2] astro_data_001.pdf (第5页)
```

### 示例2: 通用问题 (直接推理)
```
👤 You: 你好，介绍一下自己

🤖 使用直接推理...

🤖 AstroSage:
你好!我是AstroSage，一个专门训练用于天文领域问答的AI助手...
```

---

## 🛠️ 故障排除

### 问题: "未找到训练好的模型"
**解决**: 
```bash
# 检查模型是否存在
ls train_qwen/output_qwen25/merged_model/

# 如果不存在，先合并LoRA
python train_qwen/merge_lora.py
```

### 问题: "CUDA out of memory"
**解决**:
```bash
# 关闭其他GPU程序
# 或使用CPU模式
python start_astrosage_complete.py --no-rag
```

### 问题: "RAG检索失败"
**解决**:
```bash
# 检查知识库是否存在
ls output/qa_hybrid/

# 禁用RAG运行
python start_astrosage_complete.py --no-rag
```

---

## 📖 相关文件

- `start_astrosage_complete.py` - 主启动脚本
- `start_local_rag.sh` - 启动菜单
- `rag_system/` - RAG系统代码
- `MODEL_EVALUATION_REPORT.md` - 模型评估报告
