# 问题修复总结

## 遇到的问题

### 1. Jinja2 版本问题 ❌
**错误**: `module 'jinja2' has no attribute 'pass_eval_context'`

**原因**: 
- transformers 的 `apply_chat_template` 需要较新版本的 Jinja2
- 系统中 Jinja2 2.11.3 版本过旧

**解决**: 
- 已升级 Jinja2 到 3.1.6
- 简化版脚本手动构建 prompt，绕过 Jinja2 模板

### 2. RAG Embedding 模型下载问题 ❌
**错误**: Connection reset by peer (下载 BAAI/bge-large-zh-v1.5)

**原因**:
- RAG 系统需要下载 sentence-transformers 嵌入模型
- 网络连接 HuggingFace 不稳定

**解决**:
- 简化版脚本暂不使用 RAG
- 直接使用训练好的模型推理

---

## ✅ 修复后的方案

### 简化版脚本: `start_maoastro_simple.py`

**特点**:
- ✅ 手动构建 prompt，不依赖 Jinja2 模板
- ✅ 直接使用模型推理，无需 RAG 依赖
- ✅ 启动速度更快
- ✅ 代码更简洁可靠

**使用方法**:
```bash
# 启动 maoAstro LLM
python start_maoastro_simple.py

# 或指定模型路径
python start_maoastro_simple.py --model-path models/maoAstro_llm
```

---

## 🚀 立即使用

### 步骤1: 终止之前的进程
```bash
# 如果有卡住的进程
pkill -f start_local_rag
pkill -f start_astrosage
```

### 步骤2: 启动简化版
```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python start_maoastro_simple.py
```

### 步骤3: 开始问答
```
👤 You: 什么是灾变变星?

🤖 maoAstro: 灾变变星(Cataclysmic Variable, CV)是一种由白矮星和伴星组成的双星系统...
```

---

## 📁 文件说明

| 文件 | 状态 | 说明 |
|------|------|------|
| `start_maoastro_simple.py` | ✅ **推荐** | 简化版，稳定可靠 |
| `start_astrosage_complete.py` | ⚠️ 复杂版 | 有Jinja2和RAG问题 |
| `start_local_rag.sh` | ⚠️ 菜单脚本 | 会调用复杂版 |

---

## 🔄 如果需要 RAG 功能

后续可以:
1. 预下载 embedding 模型到本地
2. 修改简化版脚本添加 RAG
3. 或使用 Ollama + RAG 模式

---

## 💡 当前版本特点

**maoAstro LLM (简化版)**:
- 基于 Qwen2.5-7B + LoRA 微调
- 使用 3,413 条天文 QA 训练
- 无需外部依赖即可运行
- 支持中文天文专业问答

**启动后你可以问**:
- "什么是灾变变星?"
- "赫罗图的主序带是什么?"
- "如何测量变星周期?"
- "白矮星是如何形成的?"

---

## 验证测试

```bash
# 快速测试
python start_maoastro_simple.py

# 输入测试问题
> 什么是灾变变星?
> CV的轨道周期是多少?
> 赫罗图上的主序带特征?
```

如果一切正常，你会得到专业、准确的天文回答！
