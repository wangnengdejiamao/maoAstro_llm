# Kimi API + Ollama 集成方案

## 概述

本方案结合 Kimi API（云端大模型）和本地 Ollama qwen3:8b，实现：
1. **数据生成**：用 Kimi API 生成高质量训练数据
2. **RAG 系统**：向量检索增强问答
3. **Tool 集成**：天文数据查询工具
4. **智能路由**：自动选择最合适的模型

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户查询                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     查询分析 & 路由                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │
│  │ 需要 Tool?  │───→│  简单查询   │───→│ Ollama qwen3:8b │     │
│  └─────────────┘    └─────────────┘    └─────────────────┘     │
│         │                 │                                      │
│         ↓                 ↓                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │
│  │  执行 Tool  │    │  复杂推理   │───→│   Kimi API      │     │
│  └─────────────┘    └─────────────┘    └─────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RAG 知识增强                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  向量数据库 (FAISS) + Embedding (BGE)                   │   │
│  │  ├─ 灾变变星知识                                         │   │
│  │  ├─ 变星分类学                                           │   │
│  │  └─ 观测方法指南                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      生成最终回答                                │
└─────────────────────────────────────────────────────────────────┘
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `kimi_data_generator.py` | 使用 Kimi API 生成训练数据、RAG 文档、Tool 示例 |
| `rag_tool_system.py` | 完整的 RAG + Tool + LLM 集成系统 |
| `model_distillation_ollama.py` | 使用软标签微调小型模型（可选） |

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install openai sentence-transformers faiss-cpu requests

# 确保 Ollama 运行
ollama serve
ollama list  # 确认 qwen3:8b 已安装
```

### 2. 生成训练数据（使用 Kimi API）

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python langgraph_demo/kimi_data_generator.py
```

这将生成：
- `output/kimi_generated/training_data.json` - 训练数据
- `output/kimi_generated/rag_documents.json` - RAG 知识库文档
- `output/kimi_generated/tool_examples.json` - Tool 调用示例
- `output/kimi_generated/summary.json` - 生成报告

### 3. 启动 RAG + Tool 系统

```bash
python langgraph_demo/rag_tool_system.py
```

## 使用方式

### 方式一：RAG + Tool 增强（推荐，无需训练）

```python
from langgraph_demo.rag_tool_system import AstronomyAssistant

# 初始化助手
assistant = AstronomyAssistant(use_kimi=True)

# 加载知识库
assistant.load_kimi_documents()

# 查询（纯知识）
result = assistant.query("什么是造父变星？")
print(result['response'])

# 查询（带数据）
result = assistant.query("分析 EV UMa", ra=13.1316, dec=53.8585)
print(result['response'])
```

### 方式二：训练小型专用模型

如果你希望训练一个完全离线的专用模型：

```bash
# 1. 先运行 Kimi 数据生成
python langgraph_demo/kimi_data_generator.py

# 2. 修改配置文件使用生成的数据
# 编辑 model_distillation_ollama.py 中的路径

# 3. 运行蒸馏训练
python langgraph_demo/model_distillation_ollama.py
```

### 方式三：混合模式（最佳效果）

```python
# 简单查询 → Ollama (快速、免费)
# 复杂分析 → Kimi API (准确、专业)
# RAG 增强 → 本地知识库
# 数据查询 → Tool 工具
```

## 交互命令

启动交互模式后可用命令：

```
/exit   - 退出程序
/tools  - 查看可用工具
/docs   - 查看知识库状态
```

## API Key

当前配置的 Kimi API Key:
```
19cb2d77-5ef2-8672-8000-0000a0d97edd
```

如需更换，修改：
- `kimi_data_generator.py` 中的 `KimiConfig`
- `rag_tool_system.py` 中的 `AstronomyAssistant`

## 模型对比

| 特性 | Ollama qwen3:8b | Kimi API |
|------|-----------------|----------|
| **成本** | 免费 | 按 token 计费 |
| **速度** | 本地运行，快 | 网络依赖 |
| **质量** | 良好 | 优秀 |
| **离线** | 是 | 否 |
| **适用场景** | 日常查询 | 复杂分析 |

## 扩展建议

### 1. 添加更多工具

```python
# 在 AstronomyTools 类中添加新方法
def query_ztf(self, ra: float, dec: float):
    """查询 ZTF 光变曲线"""
    # 实现 ZTF API 调用
    pass
```

### 2. 扩展知识库

```python
# 添加更多主题到 kimi_data_generator.py
topics = [
    "你的新主题 1",
    "你的新主题 2"
]
```

### 3. 自定义嵌入模型

```python
# 修改 RAGConfig
config = RAGConfig(
    embedding_model="你的模型路径"
)
```

## 故障排除

### Ollama 连接失败
```bash
# 检查服务状态
ollama serve
# 或
systemctl status ollama
```

### Kimi API 错误
- 检查 API Key 是否有效
- 检查网络连接
- 查看 API 额度

### 向量库加载失败
```bash
# 重新生成
rm -rf langgraph_demo/output/vector_db
python langgraph_demo/kimi_data_generator.py
```

## 后续优化方向

1. **模型微调**：用 Kimi 生成的数据微调 Qwen2.5-7B
2. **Agent 框架**：集成 LangGraph 实现多步骤任务
3. **Web 界面**：使用 Gradio 或 Streamlit 创建 UI
4. **分布式**：多用户共享知识库和模型
