# AstroSage 项目完全指南

> **项目简介**: AstroSage 是一个天文领域的垂直大模型智能分析平台，集成了数据采集、PDF解析、问答生成、模型训练和RAG检索增强等完整流程。

---

## 📚 目录

1. [项目架构](#项目架构)
2. [核心流程](#核心流程)
3. [文件清单](#文件清单)
4. [使用指南](#使用指南)
5. [开发文档](#开发文档)

---

## 项目架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AstroSage 系统架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   数据采集   │    │   PDF解析    │    │   QA生成     │                  │
│  │              │    │              │    │              │                  │
│  │ - astroquery │───→│ - pdfplumber │───→│ - 规则生成   │                  │
│  │ - 天文数据库 │    │ - 文本提取   │    │ - API增强    │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                  │                          │
│                                                  ↓                          │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              训练数据 (3,413条 QA对)                      │              │
│  │         train_qwen/data/qwen_train.json                  │              │
│  └──────────────────────────┬───────────────────────────────┘              │
│                             │                                               │
│                             ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              模型训练 (LoRA微调)                          │              │
│  │     Qwen2.5-7B + LoRA (r=64, 可训练161M参数)              │              │
│  │              显存需求: ~12GB                              │              │
│  └──────────────────────────┬───────────────────────────────┘              │
│                             │                                               │
│                             ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              AstroRAG 双轨检索系统                        │              │
│  │                                                          │              │
│  │   ┌─────────────┐         ┌─────────────┐                │              │
│  │   │  向量检索   │ ←─────→ │ 关键词检索  │                │              │
│  │   │  (Chroma)   │  融合   │ (倒排索引)  │                │              │
│  │   └──────┬──────┘         └──────┬──────┘                │              │
│  │          └───────────┬───────────┘                        │              │
│  │                      ↓                                    │              │
│  │              幻觉检测 + 引用溯源                          │              │
│  └──────────────────────────┬───────────────────────────────┘              │
│                             │                                               │
│                             ↓                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              用户问答接口                                │              │
│  │         - Ollama本地部署                                 │              │
│  │         - API服务                                        │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层次 | 技术 | 用途 |
|------|------|------|
| **数据处理** | pdfplumber, astroquery | PDF解析、天文数据获取 |
| **模型训练** | transformers, peft, bitsandbytes | LoRA微调、4-bit量化 |
| **向量检索** | chromadb, sentence-transformers | 语义搜索 |
| **关键词检索** | Whoosh/自定义倒排索引 | 精确匹配 |
| **RAG框架** | LangChain风格自定义 | 检索增强生成 |
| **部署** | Ollama, FastAPI | 本地/服务化部署 |

---

## 核心流程

### 流程1: 数据处理 (Data Pipeline)

```
天文论文PDF
    ↓
process_papers_smart.py  ─────────────────┐
    ↓                                      │
文本提取 (pdfplumber)                     │
    ↓                                      │
元数据解析 (标题/作者/摘要)                │
    ↓                                      │
generate_astronomy_qa_hybrid.py ◄─────────┘
    ↓
规则生成 QA (本地) ────┐
    ↓                   │
API增强 QA (Kimi) ─────┼──→ QA对存储
    ↓                   │
引用溯源标注 ◄─────────┘
    ↓
output/qa_hybrid/
    ├── qa_dataset_full.json      # 完整数据集 (20,609条)
    ├── cache/*.json              # 按PDF缓存
    └── stats.json                # 统计信息
```

**核心文件**:
- `process_papers_smart.py` - PDF处理入口
- `generate_astronomy_qa_hybrid.py` - QA生成入口
- `analyze_qa_results.py` - 数据分析

### 流程2: 模型训练 (Training Pipeline)

```
训练数据准备
    ↓
train_qwen/convert_to_qwen_format.py
    ↓
格式转换 (ShareGPT → Qwen格式)
    ↓
train_qwen/train_with_qwen25.py
    ↓
模型下载 (Qwen2.5-7B-Instruct)
    ↓
LoRA配置 (r=64, alpha=16)
    ↓
4-bit量化加载
    ↓
训练循环 (3 epochs)
    ↓
保存LoRA权重
train_qwen/output_qwen25/final_model/
```

**核心文件**:
- `train_qwen/convert_to_qwen_format.py` - 数据格式转换
- `train_qwen/train_with_qwen25.py` - 训练入口
- `train_qwen/inference.py` - 推理测试

### 流程3: RAG推理 (RAG Pipeline)

```
用户提问
    ↓
rag_pipeline.py
    ↓
┌─────────────────────┐
│   双轨并行检索       │
│                     │
│  ① 向量检索         │
│     ChromaDB        │
│     语义相似度       │
│                     │
│  ② 关键词检索       │
│     倒排索引        │
│     精确匹配        │
└──────────┬──────────┘
           ↓
    RRF重排序融合
           ↓
    Top-K 文档召回
           ↓
    LLM生成回答
           ↓
    幻觉检测
    - 指示词检查
    - 数值验证
           ↓
    引用溯源标注
           ↓
结构化输出 (answer + citations + confidence)
```

**核心文件**:
- `rag_system/retrieval/rag_pipeline.py` - RAG主流程
- `rag_system/retrieval/hybrid_retriever.py` - 混合检索
- `rag_system/vector_store/chroma_store.py` - 向量存储
- `rag_system/inverted_index/keyword_index.py` - 关键词索引

---

## 文件清单

### 核心文件 (必须保留)

| 类别 | 文件 | 说明 |
|------|------|------|
| **数据处理** | `process_papers_smart.py` | PDF处理和QA生成 |
| | `generate_astronomy_qa_hybrid.py` | 混合模式QA生成 |
| | `analyze_qa_results.py` | QA数据分析 |
| **模型训练** | `train_qwen/train_with_qwen25.py` | LoRA训练 (当前使用) |
| | `train_qwen/convert_to_qwen_format.py` | 数据格式转换 |
| | `train_qwen/inference.py` | 推理测试 |
| **RAG系统** | `rag_system/retrieval/rag_pipeline.py` | RAG主流程 |
| | `rag_system/retrieval/hybrid_retriever.py` | 双轨检索 |
| | `rag_system/vector_store/chroma_store.py` | 向量存储 |
| | `rag_system/inverted_index/keyword_index.py` | 关键词索引 |
| **VSP工具** | `lib/vsp/VSP_Lib.py` | VSP核心库 |
| | `lib/vsp/VSP_LAMOST.py` | LAMOST查询 |
| | `lib/vsp/VSP_GAIA.py` | GAIA查询 |
| | `lib/vsp/VSP_EBV.py` | 消光计算 |
| **工具脚本** | `start_local_rag.sh` | 启动RAG服务 |
| | `monitor_training.sh` | 训练监控 |

### 归档文件 (已废弃)

位于 `archive/deprecated/` 和 `archive/old_scripts/`:

```
archive/
├── deprecated/              # 已归档的重复文件
│   ├── generate_astronomy_qa.py
│   ├── train_qwen_lora.py
│   ├── train_lora_simple.py
│   ├── train_lora_local.py
│   ├── ultimate_analysis.py
│   └── ...
└── old_scripts/             # 早期开发脚本
    ├── main.py
    ├── ev_uma_analysis.py
    └── ... (20+文件)
```

### 实验性文件 (可选保留)

```
langgraph_demo/              # LangGraph实验
├── langgraph_demo.py        # 核心演示
├── ollama_rag_complete.py   # RAG完整版
└── ... (其他可删除)

src/                         # 早期分析脚本
├── astro_tools.py           # 工具函数
├── hr_diagram_plotter.py    # 赫罗图绘制
└── ... (许多版本迭代文件)

model_evaluation/            # 模型评估
├── astronomy_evaluator.py   # 评估器
└── ...
```

---

## 使用指南

### 快速开始

```bash
# 1. 环境准备
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
pip install -r requirements.txt

# 2. 启动Ollama+RAG (无需训练)
ollama serve
./start_local_rag.sh

# 3. 训练模型 (可选)
export HF_ENDPOINT=https://hf-mirror.com
python train_qwen/train_with_qwen25.py
```

### 数据流程

```bash
# 1. 处理PDF生成QA
python process_papers_smart.py

# 2. 分析生成的数据
python analyze_qa_results.py

# 3. 转换训练格式
python train_qwen/convert_to_qwen_format.py
```

### 训练流程

```bash
# 1. 监控训练
./monitor_training.sh

# 2. 查看日志
tail -f train_qwen/training_qwen25_7b.log

# 3. 推理测试
python train_qwen/inference.py
```

---

## 开发文档

### 添加新的QA主题

编辑 `generate_astronomy_qa_hybrid.py`:

```python
# 在 QAGenerator 类中添加新的主题处理器
def _generate_new_topic_questions(self, text, page_num):
    """生成新主题的问题"""
    questions = []
    
    # 检测关键词
    if "新主题关键词" in text:
        qa = {
            "question": "问题内容?",
            "answer": "答案内容",
            "category": "新主题",
            "source": f"{self.pdf_name} 第{page_num}页"
        }
        questions.append(qa)
    
    return questions
```

### 调整RAG检索权重

编辑 `rag_system/retrieval/hybrid_retriever.py`:

```python
retriever = HybridRetriever(
    vector_weight=0.7,      # 增加向量检索权重
    keyword_weight=0.3,     # 减少关键词检索权重
)
```

### 自定义幻觉检测

编辑 `rag_system/retrieval/rag_pipeline.py`:

```python
class HallucinationDetector:
    def __init__(self):
        # 添加自定义指示词
        self.hallucination_indicators = [
            "不确定", "可能", "也许",
            "your_custom_indicator",  # 添加新的
        ]
```

---

## 项目统计

| 指标 | 数值 |
|------|------|
| 总文件数 | ~120个Python文件 |
| 核心文件 | ~20个 |
| 已归档 | ~40个 |
| QA数据 | 20,609条 |
| 训练数据 | 3,413条 |
| 模型参数 | 7.7B (LoRA 161M) |

---

## 更新日志

- **2026-03-12**: 代码清理和文档整理
- **2026-03-11**: 集成Kimi API生成高质量QA
- **2026-03-10**: 完成双轨RAG系统开发
- **2026-03-06**: 项目初始化

---

**维护者**: AstroSage Team  
**许可证**: MIT License
