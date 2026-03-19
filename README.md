# maoAstro LLM: 天文AI智能分析与训练平台 🔭

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

**maoAstro LLM** 是一个集成天文数据分析、大模型训练与部署的智能平台。该平台整合了现代天文观测数据（ZTF、TESS、LAMOST、SDSS、Gaia等）与本地部署的大语言模型（Qwen、Llama等），提供从数据处理到模型训练再到智能问答的完整解决方案。

### 核心特性

| 功能模块 | 描述 | 状态 |
|---------|------|------|
| 🤖 **大模型训练** | 支持Qwen、Llama等模型的LoRA微调 | ✅ |
| 📚 **RAG检索系统** | 双轨检索（向量+关键词）增强问答 | ✅ |
| 📝 **QA自动生成** | 从PDF论文自动生成训练数据 | ✅ |
| 📊 **光变曲线分析** | ZTF/TESS光变曲线获取与周期分析 | ✅ |
| 🔬 **光谱分析** | LAMOST/SDSS光谱检查与处理 | ✅ |
| 🌟 **赫罗图** | 基于LAMOST DR10背景的赫罗图绘制 | ✅ |
| 📈 **SED分析** | 多波段能谱分布(SED)与黑体拟合 | ✅ |
| 🗄️ **数据查询** | SIMBAD/VizieR/Gaia数据自动查询 | ✅ |
| 🌌 **消光查询** | CSFD/SFD银河系消光地图查询 | ✅ |

---

## 🚀 快速开始

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/wangnengdejiamao/maoAstro_llm.git
cd maoAstro_llm

# 安装依赖
pip install -r requirements.txt
```

### 1. 启动智能问答系统（推荐）

```bash
# 启动带RAG的大模型问答
python start_maoastro_with_simple_rag.py
```

### 2. 训练自己的天文模型

```bash
# 使用Llama3.1中文进行训练
python train_alternative_model.py --model llama3-chinese

# 或使用AstroSage 8B继续微调
python export_astrosage_simple.py
python train_qwen/astrosage_export/finetune_transformers.py
```

### 3. 导出到Ollama部署

```bash
python export_llama31_to_ollama.py \
    --model train_qwen/astrosage_continued/final_model \
    --name maoAstro-finetuned
```

---

## 📁 项目结构

```
maoAstro_llm/
├── main.py                          # 天文数据分析主入口
├── query_extinction.py              # 消光查询工具
├── start_maoastro_simple.py         # 简化版启动 ⭐
├── start_maoastro_with_simple_rag.py # RAG问答启动 ⭐
│
├── train_qwen/                      # 模型训练系统 ⭐
│   ├── README.md
│   ├── train_with_qwen25.py        # Qwen训练
│   ├── train_alternative_model.py  # 替代模型训练
│   ├── finetune_transformers.py    # Transformers微调
│   ├── inference.py                # 模型推理
│   ├── export_llama31_to_ollama.py # 导出到Ollama
│   ├── astrosage_export/           # AstroSage导出工具
│   ├── astrosage_continued/        # 训练输出
│   └── data/                       # 训练数据
│
├── rag_system/                      # RAG检索系统 ⭐
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # 混合检索
│   │   ├── rag_pipeline.py        # RAG管道
│   │   └── rag_with_ollama.py     # Ollama集成
│   ├── vector_store/               # 向量存储
│   └── inverted_index/             # 关键词索引
│
├── generate_astronomy_qa_hybrid.py  # QA自动生成 ⭐
├── analyze_qa_results.py            # QA分析
├── process_papers_smart.py          # PDF处理
│
├── src/                             # 天文分析源代码
│   ├── integrated_analysis.py
│   ├── complete_analysis_system.py
│   ├── astro_tools.py
│   ├── sed_plotter.py
│   ├── hr_diagram_plotter.py
│   └── ollama_qwen_interface.py
│
├── lib/vsp/                         # VSP (Variable Star Pipeline)
├── VSP/                             # VSP Notebooks
├── data/                            # 数据文件
├── output/                          # 输出目录
├── docs/                            # 文档
└── README.md                        # 本文件
```

---

## 🎯 核心功能详解

### 1. 大模型训练系统

支持多种模型的LoRA微调：

```bash
# Qwen2.5训练
python train_qwen/train_with_qwen25.py

# Llama3.1训练  
python train_alternative_model.py --model llama3-chinese

# AstroSage 8B继续微调
python export_astrosage_simple.py
python train_qwen/astrosage_export/finetune_transformers.py
```

**训练配置**：
- 基础模型: Qwen2.5-7B / Llama3.1-8B / AstroSage-8B
- 训练数据: 3,413条天文QA对
- 方法: LoRA (r=64, alpha=16)
- 显存需求: ~12GB (4-bit量化)

### 2. RAG双轨检索系统

结合向量检索和关键词检索：

```python
from rag_system.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    vector_weight=0.6,
    keyword_weight=0.4
)
results = retriever.retrieve("CV的轨道周期是多少?", top_k=3)
```

**特点**：
- 向量检索: 语义理解
- 关键词检索: 精确匹配
- 引用溯源: 显示答案来源

### 3. QA自动生成

从PDF论文自动生成训练数据：

```bash
# 配置API Key (环境变量)
export MOONSHOT_API_KEYS="sk-your-key"

# 生成QA
python generate_astronomy_qa_hybrid.py \
    --input ./papers \
    --output ./qa_output \
    --use-api
```

**生成模式**：
- 规则模式: 基于模板 (无需API)
- API模式: 大模型生成 (需要API Key)

### 4. 天文数据分析

```bash
# 基础分析
python main.py --ra 13.1316 --dec 53.8585 --name "MyStar"

# 完整分析
python main.py --ra 13.1316 --dec 53.8585 --mode complete

# 仅消光查询
python main.py --ra 13.1316 --dec 53.8585 --mode extinction
```

---

## 📚 新增文档

| 文档 | 说明 |
|------|------|
| `PROJECT_OVERVIEW.md` | 项目完整概览 |
| `TRAINING_GUIDE.md` | 模型训练完整指南 |
| `LLAMA31_TRAINING_GUIDE.md` | Llama3.1训练专用 |
| `RAG_FIX_GUIDE.md` | RAG系统修复指南 |
| `START_GUIDE.md` | 快速启动指南 |
| `MODEL_EVALUATION_REPORT.md` | 模型评估报告 |
| `UPDATE_COMPLETE.md` | 本次更新说明 |

---

## 🔐 API Key 配置

部分功能需要配置API Key：

```bash
# 方法1: 环境变量 (推荐)
export MOONSHOT_API_KEYS="sk-your-key-1,sk-your-key-2"

# 方法2: 修改脚本
# 编辑 generate_astronomy_qa_hybrid.py
API_KEYS = ["sk-your-actual-key"]
```

**注意**: 所有示例API Key已清理为 `YOUR_API_KEY_HERE`，使用前需配置。

---

## 📊 数据要求

### 消光地图
- `csfd_ebv.fits` - CSFD E(B-V) 修正消光地图
- `sfd_ebv.fits` - 原始SFD消光地图
- `lss_intensity.fits` - LSS强度图

### 训练数据
- `train_qwen/data/qwen_train.json` - 训练集 (3,413条)
- `train_qwen/data/qwen_val.json` - 验证集 (380条)

---

## 🛠️ 故障排除

### Ollama连接失败
```bash
# 检查服务
curl http://localhost:11434/api/tags

# 启动服务
ollama serve

# 拉取模型
ollama pull qwen3:8b
```

### 训练显存不足
```bash
# 修改训练脚本减小batch size
# 编辑 train_qwen/train_with_qwen25.py
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

---

## 📝 使用示例

### 示例1: 训练并部署模型

```bash
# 1. 训练模型
python train_alternative_model.py --model llama3-chinese --epochs 3

# 2. 导出到Ollama
python export_llama31_to_ollama.py \
    --model train_qwen/maoAstro-llama3-chinese/final_model \
    --name maoAstro-llama3

# 3. 使用
ollama run maoAstro-llama3
```

### 示例2: RAG问答

```bash
python start_maoastro_with_simple_rag.py

# 输入问题
> 什么是灾变变星?
> /rag 赫罗图上的主序带特征
```

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发规范
1. 遵循PEP 8代码风格
2. 添加适当的文档字符串
3. 不要提交API Key
4. 大文件(>100MB)使用Git LFS

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢

- **LAMOST**: 提供光谱数据
- **ZTF**: 提供光变曲线数据
- **TESS**: 提供系外行星搜寻数据
- **Gaia**: 提供天体测量数据
- **Qwen**: 阿里云通义千问大模型
- **Llama**: Meta开源大模型

---

**版本**: 3.0  
**更新日期**: 2026-03-20  
**更新内容**: 新增模型训练系统和RAG检索系统
