# maoAstro LLM

<p align="center">
  <strong>天文领域专用大语言模型与智能分析平台</strong><br>
  <em>Integrating Modern Astronomical Data (ZTF, TESS, LAMOST, SDSS, Gaia) with Domain-Specific LLM</em>
</p>

<p align="center">
  <a href="#features">功能特性</a> •
  <a href="#quick-start">快速开始</a> •
  <a href="#model">模型</a> •
  <a href="#tools">工具箱</a> •
  <a href="#citation">引用</a>
</p>

---

## 📋 目录

1. [项目简介](#项目简介)
2. [功能特性](#功能特性)
3. [模型](#模型)
4. [快速开始](#快速开始)
5. [工具箱详解](#工具箱详解)
6. [数据支持](#数据支持)
7. [参考文献](#参考文献)
8. [版权与许可](#版权与许可)

---

## 项目简介

**maoAstro LLM** 是一个集成现代天文观测数据与人工智能大模型的智能分析平台。该平台整合了 ZTF、TESS、LAMOST、SDSS、Gaia 等主流天文观测数据，结合本地部署的大语言模型，为天文学家提供一站式的天体物理分析解决方案。

### 核心能力

- 🔭 **多源数据整合**: 支持 ZTF、TESS、LAMOST、SDSS、Gaia 等主流天文数据库的统一查询与分析
- 🤖 **领域专用LLM**: 基于 Qwen2.5-7B 微调的天文专用模型，或直接使用 AstroSage-LLaMA-3.1-8B
- 📚 **RAG知识增强**: 双轨检索系统（向量检索+关键词检索），支持PDF文献知识库
- 📊 **多模态分析**: 光变曲线、光谱、能谱分布(SED)、赫罗图等多种分析维度
- 🌐 **跨语言支持**: 支持中英文天文文献处理与问答

---

## 功能特性

### 1. 天文数据分析 (src/)

| 模块 | 文件 | 功能描述 |
|------|------|----------|
| **统一查询** | `unified_astro_query.py` | 整合 ZTF/TESS/LAMOST/SDSS/Gaia 的统一天文数据查询接口 |
| **光变分析** | `lightcurve_processor.py` | ZTF/TESS 光变曲线下载、周期分析、时域特征提取 |
| **光谱分析** | `spectrum_analyzer.py` | LAMOST/SDSS 光谱数据处理、谱线识别、红移测量 |
| **能谱分析** | `sed_plotter.py` | 能谱分布(SED)计算与可视化、多波段测光整合 |
| **赫罗图** | `hr_diagram_plotter.py` | Gaia 数据驱动的赫罗图绘制与恒星分类 |
| **智能分析** | `intelligent_astro_analyzer.py` | AI增强的天体综合分析，整合RAG与大模型推理 |
| **完整分析** | `complete_analysis_system.py` | 一站式天体分析流程，从查询到可视化报告 |

### 2. AI大模型系统

#### 2.1 训练系统 (train_qwen/)

```
train_qwen/
├── train_with_qwen25.py      # Qwen2.5-7B LoRA 训练脚本
├── inference.py              # 模型推理与文本生成
├── merge_lora.py             # LoRA 权重合并
├── convert_to_qwen_format.py # 数据格式转换
└── data/qwen_train.json      # 20,609 QA训练数据
```

**训练参数**:
- 基础模型: Qwen/Qwen2.5-7B-Instruct
- 训练方法: LoRA (Low-Rank Adaptation)
- LoRA rank: 64, alpha: 16
- 训练数据: 20,609 条天文领域QA对
- 训练轮次: 3 epochs
- 损失变化: 0.34 → 0.21

#### 2.2 RAG检索系统 (rag_system/)

```
rag_system/
├── vector_store/chroma_store.py      # ChromaDB 向量数据库
├── inverted_index/keyword_index.py   # Whoosh 关键词索引
└── retrieval/
    ├── hybrid_retriever.py           # 混合检索（向量+关键词）
    └── rag_pipeline.py               # RAG 完整流程
```

**检索策略**:
- 向量检索: 基于 ChromaDB 的语义相似度搜索
- 关键词检索: 基于 Whoosh 的精确匹配
- 混合权重: 向量权重 0.6 + 关键词权重 0.4
- 引用溯源: 每个回答附带文献来源

### 3. 辅助工具

| 工具 | 功能 |
|------|------|
| `generate_astronomy_qa_hybrid.py` | 从PDF生成QA对（规则+API混合模式） |
| `use_astrosage_with_rag.py` | 使用 AstroSage 模型 + RAG |
| `evaluate_model.py` | 模型性能评估 |
| `test_manual_eval.py` | 人工评估测试 |
| `export_astrosage_simple.py` | 导出为 Ollama 格式 |
| `download_data.py` | 天文数据批量下载 |

---

## 模型

### 推荐模型: AstroSage-LLaMA-3.1-8B

来自 [AstroMLab](https://astromlab.org/ollama.html) 的天文领域专用模型。

**模型特点**:
- 架构: LLaMA 3.1 8B
-  specialization: 天体物理、天文学
- 量化版本: BF16 (高精度) / Q8_0 (高效)
- 部署: 支持 llama.cpp / Ollama
- 硬件: CPU 友好，普通笔记本可运行

**下载与使用**:

```bash
# 方法1: 使用 Ollama
ollama pull astromlab/astrosage-llama3.1-8b

# 方法2: 从 HuggingFace 下载 GGUF
# https://huggingface.co/astromlab/AstroSage-LLaMA-3.1-8B

# 启动模型
python src/intelligent_astro_analyzer.py --model astromlab/astrosage-llama3.1-8b
```

### 自训练模型: maoAstro-Qwen2.5-7B

**训练数据构成**:
- PDF文献: 242 篇天体物理论文
- QA对总数: 20,609 条
  - API生成: 3,793 条 (高质量)
  - 规则生成: 16,816 条 (基础)
- 主题覆盖: 灾变变星、白矮星、吸积盘、光变曲线、光谱分析等

**模型性能**:
- 训练损失: 0.34 → 0.21
- 评估指标: 见 `evaluate_model.py` 输出
- 推理速度: ~50 tokens/s (RTX 3080 Ti)

---

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/wangnengdejiamao/maoAstro_llm.git
cd maoAstro_llm

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用 AstroSage 模型（推荐）

```bash
# 1. 安装 Ollama
# 访问 https://ollama.com 下载安装

# 2. 拉取模型
ollama pull astromlab/astrosage-llama3.1-8b

# 3. 启动智能分析
python src/intelligent_astro_analyzer.py
```

### 使用自训练模型

```bash
# 1. 准备训练数据
python generate_astronomy_qa_hybrid.py \
    --input ./data/papers \
    --output ./output/qa_hybrid \
    --use-api

# 2. 训练模型
cd train_qwen
python train_with_qwen25.py

# 3. 合并 LoRA 权重
python merge_lora.py

# 4. 启动推理
python start_maoastro_with_simple_rag.py
```

### 天文数据分析示例

```bash
# 统一查询天体的多波段数据
python src/unified_astro_query.py \
    --ra 189.0856 \
    --dec 11.2324 \
    --radius 2.0

# 分析光变曲线
python src/lightcurve_processor.py \
    --target "ZTF J1901+1108" \
    --period-search

# 绘制赫罗图
python src/hr_diagram_plotter.py \
    --catalog gaia \
    --region "180-200,10-15"
```

---

## 工具箱详解

### 数据查询类

#### unified_astro_query.py
统一天文数据查询接口，整合多个天文数据库。

**支持数据源**:
- **ZTF**: Zwicky Transient Facility 光学时域巡天
- **TESS**: 凌星系外行星巡天卫星光变数据
- **LAMOST**: 郭守敬望远镜光谱数据 (DR10)
- **SDSS**: 斯隆数字巡天光谱与测光
- **Gaia**: 盖亚空间望远镜天体测量与测光

**功能**:
- 坐标交叉匹配 (cone search)
- 多波段测光数据整合
- 光谱自动下载与预处理
- 数据质量标记与筛选

#### complete_astro_download.py
批量天文数据下载管理器。

**特性**:
- 断点续传
- 多线程并行下载
- 自动解压与格式转换
- 下载进度可视化

### 数据分析类

#### lightcurve_processor.py
光变曲线专业分析工具。

**功能模块**:
- 周期搜索 (Lomb-Scargle, BLS, PDM)
- 相位折叠与可视化
- 时域特征提取 (振幅、偏度、峰度)
- 变星分类辅助

#### spectrum_analyzer.py
光谱分析工具箱。

**功能**:
- 谱线识别 (Hα, Hβ, Ca II, Na I D等)
- 红移/视向速度测量
- 光谱分类 (基于MK系统)
- 大气参数估计 (Teff, log g, [Fe/H])

#### sed_plotter.py
能谱分布(SED)分析工具。

**功能**:
- 多波段测光整合
- 黑体拟合与温度估计
- 消光校正
- 恒星参数推导 (L, R, d)

#### hr_diagram_plotter.py
赫罗图绘制与恒星演化分析。

**功能**:
- Gaia BP/RP/ G 波段测光
- 绝对星等计算 (距离校正)
- 等龄线/等质量线叠加
- 恒星分类 (主序、巨星、白矮星)

### AI增强类

#### intelligent_astro_analyzer.py
智能天文分析器，整合RAG与大模型。

**工作流程**:
1. 输入天体坐标或名称
2. 自动查询多源数据
3. RAG检索相关文献知识
4. LLM生成综合分析报告
5. 输出结构化结果

**输出内容**:
- 天体分类与物理参数
- 科学重要性评估
- 后续观测建议
- 参考文献列表

#### astro_agent.py
天文AI智能体，支持工具调用。

**可用工具**:
- 数据查询工具
- 光变分析工具
- 光谱分析工具
- 可视化工具

### RAG系统类

#### enhanced_rag_system.py
增强版RAG知识库系统。

**特性**:
- 双轨检索（向量+关键词）
- 重排序(Re-ranking)优化
- 多文档摘要融合
- 引用溯源与验证

#### rag_with_distillation.py
RAG与模型蒸馏整合系统。

**功能**:
- 知识蒸馏生成合成数据
- 模型能力评估与对比
- 持续学习与知识更新

---

## 数据支持

### 星表与数据库

| 数据源 | 类型 | 数据量 | 访问方式 |
|--------|------|--------|----------|
| ZTF DR20 | 光学时域 | 50亿+ 光源 | ZTF Archive |
| TESS | 空间光变 | 2亿+ 恒星 | MAST/TESSCut |
| LAMOST DR10 | 低分辨率光谱 | 2000万+ 光谱 | CASDA/VizieR |
| SDSS DR17 | 光谱+成像 | 5亿+ 天体 | SDSS SkyServer |
| Gaia DR3 | 天体测量+测光 | 18亿+ 恒星 | Gaia Archive |

### 本地数据

`lib/` 目录包含以下预下载数据:
- GCVS 2022: 变星总表
- VSX: 变星指数
- MWDD: 白矮星数据库
- SDSS DR7 白矮星星表
- NASA系外行星档案

---

## 参考文献

### 天文数据

1. **ZTF**: Bellm, E. C., et al. (2019). The Zwicky Transient Facility: System Overview, Performance, and First Results. *Publications of the Astronomical Society of the Pacific*, 131(995), 018002.

2. **TESS**: Ricker, G. R., et al. (2015). Transiting Exoplanet Survey Satellite (TESS). *Journal of Astronomical Telescopes, Instruments, and Systems*, 1(1), 014003.

3. **LAMOST**: Cui, X.-Q., et al. (2012). The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST). *Research in Astronomy and Astrophysics*, 12(9), 1197.

4. **SDSS**: York, D. G., et al. (2000). The Sloan Digital Sky Survey: Technical Summary. *The Astronomical Journal*, 120(3), 1579.

5. **Gaia**: Gaia Collaboration (2016). The Gaia mission. *Astronomy & Astrophysics*, 595, A1.

### 大语言模型

6. **Qwen2.5**: Qwen Team (2024). Qwen2.5: A Party of Foundation Models. arXiv:2412.15115.

7. **LLaMA 3.1**: Dubey, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

8. **AstroSage**: de Haan, T., et al. (2024). AstroSage: A Large Language Model for Astronomy. *AstroMLab Technical Report*.

### 方法论文献

9. **LoRA**: Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

10. **RAG**: Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

---

## 版权与许可

### 代码许可

本项目采用 **MIT License** 开源许可。

```
MIT License

Copyright (c) 2024-2026 maoAstro Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### 数据使用声明

- **ZTF 数据**: 基于 ZTF 公共数据发布，遵循 IPAC/NExScI 数据使用政策
- **TESS 数据**: NASA 公共领域数据
- **LAMOST 数据**: 遵循 LAMOST 数据发布政策
- **SDSS 数据**: 遵循 SDSS 数据发布政策
- **Gaia 数据**: ESA/Gaia/DPAC 公共数据

### 第三方模型

- **AstroSage-LLaMA-3.1-8B**: 版权归属 [AstroMLab](https://astromlab.org/)，遵循相应许可协议
- **Qwen2.5**: 版权归属 Alibaba Cloud，遵循 Qwen License

### 致谢

感谢以下项目和机构的支持:
- [AstroMLab](https://astromlab.org/) 提供 AstroSage 模型
- [AstroPy](https://www.astropy.org/) 天文 Python 生态系统
- [HuggingFace](https://huggingface.co/) 模型托管服务
- [Ollama](https://ollama.com/) 本地模型部署工具

---

## 联系方式

- 项目主页: https://github.com/wangnengdejiamao/maoAstro_llm
- 问题反馈: [GitHub Issues](https://github.com/wangnengdejiamao/maoAstro_llm/issues)
- 邮件联系: [待添加]

---

<p align="center">
  <sub>Made with ❤️ for Astronomy</sub>
</p>
