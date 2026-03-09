# maoAstro LLM - 天文AI智能分析平台 🔭

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AstroPy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg)](https://www.astropy.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.com/)

## 项目简介

**maoAstro LLM** 是一个集成多种天文数据分析工具和人工智能大模型的智能分析平台。该平台整合了现代天文观测数据（ZTF、TESS、LAMOST、SDSS、Gaia等）与本地部署的大语言模型（通过 Ollama），为天文学家提供一站式的天体分析解决方案。

本项目特别适用于变星、激变变星（Cataclysmic Variables）、白矮星、双星系统等天体的综合分析。

---

## ✨ 核心特性

| 功能模块 | 描述 | 状态 |
|---------|------|------|
| 🤖 **LLM 智能分析** | 基于 Ollama 本地部署的大模型，支持多模态分析（文本+图像） | ✅ |
| 📚 **RAG 知识检索** | 天文领域知识库，支持向量检索和关键词匹配 | ✅ |
| 📊 **光变曲线分析** | ZTF/TESS 光变曲线获取与 Lomb-Scargle 周期分析 | ✅ |
| 🔬 **光谱分析** | LAMOST/SDSS 光谱检查与处理 | ✅ |
| 🌟 **赫罗图** | 基于 LAMOST DR10 背景的赫罗图绘制 | ✅ |
| 📈 **SED 分析** | 多波段能谱分布与黑体拟合 | ✅ |
| 🗄️ **数据查询** | SIMBAD/VizieR/Gaia/ astroquery 数据自动查询 | ✅ |
| 🌌 **消光查询** | CSFD/SFD 银河系消光地图查询 | ✅ |
| 🛰️ **TPF 像素图** | TESS 像素响应函数可视化 | ✅ |

---

## 📁 项目结构

```
maoAstro_llm/
├── src/                              # 源代码 (44个 Python 文件)
│   ├── ollama_qwen_interface.py      # Ollama LLM 接口与 Prompt 模板
│   ├── rag_system.py                 # 基础 RAG 知识库系统
│   ├── enhanced_rag_system.py        # 增强版 RAG（向量检索）
│   ├── intelligent_astro_analyzer.py # 智能天文分析器
│   ├── rag_with_distillation.py      # 知识蒸馏 RAG 系统
│   ├── astro_tools.py                # 基础天文工具（消光查询等）
│   ├── extended_tools.py             # 扩展工具（SIMBAD、ZTF、TESS）
│   ├── sed_plotter.py                # SED 绘图器
│   ├── hr_diagram_plotter.py         # 赫罗图绘制器
│   ├── lightcurve_processor.py       # 光变曲线处理
│   ├── spectrum_analyzer.py          # 光谱分析器
│   ├── multilingual_support.py       # 多语言支持
│   ├── multimodal_fusion.py          # 多模态融合
│   ├── explainable_analysis.py       # 可解释分析
│   ├── realtime_knowledge_update.py  # 实时知识更新
│   └── ...                           # 其他分析模块
│
├── VSP/                              # VSP Jupyter Notebooks (10个)
│   ├── VSP_LAMOST.ipynb              # LAMOST 数据处理
│   ├── VSP_ZTF.ipynb                 # ZTF 光变曲线
│   ├── VSP_TESS.ipynb                # TESS 数据处理
│   ├── VSP_GAIA.ipynb                # Gaia 数据查询
│   ├── VSP_Period.ipynb              # 周期分析
│   ├── VSP_Photometry.ipynb          # 测光处理
│   ├── VSP_ASASSN.ipynb              # ASAS-SN 数据
│   ├── VSP_Kinematics.ipynb          # 运动学分析
│   ├── VSP_EBV.ipynb                 # 消光计算
│   └── VSP_Auxiliary.ipynb           # 辅助工具
│
├── scripts/                          # 实用脚本
│   ├── download_extinction.py        # 下载消光地图数据
│   ├── test_llm.py                   # 测试 LLM 功能
│   └── check_setup.py                # 环境配置检查
│
├── docs/                             # 文档
├── examples/                         # 示例代码
├── tests/                            # 测试文件
├── data/                             # 数据目录（消光地图）
├── models/                           # 模型目录（Ollama 管理）
├── output/                           # 输出目录
├── README.md                         # 本文件
├── SETUP_GUIDE.md                    # 详细配置指南
├── requirements.txt                  # Python 依赖
└── setup.py                          # 安装脚本
```

---

## 📦 依赖包清单

### 基础科学计算
| 包名 | 版本 | 说明 |
|------|------|------|
| numpy | >=1.21.0 | 数值计算 |
| pandas | >=1.3.0 | 数据处理 |
| scipy | >=1.7.0 | 科学计算 |

### 天文数据处理
| 包名 | 版本 | 说明 |
|------|------|------|
| astropy | >=5.0.0 | 天文计算核心库 |
| astroquery | >=0.4.6 | 天文数据查询接口 |
| healpy | >=1.16.0 | HEALPix 球面像素化处理（消光地图） |
| lightkurve | >=2.3.0 | TESS/Kepler 光变曲线分析 |
| astroplan | >=0.8.0 | 天文观测计划 |
| dustmaps | >=1.0.0 | 消光地图查询 |

### AI/ML 与 LLM
| 包名 | 版本 | 说明 |
|------|------|------|
| transformers | >=4.35.0 | Hugging Face 模型库 |
| accelerate | >=0.24.0 | 模型加速 |
| bitsandbytes | >=0.41.0 | 量化支持 |
| ollama | >=0.1.0 | Ollama Python 接口 |
| openai | >=1.0.0 | OpenAI API 接口（兼容） |

### 数据可视化
| 包名 | 版本 | 说明 |
|------|------|------|
| matplotlib | >=3.5.0 | 绘图基础库 |
| seaborn | >=0.11.0 | 高级统计可视化 |
| Pillow | >=9.0.0 | 图像处理 |

### 其他工具
| 包名 | 版本 | 说明 |
|------|------|------|
| requests | >=2.27.0 | HTTP 请求 |
| tqdm | >=4.62.0 | 进度条 |
| jupyter | >=1.0.0 | Jupyter Notebook |
| ipython | >=8.0.0 | IPython 交互环境 |

### 可选依赖
| 包名 | 说明 |
|------|------|
| torch | PyTorch 深度学习框架（根据 CUDA 版本手动安装） |
| gradio | Web 界面（可选） |
| streamlit | Web 应用（可选） |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/wangnengdejiamao/maoAstro_llm.git
cd maoAstro_llm
```

### 2. 创建 AstroMLab 环境（强烈推荐⭐）

**AstroMLab** 是专为天文数据分析优化的 Python 环境：

```bash
# 创建 conda 环境
conda create -n astromlab python=3.9 -y
conda activate astromlab

# 安装天文包（优先使用 conda-forge）
conda install -c conda-forge astropy astroquery healpy -y
conda install -c conda-forge lightkurve astroplan -y

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 下载数据文件

#### 消光地图数据 (~900MB)

```bash
# 自动下载
python scripts/download_extinction.py

# 或手动下载后放到 data/ 目录
# - csfd_ebv.fits (CSFD 修正消光地图)
# - sfd_ebv.fits (SFD 原始消光地图)
# - lss_intensity.fits (LSS 强度图)
# - lss_error.fits (LSS 误差图)
# - mask.fits (数据质量掩码)
```

数据来源：
- **CSFD**: https://github.com/CPPMariner/CSFD (修正 SFD 消光地图)
- **SFD**: https://www.legacysurvey.org/ (原始消光地图)

### 4. 安装 Ollama 和 LLM 模型

```bash
# 安装 Ollama
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: 从 https://ollama.com/download 下载安装

# 下载推荐模型
ollama pull qwen3:8b      # 通义千问 8B（推荐⭐，中文优秀）
ollama pull llama3.1:8b   # Llama 3.1 8B（英文优秀）
ollama pull deepseek-r1:8b  # DeepSeek R1（推理能力强）

# 启动服务
ollama serve
```

### 5. 验证安装

```bash
# 检查所有组件
python scripts/check_setup.py

# 测试 LLM
python scripts/test_llm.py --model qwen3:8b
```

---

## 🤖 LLM 模型部署指南

### 模型推荐

| 模型 | 大小 | 特点 | 适用场景 |
|------|------|------|----------|
| **qwen3:8b** | ~4.7GB | 中文优秀，通用能力强 | 中文天文分析（推荐⭐） |
| **llama3.1:8b** | ~4.7GB | 英文优秀，开源生态好 | 英文文献分析 |
| **deepseek-r1:8b** | ~4.9GB | 推理能力强 | 复杂物理分析 |

### 自定义模型（AstroSage）

可以基于上述模型创建自定义的 AstroSage 天文专用模型：

```bash
# 创建 Modelfile
cat > Modelfile.astrosage << 'EOF'
FROM qwen3:8b

SYSTEM """你是一位专业的天体物理学家，擅长分析各类天体观测数据。你精通变星、激变变星、白矮星、双星系统等天体的物理特性。在分析时，你会基于观测事实给出科学合理的解释，不会编造数据或引用不存在的公式。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 构建模型
ollama create astrosage-local -f Modelfile.astrosage

# 使用模型
python -c "from src.ollama_qwen_interface import OllamaQwenInterface; \
           ollama = OllamaQwenInterface(model_name='astrosage-local')"
```

---

## 📚 RAG 知识库系统

本项目包含两种 RAG（检索增强生成）系统：

### 1. 基础 RAG (`rag_system.py`)

内置天文知识库，包含以下主题：

- **am_cvn**: AM CVn 型星（超紧密氦双星）
- **polar**: 高偏振星/Polar（强磁激变变星）
- **cataclysmic_variable**: 激变变星（CVs）
- **rr_lyrae**: RR Lyrae 变星
- **eclipsing_binary**: 食双星
- **white_dwarf**: 白矮星
- **delta_scuti**: δ Scuti 变星

使用示例：

```python
from src.rag_system import AstronomyRAG

# 初始化知识库
rag = AstronomyRAG()

# 搜索知识
result = rag.search("AM CVn 周期-光度关系")
print(result)

# 获取特定主题
knowledge = rag.get("polar")
print(knowledge)
```

### 2. 增强版 RAG (`enhanced_rag_system.py`)

支持向量检索的增强版系统：

```python
from src.enhanced_rag_system import EnhancedAstronomyRAG

# 初始化（自动构建向量索引）
rag = EnhancedAstronomyRAG(use_vector_store=True)

# 混合检索（向量 + 关键词）
result = rag.search("cataclysmic variable classification", top_k=5)
print(result)

# 按天体类型搜索
result = rag.search_by_object_type("polar")
print(result)
```

### 3. 智能分析器 (`intelligent_astro_analyzer.py`)

整合 RAG + LLM 的智能分析系统：

```python
from src.intelligent_astro_analyzer import IntelligentAstroAnalyzer

# 初始化分析器
analyzer = IntelligentAstroAnalyzer(
    ollama_model="qwen3:8b",
    use_rag=True,
    use_vision=True
)

# 分析天体数据
target_data = {
    'name': 'AM_Her',
    'ra': 270.35,
    'dec': 49.50,
    'period': 0.1289,  # 天
    'magnitudes': {'V': 12.3, 'G': 11.8},
    'extinction': {'A_V': 0.15}
}

result = analyzer.analyze_target(target_data)
print(result.raw_analysis)
```

---

## 📝 Prompt 模板系统

### 内置 Prompt 模板

`ollama_qwen_interface.py` 提供了多种专用 Prompt 模板：

#### 1. SED 图分析

```python
from src.ollama_qwen_interface import OllamaQwenInterface

ollama = OllamaQwenInterface()
result = ollama.analyze_sed_plot("path/to/sed.png")
```

Prompt 内容：
```
请分析这张光谱能量分布(SED)图：
1. 描述图中显示的波段覆盖范围
2. 分析流量随波长的变化趋势
3. 判断这可能是什么类型的天体（恒星、星系、激变变星等）
4. 指出任何异常或特殊特征
5. 给出后续观测建议
```

#### 2. 赫罗图分析

```python
result = ollama.analyze_hr_diagram("path/to/hr.png")
```

#### 3. 光变曲线分析

```python
result = ollama.analyze_light_curve("path/to/lc.png")
```

#### 4. 综合数据分析

```python
result = ollama.analyze_target_summary(data_json)
```

### 自定义 Prompt

```python
from src.ollama_qwen_interface import OllamaQwenInterface

ollama = OllamaQwenInterface(model_name="qwen3:8b")

system_prompt = "你是一位专业的恒星物理学家，擅长分析脉动变星。"

user_prompt = """
分析以下 RR Lyrae 变星的观测数据：
- 周期: 0.55 天
- 振幅: 0.8 星等
- 光谱型: A5
- 金属丰度: [Fe/H] = -1.5

请判断这是 RRab 还是 RRc 型，并解释原因。
"""

response = ollama.analyze_text(user_prompt, system_prompt=system_prompt)
print(response)
```

---

## 🌐 数据版权与引用

### 数据来源

本项目使用的天文数据来自以下公开数据库和 survey：

#### 1. LAMOST (郭守敬望远镜)
- **来源**: 中国国家天文台 LAMOST 项目
- **网址**: http://www.lamost.org/
- **数据发布**: LAMOST DR10 (及后续版本)
- **使用政策**: 遵守 LAMOST 数据发布政策
- **引用**: 
  ```
  Cui et al. 2012, RAA, 12, 1197
  Zhao et al. 2012, RAA, 12, 723
  ```

#### 2. ZTF (Zwicky Transient Facility)
- **来源**: 美国加州理工学院 / Palomar 天文台
- **网址**: https://www.ztf.caltech.edu/
- **数据发布**: ZTF Public Data Releases
- **使用政策**: 遵循 ZTF 数据使用条款
- **引用**:
  ```
  Bellm et al. 2019, PASP, 131, 068002
  Graham et al. 2019, PASP, 131, 078001
  ```

#### 3. TESS (Transiting Exoplanet Survey Satellite)
- **来源**: NASA
- **网址**: https://tess.mit.edu/
- **数据发布**: MAST Archive
- **使用政策**: 公共领域数据
- **引用**:
  ```
  Ricker et al. 2015, JATIS, 1, 014003
  ```

#### 4. SDSS (Sloan Digital Sky Survey)
- **来源**: SDSS 合作项目
- **网址**: https://www.sdss.org/
- **数据发布**: SDSS DR16/DR17
- **使用政策**: 遵循 SDSS 数据使用政策
- **引用**:
  ```
  Blanton et al. 2017, AJ, 154, 28
  ```

#### 5. Gaia
- **来源**: 欧洲航天局 (ESA)
- **网址**: https://www.cosmos.esa.int/web/gaia
- **数据发布**: Gaia DR3
- **使用政策**: 遵循 ESA/Gaia 数据使用条款
- **引用**:
  ```
  Gaia Collaboration et al. 2022, A&A, 674, A1
  ```

#### 6. 消光地图 (CSFD/SFD)
- **SFD 来源**: Schlegel, Finkbeiner & Davis 1998
- **CSFD 来源**: 修正版 SFD 地图
- **引用**:
  ```
  Schlegel, D. J., Finkbeiner, D. P., & Davis, M. 1998, ApJ, 500, 525
  ```

#### 7. SIMBAD/VizieR
- **来源**: 斯特拉斯堡天文数据 Centre (CDS)
- **网址**: https://simbad.u-strasbg.fr/ / https://vizier.u-strasbg.fr/
- **引用**:
  ```
  Wenger et al. 2000, A&AS, 143, 9 (SIMBAD)
  Ochsenbein et al. 2000, A&AS, 143, 23 (VizieR)
  ```

### 科学使用要求

1. **正确引用**: 使用本项目分析的数据发表科研成果时，请正确引用原始数据源
2. **致谢**: 建议在论文致谢部分提及相关 survey 和数据库
3. **数据质量**: 注意各数据集的观测限制和系统误差
4. **版权**: 本项目代码采用 MIT 许可证，但数据版权归原始数据提供方所有

---

## 🔬 使用示例

### 示例 1: 消光查询

```python
from src.astro_tools import query_extinction

# 查询坐标 (RA=13.1316°, DEC=53.8585°) 的消光
result = query_extinction(13.1316, 53.8585)
print(f"A_V = {result['av']:.3f} mag")
print(f"E(B-V) = {result['ebv']:.3f} mag")
```

### 示例 2: SIMBAD 查询

```python
from src.extended_tools import SIMBADQuery

simbad = SIMBADQuery()
result = simbad.query_object("AM Her")
print(result)
```

### 示例 3: ZTF 光变曲线

```python
from src.extended_tools import ZTFAnalyzer

ztf = ZTFAnalyzer()
lc = ztf.get_lightcurve(ra=270.35, dec=49.50, radius=3.0)
ztf.plot_lightcurve(lc)
```

### 示例 4: 完整分析流程

```python
from src.complete_analysis_system import CompleteAnalysisSystem

analysis = CompleteAnalysisSystem()
result = analysis.analyze(
    ra=13.1316,
    dec=53.8585,
    name="MyStar",
    run_ztf=True,
    run_tess=True,
    check_spectra=True
)
```

---

## ⚙️ 配置说明

### Ollama 配置

编辑 `src/ollama_qwen_interface.py` 或运行时指定：

```python
OllamaQwenInterface(
    model_name="qwen3:8b",           # 模型名称
    base_url="http://localhost:11434"  # 服务地址
)
```

### 数据路径配置

在 `src/astro_tools.py` 中修改：

```python
Config(
    EXTINCTION_FILE="./data/csfd_ebv.fits",  # 消光文件路径
    OUTPUT_DIR="./output"                     # 输出目录
)
```

---

## 🛠️ 故障排除

### 问题 1: Ollama 连接失败

```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 启动服务
ollama serve

# 检查模型是否下载
ollama list
```

### 问题 2: healpy 安装失败

```bash
# 使用 conda 安装（推荐）
conda install -c conda-forge healpy

# 或先安装依赖
sudo apt-get install libcfitsio-dev  # Ubuntu/Debian
brew install cfitsio                  # macOS
```

### 问题 3: 消光文件缺失

```bash
# 运行下载脚本
python scripts/download_extinction.py

# 或手动下载后放入 data/ 目录
```

### 问题 4: astroquery 查询超时

```python
# 增加超时时间
from astroquery.simbad import Simbad
Simbad.TIMEOUT = 60  # 秒
```

---

## 📖 文档

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - 详细安装配置指南
- [data/README.md](data/README.md) - 数据文件说明
- [docs/](docs/) - 技术文档
- [examples/](examples/) - 示例代码

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目代码采用 [MIT License](LICENSE) 开源许可证。

**注意**: 本项目中使用的第三方天文数据受各自数据提供方的版权和使用条款约束，使用时请遵守相关规定。

---

## 🙏 致谢

感谢以下项目和机构提供的数据与服务：

- **LAMOST** - 提供光谱数据
- **ZTF** - 提供光变曲线数据
- **TESS** - 提供系外行星搜寻数据
- **Gaia** - 提供天体测量数据
- **SDSS** - 提供光谱和成像数据
- **SIMBAD/VizieR** - 提供天体数据库服务
- **CSFD/SFD** - 提供消光地图
- **Qwen/Llama** - 提供大语言模型
- **Ollama** - 提供本地 LLM 运行框架
- **AstroPy** - 提供天文计算工具

---

## 📧 联系方式

如有问题或建议，欢迎提交 [GitHub Issue](https://github.com/wangnengdejiamao/maoAstro_llm/issues)。

---

**版本**: 2.0  
**更新日期**: 2026-03-09  
**维护者**: wangnengdejiamao
