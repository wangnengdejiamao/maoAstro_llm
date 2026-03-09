# maoAstro LLM - 天文AI智能分析平台 🔭

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AstroPy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg)](https://www.astropy.org/)

## 项目简介

**maoAstro LLM** 是一个集成多种天文数据分析工具和人工智能大模型的智能分析平台。该平台整合了现代天文观测数据（ZTF、TESS、LAMOST、SDSS、Gaia等）与本地部署的 Qwen 大语言模型，为天文学家提供一站式的天体分析解决方案。

### 核心特性

| 功能模块 | 描述 | 状态 |
|---------|------|------|
| 🤖 **LLM 智能分析** | 基于 Ollama 本地部署的 Qwen3/Llama3 大模型，提供智能数据解读 | ✅ |
| 📊 **光变曲线分析** | ZTF/TESS 光变曲线获取与 Lomb-Scargle 周期分析 | ✅ |
| 🔬 **光谱分析** | LAMOST/SDSS 光谱检查与处理 | ✅ |
| 🌟 **赫罗图** | 基于 LAMOST DR10 背景的赫罗图绘制 | ✅ |
| 📈 **SED 分析** | 多波段能谱分布 (SED) 与黑体拟合 | ✅ |
| 🗄️ **数据查询** | SIMBAD/VizieR/Gaia 数据自动查询 | ✅ |
| 🌌 **消光查询** | CSFD/SFD 银河系消光地图查询 | ✅ |
| 🛰️ **TPF 像素图** | TESS 像素响应函数 (TPF) 可视化 | ✅ |

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/maoAstro_llm.git
cd maoAstro_llm
```

### 2. 配置环境

```bash
# 创建 AstroMLab 环境（推荐）
conda create -n astromlab python=3.9 -y
conda activate astromlab

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载数据文件

```bash
# 下载消光地图 (~900MB)
python scripts/download_extinction.py

# 安装 LLM 模型
ollama pull qwen3:8b
```

> 📖 详细配置指南: [SETUP_GUIDE.md](SETUP_GUIDE.md)

### 4. 验证安装

```bash
# 检查所有组件
python scripts/check_setup.py

# 测试 LLM
python scripts/test_llm.py
```

### 5. 运行分析

```python
from src.ollama_qwen_interface import OllamaQwenInterface
from src.astro_tools import query_extinction

# 查询消光
result = query_extinction(13.1316, 53.8585)
print(f"A_V = {result['av']:.3f} mag")

# LLM 分析
ollama = OllamaQwenInterface(model_name="qwen3:8b")
response = ollama.analyze_text("分析这个恒星的特征")
print(response)
```

---

## 📁 项目结构

```
maoAstro_llm/
├── src/                         # 源代码
│   ├── astro_tools.py           # 基础天文工具
│   ├── extended_tools.py        # 扩展工具 (SIMBAD等)
│   ├── sed_plotter.py           # SED 绘图器
│   ├── hr_diagram_plotter.py    # 赫罗图绘制器
│   ├── ollama_qwen_interface.py # Ollama LLM 接口
│   └── ...
│
├── VSP/                         # VSP Jupyter Notebooks
│   ├── VSP_LAMOST.ipynb
│   ├── VSP_ZTF.ipynb
│   ├── VSP_TESS.ipynb
│   └── ...
│
├── data/                        # 消光地图数据（需下载）
│   ├── csfd_ebv.fits
│   ├── sfd_ebv.fits
│   └── ...
│
├── models/                      # LLM 模型（Ollama 管理）
├── output/                      # 输出目录
├── scripts/                     # 实用脚本
│   ├── download_extinction.py   # 下载消光文件
│   ├── test_llm.py              # 测试 LLM
│   └── check_setup.py           # 环境检查
│
├── examples/                    # 示例代码
├── docs/                        # 文档
├── tests/                       # 测试文件
├── README.md                    # 本文件
├── SETUP_GUIDE.md               # 详细配置指南
└── requirements.txt             # 依赖清单
```

---

## 📦 数据与模型

### 消光地图数据 (~900MB)

用于银河系消光计算，需要单独下载：

```bash
python scripts/download_extinction.py
```

| 文件 | 大小 | 说明 |
|------|------|------|
| `csfd_ebv.fits` | ~200MB | CSFD 修正消光地图 |
| `sfd_ebv.fits` | ~200MB | SFD 原始消光地图 |
| `lss_intensity.fits` | ~200MB | LSS 强度图 |
| `lss_error.fits` | ~200MB | LSS 误差图 |
| `mask.fits` | ~100MB | 数据质量掩码 |

### LLM 模型

使用 [Ollama](https://ollama.com) 本地运行大语言模型：

```bash
# 安装 Ollama
# 访问 https://ollama.com/download

# 下载推荐模型
ollama pull qwen3:8b      # 通义千问，中文优秀（推荐⭐）
ollama pull llama3.1:8b   # Meta Llama，英文优秀
ollama pull deepseek-r1:8b  # DeepSeek，推理能力强
```

---

## 🔧 功能详解

### 1. LLM 智能分析

基于 Ollama 本地部署的大模型，提供：
- 自然语言交互式数据分析
- 自动解读天文图表
- 科学建议与异常检测

```python
from src.ollama_qwen_interface import OllamaQwenInterface

ollama = OllamaQwenInterface(model_name="qwen3:8b")
analysis = ollama.analyze_text(
    "分析这个恒星的光变曲线特征", 
    system_prompt="你是一位天文学家..."
)
```

### 2. ZTF/TESS 光变曲线分析

- 自动从 ZTF/TESS 数据库获取光变曲线
- Lomb-Scargle 周期分析
- 相位折叠与可视化

### 3. 赫罗图 (HR Diagram)

- 基于 LAMOST DR10 背景恒星
- 显示目标星在 HR 图上的位置
- 支持消光修正

### 4. SED 分析与黑体拟合

- 多波段数据收集 (GALEX, Gaia, 2MASS, WISE)
- 黑体谱拟合计算温度
- 残差分析

### 5. 消光查询

```python
from src.astro_tools import query_extinction

result = query_extinction(13.1316, 53.8585)
print(f"A_V = {result['av']:.3f} mag")
```

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | 完整配置指南（GitHub 上传、环境配置、数据下载） |
| [data/README.md](data/README.md) | 消光文件说明 |
| [examples/](examples/) | 示例代码 |
| [docs/](docs/) | 详细文档 |

---

## 🛠️ 故障排除

### Ollama 连接失败

```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 启动服务
ollama serve

# 下载模型
ollama pull qwen3:8b
```

### 消光文件缺失

```bash
# 重新下载
python scripts/download_extinction.py --force
```

### 环境检查

```bash
# 运行完整检查
python scripts/check_setup.py
```

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

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢

- **LAMOST**: 提供光谱数据
- **ZTF**: 提供光变曲线数据
- **TESS**: 提供系外行星搜寻数据
- **Gaia**: 提供天体测量数据
- **SFD/CSFD**: 提供消光地图
- **Qwen**: 阿里云通义千问大模型
- **Ollama**: 本地 LLM 运行框架

---

## 📧 联系方式

如有问题或建议，欢迎提交 [GitHub Issue](https://github.com/YOUR_USERNAME/maoAstro_llm/issues)。

---

**版本**: 2.0  
**更新日期**: 2026-03-09
