# maoAstro LLM - 完整配置教程

本文档指导你如何上传项目到 GitHub，并配置完整的运行环境（包括消光文件和 LLM 模型）。

---

## 📋 目录

1. [GitHub 上传](#github-上传)
2. [环境配置](#环境配置)
3. [下载消光文件](#下载消光文件)
4. [下载 LLM 模型](#下载-llm-模型)
5. [AstroMLab 环境建议](#astromlab-环境建议)
6. [验证安装](#验证安装)

---

## GitHub 上传

### 步骤 1: 创建 GitHub 仓库

1. 访问 https://github.com/new
2. Repository name: `maoAstro_llm`
3. Description: 天文AI智能分析平台
4. 选择 **Public**（公开）或 **Private**（私有）
5. **不要**勾选 "Add a README file"
6. 点击 **Create repository**

### 步骤 2: 本地配置 Git

```bash
# 进入项目目录
cd /mnt/c/Users/Administrator/Desktop/maoAstro_llm

# 初始化 Git
git init

# 配置用户信息（替换为你的 GitHub 信息）
git config user.name "你的GitHub用户名"
git config user.email "你的邮箱@example.com"

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: maoAstro LLM 天文AI分析平台"

# 连接远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/maoAstro_llm.git

# 推送代码
git branch -M main
git push -u origin main
```

### 步骤 3: 验证上传

访问 `https://github.com/YOUR_USERNAME/maoAstro_llm` 确认代码已上传。

---

## 环境配置

### 方式 A: 使用 AstroMLab 环境（推荐⭐）

**AstroMLab** 是专为天文数据分析设计的 Python 环境，预装了大量天文包。

#### 安装 AstroMLab

```bash
# 安装 Miniconda（如果尚未安装）
# Windows: 从 https://docs.conda.io/en/latest/miniconda.html 下载安装
# macOS/Linux:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建 AstroMLab 环境
conda create -n astromlab python=3.9 -y

# 激活环境
conda activate astromlab

# 安装核心天文包
conda install -c conda-forge astropy astroquery -y
conda install -c conda-forge healpy -y
conda install -c conda-forge matplotlib numpy scipy pandas -y

# 安装深度学习相关
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate

# 安装其他依赖
pip install lightkurve astroplan
pip install requests beautifulsoup4 tqdm
```

#### 安装本项目依赖

```bash
# 在项目目录下
pip install -r requirements.txt
```

### 方式 B: 使用标准虚拟环境

```bash
# 创建虚拟环境
cd /mnt/c/Users/Administrator/Desktop/maoAstro_llm
python -m venv venv

# 激活环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 下载消光文件

消光文件用于计算银河系的消光（ reddening ）值，是天文观测数据校正的重要输入。

### 需要的文件

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `csfd_ebv.fits` | ~200 MB | CSFD 修正消光地图（推荐） |
| `sfd_ebv.fits` | ~200 MB | SFD 原始消光地图 |
| `lss_intensity.fits` | ~200 MB | 大尺度结构强度图 |
| `lss_error.fits` | ~200 MB | 大尺度结构误差图 |
| `mask.fits` | ~100 MB | 数据质量掩码 |

**总大小: ~900 MB**

### 下载方法

#### 方法 1: 自动下载脚本（推荐）

```bash
# 运行下载脚本
python scripts/download_extinction.py
```

#### 方法 2: 手动下载

1. **CSFD 消光地图**（推荐）
   - 访问: https://github.com/CPPMariner/CSFD
   - 下载最新发布的消光地图
   - 将文件解压到 `data/` 目录

2. **SFD 原始地图**（备选）
   - 访问: https://www.legacysurvey.org/
   - 或 https://github.com/kbarbary/sfddata
   - 下载 SFD 消光地图

#### 方法 3: 百度网盘/阿里云盘（国内用户）

如果官方下载较慢，可以使用网盘镜像：

```
百度网盘链接: https://pan.baidu.com/s/xxxxxxxxx
提取码: xxxx

阿里云盘链接: https://www.aliyundrive.com/s/xxxxxxxxx
```

> ⚠️ 注意：请优先从官方渠道下载，确保数据完整性。

### 文件放置位置

```
maoAstro_llm/
├── data/
│   ├── csfd_ebv.fits          ← 放在这里
│   ├── sfd_ebv.fits           ← 放在这里
│   ├── lss_intensity.fits     ← 放在这里
│   ├── lss_error.fits         ← 放在这里
│   ├── mask.fits              ← 放在这里
│   └── README.md
```

### 验证消光文件

```bash
# 运行测试脚本
python -c "
from astropy.io import fits
import os

data_dir = 'data'
files = ['csfd_ebv.fits', 'sfd_ebv.fits', 'lss_intensity.fits', 'lss_error.fits', 'mask.fits']

print('检查消光文件...')
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        with fits.open(path) as hdul:
            print(f'✓ {f}: {size:.1f} MB, {len(hdul)} HDUs')
    else:
        print(f'✗ {f}: 缺失')
"
```

---

## 下载 LLM 模型

本项目使用 **Ollama** 本地运行大语言模型，无需联网即可进行 AI 分析。

### 步骤 1: 安装 Ollama

#### Windows
```powershell
# 方法 1: 下载安装程序
# 访问 https://ollama.com/download/windows

# 方法 2: 使用 winget
winget install Ollama.Ollama
```

#### macOS
```bash
# 方法 1: 下载安装程序
# 访问 https://ollama.com/download/mac

# 方法 2: 使用 Homebrew
brew install ollama
```

#### Linux
```bash
# 一键安装
curl -fsSL https://ollama.com/install.sh | sh

# 或手动安装
sudo apt install ollama  # Ubuntu/Debian
sudo yum install ollama  # CentOS/RHEL
```

### 步骤 2: 启动 Ollama 服务

```bash
# 启动服务（Windows/macOS 通常自动启动）
ollama serve

# 在另一个终端验证服务状态
ollama list
```

### 步骤 3: 下载推荐模型

#### 推荐模型 1: Qwen3 8B（默认推荐⭐）

通义千问，中文表现优秀，适合天文领域问答。

```bash
# 下载模型（约 4.7 GB）
ollama pull qwen3:8b

# 验证安装
ollama run qwen3:8b
# 输入测试: "你好，请介绍太阳系的八大行星"
# 按 Ctrl+D 退出
```

#### 推荐模型 2: Llama 3.1 8B

Meta 开源模型，英文表现优秀。

```bash
ollama pull llama3.1:8b
```

#### 推荐模型 3: DeepSeek R1 8B

推理能力强的模型，适合复杂分析。

```bash
ollama pull deepseek-r1:8b
```

#### 轻量级模型（低配设备）

如果电脑配置较低（< 8GB 内存）：

```bash
# Qwen 1.8B（约 1.1 GB）
ollama pull qwen:1.8b

# Llama 3.2 1B（约 600 MB）
ollama pull llama3.2:1b
```

### 步骤 4: 配置项目使用模型

编辑 `src/ollama_qwen_interface.py`（或创建配置文件）：

```python
# config.py
OLLAMA_CONFIG = {
    "model_name": "qwen3:8b",      # 修改为你下载的模型
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "timeout": 120,
}
```

### 步骤 5: 测试 LLM 功能

```bash
# 运行测试脚本
python scripts/test_llm.py
```

预期输出：
```
✓ Ollama 服务运行正常
✓ 模型 qwen3:8b 可用
✓ LLM 响应测试通过
模型回复: "太阳系的八大行星按照距离太阳由近到远分别是：水星、金星、地球、火星..."
```

---

## AstroMLab 环境建议

### 什么是 AstroMLab?

**AstroMLab** 是一个专为天文数据分析优化的 Python 环境，特点：
- 🚀 预装 100+ 天文相关包
- 🔬 集成 AstroPy、NumPy、SciPy 等核心库
- 🤖 内置 PyTorch/TensorFlow 用于机器学习
- 📊 支持 JupyterLab 交互式分析
- 🐳 提供 Docker 镜像便于部署

### 推荐的 AstroMLab 配置

#### 基础配置（个人电脑）

```yaml
# environment.yml
name: astromlab
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - astropy>=5.0
  - astroquery>=0.4
  - healpy>=1.16
  - numpy>=1.21
  - scipy>=1.7
  - matplotlib>=3.5
  - pandas>=1.3
  - jupyterlab>=3.0
  - pip
  - pip:
    - lightkurve>=2.3
    - astroplan>=0.8
    - ollama>=0.1
    - openai>=1.0
```

创建环境：
```bash
conda env create -f environment.yml
conda activate astromlab
```

#### 高性能配置（服务器/工作站）

```yaml
# environment_gpu.yml
name: astromlab-gpu
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - astropy>=6.0
  - astroquery>=0.4
  - healpy>=1.16
  - numpy>=1.24
  - scipy>=1.10
  - matplotlib>=3.7
  - pandas>=2.0
  - jupyterlab>=4.0
  - pytorch>=2.0
  - pytorch-cuda=12.1
  - cudatoolkit
  - pip
  - pip:
    - lightkurve>=2.4
    - transformers>=4.30
    - accelerate>=0.20
    - bitsandbytes>=0.40
    - ollama>=0.1
```

### Docker 部署（推荐服务器使用）

```dockerfile
# Dockerfile
FROM continuumio/miniconda3:latest

LABEL maintainer="maoAstro"
LABEL description="AstroMLab for maoAstro LLM"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制环境配置
COPY environment.yml /tmp/

# 创建环境
RUN conda env create -f /tmp/environment.yml

# 激活环境
SHELL ["conda", "run", "-n", "astromlab", "/bin/bash", "-c"]

# 安装 Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 设置工作目录
WORKDIR /workspace

# 暴露端口
EXPOSE 8888 11434

# 启动命令
CMD ["conda", "run", "-n", "astromlab", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

构建和运行：
```bash
# 构建镜像
docker build -t astromlab:latest .

# 运行容器
docker run -it --rm \
  -p 8888:8888 \
  -p 11434:11434 \
  -v $(pwd):/workspace \
  astromlab:latest
```

---

## 验证安装

### 运行完整检查脚本

```bash
# 检查所有组件
python scripts/check_setup.py
```

### 手动检查清单

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| Python 版本 | `python --version` | 3.9+ |
| Conda 环境 | `conda info --envs` | astromlab 存在且激活 |
| 核心包 | `python -c "import astropy; print(astropy.__version__)"` | 5.0+ |
| Ollama 服务 | `curl http://localhost:11434/api/tags` | 返回模型列表 |
| 消光文件 | `ls data/*.fits` | 5 个文件存在 |
| LLM 模型 | `ollama list` | qwen3:8b 或其他模型 |

### 运行示例分析

```bash
# 测试消光查询
python -c "
from src.astro_tools import query_extinction
result = query_extinction(13.1316, 53.8585)
print(f'A_V = {result[\"av\"]:.3f} mag')
"

# 测试 LLM 接口
python -c "
from src.ollama_qwen_interface import OllamaQwenInterface
ollama = OllamaQwenInterface()
response = ollama.analyze_text('简述恒星演化')
print(response)
"
```

---

## 常见问题

### Q1: 消光文件下载失败？

**A:** 
- 检查网络连接
- 使用代理或 VPN 访问 GitHub
- 尝试百度网盘镜像
- 联系项目维护者获取备用下载链接

### Q2: Ollama 无法启动？

**A:**
- Windows: 检查 Windows 服务中 Ollama 是否运行
- Linux: 运行 `sudo systemctl start ollama`
- 检查端口 11434 是否被占用: `lsof -i :11434`
- 防火墙设置: 确保允许 11434 端口

### Q3: 模型下载太慢？

**A:**
- 使用国内镜像（如果可用）
- 检查网络带宽
- 尝试更小模型（如 qwen:1.8b）
- 使用 HuggingFace 镜像下载 GGUF 文件后导入 Ollama

### Q4: 内存不足？

**A:**
- 使用更小的模型（1B-3B 参数）
- 关闭其他应用程序
- 使用量化版本（Q4_K_M 等）
- 考虑使用云服务器

### Q5: 如何在服务器上部署？

**A:**
```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/maoAstro_llm.git
cd maoAstro_llm

# 2. 创建环境
conda env create -f environment.yml
conda activate astromlab

# 3. 下载数据
python scripts/download_extinction.py

# 4. 安装并启动 Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3:8b

# 5. 运行服务
python src/server.py
```

---

## 下一步

完成上述配置后：

1. 📖 阅读 [README.md](README.md) 了解项目功能
2. 🚀 运行示例代码 `examples/basic_analysis.py`
3. 🔬 开始你的天文数据分析
4. 🐛 遇到问题在 GitHub Issues 提问

---

**祝使用愉快！** 🌟

如有问题，请通过 GitHub Issues 联系。
