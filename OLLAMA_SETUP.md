# Ollama 安装与配置指南

本指南将帮助你安装 Ollama 并配置 qwen-vl 模型供 Astro-AI 项目使用。

## 📋 前置条件

- Ubuntu 20.04+ / WSL2
- 至少 16GB 内存（用于运行 8B 模型）
- 约 20GB 磁盘空间

## 🔧 步骤 1: 安装 Ollama

### 方式一：官方安装脚本（推荐）

```bash
# 运行官方安装脚本
curl -fsSL https://ollama.com/install.sh | sh

# 或者使用 sudo
curl -fsSL https://ollama.com/install.sh | sudo sh
```

### 方式二：手动安装

```bash
# 下载 Ollama 二进制文件
sudo curl -L -o /usr/local/bin/ollama https://ollama.com/download/ollama-linux-amd64
sudo chmod +x /usr/local/bin/ollama

# 创建 Ollama 用户
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
```

### 方式三：Docker 安装

```bash
# 使用 Docker 运行 Ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## 🚀 步骤 2: 启动 Ollama 服务

### 前台运行（测试用）

```bash
ollama serve
```

### 后台运行（推荐）

```bash
# 使用 systemd（如果已安装）
sudo systemctl start ollama
sudo systemctl enable ollama

# 或者使用 nohup
nohup ollama serve > ~/.ollama/ollama.log 2>&1 &
```

### 验证服务状态

```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 或使用 ollama 命令
ollama list
```

## 📥 步骤 3: 下载 Qwen-VL 模型

```bash
# 拉取 qwen2-vl 模型（Ollama 官方支持的视觉语言模型）
ollama pull qwen2-vl:7b

# 或者拉取更大的版本
ollama pull qwen2-vl:72b

# 查看已安装的模型
ollama list
```

> **注意**: Ollama 官方目前主要支持 `qwen2-vl` 系列模型。如果你的项目需要特定的 `qwen-vl-chat-8b`，可以使用自定义 Modelfile 导入。

## 📝 步骤 4: 配置项目使用 Ollama

### 使用自定义 Modelfile（高级）

如果你想使用特定的 Qwen-VL 模型，可以创建 Modelfile：

```bash
# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM ./models/Qwen-VL-Chat-Int4

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20

SYSTEM """你是一个专业的天文分析助手，擅长分析天体物理数据和分类天体类型。"""
EOF

# 构建自定义模型
ollama create qwen-vl-astro -f Modelfile
```

## 🔌 步骤 5: 测试 Ollama 连接

```bash
# 测试简单对话
ollama run qwen2-vl:7b "你好，请介绍一下你自己"

# 测试带图片的分析（需要图片路径）
ollama run qwen2-vl:7b "分析这张天文图像" --image /path/to/image.png
```

## ⚙️ 项目配置

项目已集成了 `OllamaInterface` 类，位于 `src/ollama_interface.py`。使用方式：

```python
from src.ollama_interface import OllamaInterface

# 初始化接口
ollama = OllamaInterface(model_name="qwen2-vl:7b")

# 分析文本
result = ollama.analyze("分析这个天体的特征: 消光 A_V=1.12, 周期 3.09小时")
print(result)

# 分析图片
result = ollama.analyze("描述这张天文图像", image_path="path/to/image.png")
```

## 🛠️ 故障排除

### 问题：Ollama 服务无法启动

```bash
# 检查端口占用
lsof -i :11434

# 杀死占用进程
kill -9 <PID>

# 重新启动
ollama serve
```

### 问题：模型下载失败

```bash
# 检查网络连接
ping ollama.com

# 使用代理（如果需要）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 重新拉取
ollama pull qwen2-vl:7b
```

### 问题：内存不足

```bash
# 使用更小的模型
ollama pull qwen2-vl:7b

# 或增加交换空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 问题：项目无法连接到 Ollama

```bash
# 检查 Ollama 服务是否运行
curl http://localhost:11434/api/tags

# 设置环境变量（如果使用非标准端口）
export OLLAMA_HOST=http://localhost:11434

# 测试 Python 连接
python -c "import ollama; print(ollama.list())"
```

## 📚 常用命令

```bash
# 列出已安装模型
ollama list

# 删除模型
ollama rm qwen2-vl:7b

# 查看模型信息
ollama show qwen2-vl:7b

# 运行交互式对话
ollama run qwen2-vl:7b

# 复制模型
ollama cp qwen2-vl:7b my-qwen

# 导出模型
ollama show qwen2-vl:7b --modelfile > exported.txt
```

## 🔗 参考链接

- [Ollama 官网](https://ollama.com)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Qwen-VL 模型](https://github.com/QwenLM/Qwen-VL)
- [Ollama Python 库](https://github.com/ollama/ollama-python)

---

完成以上步骤后，你的 Astro-AI 项目就可以通过 Ollama 调用 Qwen-VL 模型进行天文数据分析了！🔭✨
