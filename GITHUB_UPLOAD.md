# GitHub 上传快速指南

5 分钟完成项目上传到 GitHub。

---

## 步骤 1: 创建 GitHub 仓库（1分钟）

1. 访问 https://github.com/new
2. **Repository name**: `maoAstro_llm`
3. **Description**: 天文AI智能分析平台
4. **Visibility**: 选择 Public（公开）或 Private（私有）
5. **不要勾选** "Add a README file"（我们已有）
6. 点击 **Create repository**

---

## 步骤 2: 上传代码（2分钟）

```bash
# 进入项目目录
cd /mnt/c/Users/Administrator/Desktop/maoAstro_llm

# 初始化 Git
git init

# 配置用户信息（替换为你的信息）
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

**提示**: 如果提示输入密码，使用 GitHub Personal Access Token 代替密码。
- 生成 Token: https://github.com/settings/tokens
- 勾选 `repo` 权限

---

## 步骤 3: 验证上传（1分钟）

访问 `https://github.com/YOUR_USERNAME/maoAstro_llm`

确认能看到以下文件：
- ✅ `README.md`
- ✅ `src/` 目录
- ✅ `scripts/` 目录
- ✅ `requirements.txt`

---

## 后续配置（可选）

### 添加项目话题

在 GitHub 仓库页面右侧，点击齿轮图标添加 Topics：
- `astronomy`
- `machine-learning`
- `astrophysics`
- `ollama`
- `python`

### 启用 Issues

Settings → General → Features → ☑️ Issues

---

## 项目大小

清理后的项目约 **8MB**，适合 GitHub 托管。

| 目录 | 大小 | 说明 |
|------|------|------|
| `src/` | ~3MB | 源代码 |
| `VSP/` | ~3MB | Notebooks |
| `docs/` | ~1MB | 文档 |
| `scripts/` | ~20KB | 工具脚本 |
| 其他 | ~1MB | 配置文件 |

**注意**: 大文件（消光地图、模型）未包含在仓库中，需要单独下载。

---

## 完整配置流程

上传完成后，配置完整运行环境：

```bash
# 1. 克隆项目（新机器）
git clone https://github.com/YOUR_USERNAME/maoAstro_llm.git
cd maoAstro_llm

# 2. 创建环境
conda create -n astromlab python=3.9 -y
conda activate astromlab
pip install -r requirements.txt

# 3. 下载消光文件
python scripts/download_extinction.py

# 4. 安装 Ollama 和模型
# 访问 https://ollama.com/download 安装
ollama pull qwen3:8b

# 5. 验证安装
python scripts/check_setup.py
```

详细指南: [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 常见问题

### Q: 推送时提示 "Permission denied"?

```bash
# 使用 HTTPS 重新添加远程仓库
git remote remove origin
git remote add origin https://USERNAME:TOKEN@github.com/USERNAME/maoAstro_llm.git
```

### Q: 文件太大上传失败?

GitHub 限制单文件 100MB。检查是否有大文件：
```bash
find . -type f -size +100M
```

### Q: 如何更新已上传的项目?

```bash
# 修改文件后
git add .
git commit -m "更新描述"
git push
```

---

**祝开源顺利！** 🌟
