# GitHub 上传指南

本文档指导如何将 `astro-ai-demo` 项目清理并上传到 GitHub。

---

## 📋 目录

1. [准备工作](#准备工作)
2. [项目清理](#项目清理)
3. [创建 GitHub 仓库](#创建-github-仓库)
4. [上传项目](#上传项目)
5. [验证上传](#验证上传)
6. [常见问题](#常见问题)

---

## 准备工作

### 1. 安装 Git

如果还没有安装 Git，请根据你的操作系统安装：

**Windows:**
```bash
# 下载安装程序
https://git-scm.com/download/win
```

**macOS:**
```bash
# 使用 Homebrew
brew install git

# 或下载安装程序
https://git-scm.com/download/mac
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

### 2. 配置 Git

```bash
# 设置用户名和邮箱（使用你的 GitHub 账号信息）
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"

# 验证配置
git config --list
```

### 3. 检查 GitHub 账号

确保你有一个 GitHub 账号：https://github.com/join

---

## 项目清理

### 清理前的大小

当前项目约 **48GB**，包含：

| 目录/文件 | 大小 | 说明 |
|-----------|------|------|
| `lib/` | 33GB | LAMOST DR10 等大型天文数据 |
| `models/` | 14GB | LLM 模型文件 |
| `data/` | 898MB | 消光地图等数据 |
| `contamination_*.joblib` | 920MB | 污染检测模型 |
| 其他 | ~100MB | 代码和文档 |

### 使用清理脚本

项目提供了自动清理脚本：

```bash
# 1. 先预览将要删除的文件（推荐先运行这个）
python clean_for_github.py --dry-run

# 2. 确认无误后，执行清理
python clean_for_github.py

# 或跳过确认直接清理
python clean_for_github.py --force
```

### 手动清理（可选）

如果不想使用脚本，可以手动删除以下目录：

```bash
# 大型数据文件（33GB）
rm -rf lib/

# 模型文件（14GB）
rm -rf models/
rm -f contamination_*.joblib contamination_*.npy

# 消光地图数据（800MB）
rm -f data/*.fits
rm -rf data/large_files/

# 下载的光谱数据（116MB）
rm -rf SDSS_Spectra_Downloads/

# 输出文件
rm -rf output/data/ output/figures/
rm -f output/*.json output/*.png output/*.txt

# 缓存和废弃文件
rm -rf cache/
rm -rf Useless/
rm -rf archive/
rm -rf downloads/

# IDE 配置
rm -rf .idea/

# Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Jupyter 检查点
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# 大型 Notebook（170MB）
rm -f VSP/VSP_Integrate.ipynb
```

### 清理后的大小

清理后的项目约 **10-20MB**，包含：

- ✅ 源代码 (`src/`)
- ✅ 文档 (`docs/`)
- ✅ 测试文件 (`tests/`)
- ✅ 示例代码 (`examples/`)
- ✅ 配置文件 (README, requirements.txt 等)
- ✅ 小型 Notebooks (`VSP/` 除 VSP_Integrate.ipynb 外)

---

## 创建 GitHub 仓库

### 方法 1: 通过网页创建（推荐新手）

1. 登录 GitHub：https://github.com
2. 点击右上角 **+** 按钮 → **New repository**
3. 填写仓库信息：
   - **Repository name**: `astro-ai-demo`（或你喜欢的名字）
   - **Description**: 天文AI智能分析平台
   - **Visibility**: Public（公开）或 Private（私有）
   - **Initialize this repository with**: ❌ 不要勾选（我们已有本地项目）
4. 点击 **Create repository**

### 方法 2: 通过 GitHub CLI

```bash
# 安装 GitHub CLI
# macOS
brew install gh

# Windows
winget install --id GitHub.cli

# Ubuntu
sudo apt install gh

# 登录
github auth login

# 创建仓库
gh repo create astro-ai-demo --public --description "天文AI智能分析平台"
```

---

## 上传项目

### 步骤 1: 初始化本地 Git 仓库

```bash
# 进入项目目录
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 初始化 Git 仓库
git init

# 添加所有文件到暂存区
git add .

# 提交
git commit -m "Initial commit: 天文AI智能分析平台"
```

### 步骤 2: 连接远程仓库

**如果你使用 HTTPS（推荐新手）：**

```bash
# 添加远程仓库（替换为你的用户名）
git remote add origin https://github.com/你的用户名/astro-ai-demo.git
```

**如果你使用 SSH（需要配置密钥）：**

```bash
# 添加远程仓库
git remote add origin git@github.com:你的用户名/astro-ai-demo.git
```

### 步骤 3: 推送到 GitHub

```bash
# 推送主分支
git branch -M main
git push -u origin main
```

**如果遇到大文件错误：**

GitHub 限制单个文件最大 100MB。如果报错，确保已运行清理脚本：

```bash
# 检查是否有大文件
find . -type f -size +100M

# 如果误提交了大文件，移除它们
git rm --cached 大文件名
git commit --amend -m "Initial commit: 天文AI智能分析平台"
git push -f origin main
```

---

## 验证上传

### 检查仓库

1. 打开浏览器，访问：`https://github.com/你的用户名/astro-ai-demo`
2. 确认以下文件存在：
   - ✅ `README.md`
   - ✅ `src/` 目录
   - ✅ `requirements.txt`
   - ✅ `.gitignore`

### 克隆测试

```bash
# 在另一个目录测试克隆
cd /tmp
git clone https://github.com/你的用户名/astro-ai-demo.git
cd astro-ai-demo

# 检查项目结构
ls -la

# 安装依赖（可选）
pip install -r requirements.txt
```

---

## 后续配置

### 1. 添加数据下载说明

清理后的大文件需要通过其他方式获取。编辑 `README.md` 添加数据下载说明：

```markdown
## 数据文件

本项目需要以下数据文件，由于体积较大未包含在仓库中：

### 消光地图数据 (~800MB)
```bash
python download_data.py --extinction
```

### 模型文件 (~900MB)
```bash
python download_data.py --models
```

### 完整星表数据 (~30GB)
请访问 LAMOST 官网下载：http://www.lamost.org/dr10/
```

### 2. 启用 GitHub 功能

在仓库页面，点击 **Settings** 标签：

- **Issues**: 开启（用于 bug 报告和功能请求）
- **Discussions**: 开启（用于讨论）
- **Wiki**: 可选（用于详细文档）
- **Projects**: 可选（用于项目管理）

### 3. 添加项目话题（Topics）

在仓库页面右侧，点击齿轮图标添加话题：

- `astronomy`
- `machine-learning`
- `data-analysis`
- `python`
- `astrophysics`

### 4. 设置分支保护（可选）

Settings → Branches → Add rule：
- Branch name pattern: `main`
- ☑️ Require pull request reviews before merging
- ☑️ Require status checks to pass

---

## 常见问题

### Q1: 误提交了大文件怎么办？

```bash
# 从 Git 历史中完全移除大文件
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch 大文件名' \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

或使用 BFG Repo-Cleaner（更快）：

```bash
# 下载 BFG
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# 运行清理
java -jar bfg-1.14.0.jar --delete-files 大文件名

# 清理并推送
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all
```

### Q2: 超过 GitHub 仓库大小限制？

GitHub 限制：
- 单个文件：100MB（硬限制）
- 推荐仓库大小：1GB 以下
- 仓库总大小上限：5GB（软限制）

如果仓库太大：
1. 确保 `.gitignore` 正确配置
2. 使用 Git LFS（Large File Storage）管理大文件
3. 将数据文件放在外部存储（如 Hugging Face、Figshare、Zenodo）

### Q3: 如何添加协作者？

Settings → Manage access → Invite a collaborator

### Q4: 如何创建发布版本？

1. 在 GitHub 页面点击右侧的 **Create a new release**
2. 点击 **Choose a tag**，输入版本号（如 `v1.0.0`）
3. 填写发布标题和说明
4. 点击 **Publish release**

---

## 快速命令参考

```bash
# 完整流程（复制粘贴执行）

# 1. 清理项目
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python clean_for_github.py --force

# 2. 初始化 Git
git init
git add .
git commit -m "Initial commit"

# 3. 连接远程（替换用户名）
git remote add origin https://github.com/你的用户名/astro-ai-demo.git

# 4. 推送
git branch -M main
git push -u origin main

# 5. 验证
gh repo view 你的用户名/astro-ai-demo --web
```

---

## 相关文档

- [README.md](README.md) - 项目说明
- [download_data.py](download_data.py) - 数据下载脚本
- [clean_for_github.py](clean_for_github.py) - 项目清理脚本

---

**祝你的项目开源顺利！** ⭐

如有问题，请在 GitHub Issues 中提问。
