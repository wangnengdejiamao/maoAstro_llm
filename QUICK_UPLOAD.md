# 快速上传指南（精简版）

5 分钟完成 GitHub 上传。

## 步骤 1: 清理项目（1分钟）

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 预览要删除的文件
python clean_for_github.py --dry-run

# 执行清理
python clean_for_github.py --force
```

## 步骤 2: 创建 GitHub 仓库（1分钟）

1. 访问 https://github.com/new
2. 填写 Repository name: `astro-ai-demo`
3. 选择 Public 或 Private
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 **Create repository**

## 步骤 3: 上传代码（2分钟）

```bash
# 初始化 Git
git init

# 添加文件
git add .

# 提交
git commit -m "Initial commit: 天文AI智能分析平台"

# 连接远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/astro-ai-demo.git

# 推送
git branch -M main
git push -u origin main
```

## 步骤 4: 验证（1分钟）

访问 `https://github.com/YOUR_USERNAME/astro-ai-demo`

确认能看到代码文件即表示成功！

---

## 遇到问题？

### 文件太大报错

```bash
# 检查是否有 >100MB 的文件
find . -type f -size +100M

# 重新运行清理脚本
python clean_for_github.py --force
```

### 需要用户名密码

```bash
# 使用 GitHub Token 代替密码
# 访问: https://github.com/settings/tokens
# 生成 Personal Access Token (classic)
# 勾选 "repo" 权限
# 用 Token 代替密码输入
```

### 清理误删了重要文件

```bash
# 从 Git 恢复（如果已提交）
git checkout HEAD -- 文件名

# 从备份恢复（如果有）
```

---

## 清理后的项目结构

```
astro-ai-demo/ (~15MB)
├── src/              # 源代码
├── docs/             # 文档
├── tests/            # 测试
├── examples/         # 示例
├── VSP/              # Notebooks（不含 VSP_Integrate.ipynb）
├── data/             # 数据目录（仅 README）
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── clean_for_github.py      # 清理脚本
├── download_data.py         # 数据下载脚本
└── GITHUB_UPLOAD_GUIDE.md   # 完整上传指南
```

---

## 下一步

1. ⭐ 在 GitHub 上给仓库点 Star
2. 📝 完善 README.md 添加项目截图
3. 🏷️ 添加 Topics: astronomy, machine-learning, python
4. 📦 创建 Release 版本
5. 🚀 分享你的项目！

---

完整指南: [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md)
