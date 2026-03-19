# 模型重命名指南

如果你想将模型重命名为 `maoAstro_llm`，有几种方式：

---

## 方式1: 创建符号链接 (推荐，最简单)

```bash
# 创建软链接，保留原模型
ln -s train_qwen/output_qwen25/merged_model models/maoAstro_llm

# 然后使用新名称启动
python start_astrosage_complete.py --model-path models/maoAstro_llm
```

---

## 方式2: 复制重命名

```bash
# 复制整个模型目录
cp -r train_qwen/output_qwen25/merged_model models/maoAstro_llm

# 或使用更高效的硬链接 (节省空间，同分区)
cp -rl train_qwen/output_qwen25/merged_model models/maoAstro_llm

# 启动
python start_astrosage_complete.py --model-path models/maoAstro_llm
```

---

## 方式3: 直接移动/重命名

```bash
# 重命名模型目录
mv train_qwen/output_qwen25/merged_model models/maoAstro_llm

# 修改启动脚本中的默认路径
# 编辑 start_astrosage_complete.py，修改默认路径
```

---

## 方式4: 使用Ollama导入 (部署用)

```bash
# 将模型导入Ollama，命名为 maoAstro_llm

# 1. 创建Modelfile
cat > Modelfile.maoastro << 'EOF'
FROM train_qwen/output_qwen25/merged_model

SYSTEM """你是 maoAstro_llm，一个专门训练用于天文领域问答的AI助手。
你基于Qwen2.5-7B模型，使用3,413条天文QA数据进行了微调训练。
请用专业、准确的中文回答天文问题。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 2. 创建Ollama模型
ollama create maoAstro_llm -f Modelfile.maoastro

# 3. 运行
ollama run maoAstro_llm
```

---

## 修改后的启动命令

重命名后，启动时需要指定新路径：

```bash
# 通用方式
python start_astrosage_complete.py --model-path models/maoAstro_llm

# 或修改 start_local_rag.sh 中的默认路径
# 找到这行:
# model_path="train_qwen/output_qwen25/merged_model"
# 改为:
# model_path="models/maoAstro_llm"
```

---

## 一键重命名脚本

```bash
#!/bin/bash
# rename_model.sh

NEW_NAME="maoAstro_llm"
SOURCE="train_qwen/output_qwen25/merged_model"
DEST="models/$NEW_NAME"

echo "🔄 重命名模型..."
echo "   源: $SOURCE"
echo "   目标: $DEST"

# 创建models目录
mkdir -p models

# 创建符号链接
ln -sf "$(pwd)/$SOURCE" "$DEST"

echo "✅ 完成!"
echo ""
echo "启动命令:"
echo "  python start_astrosage_complete.py --model-path $DEST"
```

保存为 `rename_model.sh`，然后：
```bash
chmod +x rename_model.sh
./rename_model.sh
```

---

## 推荐方案

| 场景 | 推荐方式 |
|------|---------|
| 测试用 | 方式1 (符号链接) |
| 正式发布 | 方式2 (复制) 或 方式4 (Ollama) |
| 临时使用 | 方式3 (移动) |

---

**注意**: 模型文件很大 (15GB)，建议使用符号链接避免重复占用空间。
