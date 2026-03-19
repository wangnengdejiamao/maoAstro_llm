# AstroSage 8B 微调监控指南

## ✅ 训练状态

**训练已启动！**
- **进程ID**: 8134
- **日志文件**: `train_qwen/astrosage_continued/training.log`
- **方案**: Transformers + LoRA (Llama-3.1-8B 中文版)
- **预计时间**: 3-6 小时 (取决于下载速度和训练速度)

---

## 📊 监控命令

### 1. 查看实时日志

```bash
# 实时查看训练进度
tail -f train_qwen/astrosage_continued/training.log

# 查看最新100行
tail -100 train_qwen/astrosage_continued/training.log
```

### 2. 查看 GPU 状态

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或一次性查看
nvidia-smi
```

### 3. 检查进程状态

```bash
# 查看训练进程
ps aux | grep finetune_transformers

# 查看 PID 8134 是否存在
ps -p 8134
```

---

## 🎯 训练流程

当前训练流程：

```
1. 下载基础模型 (Llama-3.1-8B-Chinese-Chat) ~16GB
   [当前阶段 - 预计 10-20 分钟]
   
2. 加载模型到 GPU
   
3. 配置 LoRA (r=64)
   
4. 加载训练数据 (3,413条)
   
5. 开始训练 (3 epochs)
   [预计 2-4 小时]
   
6. 保存模型
   [输出到 train_qwen/astrosage_continued/final_model/]
```

---

## 🔍 关键日志信息

当看到这些信息时表示正常：

```
✅ 基础模型: shenzhi-wang/Llama3.1-8B-Chinese-Chat
✅ 模型加载完成
✅ LoRA 配置完成
   trainable params: XX,XXX,XXX || all params: X,XXX,XXX,XXX
✅ 开始训练!
   Step 10/1281: loss=X.XXXX
   Step 20/1281: loss=X.XXXX
```

---

## ⚠️ 注意事项

### 如果下载卡住
```bash
# 查看下载进度
tail -f train_qwen/astrosage_continued/training.log | grep -E "(Downloading|Downloaded)"
```

### 如果显存不足
训练会自动使用 4-bit 量化，12GB 显存应该足够。如果仍然OOM：
```bash
# 停止当前训练
kill 8134

# 修改脚本减小 batch size
# 编辑 train_qwen/astrosage_export/finetune_transformers.py
# 将 per_device_train_batch_size=1 改为 gradient_accumulation_steps=16
```

### 如果训练中断
```bash
# 重新启动训练
./start_training.sh

# 注意: 这会从头开始，之前的进度不会保存
```

---

## 📁 输出文件

训练完成后会生成：

```
train_qwen/astrosage_continued/
├── training.log          # 训练日志
├── train.pid             # 进程ID文件
├── final_model/          # 最终模型
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── checkpoint-*/         # 中间检查点 (如果有)
```

---

## 🚀 训练完成后

### 1. 测试模型
```bash
python inference_llama31.py \
    --model train_qwen/astrosage_continued/final_model
```

### 2. 导出到 Ollama
```bash
python export_llama31_to_ollama.py \
    --model train_qwen/astrosage_continued/final_model \
    --name maoAstro-finetuned
```

### 3. 使用模型
```bash
ollama run maoAstro-finetuned
```

---

## 💡 常见问题

**Q: 训练需要多久？**
A: 
- 下载模型: 10-20 分钟 (取决于网速)
- 训练时间: 2-4 小时 (3 epochs, RTX 3080 Ti)
- 总计: 3-6 小时

**Q: 可以中断后继续吗？**
A: 当前脚本不支持断点续训。如果需要，可以修改脚本添加 `resume_from_checkpoint`。

**Q: 如何知道训练是否正常？**
A: 查看日志中的 loss 值，正常情况下 loss 应该逐渐下降 (从 ~2.0 降到 ~0.5)。

---

## 📞 停止训练

如果需要停止训练：

```bash
# 温和停止 (保存当前状态)
kill 8134

# 强制停止
kill -9 8134
```

---

**训练正在进行中，请耐心等待！** 🚀
