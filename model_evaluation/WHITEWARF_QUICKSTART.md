# 🌟 WhiteWarf: Qwen-8B 白矮星专用模型快速开始

## 简介

WhiteWarf 是基于 Qwen-8B 微调的白矮星天体物理专用模型，专长于：
- 🌟 **单白矮星** - 演化、冷却、质量测定
- 🌟 **双白矮星** - 引力波、并合、Ia 型超新星前身
- 🌟 **磁性白矮星** - 磁场产生、分类、观测
- 🌟 **吸积白矮星** - 激变变星、新星爆发、AM CVn

## 🚀 三种使用方式

### 方式一：一键完整训练（推荐）

```bash
cd model_evaluation

# 运行完整训练流程（下载文献 + Kimi 生成数据 + 微调）
bash train_whitewarf.sh
```

这会执行：
1. 从 arXiv 下载白矮星领域论文
2. 使用 Kimi API 生成高质量训练数据
3. 运行 LoRA 微调
4. 合并导出模型
5. 运行白矮星专用评估

### 方式二：分步执行

#### Step 1: 下载文献

```bash
# 下载白矮星相关论文
python download_wd_papers.py \
    --search \
    --max-results 100 \
    --start-date 2020-01-01 \
    --output-dir ./data/white_dwarf_papers
```

#### Step 2: 使用 Kimi 生成训练数据

```bash
# 生成高质量 QA 对（需要 API Key）
export KIMI_API_KEY="9cccf25-f532-861b-8000-000042a859dc"

python download_wd_papers.py \
    --generate-data \
    --use-kimi \
    --output-dir ./data/white_dwarf_papers
```

**使用 Kimi 生成的数据包括：**
- ✅ 概念问答对（12 个主题 × 5 题）
- ✅ 反幻觉训练数据（6 个常见误解）
- ✅ 计算题（冷却时标、吸积盘温度等）
- ✅ 论文摘要问答（自动从 arXiv 论文生成）

#### Step 3: 微调模型

```bash
# 使用 LLaMA-Factory 微调
# 监督微调 (SFT)
llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --do_train \
    --dataset wd_training_data \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --output_dir ./models/qwen8b-whitewarf-sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-5 \
    --num_train_epochs 4 \
    --bf16

# 合并权重
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path ./models/qwen8b-whitewarf-sft \
    --finetuning_type lora \
    --template qwen \
    --export_dir ./models/qwen8b-whitewarf-final
```

#### Step 4: 评估

```bash
# 白矮星专用评估
python wd_evaluator.py \
    --model ./models/qwen8b-whitewarf-final \
    --interface hf
```

### 方式三：Ollama 快速部署

```bash
# 1. 创建 Ollama 模型
ollama create whitewarf -f Modelfile.whitewarf

# 2. 运行模型
ollama run whitewarf

# 3. 测试
>>> 什么是 AM CVn 系统？
>>> 双白矮星如何产生引力波？
>>> 磁性白矮星的磁场是怎么产生的？
```

## 📊 评估基准

WhiteWarf 包含专门的评估体系：

| 测试类别 | 题目数 | 评估内容 |
|---------|-------|---------|
| 单白矮星 | 3 | 钱德拉塞卡极限、冷却、组成 |
| 双白矮星 | 3 | 引力波、并合、轨道演化 |
| 磁性白矮星 | 3 | 磁场比例、产生机制、分类 |
| 吸积白矮星 | 4 | CVs、新星、AM CVn |
| 反幻觉 | 2 | 常见误解检测 |

**运行评估：**

```bash
python wd_evaluator.py --model whitewarf --interface ollama
```

## 🧪 测试 Kimi API

```bash
# 测试 API 连接
python kimi_interface.py
```

预期输出：
```
✓ Kimi API 连接成功
  可用模型: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k

测试 Kimi API
============================================================

1. 测试简单生成...
响应: 钱德拉塞卡极限是白矮星能够...

2. 测试批量生成...
Q: 什么是双白矮星系统？
A: 双白矮星是由两颗白矮星组成的...

✓ 测试完成
```

## 📁 数据文件说明

| 文件 | 说明 | 生成方式 |
|------|------|---------|
| `wd_papers.json` | arXiv 论文元数据 | 自动下载 |
| `wd_training_data.json` | 训练数据 | Kimi API |
| `wd_eval.json` | 评估数据 | 模板+Kimi |
| `dataset_info.json` | 数据集配置 | 手动 |

## 🎯 关键性能指标

### 必须超越的基准

| 指标 | 目标 |
|------|------|
| 白矮星基础知识 | > 90% |
| 双白矮星系统 | > 85% |
| 磁性白矮星 | > 85% |
| 吸积白矮星 | > 85% |
| 反幻觉测试 | 100% |

### 反幻觉关键点

模型必须正确识别：
- ✅ AM CVn **不含中子星**
- ✅ 白矮星**不通过核聚变**产能
- ✅ 双白矮星并合**不总是**产生超新星
- ✅ **并非所有**白矮星都有强磁场

## 💡 进阶使用

### 使用 Kimi 生成更多训练数据

```python
from kimi_interface import KimiInterface, WhiteDwarfDataGenerator

kimi = KimiInterface()
generator = WhiteDwarfDataGenerator(kimi)

# 生成特定主题的 QA
qa_pairs = generator.generate_qa_pairs("白矮星结晶化", n_pairs=5)

# 生成计算题
calc_problems = generator.generate_calculation_problems("吸积盘温度", n_problems=3)

# 生成反幻觉数据
anti_hall_data = generator.generate_anti_hallucination_data()
```

### 处理用户提供的文献

如果你有特定的白矮星论文 PDF：

```bash
# 放置 PDF 到目录
mkdir -p ./user_papers
cp your_paper.pdf ./user_papers/

# 处理（需要你先提取文本）
python download_wd_papers.py --user-papers ./user_papers
```

**更好的方式**：提供论文的 arXiv ID，系统自动下载和处理。

## 🔧 故障排除

### Kimi API 错误

```
错误: 429 Too Many Requests
```
**解决**：脚本已内置重试机制，会自动等待并重试。

### 论文下载失败

```
错误: feedparser 未安装
```
**解决**：
```bash
pip install feedparser
```

### 显存不足

```
CUDA out of memory
```
**解决**：
```bash
# 减小 batch size
llamafactory-cli train ... --per_device_train_batch_size 2

# 或使用 8-bit 量化
llamafactory-cli train ... --quantization_bit 8
```

## 📚 参考资源

### 白矮星领域经典文献

1. **Fontaine et al. (2001)** - "The Potential of White Dwarf Cosmochronology"
2. **Nelemans (2009)** - "The Galactic Double White Dwarf Population"
3. **Ferrario et al. (2015)** - "Magnetic White Dwarfs"
4. **Kupfer et al. (2023)** - "LISA verification binaries"

### 关键综述

- **Winget & Kepler (2008)** - "Pulsating White Dwarf Stars"
- **Marsh (2011)** - "Double White Dwarf Systems"
- **Ramsay & Hakala (2021)** - "Compact Binary Evolution"

## 📞 需要帮助？

1. 查看详细文档：`ASTRONOMY_LLM_EVALUATION_GUIDE.md`
2. 运行演示：`python demo.py`
3. 测试 API：`python kimi_interface.py`
4. 运行评估：`python wd_evaluator.py --help`

## ✅ 检查清单

开始训练前，请确认：

- [ ] Kimi API Key 已设置 (`export KIMI_API_KEY=...`)
- [ ] 依赖已安装 (`pip install -r requirements.txt`)
- [ ] 有足够的磁盘空间 (~10GB)
- [ ] 有足够的显存 (~20GB for LoRA) 或使用 CPU
- [ ] 已安装 LLaMA-Factory (`pip install llmtuner`)

---

**版本**: 1.0  
**最后更新**: 2026-03-11
