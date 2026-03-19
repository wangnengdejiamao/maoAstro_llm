# 🌟 WhiteWarf: Qwen-8B 白矮星专用模型

## 项目简介

WhiteWarf 是基于 Qwen-8B 微调的天体物理专用大语言模型，专注于**白矮星天体物理学**领域：

- 🔬 **单白矮星** - 演化、冷却、质量测定、光谱分类
- 🌌 **双白矮星** - 引力波、并合、Ia 型超新星前身
- 🧲 **磁性白矮星** - 磁场产生、分类、高场星
- 💫 **吸积白矮星** - 激变变星、新星、AM CVn 系统

## 🚀 快速开始（三选一）

### 方案 A: 一键训练（需要 Kimi API）

```bash
# 使用 Kimi API 生成高质量训练数据并微调
bash train_whitewarf.sh
```

**要求**: 
- Kimi API Key 已设置（已内置：`9cccf25-f532-861b-8000-000042a859dc`）
- 网络可访问 Kimi API
- 约 20GB GPU 显存（或 CPU 模式）

### 方案 B: 一键训练（无需 API，推荐）

```bash
# 使用预定义的高质量模板数据训练
bash train_whitewarf_no_kimi.sh
```

**优点**: 
- ✅ 无需外部 API
- ✅ 数据质量经过验证
- ✅ 包含反幻觉训练
- ✅ 立即可用

### 方案 C: Ollama 快速部署

```bash
# 直接使用预配置的 Modelfile
ollama create whitewarf -f Modelfile.whitewarf
ollama run whitewarf
```

**适用**: 快速体验白矮星专家模型

## 📁 文件结构

```
model_evaluation/
├── 📘 WHITEWARF_README.md          # 本文件
├── 📗 WHITEWARF_QUICKSTART.md      # 详细快速开始指南
├── 🐍 kimi_interface.py            # Kimi API 接口
├── 🐍 download_wd_papers.py        # 白矮星论文下载器
├── 🐍 wd_evaluator.py              # 白矮星专用评估器
├── 🐍 test_pipeline.py             # 流程测试脚本
├── 📜 train_whitewarf.sh           # 一键训练（Kimi 版）
├── 📜 train_whitewarf_no_kimi.sh   # 一键训练（无 API 版）
├── ⚙️  configs/
│   └── qwen8b_whitewarf_finetune.yaml  # 微调配置
├── 📦 Modelfile.whitewarf          # Ollama 配置
└── 📊 data/
    └── dataset_info.json           # 数据集配置
```

## 🎯 核心功能

### 1. Kimi API 集成 (`kimi_interface.py`)

```python
from kimi_interface import KimiInterface, WhiteDwarfDataGenerator

kimi = KimiInterface()
generator = WhiteDwarfDataGenerator(kimi)

# 生成白矮星问答对
qa_pairs = generator.generate_qa_pairs("双白矮星并合", n_pairs=5)

# 生成反幻觉训练数据
anti_hall = generator.generate_anti_hallucination_data()

# 生成计算题
calc_probs = generator.generate_calculation_problems("吸积盘温度", 3)
```

### 2. 文献下载 (`download_wd_papers.py`)

```bash
# 下载白矮星领域论文
python download_wd_papers.py \
    --search \
    --max-results 100 \
    --start-date 2020-01-01

# 使用 Kimi 生成训练数据
python download_wd_papers.py \
    --generate-data \
    --use-kimi
```

### 3. 专用评估 (`wd_evaluator.py`)

```bash
# 评估模型在白矮星领域的表现
python wd_evaluator.py \
    --model your-model \
    --interface ollama  # 或 hf
```

**评估维度**:
- 单白矮星基础知识
- 双白矮星系统
- 磁性白矮星
- 吸积白矮星
- **反幻觉能力**（关键点！）

## 📊 训练数据构成

### 数据类型

| 类型 | 数量 | 来源 | 说明 |
|------|------|------|------|
| 概念问答 | 12 主题 | Kimi/模板 | 白矮星基础知识 |
| 反幻觉训练 | 6 条 | 专家编写 | 常见错误纠正 |
| 计算题 | 6 题 | Kimi/模板 | 物理计算 |
| arXiv 论文 | 100+ 篇 | 自动下载 | 最新研究 |

### 核心训练主题

1. **单白矮星**
   - 钱德拉塞卡极限 (~1.4 M☉)
   - 光谱分类 (DA, DB, DC, DQ, DZ)
   - 冷却序列

2. **双白矮星**
   - 引力波辐射 (mHz 频段)
   - 并合过程
   - 不是**总是**产生超新星！

3. **磁性白矮星**
   - 10-20% 白矮星有磁场
   - 场强 10^3 - 10^9 G

4. **吸积白矮星**
   - 激变变星 (CVs)
   - AM CVn 系统 (5-65 分钟周期)
   - **不含中子星！**

## 🎓 反幻觉关键点

模型必须正确识别以下常见误解：

| 误解 | 真相 | 重要性 |
|------|------|--------|
| AM CVn 含中子星 | ❌ 是双白矮星 | ⭐⭐⭐ |
| 所有 DWD 并合产生 SN | ❌ 只有 >1.4 M☉ 可能 | ⭐⭐⭐ |
| 所有 WD 有强磁场 | ❌ 仅 10-20% | ⭐⭐ |
| WD 通过核聚变产能 | ❌ 只是冷却 | ⭐⭐ |
| CV 光变是潮汐引起 | ❌ 是吸积盘不稳定性 | ⭐⭐⭐ |

## 💻 使用示例

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen8b-whitewarf-final",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen8b-whitewarf-final"
)

# 系统提示词
system_prompt = """你是白矮星天体物理学专家。"""

# 对话
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "AM CVn 系统包含什么？"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Ollama CLI

```bash
ollama run whitewarf

>>> 什么是双白矮星并合的产物？
>>> 磁性白矮星的磁场是怎么产生的？
>>> AM CVn 的周期范围是多少？
```

## 📈 评估基准

WhiteWarf 包含 15 题专用评估：

| 子领域 | 题目数 | 关键概念 |
|--------|--------|----------|
| 单白矮星 | 3 | 钱德拉塞卡极限、冷却、组成 |
| 双白矮星 | 3 | 引力波、并合条件、轨道演化 |
| 磁性白矮星 | 3 | 磁场比例、产生机制、分类 |
| 吸积白矮星 | 4 | CVs、新星、AM CVn 特性 |
| 反幻觉 | 2 | 常见误解检测 |

**运行评估**:
```bash
python wd_evaluator.py --model whitewarf --interface ollama
```

## 🔧 故障排除

### 问题 1: Kimi API 401 错误

```
错误: Invalid Authentication
```

**解决**: 
```bash
# 使用无需 API 的版本
bash train_whitewarf_no_kimi.sh
```

### 问题 2: 显存不足

```
CUDA out of memory
```

**解决**:
```bash
# 减小 batch size
# 在 yaml 配置中修改:
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# 或使用 8-bit 量化
--quantization_bit 8
```

### 问题 3: arXiv 下载失败

```bash
# 安装 feedparser
pip install feedparser

# 或使用模板数据（无需下载）
bash train_whitewarf_no_kimi.sh
```

## 📚 参考资料

### 经典论文
1. **Fontaine et al. (2001)** - "The Potential of White Dwarf Cosmochronology"
2. **Nelemans (2009)** - "The Galactic Double White Dwarf Population"
3. **Ferrario et al. (2015)** - "Magnetic White Dwarfs"

### 在线资源
- [AstroMLab](https://astromlab.org/) - 天文 LLM 基准
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 微调框架
- [arXiv astro-ph](https://arxiv.org/archive/astro-ph) - 最新论文

## 🤝 贡献

如果你有：
- 白矮星领域论文 PDF
- 专业数据集
- 改进建议

欢迎贡献！

## 📝 许可证

与主项目保持一致。

---

**版本**: 1.0  
**最后更新**: 2026-03-11

---

## 🎯 下一步

1. **运行测试**:
   ```bash
   python test_pipeline.py
   ```

2. **开始训练**（推荐无 API 版本）:
   ```bash
   bash train_whitewarf_no_kimi.sh
   ```

3. **部署到 Ollama**:
   ```bash
   ollama create whitewarf -f Modelfile.whitewarf
   ollama run whitewarf
   ```

4. **评估模型**:
   ```bash
   python wd_evaluator.py --model whitewarf --interface ollama
   ```

祝训练顺利！🌟
