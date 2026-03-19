# 天文领域大模型评估与微调系统 🔭

## 📖 简介

本系统提供了一套完整的天文领域大语言模型 (LLM) 评估和微调方案，帮助你：

1. **评估模型性能** - 全面评估天文领域知识掌握程度
2. **对比基准模型** - 与 AstroMLab-8B (80.9%) 直接对比
3. **构建训练数据** - 从 arXiv 论文生成高质量训练数据
4. **微调 Qwen-8B** - 使用 LLaMA-Factory 进行两阶段微调

## 🎯 核心目标

| 模型 | 参数量 | AstroMLab-1 准确率 | 目标 |
|------|--------|-------------------|------|
| AstroMLab-8B (基准) | 8B | **80.9%** | 必须超越 |
| **你的 Qwen-8B** | 8B | **> 81%** | 🎯 目标 |

## 📁 文件结构

```
model_evaluation/
├── README.md                           # 本文件
├── ASTRONOMY_LLM_EVALUATION_GUIDE.md   # 详细技术文档
├── QUICKSTART.md                       # 快速开始指南
├── requirements.txt                    # 依赖列表
├── astronomy_evaluator.py              # 核心评估器
├── run_eval.py                         # 评估运行脚本
├── build_dataset.py                    # 数据集构建工具
├── train_qwen_astro.sh                 # 训练脚本
├── demo.py                             # 演示脚本
├── configs/
│   └── qwen8b_astro_finetune.yaml     # 微调配置
└── data/
    └── dataset_info.json              # 数据集配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd model_evaluation
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python demo.py
```

### 3. 评估现有模型

```bash
# 评估 Ollama 部署的 Qwen3-8B
python run_eval.py \
    --model qwen3:8b \
    --interface ollama \
    --dataset custom \
    --data-path ./data/eval_dataset.json

# 对比多个模型
python run_eval.py \
    --models qwen3:8b llama3.1:8b gemma2:9b \
    --compare
```

### 4. 构建评估数据

```bash
python build_dataset.py \
    --build-eval \
    --n-samples 1000 \
    --output ./data/eval_dataset.json
```

## 📊 评估系统特性

### 评估维度

| 维度 | 说明 | 示例 |
|------|------|------|
| **基础知识** | 天文术语、概念理解 | "什么是赫罗图？" |
| **应用任务** | 天体分类、参数估计 | "这是什么类型的变星？" |
| **分析推理** | 多步计算、物理推导 | "计算恒星距离" |
| **反幻觉** | 检测错误概念 | "AM CVn 包含中子星？" |

### 幻觉检测点

系统内置关键天文概念检测：

- ✅ AM CVn **不含中子星**
- ✅ AM CVn **没有标准 P-L 公式**
- ✅ CV/AM CVn 光变 **不是潮汐引起**
- ✅ 潮汐 **不直接导致周期性光变**

### 输出指标

```
【总体性能】
  准确率: 82.5%
  vs AstroMLab-8B: +1.6% ↑

【子领域准确率】
  stellar      : 84.0% ████████████████
  exoplanet    : 80.5% ███████████████▌
  cosmology    : 82.0% ███████████████▊

【幻觉检测】
  幻觉率: 8.5%

【置信度校准】
  ECE: 0.045
```

## 🔧 微调方案

### 两阶段训练流程

```
┌────────────────────────────────────────────────────────┐
│  Stage 1: 持续预训练 (CPT)                             │
│  • 数据: arXiv 天文论文 (5K+ 篇)                        │
│  • 学习率: 2e-5                                        │
│  • LoRA rank: 64                                       │
├────────────────────────────────────────────────────────┤
│  Stage 2: 监督微调 (SFT)                               │
│  • 数据: 天文 QA 对 + 反幻觉数据 (10K+ 条)              │
│  • 学习率: 5e-5                                        │
│  • LoRA rank: 128                                      │
└────────────────────────────────────────────────────────┘
```

### 运行训练

```bash
# 一键运行完整训练
bash train_qwen_astro.sh

# 或手动分步执行
# 1. 准备数据
python build_dataset.py --build-training --n-samples 10000

# 2. 持续预训练
llamafactory-cli train --stage pt --dataset arxiv_astro_cpt ...

# 3. 监督微调  
llamafactory-cli train --stage sft --dataset astro_qa_sft ...

# 4. 合并导出
llamafactory-cli export --adapter_name_or_path ./models/qwen8b-astro-sft ...
```

## 📈 超越 AstroMLab-8B 的关键策略

### 1. 数据优势
- ✅ 使用 2024-2025 年最新 arXiv 数据
- ✅ 重点强化反幻觉训练
- ✅ 平衡各天文子领域

### 2. 模型优势
- ✅ Qwen 原生中文支持优于 LLaMA
- ✅ 更强的多语言和推理能力
- ✅ 更新的基础架构

### 3. 训练优势
- ✅ 两阶段训练策略 (CPT → SFT)
- ✅ 分层 LoRA 配置
- ✅ 课程学习 + 对比学习

## 💻 使用示例

### Python API

```python
from astronomy_evaluator import AstronomyEvaluator, OllamaModelInterface

# 创建模型接口
model = OllamaModelInterface("qwen3:8b")

# 创建评估器
evaluator = AstronomyEvaluator(model)

# 加载数据集
dataset = evaluator.load_custom_dataset("./data/eval_dataset.json")

# 运行评估
summary = evaluator.evaluate(dataset)

# 生成报告
report = evaluator.generate_report(summary)
print(report)

# 保存结果
evaluator.save_results(summary)
```

### 幻觉检测 API

```python
from astronomy_evaluator import AstronomyEvaluator

evaluator = AstronomyEvaluator()

# 检测文本
has_hallucination = evaluator._detect_hallucination(
    "AM CVn 是一个包含中子星的双星系统"
)
print(f"幻觉检测: {has_hallucination}")  # True
```

## 📚 参考资源

### 论文
- [AstroMLab 1: Who wins astronomy jeopardy!?](https://arxiv.org/abs/2407.11194)
- [AstroMLab 3: 8B Model Performance](https://astromlab.org/)

### 开源资源
- [AstroMLab 官网](https://astromlab.org/)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen 模型](https://huggingface.co/Qwen)

## 🎓 进阶使用

### 自定义评估指标

```python
# 在 astronomy_evaluator.py 中添加自定义指标
def custom_metric(results):
    """计算自定义指标"""
    # 你的计算逻辑
    return score
```

### 添加新的幻觉检测点

```python
# 在 HALLUCINATION_CHECKS 中添加
HALLUCINATION_CHECKS = {
    "your_check": {
        "pattern": r"...",
        "context": r"...",
        "should_exist": False,
        "description": "..."
    }
}
```

## ❓ 常见问题

**Q: 如何确认我的模型超越了 AstroMLab-8B？**
```bash
python run_eval.py \
    --models your-model AstroSage-LLaMA-3.1-8B \
    --compare \
    --dataset astromlab1
```

**Q: 评估需要多久？**
- 100 题: ~2-5 分钟 (取决于模型)
- 1000 题: ~20-50 分钟
- AstroMLab-1 (4425 题): ~1-3 小时

**Q: 需要多少 GPU 显存？**
- 8-bit 推理: ~10GB
- LoRA 微调: ~20GB
- 全参数微调: ~80GB

## 📄 许可证

与主项目保持一致。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**版本**: 1.0  
**更新日期**: 2026-03-11
