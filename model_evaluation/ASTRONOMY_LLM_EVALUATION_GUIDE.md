# 天文领域大模型评估与优化完整指南

## 📊 基准对比目标

| 模型 | 参数量 | AstroMLab-1 准确率 | 目标 |
|------|--------|-------------------|------|
| AstroSage-LLaMA-3.1-8B | 8B | **80.9%** | 基准线 |
| **你的 Qwen-8B 微调模型** | 8B | **> 81%** | 超越目标 |

> **AstroMLab-8B** = **AstroSage-LLaMA-3.1-8B**，是 AstroMLab 团队发布的天文领域专用模型

---

## 🎯 评估框架设计

### 1. 评估维度 (4大维度 + 12个子维度)

```
天文领域LLM评估金字塔
┌─────────────────────────────────────────────────────────┐
│                    Level 4: 科研能力                     │
│     (论文理解、假设生成、实验设计、结果解读)               │
├─────────────────────────────────────────────────────────┤
│                    Level 3: 分析推理                     │
│     (数据解释、多步计算、物理推导、误差分析)               │
├─────────────────────────────────────────────────────────┤
│                    Level 2: 应用任务                     │
│     (天体分类、参数估计、坐标转换、数据查询)               │
├─────────────────────────────────────────────────────────┤
│                    Level 1: 基础知识                     │
│     (概念理解、术语掌握、常数记忆、公式应用)               │
└─────────────────────────────────────────────────────────┘
```

### 2. 评估数据集构成

| 类别 | 数量 | 说明 | 难度 |
|------|------|------|------|
| **AstroMLab-1 MCQ** | 4,425题 | Annual Review of Astronomy and Astrophysics 多选题 | ⭐⭐⭐ |
| **概念理解题** | 500题 | 天文术语、物理概念解释 | ⭐⭐ |
| **计算题** | 300题 | 恒星物理、宇宙学计算 | ⭐⭐⭐⭐ |
| **分类任务** | 200题 | 变星分类、星系分类 | ⭐⭐⭐ |
| **数据分析题** | 150题 | 光变曲线、光谱解释 | ⭐⭐⭐⭐ |
| **反幻觉测试** | 100题 | 常见错误概念检测 | ⭐⭐⭐⭐⭐ |

### 3. 天文子领域覆盖

- [x] 恒星天体物理 (Stellar Astrophysics)
- [x] 系外行星 (Exoplanets)
- [x] 星系天文学 (Galactic Astronomy)
- [x] 宇宙学 (Cosmology)
- [x] 高能天体物理 (High-Energy Astrophysics)
- [x] 观测仪器学 (Instrumentation)
- [x] 太阳物理 (Solar Physics)
- [x] 天体力学 (Celestial Mechanics)

---

## 🔬 核心评估指标

### 1. 准确率指标

```python
# 总体准确率
overall_accuracy = correct_predictions / total_questions

# 子领域准确率 (用于诊断弱点)
subfield_accuracy = {
    'stellar': correct_stellar / total_stellar,
    'exoplanet': correct_exoplanet / total_exoplanet,
    'cosmology': correct_cosmology / total_cosmology,
    # ...
}

# 难度分层准确率
accuracy_by_difficulty = {
    'easy': correct_easy / total_easy,      # 概念题
    'medium': correct_medium / total_medium, # 应用题
    'hard': correct_hard / total_hard,      # 推理题
}
```

### 2. 幻觉检测指标 (Anti-Hallucination)

```python
# 关键概念错误率 (Critical Concept Error Rate)
ccer = hallucinated_concepts / total_concept_questions

# 常见天文幻觉检测点:
hallucination_checks = {
    'am_cvn_composition': 'AM CVn 不含中子星',
    'pl_relation': 'P-L 关系不能杜撰公式',
    'cv_variability': 'CV 光变不是潮汐导致',
    'tidal_light': '潮汐不直接导致周期性光变',
    # ...
}
```

### 3. 置信度校准指标

```python
# 置信度-准确率相关性 (Calibration)
# 理想情况: 模型说"80%置信"时，准确率应该接近80%
calibration_error = |confidence - accuracy|

# ECE (Expected Calibration Error)
ece = sum(|bin_accuracy - bin_confidence| * bin_samples) / total_samples
```

---

## 🛠️ 评估系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Astronomy LLM Evaluator                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dataset     │  │   Model      │  │  Metrics     │      │
│  │  Manager     │→ │  Interface   │→ │  Calculator  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  AstroMLab-1 │  │  Custom      │  │  Comparison  │      │
│  │  Benchmark   │  │  Benchmark   │  │  Reporter    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 评估流程

### Phase 1: 基础能力评估

```bash
# 1. 运行 AstroMLab-1 基准测试
python model_evaluation/run_astromlab1_eval.py \
    --model your-qwen8b-astro \
    --benchmark astromlab1 \
    --output results/astromlab1_results.json

# 2. 运行自定义天文评估
python model_evaluation/run_astronomy_eval.py \
    --model your-qwen8b-astro \
    --dataset custom_astronomy \
    --categories all
```

### Phase 2: 对比评估

```bash
# 对比多个模型
python model_evaluation/compare_models.py \
    --models your-qwen8b-astro astrosage-8b qwen3-8b \
    --benchmark astromlab1 \
    --output report/comparison_report.html
```

### Phase 3: 错误分析

```bash
# 生成错误分析报告
python model_evaluation/error_analysis.py \
    --results results/astromlab1_results.json \
    --output report/error_analysis.html
```

---

## 🧪 数据集构建方法

### 1. 从 arXiv 论文构建训练数据

```bash
# 下载天文领域论文
python model_evaluation/build_dataset.py \
    --source arxiv \
    --categories astro-ph.SR astro-ph.EP astro-ph.CO \
    --output data/arxiv_papers.json

# 生成问答对
python model_evaluation/generate_qa_pairs.py \
    --input data/arxiv_papers.json \
    --output data/training_qa.json \
    --format sharegpt
```

### 2. 从专业书籍/综述构建

```bash
# 解析 Annual Review of Astronomy and Astrophysics
python model_evaluation/parse_annual_review.py \
    --input data/annual_review/ \
    --output data/araa_qa.json
```

---

## 🚀 模型微调方案 (Qwen-8B)

### 推荐训练流程

```
┌──────────────────────────────────────────────────────────┐
│                    两阶段训练流程                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1: 持续预训练 (Continual Pre-training)            │
│  ─────────────────────────────────────────────          │
│  • 数据: arXiv 天文论文摘要 + 引言 + 结论                  │
│  • 目标: 增强天文领域知识表示                              │
│  • 学习率: 2e-5                                           │
│  • 步数: 10,000 steps                                     │
│                                                          │
│  Stage 2: 监督微调 (Supervised Fine-tuning)              │
│  ─────────────────────────────────────────────          │
│  • 数据: 高质量天文问答对 (10K+ 条)                        │
│  • 目标: 提升指令跟随和推理能力                            │
│  • 学习率: 5e-5                                           │
│  • 步数: 5,000 steps                                      │
│                                                          │
│  Stage 3: 模型融合 (Model Merging) [可选]                  │
│  ─────────────────────────────────────────────          │
│  • 方法: SLERP / TIES / DARE                              │
│  • 目标: 结合多个微调模型优势                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 微调配置文件

```yaml
# model_evaluation/configs/qwen8b_astro_finetune.yaml
base_model: Qwen/Qwen2.5-7B-Instruct  # 或 Qwen3-8B

# Stage 1: Continual Pre-training
cpt:
  data: data/arxiv_astro_text.jsonl
  learning_rate: 2.0e-5
  num_epochs: 3
  batch_size: 4
  max_seq_length: 8192
  lora:
    r: 64
    alpha: 128
    target_modules: [q_proj, k_proj, v_proj, o_proj]

# Stage 2: Supervised Fine-tuning
sft:
  data: data/astro_qa_sharegpt.json
  learning_rate: 5.0e-5
  num_epochs: 3
  batch_size: 4
  max_seq_length: 8192
  lora:
    r: 128
    alpha: 256
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

---

## 📈 优化策略

### 1. 数据质量优化

| 策略 | 说明 | 预期提升 |
|------|------|---------|
| **数据去重** | 使用 MinHash 去重 | +1-2% |
| **质量筛选** | 基于困惑度筛选 | +2-3% |
| **难度均衡** | 确保各难度样本均衡 | +1% |
| **幻觉标注** | 标注常见错误概念 | +3-5% |

### 2. 训练策略优化

```python
# 关键优化技巧

# 1. 课程学习 (Curriculum Learning)
# 先易后难: 概念题 → 应用题 → 推理题

# 2. 多任务学习 (Multi-task Learning)
tasks = [
    'multiple_choice',   # 多选题
    'open_qa',          # 开放问答
    'calculation',      # 计算题
    'classification',   # 分类任务
]

# 3. 对比学习 (Contrastive Learning)
# 对易混淆概念进行对比训练
pairs = [
    ('AM CVn vs CV', 'AM CVn没有中子星，CV可能有'),
    ('Pulsating vs Eclipsing', '脉动vs掩食'),
]
```

### 3. 推理优化

```python
# 自洽性解码 (Self-Consistency)
def self_consistency_generate(model, question, n_samples=5):
    """多次采样，选择最一致的答案"""
    answers = [model.generate(question, temperature=0.7) for _ in range(n_samples)]
    return majority_vote(answers)

# 链式思考 (Chain-of-Thought)
cot_prompt = """请逐步推理这个问题：
1. 首先分析已知条件
2. 然后应用相关物理公式
3. 进行计算
4. 最后给出答案
"""
```

---

## 🔧 快速开始

### 步骤 1: 安装依赖

```bash
pip install -r model_evaluation/requirements.txt
```

### 步骤 2: 下载基准数据集

```bash
python model_evaluation/download_benchmarks.py --all
```

### 步骤 3: 运行评估

```bash
# 评估你的模型
python model_evaluation/evaluate.py \
    --model path/to/your-qwen8b-astro \
    --benchmark astromlab1 \
    --output results/

# 对比 AstroMLab-8B
python model_evaluation/compare.py \
    --models your-qwen8b-astro AstroSage-LLaMA-3.1-8B \
    --benchmark astromlab1
```

### 步骤 4: 查看报告

```bash
# 生成 HTML 报告
python model_evaluation/generate_report.py \
    --results results/ \
    --output report.html
```

---

## 📊 预期结果

### 超越 AstroMLab-8B 的关键点

1. **更好的中文支持**: Qwen 原生中文能力强于 LLaMA
2. **更新的知识**: 使用更新的 arXiv 数据 (2024-2025)
3. **更强的推理**: 采用更新的训练技术 (DPO, ORPO)
4. **更低幻觉**: 针对性的反幻觉训练

### 目标性能

| 指标 | AstroMLab-8B | 你的 Qwen-8B 目标 |
|------|-------------|------------------|
| AstroMLab-1 | 80.9% | **≥ 82%** |
| 概念理解 | ~85% | **≥ 87%** |
| 计算准确率 | ~75% | **≥ 78%** |
| 幻觉率 | ~15% | **≤ 10%** |

---

## 📚 参考资源

### 论文
1. [AstroMLab 1: Who wins astronomy jeopardy!?](https://arxiv.org/abs/2407.11194)
2. [AstroMLab 3: Achieving GPT-4o Level Performance with 8B](https://astromlab.org/)
3. [AstroMLab 4: 70B Model and Benchmarking](https://astromlab.org/)

### 开源资源
- [AstroMLab 数据集](https://huggingface.co/datasets/astromlab)
- [AstroSage 模型](https://huggingface.co/astromlab)
- [LLaMA-Factory 微调框架](https://github.com/hiyouga/LLaMA-Factory)

---

## ❓ FAQ

**Q: 为什么 AstroMLab-8B 使用 LLaMA-3.1-8B 而不是 Qwen？**
A: AstroMLab 早期工作主要基于 LLaMA 系列。Qwen 系列后续展现出了更强的多语言和推理能力。

**Q: 需要多少 GPU 显存？**
A: 
- 8-bit 推理: ~10GB
- LoRA 微调: ~20GB (A100 40GB 或 RTX 4090)
- 全参数微调: ~80GB (需要 A100 80GB 或多卡)

**Q: 训练需要多长时间？**
A:
- CPT (10K steps): ~8 小时 (A100)
- SFT (5K steps): ~4 小时 (A100)
- 总计: ~12 小时

---

*文档版本: 1.0*  
*最后更新: 2026-03-11*
