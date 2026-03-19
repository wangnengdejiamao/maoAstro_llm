# 🌟 AstroSage - Qwen2.5 训练完整指南

## 📊 数据生成结果

✅ **问答对生成完成！**

```
总问答对: 20,609 条
├── API生成 (Kimi高质量): 3,793 条 ⭐
└── 规则生成: 16,816 条

问题类型分布:
├── SED (能谱分布):        8,325 条
├── Light Curve (光变曲线): 3,170 条
├── HR Diagram (赫罗图):    3,183 条
├── Period (周期):          1,483 条
├── CV (灾变变星):          1,020 条
├── General (通用):         1,238 条
├── X-ray (X射线):            892 条
├── Binary (双星):            654 条
└── Spectrum (光谱):          644 条
```

---

## 🚀 快速开始训练

### 1️⃣ 准备环境

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 安装依赖
pip install torch transformers peft datasets accelerate bitsandbytes

# 或使用 requirements.txt
pip install -r train_qwen/requirements.txt
```

### 2️⃣ 数据已准备就绪

```bash
# 数据位置
train_qwen/data/
├── qwen_train.json       # Qwen格式训练集 (3,413条)
├── qwen_val.json         # Qwen格式验证集 (380条)
├── alpaca_train.json     # Alpaca格式训练集
├── alpaca_val.json       # Alpaca格式验证集
└── dataset_info.json     # 数据集统计
```

**注意**: 当前使用的是筛选后的 **3,793 条高质量 API 生成数据**

### 3️⃣ 开始训练

```bash
# 方式1: 使用启动脚本 (推荐)
chmod +x train_qwen/train.sh
./train_qwen/train.sh --model-size 7b

# 方式2: 直接使用 Python
python train_qwen/train_qwen_lora.py \
    --model-size 7b \
    --train-data train_qwen/data/qwen_train.json \
    --val-data train_qwen/data/qwen_val.json \
    --output-dir train_qwen/output \
    --use-4bit \
    --lora-r 64 \
    --lora-alpha 16 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --lr 2e-4 \
    --epochs 3
```

---

## 📁 项目文件结构

```
astro-ai-demo/
├── 📂 train_qwen/                     # Qwen 训练代码
│   ├── convert_to_qwen_format.py     # 数据格式转换
│   ├── train_qwen_lora.py            # LoRA 训练脚本
│   ├── inference.py                  # 推理测试脚本
│   ├── train.sh                      # 启动脚本
│   ├── requirements.txt              # 依赖列表
│   ├── README.md                     # 详细文档
│   └── 📂 data/                      # 训练数据
│       ├── qwen_train.json           # 训练集
│       ├── qwen_val.json             # 验证集
│       └── dataset_info.json         # 统计信息
│
├── 📂 output/qa_hybrid/               # 原始生成数据
│   ├── qa_dataset_full.json          # 完整数据集 (20,609条)
│   ├── qa_hr_diagram.json            # 赫罗图数据
│   ├── qa_sed.json                   # SED数据
│   ├── qa_light_curve.json           # 光变曲线数据
│   └── ...                           # 其他类型
│
├── generate_astronomy_qa_hybrid.py   # 问答数据生成器
├── analyze_qa_results.py             # 数据分析工具
└── QA_GENERATION_SUMMARY.md          # 生成总结报告
```

---

## 🎯 训练配置建议

### 根据显存选择配置

| 显存大小 | 推荐配置 | 命令 |
|---------|---------|------|
| 12-16GB | 7B + 4-bit | `./train.sh --model-size 7b` |
| 8-12GB | 3B + 4-bit | `./train.sh --model-size 3b` |
| 24GB+ | 7B + 无量化 | `./train.sh --model-size 7b --no-quant` |
| 40GB+ | 14B + 4-bit | `./train.sh --model-size 14b` |

### 训练参数

```bash
# 默认配置（适合大多数情况）
--model-size 7b           # 模型大小
--use-4bit                # 4-bit量化
--lora-r 64               # LoRA rank
--lora-alpha 16           # LoRA alpha
--batch-size 1            # 批大小
--gradient-accumulation 8 # 梯度累积
--lr 2e-4                 # 学习率
--epochs 3                # 训练轮数
--max-seq-length 2048     # 最大序列长度
```

---

## 🤖 推理测试

### 训练完成后测试

```bash
# 交互式对话
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    -i

# 测试预设问题
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    --test

# 单问题测试
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    -q "什么是赫罗图？"
```

### 预期效果

训练后的模型应该能够：
- ✅ 回答关于赫罗图的专业问题
- ✅ 解释 SED (能谱分布) 分析方法
- ✅ 分析光变曲线特征
- ✅ 讨论双星系统轨道周期
- ✅ 解释 X射线天体物理
- ✅ 识别光谱特征

---

## 📈 数据质量对比

### 规则生成 vs API 生成

| 指标 | 规则生成 | API生成 (Kimi) |
|-----|---------|---------------|
| 数量 | 16,816 | 3,793 |
| 问题深度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 答案准确性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 专业术语 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 具体数值 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 物理意义 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### API 生成示例

```
问题: 在 Gaia 色–绝对星等赫罗图上，标准灾变双星（CV）的轨道周期 
      P_orb 如何随位置变化？

答案: Abrahams 等人利用 Gaia DR2/EDR3 发现，70 min ≤ P_orb ≤ 8 hr 的
      标准 CV 在 G_BP–G_RP 色–M_G 赫罗图上呈近似单调序列：
      P_orb 沿"白矮星主序—主序星"连线大致正交方向递增...
      
      具体数值：
      - P_orb ≈ 2.2 hr → ⟨G_BP–G_RP⟩ ≃ 0.40 mag, M_G ≃ 9.5 mag
      - P_orb ≈ 3.2 hr → ⟨G_BP–G_RP⟩ ≃ 0.65 mag, M_G ≃ 8.7 mag
```

---

## 🔧 进阶用法

### 使用全部数据训练

```bash
# 转换全部数据（包括规则生成）
python train_qwen/convert_to_qwen_format.py \
    --input output/qa_hybrid/qa_dataset_full.json \
    --output-dir train_qwen/data_full \
    --train-ratio 0.9

# 使用全部数据训练
python train_qwen/train_qwen_lora.py \
    --train-data train_qwen/data_full/qwen_train.json \
    --val-data train_qwen/data_full/qwen_val.json \
    ...
```

### 多轮对话训练

当前数据支持多轮对话格式，可扩展为多轮问答数据。

### 合并 LoRA 权重

训练完成后自动合并，或手动合并：
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "train_qwen/output/final_model")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("train_qwen/output/merged_model")
```

---

## 📚 相关文档

| 文档 | 说明 |
|-----|------|
| `train_qwen/README.md` | 详细训练文档 |
| `QA_GENERATION_SUMMARY.md` | 数据生成总结 |
| `QA_GENERATION_STATUS.md` | 生成状态记录 |

---

## ⚡ 一键启动命令

```bash
# 完整流程（数据已准备，直接训练）
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

# 安装依赖
pip install -r train_qwen/requirements.txt

# 开始训练
chmod +x train_qwen/train.sh
./train_qwen/train.sh --model-size 7b

# 训练完成后测试
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    -i
```

---

## ✨ 特色功能

1. **多主题覆盖**: 赫罗图、SED、光变曲线、周期、X射线、光谱、CV、双星
2. **高质量数据**: 3,793 条 API 生成的高质量问答对
3. **来源可追溯**: 每个问答包含源文件名和页码
4. **多种格式**: 支持 Qwen、Alpaca、ShareGPT 格式
5. **完整工具链**: 数据生成 → 格式转换 → 训练 → 推理

---

## 🎉 恭喜！数据已准备就绪！

你现在可以：
1. ✅ 直接使用 `train_qwen/data/qwen_train.json` 开始训练
2. ✅ 运行 `./train_qwen/train.sh --model-size 7b` 微调 Qwen
3. ✅ 使用 `inference.py` 测试训练后的模型

**祝训练顺利！** 🚀
