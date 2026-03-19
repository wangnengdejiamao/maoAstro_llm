# 🌟 AstroSage - Qwen2.5 天文领域微调

使用生成的天文问答数据集对 Qwen2.5 进行 LoRA 微调。

## 📊 数据集统计

```
总问答对: 20,609
├── API生成 (高质量): 3,793
└── 规则生成: 16,816

问题类型分布:
├── SED (能谱分布): 8,325
├── Light Curve (光变曲线): 3,170
├── HR Diagram (赫罗图): 3,183
├── Period (周期): 1,483
├── CV (灾变变星): 1,020
├── General (通用): 1,238
├── X-ray (X射线): 892
├── Binary (双星): 654
└── Spectrum (光谱): 644
```

## 🚀 快速开始

### 1. 准备数据

```bash
# 转换数据为 Qwen 格式
python train_qwen/convert_to_qwen_format.py \
    --input output/qa_hybrid/qa_dataset_full.json \
    --output-dir train_qwen/data \
    --format all \
    --filter-quality \
    --prefer-api \
    --train-ratio 0.9
```

生成的文件：
- `train_qwen/data/qwen_train.json` - 训练集 (90%)
- `train_qwen/data/qwen_val.json` - 验证集 (10%)
- `train_qwen/data/alpaca_train.json` - Alpaca格式训练集
- `train_qwen/data/dataset_info.json` - 数据集统计

### 2. 安装依赖

```bash
pip install -r train_qwen/requirements.txt
```

### 3. 开始训练

#### 方式1: 使用启动脚本

```bash
# 7B 模型，4-bit量化（推荐，单卡24GB显存可运行）
chmod +x train_qwen/train.sh
./train_qwen/train.sh --model-size 7b

# 3B 模型，适合较低显存
./train_qwen/train.sh --model-size 3b

# 不使用量化（需要更多显存）
./train_qwen/train.sh --model-size 7b --no-quant

# 启用 wandb 记录
./train_qwen/train.sh --model-size 7b --use-wandb
```

#### 方式2: 直接使用 Python

```bash
# 7B 模型，4-bit量化
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

### 4. 推理测试

```bash
# 交互式对话
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    -i

# 测试预设问题
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    --test

# 单问题推理
python train_qwen/inference.py \
    --model train_qwen/output/merged_model \
    -q "什么是赫罗图？"

# 使用 LoRA 模型（未合并）
python train_qwen/inference.py \
    --model train_qwen/output/final_model \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --use-peft \
    -i
```

## 📁 文件结构

```
train_qwen/
├── convert_to_qwen_format.py   # 数据格式转换
├── train_qwen_lora.py          # LoRA 训练脚本
├── train.sh                    # 启动脚本
├── inference.py                # 推理脚本
├── requirements.txt            # 依赖列表
├── data/                       # 训练数据
│   ├── qwen_train.json
│   ├── qwen_val.json
│   └── dataset_info.json
└── output/                     # 训练输出
    ├── final_model/            # LoRA 权重
    └── merged_model/           # 合并后的完整模型
```

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| model-size | 7b | 模型大小: 0.5b/3b/7b/14b/32b |
| use-4bit | True | 4-bit量化，节省显存 |
| lora-r | 64 | LoRA rank |
| lora-alpha | 16 | LoRA alpha |
| batch-size | 1 | 批大小 |
| gradient-accumulation | 8 | 梯度累积步数 |
| lr | 2e-4 | 学习率 |
| epochs | 3 | 训练轮数 |
| max-seq-length | 2048 | 最大序列长度 |

## 💻 显存需求

| 模型 | 量化 | 显存需求 |
|-----|------|---------|
| Qwen2.5-0.5B | 无 | ~4GB |
| Qwen2.5-3B | 4-bit | ~6GB |
| Qwen2.5-7B | 4-bit | ~12GB |
| Qwen2.5-7B | 无 | ~20GB |
| Qwen2.5-14B | 4-bit | ~20GB |
| Qwen2.5-32B | 4-bit | ~40GB |

## 📝 数据格式

### Qwen 官方格式
```json
{
  "id": "astro_qa_0",
  "conversations": [
    {"from": "system", "value": "你是 AstroSage..."},
    {"from": "user", "value": "什么是赫罗图？"},
    {"from": "assistant", "value": "赫罗图(Hertzsprung-Russell Diagram)是..."}
  ],
  "metadata": {
    "question_type": "hr_diagram",
    "source_file": "...",
    "page_number": 1
  }
}
```

### Alpaca 格式
```json
{
  "instruction": "什么是赫罗图？",
  "input": "",
  "output": "赫罗图(Hertzsprung-Russell Diagram)是...",
  "history": []
}
```

## 🔧 高级用法

### 筛选高质量数据

```bash
# 只使用 API 生成的高质量数据
python train_qwen/convert_to_qwen_format.py \
    --filter-quality \
    --prefer-api
```

### 多轮对话训练

数据格式支持多轮对话：
```json
{
  "conversations": [
    {"from": "user", "value": "问题1"},
    {"from": "assistant", "value": "回答1"},
    {"from": "user", "value": "追问"},
    {"from": "assistant", "value": "回答2"}
  ]
}
```

### 自定义系统提示词

修改 `train_qwen/train_qwen_lora.py` 中的 `SYSTEM_PROMPT`：
```python
SYSTEM_PROMPT = """你是 AstroSage，一位专业的天文学专家助手...
"""
```

## 📈 监控训练

### 使用 Weights & Biases

```bash
# 安装 wandb
pip install wandb
wandb login

# 训练时启用
./train_qwen/train.sh --use-wandb
```

### 本地监控

```bash
# 查看训练日志
tail -f train_qwen/output/training.log

# 查看检查点
ls -lh train_qwen/output/checkpoint-*
```

## 🎯 微调建议

1. **小模型开始**: 先用 3B 模型测试，确认数据质量后再训练 7B/14B
2. **学习率**: 建议 1e-4 ~ 5e-4，太大容易过拟合
3. **数据质量**: API 生成的 3,793 条数据质量更高，可优先使用
4. **验证集**: 保留 10% 数据用于验证，防止过拟合
5. **早停**: 监控验证 loss，通常 2-3 个 epoch 即可

## 🐛 常见问题

### CUDA Out of Memory
```bash
# 减小 batch size 或增大梯度累积
./train_qwen/train.sh --batch-size 1 --gradient-accumulation 16

# 使用更小模型
./train_qwen/train.sh --model-size 3b
```

### 模型加载慢
```bash
# 使用更快的存储或预下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### 依赖冲突
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r train_qwen/requirements.txt
```

## 📚 参考资料

- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2.5)
- [PEFT LoRA 文档](https://huggingface.co/docs/peft)
- [Transformers 训练文档](https://huggingface.co/docs/transformers/training)

## 📄 许可证

本项目使用 Apache 2.0 许可证。
