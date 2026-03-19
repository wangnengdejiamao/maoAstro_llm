#!/bin/bash
# 白矮星专用模型一键训练脚本
# 集成 Kimi API 生成训练数据

set -e

echo "========================================"
echo "🌟 Qwen-8B 白矮星专用模型训练"
echo "========================================"
echo "侧重领域:"
echo "  • 单白矮星演化与冷却"
echo "  • 双白矮星系统"
echo "  • 磁性白矮星"
echo "  • 吸积白矮星与激变变星"
echo "========================================"

# 配置
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
PROJECT_NAME="qwen8b-whitewarf"
DATA_DIR="./data/white_dwarf_papers"
MODEL_DIR="./models"
KIMI_API_KEY="9cccf25-f532-861b-8000-000042a859dc"

# 导出 API Key
export KIMI_API_KEY

# 检查依赖
echo ""
echo "[1/6] 检查依赖..."
python -c "import requests; import json; print('✓ 依赖检查通过')" || {
    echo "安装依赖..."
    pip install requests feedparser -q
}

# 创建目录
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

echo ""
echo "[2/6] 下载白矮星领域文献..."
echo "========================================"

# 搜索 arXiv 论文
python download_wd_papers.py \
    --search \
    --max-results 100 \
    --start-date 2020-01-01 \
    --output-dir $DATA_DIR

echo ""
echo "[3/6] 使用 Kimi API 生成训练数据..."
echo "========================================"

# 生成训练数据（使用 Kimi）
python download_wd_papers.py \
    --generate-data \
    --use-kimi \
    --output-dir $DATA_DIR

# 检查数据生成结果
WD_DATA_FILE="$DATA_DIR/wd_training_data.json"
if [ -f "$WD_DATA_FILE" ]; then
    DATA_COUNT=$(python -c "import json; print(len(json.load(open('$WD_DATA_FILE'))))")
    echo "✓ 生成训练数据: $DATA_COUNT 条"
else
    echo "⚠ 数据生成可能失败，使用备用方案..."
fi

echo ""
echo "[4/6] 检查 LLaMA-Factory..."
echo "========================================"

if ! command -v llamafactory-cli &> /dev/null; then
    echo "安装 LLaMA-Factory..."
    pip install "llamafactory[torch,metrics]" -q
fi

echo "✓ LLaMA-Factory 已就绪"

echo ""
echo "[5/6] 开始模型训练..."
echo "========================================"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠ 未检测到 NVIDIA GPU，将使用 CPU 训练（极慢）"
fi

# Stage 1: 持续预训练 (可选，如果有大量领域文本)
read -p "是否运行持续预训练 (CPT)? 需要大量计算资源 [y/N]: " run_cpt
if [[ $run_cpt =~ ^[Yy]$ ]]; then
    echo ""
    echo ">>> Stage 1: 持续预训练 (CPT)"
    llamafactory-cli train \
        --stage pt \
        --model_name_or_path $BASE_MODEL \
        --do_train \
        --dataset wd_cpt \
        --finetuning_type lora \
        --lora_target all \
        --lora_rank 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --output_dir $MODEL_DIR/${PROJECT_NAME}-cpt \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1.5e-5 \
        --num_train_epochs 3 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --bf16 \
        --plot_loss \
        --overwrite_output_dir
    
    CPT_ADAPTER="--adapter_name_or_path $MODEL_DIR/${PROJECT_NAME}-cpt"
else
    echo "跳过 CPT，直接从基础模型开始 SFT"
    CPT_ADAPTER=""
fi

# Stage 2: 监督微调
echo ""
echo ">>> Stage 2: 监督微调 (SFT)"

llamafactory-cli train \
    --stage sft \
    --model_name_or_path $BASE_MODEL \
    $CPT_ADAPTER \
    --do_train \
    --dataset wd_training_data \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir $MODEL_DIR/${PROJECT_NAME}-sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-5 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --plot_loss \
    --overwrite_output_dir

echo ""
echo ">>> Stage 3: 合并权重"

# 合并 LoRA 权重
llamafactory-cli export \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $MODEL_DIR/${PROJECT_NAME}-sft \
    --finetuning_type lora \
    --template qwen \
    --export_dir $MODEL_DIR/${PROJECT_NAME}-final \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false

echo ""
echo "[6/6] 评估模型..."
echo "========================================"

# 生成白矮星专用评估数据
if [ ! -f "$DATA_DIR/wd_eval.json" ]; then
    echo "生成白矮星评估数据..."
    python -c "
import json
from download_wd_papers import WhiteDwarfPaperDownloader

downloader = WhiteDwarfPaperDownloader('$DATA_DIR')
eval_data = downloader.generate_synthetic_qa(use_kimi=False)  # 模板生成评估数据
with open('$DATA_DIR/wd_eval.json', 'w') as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)
print(f'生成 {len(eval_data)} 条评估数据')
"
fi

# 运行评估
echo "评估白矮星专用模型..."
python run_eval.py \
    --model $MODEL_DIR/${PROJECT_NAME}-final \
    --interface hf \
    --dataset custom \
    --data-path $DATA_DIR/wd_eval.json \
    --output-dir ./eval_results

echo ""
echo "========================================"
echo "✅ 训练完成!"
echo "========================================"
echo "模型位置: $MODEL_DIR/${PROJECT_NAME}-final"
echo "评估结果: ./eval_results/"
echo ""
echo "使用方法:"
echo "  # Ollama 部署"
echo "  ollama create whitewarf -f ./Modelfile.whitewarf"
echo ""
echo "  # Python 调用"
echo "  from transformers import AutoModelForCausalLM"
echo "  model = AutoModelForCausalLM.from_pretrained('$MODEL_DIR/${PROJECT_NAME}-final')"
echo ""
echo "========================================"

# 生成使用示例
cat > $MODEL_DIR/${PROJECT_NAME}-example.py << 'EOF'
#!/usr/bin/env python3
"""
白矮星专用模型使用示例
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型
model_path = "./qwen8b-whitewarf-final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 系统提示词
system_prompt = """你是白矮星天体物理学专家，专门研究单白矮星、双白矮星、磁性白矮星和吸积白矮星。
回答要精确，包含定量数据，避免常见误解。"""

# 测试问题
questions = [
    "什么是白矮星的钱德拉塞卡极限？",
    "双白矮星系统如何产生引力波？",
    "磁性白矮星的典型磁场强度是多少？",
    "吸积白矮星的新星爆发机制是什么？",
]

for question in questions:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    print(f"\nQ: {question}")
    print(f"A: {response}")
EOF

echo "示例脚本已生成: $MODEL_DIR/${PROJECT_NAME}-example.py"
