#!/bin/bash
# Qwen-8B 天文领域微调脚本
# 使用 LLaMA-Factory 框架

set -e

echo "========================================"
echo "Qwen-8B 天文领域微调"
echo "========================================"

# 配置
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 或 Qwen/Qwen3-8B
PROJECT_NAME="qwen8b-astro"
DATA_DIR="./data"
MODEL_DIR="./models"

# 检查依赖
echo "检查依赖..."
if ! command -v llamafactory-cli &> /dev/null; then
    echo "安装 LLaMA-Factory..."
    pip install llmtuner
fi

# 创建目录
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR

echo "========================================"
echo "Stage 1: 持续预训练 (CPT)"
echo "========================================"

# 准备 CPT 数据
echo "准备持续预训练数据..."
python build_dataset.py \
    --download-arxiv \
    --categories astro-ph.SR astro-ph.EP astro-ph.CO astro-ph.GA astro-ph.HE \
    --max-papers 5000 \
    --output $DATA_DIR/arxiv_papers.json

# 运行 CPT
echo "开始持续预训练..."
llamafactory-cli train \
    --stage pt \
    --model_name_or_path $BASE_MODEL \
    --do_train \
    --dataset_dir $DATA_DIR \
    --dataset arxiv_astro_cpt \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir $MODEL_DIR/${PROJECT_NAME}-cpt \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --plot_loss \
    --overwrite_output_dir

echo "✓ 持续预训练完成"

echo "========================================"
echo "Stage 2: 监督微调 (SFT)"
echo "========================================"

# 准备 SFT 数据
echo "准备监督微调数据..."
python build_dataset.py \
    --build-training \
    --n-samples 10000 \
    --output $DATA_DIR/astro_qa.json \
    --format sharegpt

# 运行 SFT
echo "开始监督微调..."
llamafactory-cli train \
    --stage sft \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $MODEL_DIR/${PROJECT_NAME}-cpt \
    --do_train \
    --dataset_dir $DATA_DIR \
    --dataset astro_qa_sft \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir $MODEL_DIR/${PROJECT_NAME}-sft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --plot_loss \
    --overwrite_output_dir

echo "✓ 监督微调完成"

echo "========================================"
echo "Stage 3: 合并权重"
echo "========================================"

# 合并 LoRA 权重
echo "合并模型权重..."
llamafactory-cli export \
    --model_name_or_path $BASE_MODEL \
    --adapter_name_or_path $MODEL_DIR/${PROJECT_NAME}-sft \
    --finetuning_type lora \
    --template qwen \
    --export_dir $MODEL_DIR/${PROJECT_NAME}-final \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false

echo "✓ 模型合并完成"

echo "========================================"
echo "Stage 4: 模型评估"
echo "========================================"

# 准备评估数据
python build_dataset.py \
    --build-eval \
    --n-samples 1000 \
    --output $DATA_DIR/eval_dataset.json

# 运行评估
echo "评估微调模型..."
python run_eval.py \
    --model $MODEL_DIR/${PROJECT_NAME}-final \
    --interface hf \
    --dataset custom \
    --data-path $DATA_DIR/eval_dataset.json \
    --output-dir ./eval_results

echo "评估基准模型 (Qwen-8B)..."
python run_eval.py \
    --model $BASE_MODEL \
    --interface hf \
    --dataset custom \
    --data-path $DATA_DIR/eval_dataset.json \
    --output-dir ./eval_results

echo "========================================"
echo "训练完成!"
echo "========================================"
echo "模型位置: $MODEL_DIR/${PROJECT_NAME}-final"
echo "评估结果: ./eval_results/"
echo "========================================"
