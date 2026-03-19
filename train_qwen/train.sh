#!/bin/bash
# Qwen LoRA 微调启动脚本

set -e

echo "=========================================="
echo "🚀 AstroSage - Qwen 训练启动脚本"
echo "=========================================="
echo ""

# 默认配置
MODEL_SIZE="7b"           # 模型大小: 0.5b, 3b, 7b, 14b, 32b
USE_4BIT=true             # 使用4-bit量化
USE_8BIT=false            # 使用8-bit量化
LORA_R=64                 # LoRA rank
LORA_ALPHA=16             # LoRA alpha
BATCH_SIZE=1              # 批大小
GRAD_ACCUM=8              # 梯度累积
LR=2e-4                   # 学习率
EPOCHS=3                  # 训练轮数
MAX_SEQ_LEN=2048          # 最大序列长度
USE_WANDB=false           # 使用 wandb

# 路径配置
DATA_DIR="train_qwen/data"
OUTPUT_DIR="train_qwen/output"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --use-8bit)
            USE_4BIT=false
            USE_8BIT=true
            shift
            ;;
        --no-quant)
            USE_4BIT=false
            USE_8BIT=false
            shift
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "📋 训练配置:"
echo "   模型: Qwen2.5-${MODEL_SIZE}-Instruct"
echo "   量化: $([ "$USE_4BIT" = true ] && echo "4-bit" || ([ "$USE_8BIT" = true ] && echo "8-bit" || echo "无"))"
echo "   LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "   批次: batch_size=${BATCH_SIZE}, accum=${GRAD_ACCUM}"
echo "   训练: lr=${LR}, epochs=${EPOCHS}"
echo ""

# 检查数据是否存在
if [ ! -f "${DATA_DIR}/qwen_train.json" ]; then
    echo "❌ 错误: 训练数据不存在!"
    echo "   请先运行数据转换:"
    echo "   python train_qwen/convert_to_qwen_format.py"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import torch; import transformers; import peft; import datasets" 2>/dev/null || {
    echo "❌ 依赖未安装，正在安装..."
    pip install -r train_qwen/requirements.txt
}

echo "✅ 依赖检查通过"
echo ""

# 启动训练
echo "🚀 开始训练..."
echo ""

python train_qwen/train_qwen_lora.py \
    --model-size ${MODEL_SIZE} \
    --train-data ${DATA_DIR}/qwen_train.json \
    --val-data ${DATA_DIR}/qwen_val.json \
    --output-dir ${OUTPUT_DIR} \
    $( [ "$USE_4BIT" = true ] && echo "--use-4bit" ) \
    $( [ "$USE_8BIT" = true ] && echo "--use-8bit" ) \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation ${GRAD_ACCUM} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --max-seq-length ${MAX_SEQ_LEN} \
    $( [ "$USE_WANDB" = true ] && echo "--use-wandb" )

echo ""
echo "=========================================="
echo "✅ 训练完成!"
echo "=========================================="
echo ""
echo "模型保存在: ${OUTPUT_DIR}/final_model/"
echo "合并模型在: ${OUTPUT_DIR}/merged_model/"
echo ""
echo "推理测试:"
echo "   python train_qwen/inference.py --model ${OUTPUT_DIR}/merged_model"
