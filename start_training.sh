#!/bin/bash
# AstroSage 8B Transformers 微调启动脚本

cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo

echo "🚀 启动 AstroSage 8B 微调 (Transformers 方案)..."
echo "日志文件: train_qwen/astrosage_continued/training.log"
echo ""

# 创建输出目录
mkdir -p train_qwen/astrosage_continued

# 启动训练并记录日志
nohup python train_qwen/astrosage_export/finetune_transformers.py > train_qwen/astrosage_continued/training.log 2>&1 &

PID=$!
echo $PID > train_qwen/astrosage_continued/train.pid

echo "✅ 训练已启动 (PID: $PID)"
echo ""
echo "查看实时日志:"
echo "  tail -f train_qwen/astrosage_continued/training.log"
echo ""
echo "查看GPU使用:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止训练:"
echo "  kill $PID"
