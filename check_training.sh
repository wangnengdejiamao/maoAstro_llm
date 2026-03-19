#!/bin/bash
# 训练进度监控脚本

echo "=========================================="
echo "🔍 训练状态检查"
echo "=========================================="
echo ""

# 检查进程
PID=$(pgrep -f "train_qwen_lora.py" | head -1)
if [ -n "$PID" ]; then
    echo "✅ 训练进程运行中 (PID: $PID)"
    echo ""
    echo "📊 GPU使用情况:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "   GPU信息不可用"
    echo ""
    echo "📈 最近日志:"
    tail -20 train_qwen/training.log 2>/dev/null | grep -v "^Requirement" | grep -v "Downloading" | tail -10
else
    echo "⚠️ 训练进程未运行"
    echo ""
    echo "📋 最后日志:"
    tail -30 train_qwen/training.log 2>/dev/null | tail -15
fi

echo ""
echo "=========================================="
echo "提示: 使用 tail -f train_qwen/training.log 查看实时日志"
echo "=========================================="
