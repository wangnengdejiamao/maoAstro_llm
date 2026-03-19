#!/bin/bash
# 训练监控脚本

LOG_FILE="train_qwen/training_qwen25_7b.log"
OUTPUT_DIR="train_qwen/output_qwen25"

echo "💫 Qwen2.5-7B 训练监控"
echo "=========================================="

# 检查进程
PID=$(pgrep -f train_with_qwen25 | head -1)
if [ -n "$PID" ]; then
    echo "✅ 训练进程运行中 (PID: $PID)"
    echo ""
    echo "📊 GPU 状态:"
    nvidia-smi | grep -E "(GPU|MiB)" | head -4
    echo ""
else
    echo "⚠️ 训练进程未运行"
fi

# 查看最新进度
echo "📈 训练进度:"
tail -5 "$LOG_FILE" 2>/dev/null | grep -E "([0-9]+%|loss|step|Step)" | tail -3

# 统计
echo ""
echo "📂 输出文件:"
ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -5

echo ""
echo "⏱️  查看实时日志: tail -f $LOG_FILE"
