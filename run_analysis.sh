#!/bin/bash
# Astro-AI 分析启动脚本

echo "================================"
echo "Astro-AI 终极分析系统"
echo "================================"
echo ""

if [ $# -ne 3 ]; then
    echo "使用方法: ./run_analysis.sh <RA> <DEC> <NAME>"
    echo "示例: ./run_analysis.sh 196.9744 53.8585 EV_UMa"
    exit 1
fi

RA=$1
DEC=$2
NAME=$3

echo "开始分析: $NAME"
echo "坐标: RA=$RA, DEC=$DEC"
echo ""

python ultimate_analysis_fixed.py --ra $RA --dec $DEC --name "$NAME"
