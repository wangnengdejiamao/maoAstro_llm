#!/bin/bash
# AstroSage 快速测试脚本

echo "=================================="
echo "AstroSage-Llama-3.1-8B 快速测试"
echo "=================================="
echo ""
echo "测试1: 检查 Ollama 服务..."
if ollama list | grep -q "astrosage-local"; then
    echo "✓ astrosage-local 模型已安装"
else
    echo "✗ 模型未安装"
    exit 1
fi

echo ""
echo "测试2: 运行简单查询 (这可能需要一些时间加载模型)..."
echo "问题: 什么是激变变星(Cataclysmic Variable)?"
echo "---"

# 使用非交互模式运行
ollama run astrosage-local "什么是激变变星(Cataclysmic Variable)? 请简要回答。" 2>&1

echo ""
echo "---"
echo "✓ 测试完成"
