#!/bin/bash
# AstroSage 天文助手快速启动脚本

echo "=============================================="
echo "AstroSage-Llama-3.1-8B 天文助手"
echo "=============================================="

# 检查 Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✗ Ollama 未运行"
    echo "请运行: ollama serve"
    exit 1
fi

echo "✓ Ollama 运行中"

# 检查模型
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

echo ""
echo "可用模型:"
echo "$MODELS" | nl

echo ""
echo "选择方案:"
echo "1) AstroSage-Llama-3.1-8B (如果已安装)"
echo "2) llama3.1:8b (备选)"
echo "3) qwen3:8b (当前可用)"
echo "4) 安装 AstroSage"
read -p "选择 (1-4): " choice

case $choice in
    1)
        MODEL="astrosage-llama-3.1-8b"
        ;;
    2)
        MODEL="llama3.1:8b"
        ;;
    3)
        MODEL="qwen3:8b"
        ;;
    4)
        echo ""
        echo "运行安装脚本..."
        python langgraph_demo/setup_astrosage.py
        exit 0
        ;;
    *)
        MODEL="qwen3:8b"
        ;;
esac

echo ""
echo "启动助手，使用模型: $MODEL"
echo "=============================================="
python langgraph_demo/astro_assistant_astrosage.py --model $MODEL
