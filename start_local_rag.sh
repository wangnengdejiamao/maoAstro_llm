#!/bin/bash
# AstroSage 本地模型 RAG 启动脚本

set -e

echo "=========================================="
echo "🚀 AstroSage - 本地模型 RAG 启动器"
echo "=========================================="
echo ""

# 检查 Python 依赖
echo "📦 检查依赖..."
python -c "import requests" 2>/dev/null || pip install requests -q
python -c "import jieba" 2>/dev/null || pip install jieba -q
echo "✅ 依赖检查通过"
echo ""

# 选择模式
echo "请选择运行模式:"
echo "1) Ollama + RAG (快速体验)"
echo "2) AstroSage 完整系统 (训练好的模型 + RAG) ⭐推荐"
echo "3) 检查可用模型"
echo ""
read -p "输入选项 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🔄 启动 Ollama + RAG 模式"
        echo ""
        
        # 检查 Ollama
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "⚠️  Ollama 服务未启动"
            echo ""
            echo "请在新终端窗口运行:"
            echo "  ollama serve"
            echo ""
            read -p "Ollama 启动后按回车继续..."
        fi
        
        # 列出可用模型
        echo "📋 可用模型:"
        ollama list 2>/dev/null || echo "  无法获取模型列表"
        echo ""
        
        read -p "输入模型名称 (默认: qwen3): " model_name
        model_name=${model_name:-qwen3}
        
        echo ""
        echo "🚀 启动 RAG + $model_name"
        echo ""
        python rag_system/retrieval/rag_with_ollama.py --model $model_name
        ;;
    
    2)
        echo ""
        echo "🔄 AstroSage 完整系统模式"
        echo "   (训练好的模型 + RAG双轨检索)"
        echo ""
        
        # 检查训练好的模型
        if [ -d "train_qwen/output_qwen25/merged_model" ]; then
            echo "📦 找到训练好的模型:"
            echo "   train_qwen/output_qwen25/merged_model"
            model_path="train_qwen/output_qwen25/merged_model"
        elif [ -d "train_qwen/output_qwen25/final_model" ]; then
            echo "📦 找到LoRA权重:"
            echo "   train_qwen/output_qwen25/final_model"
            echo ""
            echo "⚠️  需要先合并LoRA权重才能使用"
            echo "   运行: python train_qwen/merge_lora.py"
            exit 1
        else
            echo "❌ 未找到训练好的模型"
            echo "   请先完成模型训练或检查路径"
            exit 1
        fi
        
        echo ""
        echo "🔧 启动配置:"
        echo "   模型: $model_path"
        echo "   RAG数据: output/qa_hybrid/"
        echo ""
        
        read -p "是否启用RAG? (Y/n): " use_rag
        use_rag=${use_rag:-Y}
        
        echo ""
        echo "🚀 启动 AstroSage 完整系统..."
        echo ""
        
        if [ "$use_rag" = "n" ] || [ "$use_rag" = "N" ]; then
            python start_astrosage_complete.py --model-path "$model_path" --no-rag
        else
            python start_astrosage_complete.py --model-path "$model_path"
        fi
        ;;
    
    3)
        echo ""
        echo "📋 系统状态检查"
        echo ""
        
        # 检查 Ollama
        echo "1. Ollama 状态:"
        if pgrep -x "ollama" > /dev/null; then
            echo "   ✅ 运行中"
            echo "   可用模型:"
            ollama list 2>/dev/null | grep -v "NAME" | sed 's/^/     - /' || echo "     无法获取"
        else
            echo "   ⚠️  未运行"
            echo "   启动命令: ollama serve"
        fi
        echo ""
        
        # 检查本地模型
        echo "2. 本地 HuggingFace 模型:"
        if [ -d "models/qwen/Qwen-VL-Chat-Int4" ]; then
            size=$(du -sh models/qwen/Qwen-VL-Chat-Int4 2>/dev/null | cut -f1)
            echo "   ✅ Qwen-VL-Chat-Int4 ($size)"
        else
            echo "   ❌ 未找到"
        fi
        
        if [ -d "models/astrosage-llama-3.1-8b/hf_model" ]; then
            echo "   ✅ AstroSage-Llama-3.1-8B"
        fi
        echo ""
        
        # 检查 RAG 数据
        echo "3. RAG 数据:"
        if [ -f "output/qa_hybrid/qa_dataset_full.json" ]; then
            count=$(python -c "import json; print(len(json.load(open('output/qa_hybrid/qa_dataset_full.json'))))" 2>/dev/null || echo "未知")
            echo "   ✅ 问答数据: $count 条"
        else
            echo "   ⚠️  问答数据未生成"
        fi
        echo ""
        
        # 检查训练数据
        echo "4. 训练数据:"
        if [ -f "train_qwen/data/qwen_train.json" ]; then
            count=$(python -c "import json; print(len(json.load(open('train_qwen/data/qwen_train.json'))))" 2>/dev/null || echo "未知")
            echo "   ✅ 训练集: $count 条"
        else
            echo "   ⚠️  训练数据未准备"
        fi
        echo ""
        
        # 检查训练好的模型
        echo "5. 训练好的模型:"
        if [ -d "train_qwen/output_qwen25/merged_model" ]; then
            size=$(du -sh train_qwen/output_qwen25/merged_model 2>/dev/null | cut -f1)
            echo "   ✅ AstroSage完整模型: train_qwen/output_qwen25/merged_model ($size)"
        elif [ -d "train_qwen/output_qwen25/final_model" ]; then
            echo "   ⚠️  LoRA权重: train_qwen/output_qwen25/final_model (需要合并)"
            echo "      合并命令: python train_qwen/merge_lora.py"
        fi
        if [ -d "train_qwen/output_local/final_model" ]; then
            echo "   ✅ 本地训练模型: train_qwen/output_local/final_model"
        fi
        if [ -d "train_qwen/output/final_model" ]; then
            echo "   ✅ 下载模型: train_qwen/output/final_model"
        fi
        echo ""
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "感谢使用 AstroSage!"
echo "=========================================="
