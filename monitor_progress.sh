#!/bin/bash
while true; do
    clear
    echo "=========================================="
    echo "📊 天文问答生成进度监控"
    echo "=========================================="
    echo ""
    
    # 统计缓存文件
    cache_count=$(ls output/qa_hybrid/cache/*.json 2>/dev/null | wc -l)
    echo "✅ 已处理PDF: $cache_count / 259"
    
    # 计算进度
    if [ "$cache_count" -gt 0 ]; then
        progress=$(echo "scale=1; $cache_count * 100 / 259" | bc)
        echo "📈 进度: $progress%"
        
        # 进度条
        filled=$(echo "$cache_count * 50 / 259" | bc)
        printf "["
        for i in $(seq 1 $filled); do printf "█"; done
        for i in $(seq $filled 49); do printf "░"; done
        printf "]\n"
    fi
    
    echo ""
    
    # 检查进程
    if pgrep -f "generate_astronomy_qa" > /dev/null; then
        echo "🔄 状态: 运行中"
    else
        echo "✨ 状态: 已完成"
    fi
    
    # 显示最新统计
    if [ -f "output/qa_hybrid/stats.json" ]; then
        echo ""
        echo "📋 最新统计:"
        cat output/qa_hybrid/stats.json | python3 -m json.tool 2>/dev/null | grep -E "(processed|failed|total_qa)" | head -3
    fi
    
    # 显示最近处理的文件
    echo ""
    echo "📄 最近处理:"
    ls -lt output/qa_hybrid/cache/*.json 2>/dev/null | head -3 | awk '{print "  " $9}'
    
    sleep 10
done
