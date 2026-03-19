#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统简化修复脚本 - 仅使用基础依赖
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict

# 数据目录
DATA_DIR = Path("output/qa_hybrid")
VECTOR_FILE = DATA_DIR / "simple_vectors.json"
INDEX_FILE = DATA_DIR / "simple_index.json"


def build_simple_index():
    """构建简单的关键词索引"""
    print("="*60)
    print("🔤 构建简单关键词索引")
    print("="*60)
    
    # 加载数据
    dataset_path = DATA_DIR / "qa_dataset_full.json"
    if not dataset_path.exists():
        print(f"❌ 数据文件不存在: {dataset_path}")
        return False
    
    print(f"\n📂 加载数据...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"   共 {len(qa_data)} 条QA")
    
    # 构建倒排索引
    print("\n   构建索引...")
    index = defaultdict(list)
    
    for i, qa in enumerate(qa_data):
        question = qa.get('question', '').lower()
        answer = qa.get('answer', '').lower()
        
        # 提取关键词（简单分词）
        text = question + " " + answer
        words = set(text.replace('，', ' ').replace('。', ' ').replace('？', ' ').split())
        
        for word in words:
            if len(word) > 2:  # 只保留长度>2的词
                index[word].append(i)
    
    # 保存索引
    print(f"   索引词汇量: {len(index)}")
    
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(dict(index), f, ensure_ascii=False)
    
    print(f"   ✅ 索引已保存: {INDEX_FILE}")
    
    # 测试搜索
    print("\n   测试搜索...")
    test_queries = ["灾变变星", "赫罗图", "光变曲线"]
    
    for query in test_queries:
        query_lower = query.lower()
        if query_lower in index:
            count = len(index[query_lower])
            print(f"   '{query}': {count} 条")
        else:
            print(f"   '{query}': 0 条")
    
    return True


def build_metadata():
    """构建元数据文件"""
    print("\n" + "="*60)
    print("📊 构建元数据")
    print("="*60)
    
    dataset_path = DATA_DIR / "qa_dataset_full.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 统计信息
    stats = {
        "total": len(qa_data),
        "avg_question_length": sum(len(qa.get('question', '')) for qa in qa_data) / len(qa_data),
        "avg_answer_length": sum(len(qa.get('answer', '')) for qa in qa_data) / len(qa_data),
    }
    
    # 保存
    stats_path = DATA_DIR / "simple_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 元数据已保存: {stats_path}")
    print(f"\n   统计:")
    print(f"      总QA数: {stats['total']}")
    print(f"      平均问题长度: {stats['avg_question_length']:.0f} 字符")
    print(f"      平均回答长度: {stats['avg_answer_length']:.0f} 字符")
    
    return True


def verify():
    """验证结果"""
    print("\n" + "="*60)
    print("✅ 验证结果")
    print("="*60)
    
    ok = True
    
    if INDEX_FILE.exists():
        size = INDEX_FILE.stat().st_size / 1024
        print(f"\n🔤 关键词索引: ✓ 正常 ({size:.1f} KB)")
    else:
        print(f"\n🔤 关键词索引: ✗ 不存在")
        ok = False
    
    stats_file = DATA_DIR / "simple_stats.json"
    if stats_file.exists():
        print(f"📊 元数据: ✓ 正常")
    else:
        print(f"📊 元数据: ✗ 不存在")
        ok = False
    
    if ok:
        print("\n🎉 简单索引构建完成！")
        print("\n现在可以在简化版RAG中使用关键词检索了")
    
    return ok


def main():
    print("\n" + "="*60)
    print("🔧 RAG系统简化修复")
    print("="*60)
    print("\n本脚本将构建基础的关键词索引，无需额外依赖")
    print("")
    
    input("按回车键开始...")
    
    build_simple_index()
    build_metadata()
    verify()
    
    print("\n" + "="*60)
    print("✅ 完成")
    print("="*60)


if __name__ == "__main__":
    main()
