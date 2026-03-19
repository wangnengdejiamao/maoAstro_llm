#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage QA数据集分析工具
================================================================================
功能描述:
    分析和可视化生成的天文问答数据集，提供统计报告和质量评估。

统计维度:
    - 总体统计: 总QA数、API生成数、规则生成数
    - 主题分布: 各天文主题(赫罗图、SED、光变曲线等)的占比
    - 来源分析: PDF处理情况、成功率
    - 质量评估: API vs 规则生成的质量对比
    - 样例展示: 随机抽取示例QA对

输出格式:
    - 控制台表格展示
    - 支持导出JSON统计报告

关联文件:
    - generate_astronomy_qa_hybrid.py: QA生成脚本
    - output/qa_hybrid/: 默认输入目录

使用方法:
    # 使用默认路径分析
    python analyze_qa_results.py
    
    # 指定自定义目录
    python analyze_qa_results.py --input ./custom_qa_output

作者: AstroSage Team
创建日期: 2024-03
================================================================================
"""

import json
import os
from pathlib import Path
from collections import Counter
import random

def analyze_dataset(output_dir="output/qa_hybrid"):
    """分析生成的数据集"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    # 读取统计数据
    stats_file = output_path / "stats.json"
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # 读取完整数据集
    dataset_file = output_path / "qa_dataset_full.json"
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
    else:
        qa_pairs = []
    
    # 统计PDF文件处理情况
    cache_dir = output_path / "cache"
    if cache_dir.exists():
        processed_pdfs = len(list(cache_dir.glob("*_qa.json")))
    else:
        processed_pdfs = 0
    
    # 计算PDF目录中的总文件数
    pdf_dir = Path("data/pdfs")
    if pdf_dir.exists():
        total_pdfs = len([f for f in pdf_dir.glob("*.pdf") if not f.name.startswith("._")])
    else:
        total_pdfs = 0
    
    # 生成报告
    print("="*70)
    print("📊 天文问答数据集分析报告")
    print("="*70)
    print()
    
    print("📁 处理进度:")
    print(f"  • 总PDF文件: {total_pdfs}")
    print(f"  • 已处理: {processed_pdfs}")
    print(f"  • 进度: {processed_pdfs/total_pdfs*100:.1f}%" if total_pdfs > 0 else "  • 进度: N/A")
    print()
    
    if stats:
        print("📝 问答对统计:")
        print(f"  • 总问答对数: {stats.get('total_qa', len(qa_pairs))}")
        print(f"  • 成功处理PDF: {stats.get('processed', 'N/A')}")
        print(f"  • 失败PDF: {stats.get('failed', 'N/A')}")
        print()
        
        print("📊 问题类型分布:")
        by_type = stats.get('by_type', {})
        for qtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / max(by_type.values()) * 30)
            print(f"  • {qtype:15s}: {count:4d} {bar}")
        print()
        
        print("🔧 生成方法:")
        by_method = stats.get('by_method', {})
        for method, count in by_method.items():
            print(f"  • {method}: {count}")
        print()
    
    # 随机抽样展示
    if qa_pairs:
        print("🎲 随机问答示例:")
        print("-"*70)
        
        # 按类型分组
        type_groups = {}
        for qa in qa_pairs:
            t = qa.get('question_type', 'unknown')
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(qa)
        
        # 每类展示一个示例
        for qtype in ['hr_diagram', 'sed', 'light_curve', 'period', 'xray', 'spectrum']:
            if qtype in type_groups and type_groups[qtype]:
                sample = random.choice(type_groups[qtype])
                print(f"\n📌 类型: {qtype}")
                print(f"❓ 问题: {sample['question']}")
                print(f"💡 答案: {sample['answer'][:200]}...")
                print(f"📄 来源: {sample['source_file']}, 第{sample['page_number']}页")
                print(f"🎯 置信度: {sample.get('confidence', 'N/A')}")
                print("-"*70)
    
    # 文件列表
    print("\n📂 生成的文件:")
    for f in sorted(output_path.glob("*.json")):
        size = f.stat().st_size / 1024  # KB
        print(f"  • {f.name:40s} ({size:8.1f} KB)")
    
    print()
    print("="*70)


def show_conversation_examples(output_dir="output/qa_hybrid", n=3):
    """展示对话格式示例"""
    train_file = Path(output_dir) / "train_conversations.json"
    
    if not train_file.exists():
        print("❌ 训练集文件不存在")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print("\n" + "="*70)
    print("💬 对话格式示例 (适合微调大模型)")
    print("="*70)
    
    for i, conv in enumerate(random.sample(conversations, min(n, len(conversations))), 1):
        print(f"\n--- 示例 {i} ---")
        for msg in conv['messages']:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                print(f"[系统] {content[:100]}...")
            elif role == 'user':
                print(f"[用户] {content}")
            elif role == 'assistant':
                print(f"[助手] {content[:200]}...")
        print(f"[元数据] {conv['metadata']}")


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/qa_hybrid"
    
    analyze_dataset(output_dir)
    show_conversation_examples(output_dir)
