#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
RAG知识库检查工具
================================================================================
功能描述:
    全面检查RAG系统的知识库内容，包括：
    - 数据统计分析
    - 主题分布
    - 样本质量检查
    - 检索测试

使用方法:
    python check_rag_knowledge.py

作者: AstroSage Team
================================================================================
"""

import os
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# RAG数据目录
RAG_DIR = Path("output/qa_hybrid")


def load_json_file(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"   无法加载 {filepath}: {e}")
        return None


def analyze_knowledge_base():
    """分析知识库整体情况"""
    print("="*70)
    print("📊 RAG知识库整体分析")
    print("="*70)
    
    # 1. 检查主数据集
    full_dataset_path = RAG_DIR / "qa_dataset_full.json"
    if full_dataset_path.exists():
        print(f"\n📦 加载主数据集...")
        with open(full_dataset_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        total_qa = len(full_data)
        print(f"   总QA对数: {total_qa:,}")
        
        # 分析来源
        sources = defaultdict(int)
        api_generated = 0
        rule_generated = 0
        
        for qa in full_data:
            # 统计来源
            source = qa.get('source', 'unknown')
            sources[Path(source).name if isinstance(source, str) else 'unknown'] += 1
            
            # 统计生成方式
            if qa.get('metadata', {}).get('generated_by') == 'api':
                api_generated += 1
            else:
                rule_generated += 1
        
        print(f"   API生成: {api_generated:,} ({api_generated/total_qa*100:.1f}%)")
        print(f"   规则生成: {rule_generated:,} ({rule_generated/total_qa*100:.1f}%)")
        
        # 显示主要来源
        print(f"\n📄 主要来源文档 (Top 10):")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {source[:50]:<50} {count:>5}条")
    
    # 2. 检查分类数据
    print(f"\n📁 主题分类统计:")
    categories = {
        "赫罗图": "qa_hr_diagram.json",
        "灾变变星(CV)": "qa_cv.json",
        "双星系统": "qa_binary.json",
        "光变曲线": "qa_light_curve.json",
        "周期分析": "qa_period.json",
        "SED": "qa_sed.json",
        "光谱": "qa_spectrum.json",
        "X射线": "qa_xray.json",
        "通用": "qa_general.json",
    }
    
    total_by_category = 0
    for category, filename in sorted(categories.items()):
        filepath = RAG_DIR / filename
        if filepath.exists():
            data = load_json_file(filepath)
            if data:
                count = len(data)
                total_by_category += count
                print(f"   {category:<15} {count:>6,} 条")
    
    print(f"   {'─'*30}")
    print(f"   {'合计':<15} {total_by_category:>6,} 条")
    
    # 3. 检查缓存数据
    cache_dir = RAG_DIR / "cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*_qa.json"))
        print(f"\n💾 缓存文件: {len(cache_files)} 个PDF的处理结果")


def check_sample_quality(num_samples=5):
    """检查样本质量"""
    print("\n" + "="*70)
    print("🔍 随机样本质量检查")
    print("="*70)
    
    # 加载主数据集
    full_dataset_path = RAG_DIR / "qa_dataset_full.json"
    if not full_dataset_path.exists():
        print("   主数据集不存在")
        return
    
    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 随机采样
    samples = random.sample(full_data, min(num_samples, len(full_data)))
    
    for i, qa in enumerate(samples, 1):
        print(f"\n{'─'*70}")
        print(f"样本 {i}/{num_samples}")
        print('─'*70)
        
        question = qa.get('question', 'N/A')
        answer = qa.get('answer', 'N/A')
        category = qa.get('category', 'unknown')
        source = qa.get('source', 'unknown')
        generated_by = qa.get('metadata', {}).get('generated_by', 'unknown')
        
        print(f"【分类】{category}")
        print(f"【来源】{source}")
        print(f"【生成方式】{generated_by}")
        print(f"\n【问题】\n{question[:200]}{'...' if len(question) > 200 else ''}")
        print(f"\n【回答】\n{answer[:300]}{'...' if len(answer) > 300 else ''}")


def analyze_training_data():
    """分析训练数据"""
    print("\n" + "="*70)
    print("📚 训练数据分析")
    print("="*70)
    
    train_path = Path("train_qwen/data/qwen_train.json")
    val_path = Path("train_qwen/data/qwen_val.json")
    
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"\n训练集: {len(train_data):,} 条")
        
        # 分析对话长度
        total_turns = 0
        for item in train_data:
            conversations = item.get('conversations', [])
            total_turns += len(conversations)
        
        avg_turns = total_turns / len(train_data) if train_data else 0
        print(f"   平均对话轮数: {avg_turns:.1f}")
    else:
        print(f"\n训练集不存在: {train_path}")
    
    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"\n验证集: {len(val_data):,} 条")
    else:
        print(f"\n验证集不存在: {val_path}")


def check_chroma_db():
    """检查ChromaDB向量库"""
    print("\n" + "="*70)
    print("🔢 向量数据库(ChromaDB)检查")
    print("="*70)
    
    chroma_dir = RAG_DIR / "chroma_db"
    
    if not chroma_dir.exists():
        print(f"\n⚠️  ChromaDB目录不存在: {chroma_dir}")
        print("   RAG向量检索功能不可用")
        return
    
    # 检查目录内容
    print(f"\n📂 ChromaDB目录: {chroma_dir}")
    
    # 尝试加载ChromaDB
    try:
        import chromadb
        
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = client.list_collections()
        
        print(f"\n📚 集合数量: {len(collections)}")
        
        for collection_name in collections:
            collection = client.get_collection(collection_name)
            count = collection.count()
            print(f"   • {collection_name}: {count:,} 条向量")
            
            # 显示样本
            if count > 0:
                sample = collection.get(limit=1)
                if sample and sample.get('documents'):
                    doc_preview = sample['documents'][0][:100] if sample['documents'][0] else 'N/A'
                    print(f"     样例: {doc_preview}...")
        
        print("\n✅ ChromaDB正常工作")
        
    except ImportError:
        print("\n⚠️  未安装ChromaDB，无法检查")
        print("   安装命令: pip install chromadb")
    except Exception as e:
        print(f"\n❌ ChromaDB检查失败: {e}")


def check_keyword_index():
    """检查关键词索引"""
    print("\n" + "="*70)
    print("🔤 关键词索引检查")
    print("="*70)
    
    index_dir = RAG_DIR / "keyword_index"
    
    if not index_dir.exists():
        print(f"\n⚠️  关键词索引目录不存在: {index_dir}")
        print("   关键词检索功能不可用")
        return
    
    print(f"\n📂 索引目录: {index_dir}")
    
    # 检查文件
    index_files = list(index_dir.glob("*"))
    print(f"   索引文件数: {len(index_files)}")
    
    for f in index_files[:5]:
        size = f.stat().st_size
        print(f"   • {f.name} ({size:,} bytes)")
    
    if len(index_files) > 5:
        print(f"   ... 还有 {len(index_files)-5} 个文件")


def search_knowledge_base():
    """知识库检索测试"""
    print("\n" + "="*70)
    print("🔍 知识库检索测试")
    print("="*70)
    
    # 加载数据
    full_dataset_path = RAG_DIR / "qa_dataset_full.json"
    if not full_dataset_path.exists():
        print("   主数据集不存在，跳过检索测试")
        return
    
    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 测试查询
    test_queries = [
        "灾变变星",
        "赫罗图",
        "光变曲线",
        "白矮星",
    ]
    
    print("\n测试关键词匹配:")
    for query in test_queries:
        # 简单关键词匹配
        matched = []
        for qa in full_data:
            question = qa.get('question', '')
            if query.lower() in question.lower():
                matched.append(qa)
        
        print(f"   '{query}': 找到 {len(matched)} 条相关问题")
        
        # 显示第一条匹配
        if matched:
            sample_q = matched[0].get('question', 'N/A')[:60]
            print(f"      样例: {sample_q}...")


def generate_report():
    """生成完整报告"""
    print("\n" + "="*70)
    print("📋 RAG知识库完整报告")
    print("="*70)
    
    report_lines = []
    
    # 1. 基础统计
    report_lines.append("\n## 基础统计")
    
    full_dataset_path = RAG_DIR / "qa_dataset_full.json"
    if full_dataset_path.exists():
        with open(full_dataset_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        report_lines.append(f"- 总QA对: {len(full_data):,}")
        
        # 统计平均长度
        avg_q_len = sum(len(qa.get('question', '')) for qa in full_data) / len(full_data)
        avg_a_len = sum(len(qa.get('answer', '')) for qa in full_data) / len(full_data)
        
        report_lines.append(f"- 平均问题长度: {avg_q_len:.0f} 字符")
        report_lines.append(f"- 平均回答长度: {avg_a_len:.0f} 字符")
    
    # 2. 主题分布
    report_lines.append("\n## 主题分布")
    categories = {
        "赫罗图": "qa_hr_diagram.json",
        "灾变变星": "qa_cv.json",
        "双星": "qa_binary.json",
        "光变曲线": "qa_light_curve.json",
        "周期": "qa_period.json",
        "SED": "qa_sed.json",
        "光谱": "qa_spectrum.json",
        "X射线": "qa_xray.json",
    }
    
    for name, filename in categories.items():
        filepath = RAG_DIR / filename
        if filepath.exists():
            data = load_json_file(filepath)
            if data:
                report_lines.append(f"- {name}: {len(data):,} 条")
    
    # 3. 存储情况
    report_lines.append("\n## 存储情况")
    
    total_size = 0
    for f in RAG_DIR.glob("*.json"):
        size = f.stat().st_size
        total_size += size
    
    report_lines.append(f"- JSON数据总大小: {total_size / (1024*1024):.1f} MB")
    
    # ChromaDB
    chroma_dir = RAG_DIR / "chroma_db"
    if chroma_dir.exists():
        chroma_size = sum(f.stat().st_size for f in chroma_dir.rglob("*") if f.is_file())
        report_lines.append(f"- ChromaDB大小: {chroma_size / (1024*1024):.1f} MB")
    
    # 打印报告
    for line in report_lines:
        print(line)
    
    # 保存报告
    report_path = RAG_DIR / "knowledge_base_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RAG知识库报告\n\n")
        f.write("\n".join(report_lines))
    
    print(f"\n💾 报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 RAG知识库检查工具")
    print("="*70)
    
    if not RAG_DIR.exists():
        print(f"\n❌ RAG目录不存在: {RAG_DIR}")
        return
    
    # 执行各项检查
    analyze_knowledge_base()
    check_sample_quality(num_samples=3)
    analyze_training_data()
    check_chroma_db()
    check_keyword_index()
    search_knowledge_base()
    generate_report()
    
    print("\n" + "="*70)
    print("✅ 检查完成!")
    print("="*70)


if __name__ == "__main__":
    main()
