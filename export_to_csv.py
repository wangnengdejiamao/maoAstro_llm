#!/usr/bin/env python3
"""
导出问答对数据集到CSV格式
"""

import json
import csv
from pathlib import Path

OUTPUT_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/qa_output")

def load_json(filename):
    with open(OUTPUT_DIR / filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def export_to_csv():
    """导出为CSV格式"""
    all_data = load_json("all_qa_summary.json")
    
    # 1. 导出所有问答对到一个CSV
    csv_path = OUTPUT_DIR / "all_qa_dataset.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_name', 'qa_id', 'question', 'answer', 'category', 'source_doc', 'key_points', 'q_length', 'a_length'])
        
        for doc_name, qa_list in all_data.items():
            for qa in qa_list:
                writer.writerow([
                    doc_name,
                    qa.get('id', ''),
                    qa.get('question', ''),
                    qa.get('answer', ''),
                    qa.get('category', ''),
                    qa.get('source_doc', ''),
                    '|'.join(qa.get('key_points', [])),
                    len(qa.get('question', '')),
                    len(qa.get('answer', ''))
                ])
    
    print(f"已导出: {csv_path}")
    
    # 2. 导出简化的训练格式CSV
    simple_csv_path = OUTPUT_DIR / "training_dataset.csv"
    train_data = load_json("training_dataset.json")
    
    with open(simple_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['instruction', 'output', 'source_doc', 'category'])
        
        for item in train_data:
            writer.writerow([
                item.get('instruction', ''),
                item.get('output', ''),
                item.get('source_doc', ''),
                item.get('category', '')
            ])
    
    print(f"已导出: {simple_csv_path}")
    
    # 3. 导出分类统计CSV
    from collections import Counter
    
    category_stats = []
    for doc_name, qa_list in all_data.items():
        doc_categories = Counter(qa["category"] for qa in qa_list)
        for cat, count in doc_categories.items():
            category_stats.append({
                'doc_name': doc_name,
                'category': cat,
                'count': count
            })
    
    stats_csv_path = OUTPUT_DIR / "category_statistics.csv"
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['doc_name', 'category', 'count'])
        writer.writeheader()
        writer.writerows(category_stats)
    
    print(f"已导出: {stats_csv_path}")
    
    # 4. 导出关键词统计CSV
    keyword_stats = []
    for doc_name, qa_list in all_data.items():
        for qa in qa_list:
            for kw in qa.get('key_points', []):
                keyword_stats.append({
                    'doc_name': doc_name,
                    'qa_id': qa.get('id', ''),
                    'keyword': kw,
                    'category': qa.get('category', '')
                })
    
    kw_csv_path = OUTPUT_DIR / "keywords_dataset.csv"
    with open(kw_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['doc_name', 'qa_id', 'keyword', 'category'])
        writer.writeheader()
        writer.writerows(keyword_stats)
    
    print(f"已导出: {kw_csv_path}")
    
    print("\nCSV导出完成!")
    print(f"  - 完整数据集: {csv_path}")
    print(f"  - 训练格式: {simple_csv_path}")
    print(f"  - 分类统计: {stats_csv_path}")
    print(f"  - 关键词数据: {kw_csv_path}")

if __name__ == "__main__":
    export_to_csv()
