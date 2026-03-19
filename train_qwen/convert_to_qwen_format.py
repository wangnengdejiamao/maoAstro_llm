#!/usr/bin/env python3
"""
将生成的问答数据转换为 Qwen 训练格式
支持两种格式:
1. Qwen 官方格式 (conversation)
2. Alpaca 格式 (instruction, input, output)
"""

import json
import random
from pathlib import Path
from typing import List, Dict

def load_qa_data(qa_file: str) -> List[Dict]:
    """加载问答数据"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_qwen_format(qa_pairs: List[Dict], system_prompt: str = None) -> List[Dict]:
    """
    转换为 Qwen 官方对话格式
    {
        "id": "identity_0",
        "conversations": [
            {"from": "system", "value": "你是天文助手..."},
            {"from": "user", "value": "问题"},
            {"from": "assistant", "value": "答案"}
        ]
    }
    """
    if system_prompt is None:
        system_prompt = """你是 AstroSage，一位专业的天文学专家助手。你精通：
- 恒星演化与赫罗图分析
- 能谱分布(SED)拟合与解释
- 光变曲线分析
- 双星系统与轨道周期
- X射线天文学
- 光谱分析
- 灾变变星(CV)物理

请基于专业知识，准确、详细地回答天文学问题。如果问题涉及论文内容，请提供具体的来源信息。"""
    
    converted = []
    for i, qa in enumerate(qa_pairs):
        # 构建答案（包含来源信息）
        answer = qa['answer']
        source_info = f"\n\n【参考来源: {qa['source_file']}, 第{qa['page_number']}页】"
        
        # API 生成的高质量答案不需要额外标注
        if qa.get('generation_method') == 'rule_based':
            answer += source_info
        
        conversation = {
            "id": f"astro_qa_{i}",
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "user", "value": qa['question']},
                {"from": "assistant", "value": answer}
            ],
            "metadata": {
                "question_type": qa.get('question_type', 'general'),
                "source_file": qa['source_file'],
                "page_number": qa['page_number'],
                "confidence": qa.get('confidence', 0.8),
                "generation_method": qa.get('generation_method', 'unknown')
            }
        }
        converted.append(conversation)
    
    return converted

def convert_to_alpaca_format(qa_pairs: List[Dict]) -> List[Dict]:
    """
    转换为 Alpaca 格式
    {
        "instruction": "问题",
        "input": "",
        "output": "答案",
        "history": []
    }
    """
    converted = []
    for qa in qa_pairs:
        answer = qa['answer']
        source_info = f"\n\n【参考来源: {qa['source_file']}, 第{qa['page_number']}页】"
        
        if qa.get('generation_method') == 'rule_based':
            answer += source_info
        
        item = {
            "instruction": qa['question'],
            "input": "",
            "output": answer,
            "history": [],
            "metadata": {
                "question_type": qa.get('question_type', 'general'),
                "source_file": qa['source_file'],
                "generation_method": qa.get('generation_method', 'unknown')
            }
        }
        converted.append(item)
    
    return converted

def convert_to_sharegpt_format(qa_pairs: List[Dict]) -> List[Dict]:
    """
    转换为 ShareGPT 格式
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "问题"},
            {"from": "gpt", "value": "答案"},
            {"from": "human", "value": "追问"},
            {"from": "gpt", "value": "回答"}
        ]
    }
    """
    converted = []
    for i, qa in enumerate(qa_pairs):
        answer = qa['answer']
        source_info = f"\n\n【参考来源: {qa['source_file']}, 第{qa['page_number']}页】"
        
        if qa.get('generation_method') == 'rule_based':
            answer += source_info
        
        conversation = {
            "id": f"astro_{i}",
            "conversations": [
                {"from": "human", "value": qa['question']},
                {"from": "gpt", "value": answer}
            ]
        }
        converted.append(conversation)
    
    return converted

def filter_high_quality(qa_pairs: List[Dict], min_confidence: float = 0.8, 
                        prefer_api: bool = True) -> List[Dict]:
    """筛选高质量数据"""
    filtered = []
    
    for qa in qa_pairs:
        # 置信度筛选
        if qa.get('confidence', 0) < min_confidence:
            continue
        
        # 优先 API 生成的高质量数据
        if prefer_api and qa.get('generation_method') == 'api_based':
            filtered.append(qa)
        elif not prefer_api:
            filtered.append(qa)
    
    return filtered

def split_dataset(data: List[Dict], train_ratio: float = 0.9, seed: int = 42):
    """划分训练集和验证集"""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    split_idx = int(len(data) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    return train_data, val_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="转换为 Qwen 训练格式")
    parser.add_argument("--input", default="output/qa_hybrid/qa_dataset_full.json",
                       help="输入问答数据文件")
    parser.add_argument("--output-dir", default="train_qwen/data",
                       help="输出目录")
    parser.add_argument("--format", choices=["qwen", "alpaca", "sharegpt", "all"],
                       default="all", help="输出格式")
    parser.add_argument("--filter-quality", action="store_true",
                       help="只使用高质量数据(置信度>=0.8)")
    parser.add_argument("--prefer-api", action="store_true",
                       help="优先使用API生成的数据")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                       help="训练集比例")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"📂 加载数据: {args.input}")
    qa_pairs = load_qa_data(args.input)
    print(f"✅ 加载 {len(qa_pairs)} 条问答对")
    
    # 筛选高质量数据
    if args.filter_quality:
        qa_pairs = filter_high_quality(qa_pairs, prefer_api=args.prefer_api)
        print(f"🎯 筛选后: {len(qa_pairs)} 条高质量问答对")
        
        # 统计
        api_count = sum(1 for qa in qa_pairs if qa.get('generation_method') == 'api_based')
        print(f"   - API生成: {api_count}")
        print(f"   - 规则生成: {len(qa_pairs) - api_count}")
    
    # 划分数据集
    train_data, val_data = split_dataset(qa_pairs, args.train_ratio)
    print(f"\n📊 数据集划分:")
    print(f"   - 训练集: {len(train_data)}")
    print(f"   - 验证集: {len(val_data)}")
    
    # 转换并保存
    formats_to_convert = []
    if args.format == "all":
        formats_to_convert = ["qwen", "alpaca", "sharegpt"]
    else:
        formats_to_convert = [args.format]
    
    for fmt in formats_to_convert:
        print(f"\n🔄 转换为 {fmt} 格式...")
        
        if fmt == "qwen":
            train_converted = convert_to_qwen_format(train_data)
            val_converted = convert_to_qwen_format(val_data)
        elif fmt == "alpaca":
            train_converted = convert_to_alpaca_format(train_data)
            val_converted = convert_to_alpaca_format(val_data)
        elif fmt == "sharegpt":
            train_converted = convert_to_sharegpt_format(train_data)
            val_converted = convert_to_sharegpt_format(val_data)
        
        # 保存
        train_file = output_dir / f"{fmt}_train.json"
        val_file = output_dir / f"{fmt}_val.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_converted, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_converted, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 训练集: {train_file}")
        print(f"   ✅ 验证集: {val_file}")
    
    # 生成数据集信息
    info = {
        "total_samples": len(qa_pairs),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "api_based": sum(1 for qa in qa_pairs if qa.get('generation_method') == 'api_based'),
        "rule_based": sum(1 for qa in qa_pairs if qa.get('generation_method') == 'rule_based'),
        "question_types": {}
    }
    
    # 统计问题类型
    for qa in qa_pairs:
        qtype = qa.get('question_type', 'unknown')
        info["question_types"][qtype] = info["question_types"].get(qtype, 0) + 1
    
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 数据集信息: {info_file}")
    print(f"\n✨ 完成！数据已准备好用于 Qwen 训练")

if __name__ == "__main__":
    main()
