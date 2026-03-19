#!/usr/bin/env python3
"""
RAG 增强训练模块
在训练过程中使用 RAG 检索相关知识，增强模型学习效果并减少幻觉
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.rag_pipeline import RAGPipeline


@dataclass
class AugmentedExample:
    """增强后的训练样本"""
    original_question: str
    original_answer: str
    retrieved_context: List[Dict]
    augmented_prompt: str
    has_hallucination: bool
    confidence: float


class RAGAugmentedTrainer:
    """
    RAG 增强训练器
    使用检索到的知识来增强训练数据
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        augment_probability: float = 0.5,
    ):
        """
        初始化
        
        Args:
            rag_pipeline: RAG Pipeline 实例
            augment_probability: 数据增强概率
        """
        self.rag_pipeline = rag_pipeline
        self.augment_probability = augment_probability
        
        print(f"✅ RAG 增强训练器初始化 (增强概率: {augment_probability})")
    
    def augment_dataset(
        self,
        dataset_path: str,
        output_path: str,
    ) -> str:
        """
        增强整个数据集
        
        Args:
            dataset_path: 原始数据集路径
            output_path: 输出路径
        
        Returns:
            输出文件路径
        """
        print(f"📂 加载数据集: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✅ 加载 {len(dataset)} 条数据")
        
        augmented_data = []
        
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                print(f"🔄 处理中... {i}/{len(dataset)}")
            
            # 增强单条数据
            augmented = self._augment_example(example)
            augmented_data.append(augmented)
        
        # 保存
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 增强完成！保存到: {output_path}")
        
        # 统计
        hallucination_count = sum(1 for d in augmented_data if d.get('has_hallucination', False))
        print(f"   检测到可能幻觉: {hallucination_count}/{len(augmented_data)}")
        
        return output_path
    
    def _augment_example(self, example: Dict) -> Dict:
        """
        增强单条训练数据
        """
        # 提取问题
        question = self._extract_question(example)
        
        if not question:
            return example
        
        # 使用 RAG 检索
        rag_response = self.rag_pipeline.query(question, use_llm=False)
        
        # 检测原始回答是否有幻觉
        hallucination_check = rag_response.hallucination_check
        
        # 构建增强的 prompt
        augmented_prompt = self._build_augmented_prompt(
            example,
            rag_response.retrieved_documents,
        )
        
        # 创建增强后的样本
        augmented = {
            "id": example.get("id", ""),
            "original": example,
            "augmented_prompt": augmented_prompt,
            "retrieved_documents": rag_response.retrieved_documents,
            "citations": rag_response.citations,
            "has_hallucination": hallucination_check.get("has_hallucination", False),
            "confidence": rag_response.confidence,
            "should_include": rag_response.confidence > 0.5,  # 置信度阈值
        }
        
        return augmented
    
    def _extract_question(self, example: Dict) -> str:
        """从训练样本中提取问题"""
        # Qwen 格式
        if "conversations" in example:
            for conv in example["conversations"]:
                if conv.get("from") == "user":
                    return conv.get("value", "")
        
        # Alpaca 格式
        if "instruction" in example:
            return example["instruction"]
        
        return ""
    
    def _build_augmented_prompt(
        self,
        example: Dict,
        retrieved_docs: List[Dict],
    ) -> str:
        """
        构建增强的 prompt
        在 prompt 中加入检索到的上下文
        """
        question = self._extract_question(example)
        
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "未知")
            context_parts.append(f"[{i}] {doc['document'][:300]}... (来源: {source})")
        
        context = "\n".join(context_parts)
        
        # 增强 prompt
        augmented = f"""基于以下参考信息回答问题：

参考信息：
{context}

问题：{question}

请基于上述参考信息提供准确、详细的回答。"""
        
        return augmented
    
    def filter_high_quality(
        self,
        augmented_dataset: List[Dict],
        min_confidence: float = 0.6,
        exclude_hallucination: bool = True,
    ) -> List[Dict]:
        """
        筛选高质量的训练数据
        
        Args:
            augmented_dataset: 增强后的数据集
            min_confidence: 最小置信度
            exclude_hallucination: 是否排除可能有幻觉的数据
        
        Returns:
            筛选后的数据
        """
        filtered = []
        
        for item in augmented_dataset:
            # 检查置信度
            if item.get("confidence", 0) < min_confidence:
                continue
            
            # 检查幻觉
            if exclude_hallucination and item.get("has_hallucination", False):
                continue
            
            # 保留原始格式，但标记质量
            original = item.get("original", {})
            original["quality_score"] = item.get("confidence", 0)
            original["has_verified_context"] = True
            
            filtered.append(original)
        
        print(f"🎯 筛选后: {len(filtered)}/{len(augmented_dataset)} 条高质量数据")
        
        return filtered
    
    def prepare_training_data(
        self,
        input_path: str,
        output_train: str,
        output_val: str,
        train_ratio: float = 0.9,
    ):
        """
        准备最终训练数据
        增强 -> 筛选 -> 划分
        """
        # 1. 增强数据
        augmented_path = str(Path(output_train).with_suffix('.augmented.json'))
        self.augment_dataset(input_path, augmented_path)
        
        # 2. 加载增强数据
        with open(augmented_path, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)
        
        # 3. 筛选高质量数据
        filtered = self.filter_high_quality(augmented_data)
        
        # 4. 划分训练集和验证集
        import random
        random.seed(42)
        random.shuffle(filtered)
        
        split_idx = int(len(filtered) * train_ratio)
        train_data = filtered[:split_idx]
        val_data = filtered[split_idx:]
        
        # 5. 保存
        with open(output_train, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(output_val, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 训练数据准备完成！")
        print(f"   训练集: {len(train_data)} 条 -> {output_train}")
        print(f"   验证集: {len(val_data)} 条 -> {output_val}")


if __name__ == "__main__":
    print("🚀 RAG 增强训练示例")
    print("请使用完整的 RAG Pipeline 进行数据增强")
