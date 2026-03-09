#!/usr/bin/env python3
"""
跨语言支持系统 (Multilingual Support System)
============================================
支持中英文天文文献的联合理解与检索

核心功能:
- 多语言文本对齐
- 跨语言向量空间
- 双语知识库
- 自动语言检测

作者: AstroSage AI
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass


class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        # 中文特征字符
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        # 英文特征
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect(self, text: str) -> str:
        """
        检测文本语言
        
        Returns:
            'zh', 'en', 或 'mixed'
        """
        chinese_chars = len(self.chinese_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.3:
            return 'zh' if english_ratio < 0.1 else 'mixed'
        elif english_ratio > 0.5:
            return 'en'
        else:
            return 'mixed'


class CrossLingualEncoder:
    """
    跨语言编码器
    
    将不同语言的文本映射到统一向量空间
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.dimension = 384
        self._model = None
        self.detector = LanguageDetector()
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                print("⚠ sentence-transformers未安装，使用简化版编码")
                self._model = "simple"
    
    def encode(self, text: str) -> np.ndarray:
        """
        编码文本 (支持中英文)
        
        Args:
            text: 输入文本 (中文、英文或混合)
            
        Returns:
            向量表征
        """
        self._load_model()
        
        if self._model == "simple":
            return self._simple_encode(text)
        
        # 使用多语言模型
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _simple_encode(self, text: str) -> np.ndarray:
        """简化版跨语言编码"""
        lang = self.detector.detect(text)
        
        # 针对不同语言使用不同的预处理方式
        if lang == 'zh':
            # 中文：保留字符，去除标点
            processed = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
        else:
            # 英文/混合：标准化
            processed = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # 使用哈希创建向量 (与之前类似，但考虑多语言)
        import hashlib
        words = processed.split()
        vector = np.zeros(self.dimension)
        
        for word in words:
            # 中文字符和英文单词使用不同的哈希策略
            if any('\u4e00' <= c <= '\u9fff' for c in word):
                # 中文：按字符哈希
                for char in word:
                    char_hash = hashlib.md5(char.encode('utf-8')).digest()
                    char_vec = np.frombuffer(char_hash, dtype=np.uint8).astype(float)
                    char_vec = np.tile(char_vec, self.dimension // len(char_vec) + 1)[:self.dimension]
                    vector += char_vec
            else:
                # 英文：按词哈希
                word_hash = hashlib.md5(word.encode()).digest()
                word_vec = np.frombuffer(word_hash, dtype=np.uint8).astype(float)
                word_vec = np.tile(word_vec, self.dimension // len(word_vec) + 1)[:self.dimension]
                vector += word_vec
        
        # 归一化
        if len(words) > 0:
            vector = vector / len(words)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        
        return vector
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的跨语言相似度"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


@dataclass
class BilingualEntry:
    """双语知识条目"""
    id: str
    content_zh: str
    content_en: str
    embedding_zh: Optional[np.ndarray] = None
    embedding_en: Optional[np.ndarray] = None
    embedding_aligned: Optional[np.ndarray] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BilingualKnowledgeBase:
    """
    双语知识库
    
    支持中英文知识的对齐和联合检索
    """
    
    def __init__(self, kb_dir: str = "./astro_knowledge/bilingual"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = CrossLingualEncoder()
        self.entries: List[BilingualEntry] = []
        
        # 加载内置双语知识
        self._init_default_knowledge()
    
    def _init_default_knowledge(self):
        """初始化默认双语知识"""
        default_knowledge = [
            {
                'zh': '激变变星(Cataclysmic Variable)是白矮星从伴星吸积物质的紧密双星系统',
                'en': 'Cataclysmic Variables (CVs) are close binary systems where a white dwarf accretes matter from a companion star',
                'topic': 'CV_basic'
            },
            {
                'zh': 'AM CVn型星是超紧密双星系统，由白矮星和氦星组成，轨道周期5-65分钟',
                'en': 'AM CVn systems are ultra-compact binaries consisting of a white dwarf and a helium star, with orbital periods of 5-65 minutes',
                'topic': 'AM_CVN'
            },
            {
                'zh': '周期空缺是指激变变星在2-3小时周期范围内的数量减少现象',
                'en': 'The period gap refers to the deficit of cataclysmic variables in the 2-3 hour orbital period range',
                'topic': 'period_gap'
            },
            {
                'zh': '高偏振星(Polar)是强磁场激变变星，磁场强度10-100兆高斯',
                'en': 'Polars are magnetic cataclysmic variables with field strengths of 10-100 MegaGauss',
                'topic': 'polar'
            },
            {
                'zh': '超涨(superhump)是SU UMa型矮新星中观测到的周期略长于轨道周期的光变',
                'en': 'Superhumps are periodic photometric variations observed in SU UMa dwarf novae with periods slightly longer than the orbital period',
                'topic': 'superhump'
            },
            {
                'zh': '白矮星是恒星演化的最终产物，由电子简并压力支撑',
                'en': 'White dwarfs are the end products of stellar evolution, supported by electron degeneracy pressure',
                'topic': 'white_dwarf'
            },
            {
                'zh': '吸积盘不稳定模型(DIM)解释了矮新星的爆发行为',
                'en': 'The Disk Instability Model (DIM) explains the outburst behavior of dwarf novae',
                'topic': 'DIM'
            },
            {
                'zh': '引力波辐射是短周期双星系统角动量损失的主要机制',
                'en': 'Gravitational wave radiation is the primary angular momentum loss mechanism for short-period binary systems',
                'topic': 'gravitational_wave'
            }
        ]
        
        for item in default_knowledge:
            entry = BilingualEntry(
                id=item['topic'],
                content_zh=item['zh'],
                content_en=item['en'],
                metadata={'topic': item['topic']}
            )
            self.add_entry(entry)
        
        print(f"✓ 已加载 {len(self.entries)} 条双语知识")
    
    def add_entry(self, entry: BilingualEntry, compute_embedding: bool = True):
        """添加双语条目"""
        if compute_embedding:
            # 计算各语言表征
            entry.embedding_zh = self.encoder.encode(entry.content_zh)
            entry.embedding_en = self.encoder.encode(entry.content_en)
            
            # 计算对齐表征 (平均)
            entry.embedding_aligned = (entry.embedding_zh + entry.embedding_en) / 2
            entry.embedding_aligned = entry.embedding_aligned / np.linalg.norm(entry.embedding_aligned)
        
        self.entries.append(entry)
    
    def search(self, query: str, top_k: int = 3, 
               search_mode: str = 'auto') -> List[Tuple[BilingualEntry, float]]:
        """
        双语检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            search_mode: 'zh'(中文), 'en'(英文), 'auto'(自动检测)
            
        Returns:
            (条目, 相似度)列表
        """
        # 检测查询语言
        if search_mode == 'auto':
            lang = self.encoder.detector.detect(query)
        else:
            lang = search_mode
        
        # 编码查询
        query_emb = self.encoder.encode(query)
        
        # 检索
        results = []
        for entry in self.entries:
            # 使用对齐表征进行检索
            if entry.embedding_aligned is not None:
                sim = float(np.dot(query_emb, entry.embedding_aligned))
                results.append((entry, sim))
        
        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def translate_concept(self, text: str, target_lang: str = 'auto') -> List[Tuple[str, float]]:
        """
        概念翻译/对齐
        
        找到输入文本在不同语言中的对应表达
        
        Args:
            text: 输入文本
            target_lang: 目标语言 ('zh', 'en', 'auto')
            
        Returns:
            (翻译文本, 置信度)列表
        """
        # 检索最相似的条目
        results = self.search(text, top_k=3)
        
        translations = []
        for entry, sim in results:
            if target_lang == 'zh' or target_lang == 'auto':
                translations.append((entry.content_zh, sim))
            if target_lang == 'en' or target_lang == 'auto':
                translations.append((entry.content_en, sim))
        
        return translations
    
    def get_bilingual_context(self, query: str, top_k: int = 3) -> str:
        """
        获取双语上下文 (用于增强提示词)
        
        Returns:
            格式化的双语知识文本
        """
        results = self.search(query, top_k=top_k)
        
        contexts = []
        for entry, sim in results:
            context = f"""[相关度: {sim:.3f}] {entry.metadata.get('topic', '')}
中文: {entry.content_zh}
English: {entry.content_en}
---"""
            contexts.append(context)
        
        return "\n".join(contexts)


class MultilingualRAG:
    """
    多语言RAG系统
    
    支持中英文混合的知识检索和问答
    """
    
    def __init__(self):
        self.bilingual_kb = BilingualKnowledgeBase()
        self.detector = LanguageDetector()
        
        # 可以扩展其他知识库
        self.additional_kbs = {}
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        多语言查询
        
        Args:
            question: 问题 (中文/英文)
            top_k: 返回知识数量
            
        Returns:
            包含检索结果和语言信息的字典
        """
        # 检测问题语言
        lang = self.detector.detect(question)
        
        # 检索双语知识库
        results = self.bilingual_kb.search(question, top_k=top_k)
        
        # 根据问题语言选择回答语言
        if lang == 'zh':
            knowledge_text = "\n".join([f"• {r[0].content_zh}" for r in results])
        else:
            knowledge_text = "\n".join([f"• {r[0].content_en}" for r in results])
        
        return {
            'query_language': lang,
            'retrieved_entries': results,
            'knowledge_text': knowledge_text,
            'bilingual_context': self.bilingual_kb.get_bilingual_context(question, top_k)
        }
    
    def generate_multilingual_prompt(self, question: str, 
                                     target_lang: str = 'auto') -> str:
        """
        生成多语言增强提示词
        
        Args:
            question: 问题
            target_lang: 回答目标语言
            
        Returns:
            增强提示词
        """
        if target_lang == 'auto':
            target_lang = self.detector.detect(question)
        
        # 检索知识
        rag_result = self.query(question, top_k=3)
        
        if target_lang == 'zh':
            prompt = f"""你是一位资深天体物理学家。请基于以下知识回答问题。

问题: {question}

相关知识 (中英双语):
{rag_result['bilingual_context']}

请用中文回答，可以引用上述知识。"""
        else:
            prompt = f"""You are a senior astrophysicist. Please answer the question based on the following knowledge.

Question: {question}

Relevant Knowledge (Bilingual):
{rag_result['bilingual_context']}

Please answer in English, you may reference the knowledge above."""
        
        return prompt


# ==================== 便捷函数 ====================

def detect_language(text: str) -> str:
    """检测文本语言"""
    detector = LanguageDetector()
    return detector.detect(text)


def translate_astro_term(term: str, target: str = 'auto') -> List[Tuple[str, float]]:
    """翻译天文术语"""
    kb = BilingualKnowledgeBase()
    return kb.translate_concept(term, target)


def demo_multilingual():
    """演示多语言功能"""
    print("="*70)
    print("跨语言支持系统演示")
    print("="*70)
    
    # 初始化
    ml_rag = MultilingualRAG()
    
    # 测试查询 (中文)
    print("\n📝 中文查询: '什么是激变变星？'")
    result = ml_rag.query("什么是激变变星？", top_k=2)
    print(f"   检测到语言: {result['query_language']}")
    print(f"   检索结果:")
    for entry, sim in result['retrieved_entries']:
        print(f"   • [{sim:.3f}] {entry.content_zh[:50]}...")
    
    # 测试查询 (英文)
    print("\n📝 English query: 'What is a cataclysmic variable?'")
    result = ml_rag.query("What is a cataclysmic variable?", top_k=2)
    print(f"   Detected language: {result['query_language']}")
    print(f"   Retrieved:")
    for entry, sim in result['retrieved_entries']:
        print(f"   • [{sim:.3f}] {entry.content_en[:50]}...")
    
    # 测试跨语言检索 (中文查询，获取英文知识)
    print("\n🌐 跨语言检索: 中文查询 '超涨'，获取英文知识")
    result = ml_rag.query("超涨", top_k=1)
    entry, sim = result['retrieved_entries'][0]
    print(f"   中文: {entry.content_zh}")
    print(f"   English: {entry.content_en}")
    
    # 生成多语言提示词
    print("\n📋 生成增强提示词:")
    prompt = ml_rag.generate_multilingual_prompt("什么是AM CVn型星？", target_lang='zh')
    print(prompt[:500] + "...")
    
    print("\n完成!")


# ==================== 测试 ====================

if __name__ == "__main__":
    demo_multilingual()
