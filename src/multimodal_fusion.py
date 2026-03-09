#!/usr/bin/env python3
"""
多模态融合系统 (Multimodal Fusion System)
=========================================
整合光谱图像、光变曲线、文献文本的统一表征

核心思想:
- 将不同模态数据投影到统一的语义向量空间
- 使用对比学习对齐不同模态的表征
- 支持跨模态检索和推理

作者: AstroSage AI
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import base64
from PIL import Image
import io


@dataclass
class ModalityEmbedding:
    """模态嵌入数据结构"""
    modality: str  # 'spectrum_image', 'light_curve', 'text', 'sed'
    embedding: np.ndarray
    metadata: Dict[str, Any]
    raw_data_ref: str  # 原始数据引用


class SpectrumImageEncoder:
    """
    光谱图像编码器
    
    将光谱图像(FITS/PNG)编码为向量
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.feature_dim = 512
        
    def encode(self, image_path: str) -> np.ndarray:
        """
        编码光谱图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            特征向量 (512维)
        """
        try:
            # 方法1: 使用预训练的视觉模型 (推荐)
            # from transformers import CLIPProcessor, CLIPModel
            # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 方法2: 简化版 - 使用图像统计特征 + CNN
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 提取多尺度特征
            features = self._extract_features(img_array)
            
            return features
            
        except Exception as e:
            print(f"光谱图像编码失败: {e}")
            return np.zeros(self.feature_dim)
    
    def _extract_features(self, img_array: np.ndarray) -> np.ndarray:
        """提取图像特征 (简化版)"""
        # 实际应用中应使用预训练CNN
        # 这里使用统计特征作为示例
        
        # 颜色直方图
        hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0,256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0,256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0,256))[0]
        
        # 统计特征
        mean_feat = np.mean(img_array, axis=(0,1))
        std_feat = np.std(img_array, axis=(0,1))
        
        # 组合特征
        features = np.concatenate([
            hist_r / (np.sum(hist_r) + 1e-8),
            hist_g / (np.sum(hist_g) + 1e-8),
            hist_b / (np.sum(hist_b) + 1e-8),
            mean_feat / 255.0,
            std_feat / 255.0
        ])
        
        # 填充到目标维度
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]
        
        # L2归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features


class LightCurveEncoder:
    """
    光变曲线编码器
    
    将时间序列光变数据编码为向量
    """
    
    def __init__(self):
        self.feature_dim = 256
        
    def encode(self, time: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """
        编码光变曲线
        
        Args:
            time: 时间数组 (JD)
            magnitude: 星等数组
            
        Returns:
            特征向量 (256维)
        """
        features = []
        
        # 1. 统计特征
        valid_mag = magnitude[np.isfinite(magnitude)]
        if len(valid_mag) == 0:
            return np.zeros(self.feature_dim)
        
        features.extend([
            np.mean(valid_mag),
            np.std(valid_mag),
            np.min(valid_mag),
            np.max(valid_mag),
            np.median(valid_mag),
            np.percentile(valid_mag, 25),
            np.percentile(valid_mag, 75),
        ])
        
        # 2. 形状特征
        # 偏度
        if np.std(valid_mag) > 0:
            skewness = np.mean(((valid_mag - np.mean(valid_mag)) / np.std(valid_mag)) ** 3)
            features.append(skewness)
        else:
            features.append(0)
        
        # 3. 周期特征 (如果已知)
        # 使用Lomb-Scargle找到最强周期
        try:
            from scipy.signal import lombscargle
            
            # 标准化时间
            t_norm = time - np.min(time)
            freqs = np.linspace(0.1, 100, 1000)  # 频率范围
            
            # 处理NaN
            valid_mask = np.isfinite(magnitude)
            if np.sum(valid_mask) > 10:
                power = lombscargle(t_norm[valid_mask], magnitude[valid_mask], freqs)
                max_power_idx = np.argmax(power)
                best_freq = freqs[max_power_idx]
                
                features.extend([
                    best_freq,  # 最佳频率
                    power[max_power_idx],  # 最大功率
                    np.std(power),  # 功率谱标准差
                ])
            else:
                features.extend([0, 0, 0])
        except:
            features.extend([0, 0, 0])
        
        # 4. 变异性特征
        if len(valid_mag) > 1:
            # 一阶差分
            diff = np.diff(valid_mag)
            features.extend([
                np.std(diff),
                np.mean(np.abs(diff)),
            ])
        else:
            features.extend([0, 0])
        
        # 填充到目标维度
        features = np.array(features)
        if len(features) < self.feature_dim:
            # 使用PCA-like投影填充
            additional = np.random.randn(self.feature_dim - len(features)) * 0.01
            features = np.concatenate([features, additional])
        else:
            features = features[:self.feature_dim]
        
        # L2归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features


class TextEncoder:
    """
    文本编码器
    
    编码天文文献和描述文本
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.feature_dim = 384
        self._model = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self.feature_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                print("⚠ sentence-transformers未安装，使用简化版编码")
                self._model = "simple"
    
    def encode(self, text: str) -> np.ndarray:
        """
        编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            特征向量
        """
        self._load_model()
        
        if self._model == "simple":
            return self._simple_encode(text)
        
        # 使用预训练模型
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _simple_encode(self, text: str) -> np.ndarray:
        """简化版文本编码 (无需外部模型)"""
        import re
        import hashlib
        
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 使用哈希创建向量
        vector = np.zeros(self.feature_dim)
        for word in words:
            word_hash = hashlib.md5(word.encode()).digest()
            word_vec = np.frombuffer(word_hash, dtype=np.uint8).astype(float)
            # 扩展到目标维度
            word_vec = np.tile(word_vec, self.feature_dim // len(word_vec) + 1)[:self.feature_dim]
            vector += word_vec
        
        # 归一化
        if len(words) > 0:
            vector = vector / len(words)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        
        return vector


class MultimodalFusion:
    """
    多模态融合核心类
    
    整合不同模态的表征到统一空间
    """
    
    def __init__(self, fusion_dim: int = 512):
        self.fusion_dim = fusion_dim
        
        # 各模态编码器
        self.spectrum_encoder = SpectrumImageEncoder()
        self.lightcurve_encoder = LightCurveEncoder()
        self.text_encoder = TextEncoder()
        
        # 投影矩阵 (将各模态投影到统一空间)
        self.projection_matrices = self._init_projections()
        
    def _init_projections(self) -> Dict[str, np.ndarray]:
        """初始化投影矩阵"""
        projections = {}
        
        # 光谱图像: 512 -> fusion_dim
        projections['spectrum'] = np.random.randn(512, self.fusion_dim) * 0.01
        
        # 光变曲线: 256 -> fusion_dim
        projections['lightcurve'] = np.random.randn(256, self.fusion_dim) * 0.01
        
        # 文本: 384 -> fusion_dim
        projections['text'] = np.random.randn(384, self.fusion_dim) * 0.01
        
        return projections
    
    def encode_multimodal(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        编码多模态数据
        
        Args:
            data: 包含各模态数据的字典
                {
                    'spectrum_image': path,
                    'light_curve': {'time': [], 'mag': []},
                    'text': description,
                    ...
                }
                
        Returns:
            各模态的统一表征
        """
        embeddings = {}
        
        # 1. 编码光谱图像
        if 'spectrum_image' in data and data['spectrum_image']:
            spec_emb = self.spectrum_encoder.encode(data['spectrum_image'])
            embeddings['spectrum'] = self._project(spec_emb, 'spectrum')
        
        # 2. 编码光变曲线
        if 'light_curve' in data and data['light_curve']:
            lc_data = data['light_curve']
            lc_emb = self.lightcurve_encoder.encode(
                np.array(lc_data['time']),
                np.array(lc_data['mag'])
            )
            embeddings['lightcurve'] = self._project(lc_emb, 'lightcurve')
        
        # 3. 编码文本
        if 'text' in data and data['text']:
            text_emb = self.text_encoder.encode(data['text'])
            embeddings['text'] = self._project(text_emb, 'text')
        
        return embeddings
    
    def _project(self, embedding: np.ndarray, modality: str) -> np.ndarray:
        """投影到统一空间"""
        if modality in self.projection_matrices:
            proj = self.projection_matrices[modality]
            return np.dot(embedding, proj)
        return embedding
    
    def fuse(self, embeddings: Dict[str, np.ndarray], 
             method: str = 'attention') -> np.ndarray:
        """
        融合多模态表征
        
        Args:
            embeddings: 各模态的表征
            method: 融合方法 ('concat', 'average', 'attention', 'weighted')
            
        Returns:
            融合后的统一表征
        """
        if not embeddings:
            return np.zeros(self.fusion_dim)
        
        if method == 'concat':
            # 拼接所有表征
            all_embs = list(embeddings.values())
            fused = np.concatenate(all_embs)
            # 投影到目标维度
            if len(fused) > self.fusion_dim:
                fused = fused[:self.fusion_dim]
            else:
                fused = np.pad(fused, (0, self.fusion_dim - len(fused)))
            return fused
        
        elif method == 'average':
            # 简单平均
            all_embs = np.array(list(embeddings.values()))
            return np.mean(all_embs, axis=0)
        
        elif method == 'weighted':
            # 加权平均 (根据模态可靠性)
            weights = {
                'spectrum': 0.4,
                'lightcurve': 0.35,
                'text': 0.25
            }
            
            fused = np.zeros(self.fusion_dim)
            total_weight = 0
            for modality, emb in embeddings.items():
                weight = weights.get(modality, 0.33)
                fused += emb * weight
                total_weight += weight
            
            return fused / total_weight if total_weight > 0 else fused
        
        elif method == 'attention':
            # 注意力融合 (简化版)
            all_embs = list(embeddings.values())
            stacked = np.stack(all_embs)
            
            # 计算自注意力权重
            scores = np.dot(stacked, stacked.T)
            weights = np.softmax(scores, axis=1)
            
            # 加权求和
            fused = np.sum(stacked * weights[:, :1], axis=0)
            return fused
        
        else:
            raise ValueError(f"未知的融合方法: {method}")
    
    def cross_modal_search(self, 
                          query_emb: np.ndarray,
                          database: List[Dict[str, np.ndarray]],
                          top_k: int = 5) -> List[Tuple[int, float]]:
        """
        跨模态检索
        
        例如: 用文本查询检索相似的光谱图像
        
        Args:
            query_emb: 查询向量
            database: 多模态数据库
            top_k: 返回结果数
            
        Returns:
            (索引, 相似度)列表
        """
        similarities = []
        
        for idx, item in enumerate(database):
            # 融合数据库中的多模态表征
            fused_emb = self.fuse(item, method='weighted')
            
            # 计算余弦相似度
            sim = np.dot(query_emb, fused_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(fused_emb) + 1e-8
            )
            similarities.append((idx, sim))
        
        # 排序返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class MultimodalRAG:
    """
    多模态RAG系统
    
    支持跨模态的知识检索和问答
    """
    
    def __init__(self):
        self.fusion = MultimodalFusion()
        self.knowledge_base = []
        
    def add_knowledge(self, 
                     text: str,
                     spectrum_image: Optional[str] = None,
                     light_curve: Optional[Dict] = None,
                     metadata: Optional[Dict] = None):
        """
        添加多模态知识
        
        Args:
            text: 知识文本
            spectrum_image: 光谱图像路径
            light_curve: 光变曲线数据
            metadata: 元数据
        """
        data = {
            'text': text,
            'spectrum_image': spectrum_image,
            'light_curve': light_curve
        }
        
        # 编码各模态
        embeddings = self.fusion.encode_multimodal(data)
        fused_emb = self.fusion.fuse(embeddings, method='weighted')
        
        # 存储
        self.knowledge_base.append({
            'embeddings': embeddings,
            'fused': fused_emb,
            'text': text,
            'metadata': metadata or {}
        })
    
    def query(self, 
             query_text: str,
             query_spectrum: Optional[str] = None,
             top_k: int = 5) -> List[Dict]:
        """
        多模态查询
        
        支持文本、图像或两者结合的查询
        
        Args:
            query_text: 查询文本
            query_spectrum: 查询光谱图像
            top_k: 返回结果数
            
        Returns:
            相关知识列表
        """
        # 编码查询
        query_data = {'text': query_text}
        if query_spectrum:
            query_data['spectrum_image'] = query_spectrum
        
        query_embs = self.fusion.encode_multimodal(query_data)
        query_fused = self.fusion.fuse(query_embs, method='weighted')
        
        # 检索
        results = []
        for item in self.knowledge_base:
            sim = np.dot(query_fused, item['fused']) / (
                np.linalg.norm(query_fused) * np.linalg.norm(item['fused']) + 1e-8
            )
            results.append((sim, item))
        
        # 排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [{'similarity': sim, **item} for sim, item in results[:top_k]]


# ==================== 测试 ====================

if __name__ == "__main__":
    print("="*70)
    print("多模态融合系统测试")
    print("="*70)
    
    # 初始化融合系统
    fusion = MultimodalFusion(fusion_dim=512)
    
    # 测试数据
    test_data = {
        'text': "这是一个激变变星，显示明显的爆发特征",
        'light_curve': {
            'time': np.linspace(0, 10, 100),
            'mag': 15 + 0.5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.05, 100)
        }
    }
    
    print("\n1. 编码多模态数据...")
    embeddings = fusion.encode_multimodal(test_data)
    print(f"   编码的模态: {list(embeddings.keys())}")
    for mod, emb in embeddings.items():
        print(f"   - {mod}: 维度 {len(emb)}")
    
    print("\n2. 融合表征...")
    fused = fusion.fuse(embeddings, method='weighted')
    print(f"   融合向量维度: {len(fused)}")
    print(f"   融合向量范数: {np.linalg.norm(fused):.4f}")
    
    print("\n3. 测试多模态RAG...")
    mm_rag = MultimodalRAG()
    
    # 添加知识
    mm_rag.add_knowledge(
        text="激变变星(CV)是白矮星从伴星吸积物质的双星系统",
        light_curve={'time': [1,2,3], 'mag': [15, 15.5, 15.2]},
        metadata={'category': 'CV', 'object': 'SS Cyg'}
    )
    
    mm_rag.add_knowledge(
        text="AM CVn型星是超紧密双星，周期极短",
        light_curve={'time': [1,2,3], 'mag': [18, 18.3, 18.1]},
        metadata={'category': 'AM CVn', 'object': 'AM CVn'}
    )
    
    # 查询
    results = mm_rag.query("白矮星吸积物质", top_k=2)
    print(f"   查询结果: {len(results)} 条")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r['metadata']['category']}] 相似度: {r['similarity']:.4f}")
        print(f"      {r['text'][:50]}...")
    
    print("\n测试完成!")
