#!/usr/bin/env python3
"""
增强版RAG知识库系统 (Enhanced Astronomy RAG)
===========================================
支持向量检索的天文文献知识库系统

特性:
- 向量嵌入支持 (可扩展至50万+文献)
- 混合检索: 关键词匹配 + 向量相似度
- 天文领域特定的向量化策略
- 支持文献摘要的语义检索

作者: AstroSage AI
"""

import os
import json
import glob
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import pickle


@dataclass
class Document:
    """文档块数据结构"""
    id: str
    content: str
    source: str
    topic: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'topic': self.topic,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }


class SimpleVectorStore:
    """
    简易向量存储 (纯NumPy实现)
    可用于小型到中型知识库 (数千到数十万文档)
    大规模场景建议替换为FAISS或Chroma
    """
    
    def __init__(self, dimension: int = 384, cache_dir: str = "./cache"):
        self.dimension = dimension
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents: List[Document] = []
        self.vectors: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
        
    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        if not documents:
            return
            
        # 为没有embedding的文档生成embedding
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._generate_embedding(doc.content)
            
            self.id_to_idx[doc.id] = len(self.documents)
            self.documents.append(doc)
        
        # 重建向量矩阵
        self._rebuild_vectors()
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        生成文本嵌入向量
        这里使用简化版TF-IDF + 平均词向量
        生产环境建议使用: sentence-transformers, OpenAI API, 或本地embedding模型
        """
        # 简化版：使用词频统计 + 哈希生成固定维度向量
        # 实际应用中应替换为预训练的语言模型
        
        words = self._tokenize(text)
        vector = np.zeros(self.dimension)
        
        if not words:
            return vector
        
        # 基于词频的加权平均
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            # 使用哈希将词映射到向量空间
            word_hash = hashlib.md5(word.encode()).digest()
            word_vector = np.frombuffer(word_hash, dtype=np.uint8).astype(float)
            # 扩展到目标维度
            if len(word_vector) < self.dimension:
                word_vector = np.tile(word_vector, self.dimension // len(word_vector) + 1)[:self.dimension]
            else:
                word_vector = word_vector[:self.dimension]
            
            # TF-IDF加权
            tf = count / len(words)
            idf = 1.0  # 简化处理
            vector += word_vector * tf * idf
        
        # L2归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 保留天文专业术语
        text = text.lower()
        # 保留数字和特定符号
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        words = text.split()
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                     'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
                     'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'between',
                     'and', 'but', 'or', 'yet', 'so', 'if', 'because', 'although',
                     'though', 'while', 'where', 'when', 'that', 'which', 'who',
                     'whom', 'whose', 'what', 'this', 'these', 'those', 'i', 'me',
                     'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
                     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                     'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
                     'they', 'them', 'their', 'theirs', 'themselves'}
        return [w for w in words if w not in stopwords and len(w) > 1]
    
    def _rebuild_vectors(self):
        """重建向量矩阵"""
        if not self.documents:
            self.vectors = None
            return
        
        self.vectors = np.vstack([doc.embedding for doc in self.documents])
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        向量相似度检索
        
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.vectors is None or len(self.documents) == 0:
            return []
        
        # 生成查询向量
        query_vector = self._generate_embedding(query).reshape(1, -1)
        
        # 计算余弦相似度
        similarities = np.dot(self.vectors, query_vector.T).flatten()
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 相似度阈值
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save(self, filepath: str):
        """保存向量库到磁盘"""
        data = {
            'dimension': self.dimension,
            'documents': [doc.to_dict() for doc in self.documents],
            'vectors': self.vectors.tolist() if self.vectors is not None else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """从磁盘加载向量库"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.documents = []
        for doc_dict in data['documents']:
            doc = Document(
                id=doc_dict['id'],
                content=doc_dict['content'],
                source=doc_dict['source'],
                topic=doc_dict['topic'],
                metadata=doc_dict.get('metadata', {}),
                embedding=np.array(doc_dict['embedding']) if doc_dict['embedding'] else None
            )
            self.documents.append(doc)
            self.id_to_idx[doc.id] = len(self.documents) - 1
        
        if data['vectors'] is not None:
            self.vectors = np.array(data['vectors'])
        else:
            self._rebuild_vectors()


class EnhancedAstronomyRAG:
    """
    增强版天文RAG系统
    
    特性:
    - 混合检索: 关键词匹配 + 向量相似度
    - 天文领域知识增强
    - 支持大规模文献摘要索引
    """
    
    def __init__(self, 
                 kb_path: str = None, 
                 knowledge_dir: str = None,
                 use_vector_store: bool = True,
                 vector_cache: str = "./cache/rag_vectors.pkl"):
        """
        初始化增强版RAG系统
        
        Args:
            kb_path: 知识库JSON文件路径
            knowledge_dir: 知识库文本文件目录
            use_vector_store: 是否使用向量存储
            vector_cache: 向量缓存文件路径
        """
        self.kb_path = kb_path
        self.knowledge_dir = knowledge_dir or self._find_knowledge_dir()
        self.vector_cache = vector_cache
        self.use_vector_store = use_vector_store
        
        # 初始化存储
        self.knowledge: Dict[str, str] = {}
        self.documents: List[Document] = []
        
        # 初始化向量存储
        if use_vector_store:
            self.vector_store = SimpleVectorStore(cache_dir="./cache")
            # 尝试加载缓存
            if os.path.exists(vector_cache):
                print(f"  正在加载向量缓存: {vector_cache}")
                self.vector_store.load(vector_cache)
        else:
            self.vector_store = None
        
        # 加载知识库
        self._load_knowledge()
        
        print(f"✓ 增强版RAG系统已加载:")
        print(f"  - 知识主题: {len(self.knowledge)} 个")
        print(f"  - 文档块: {len(self.documents)} 个")
        if self.vector_store:
            print(f"  - 向量索引: {len(self.vector_store.documents)} 个")
    
    def _find_knowledge_dir(self) -> str:
        """自动查找知识库目录"""
        possible_dirs = [
            "astro_knowledge",
            "../astro_knowledge", 
            "./astro_knowledge",
            "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge"
        ]
        for d in possible_dirs:
            if os.path.isdir(d):
                return d
        return "astro_knowledge"
    
    def _load_knowledge(self):
        """加载知识库"""
        # 1. 加载默认内置知识
        self.knowledge = self._default_knowledge()
        
        # 2. 加载JSON文件
        if self.kb_path and os.path.exists(self.kb_path):
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                file_knowledge = json.load(f)
                self.knowledge.update(file_knowledge)
        
        # 3. 从文本文件加载
        self._load_text_knowledge()
        
        # 4. 构建文档块
        self._build_documents()
    
    def _load_text_knowledge(self):
        """从文本文件加载知识"""
        if not os.path.isdir(self.knowledge_dir):
            return
        
        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                topic_key = os.path.splitext(os.path.basename(filepath))[0]
                if topic_key not in self.knowledge:
                    self.knowledge[topic_key] = content
                    
            except Exception as e:
                print(f"  ⚠ 加载知识文件失败 {filepath}: {e}")
    
    def _build_documents(self):
        """构建文档块并建立向量索引"""
        docs_to_index = []
        
        # 从知识库构建文档
        for topic, content in self.knowledge.items():
            # 按段落切分
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) > 50:  # 只保留有意义的段落
                    doc_id = f"{topic}_{i}"
                    doc = Document(
                        id=doc_id,
                        content=para,
                        source="builtin",
                        topic=topic,
                        metadata={'paragraph_idx': i}
                    )
                    self.documents.append(doc)
                    docs_to_index.append(doc)
        
        # 添加到向量存储
        if self.vector_store and docs_to_index:
            # 检查是否已有缓存
            if len(self.vector_store.documents) == 0:
                print(f"  正在构建向量索引...")
                self.vector_store.add_documents(docs_to_index)
                # 保存缓存
                self.vector_store.save(self.vector_cache)
                print(f"  向量索引已保存到: {self.vector_cache}")
    
    def search(self, query: str, top_k: int = 5, 
               use_vector: bool = True,
               use_keyword: bool = True) -> str:
        """
        混合检索 - 结合向量相似度和关键词匹配
        
        Args:
            query: 查询关键词
            top_k: 返回结果数量
            use_vector: 是否使用向量检索
            use_keyword: 是否使用关键词匹配
            
        Returns:
            相关知识文本
        """
        results = []
        
        # 1. 向量检索
        if use_vector and self.vector_store:
            vector_results = self.vector_store.search(query, top_k=top_k*2)
            for doc, score in vector_results:
                results.append((score + 1.0, doc.topic, doc.content, 'vector'))
        
        # 2. 关键词匹配
        if use_keyword:
            keyword_results = self._keyword_search(query, top_k=top_k*2)
            results.extend(keyword_results)
        
        # 3. 特殊关键词映射（高优先级）
        priority_results = self._priority_search(query)
        results = priority_results + results
        
        # 4. 排序去重
        results.sort(reverse=True, key=lambda x: x[0])
        
        seen_topics = set()
        unique_results = []
        for score, topic, content, source in results:
            # 使用内容哈希去重
            content_hash = hashlib.md5(content[:100].encode()).hexdigest()
            if content_hash not in seen_topics:
                seen_topics.add(content_hash)
                unique_results.append((score, topic, content, source))
                if len(unique_results) >= top_k:
                    break
        
        # 5. 格式化输出
        if unique_results:
            output = []
            for score, topic, content, source in unique_results:
                # 截断过长内容
                if len(content) > 1000:
                    content = content[:1000] + "..."
                output.append(
                    f"【{topic.upper()}】(相关度: {score:.2f}, 来源: {source})\n{content}"
                )
            return "\n\n" + "="*60 + "\n".join(output)
        
        return "未找到相关知识。"
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[float, str, str, str]]:
        """关键词搜索"""
        query_lower = query.lower()
        query_words = set(self._extract_keywords(query_lower))
        
        matches = []
        for key, content in self.knowledge.items():
            score = 0
            key_lower = key.lower().replace('_', ' ')
            
            # 完全匹配
            if query_lower in key_lower or key_lower in query_lower:
                score += 10
            
            # 词匹配
            content_lower = content.lower()
            for word in query_words:
                if len(word) < 2:
                    continue
                if word in key_lower:
                    score += 3
                if word in content_lower:
                    score += 1
            
            if score > 0:
                # 返回最相关的段落
                paragraphs = content.split('\n\n')
                best_para = max(paragraphs, 
                               key=lambda p: sum(1 for w in query_words if w in p.lower()))
                matches.append((score, key, best_para.strip(), 'keyword'))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        return matches[:top_k]
    
    def _priority_search(self, query: str) -> List[Tuple[float, str, str, str]]:
        """高优先级关键词搜索（领域特定）"""
        query_lower = query.lower()
        
        # 天文领域关键词映射
        keyword_mappings = {
            # 天体类型
            'cataclysmic variable': ['cataclysmic_variable', 'polar', 'dwarf_nova'],
            'cv': ['cataclysmic_variable'],
            'polar': ['polar', 'cataclysmic_variable', 'magnetic_cv'],
            'am cvn': ['am_cvn', 'ultra_compact_binary'],
            'am_cvn': ['am_cvn', 'ultra_compact_binary'],
            'white dwarf': ['white_dwarf', 'cataclysmic_variable'],
            'binary': ['binary_systems', 'eclipsing_binary', 'cataclysmic_variable'],
            'pulsating': ['delta_scuti', 'rr_lyrae', 'cepheid'],
            'eclipsing': ['eclipsing_binary'],
            
            # 物理过程
            'period': ['period_luminosity_relations', 'binary_systems'],
            'orbital': ['binary_systems', 'cataclysmic_variable'],
            'accretion': ['cataclysmic_variable', 'polar', 'accretion_disk'],
            'magnetic': ['polar', 'intermediate_polar'],
            'extinction': ['interstellar_medium', 'dust'],
            
            # 观测特征
            'light curve': ['photometry', 'variability'],
            'spectrum': ['spectroscopy', 'emission_lines'],
            'sed': ['sed_analysis', 'photometry'],
        }
        
        priority_results = []
        for key, topics in keyword_mappings.items():
            if key in query_lower:
                for topic in topics:
                    if topic in self.knowledge:
                        content = self.knowledge[topic][:800]
                        priority_results.append((15, topic, content, 'priority'))
        
        return priority_results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 保留天文专业术语
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # 停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 
                     'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     '这个', '那个', '什么', '怎么', '为什么', '请', '的', '了', '在'}
        
        return [w for w in words if w not in stopwords and len(w) > 1]
    
    def search_by_object_type(self, obj_type: str) -> str:
        """根据天体类型搜索相关知识"""
        type_mappings = {
            'cataclysmic': 'cataclysmic_variable',
            'cv': 'cataclysmic_variable',
            'polar': 'polar',
            'am her': 'polar',
            'dwarf nova': 'cataclysmic_variable',
            'nova': 'cataclysmic_variable',
            'rr lyrae': 'rr_lyrae',
            'delta scuti': 'delta_scuti',
            'eclipsing': 'eclipsing_binary',
            'binary': 'binary_systems',
            'white dwarf': 'white_dwarf',
            'am cvn': 'am_cvn',
        }
        
        obj_lower = obj_type.lower()
        for key, topic in type_mappings.items():
            if key in obj_lower and topic in self.knowledge:
                return f"【{topic.upper()}】\n{self.knowledge[topic][:1500]}"
        
        return self.search(obj_type, top_k=3)
    
    def get(self, key: str) -> str:
        """获取特定主题的知识"""
        if key in self.knowledge:
            return self.knowledge[key]
        
        key_lower = key.lower()
        for k, v in self.knowledge.items():
            if key_lower in k.lower() or k.lower() in key_lower:
                return v
        
        return f"未找到 '{key}' 的知识。"
    
    def list_topics(self) -> List[str]:
        """列出所有知识主题"""
        return list(self.knowledge.keys())
    
    def _default_knowledge(self) -> Dict[str, str]:
        """默认天文知识库"""
        return {
            "cataclysmic_variable": """
激变变星(Cataclysmic Variables, CVs)是紧密双星系统，由白矮星主星和晚型伴星组成:

【系统组成】
- 主星: 白矮星(质量0.6-1.2 M_sun，半径约地球大小)
- 伴星: 红矮星或亚巨星(质量<1 M_sun)
- 轨道周期: 通常1-12小时
- 距离: 两星非常接近(太阳半径量级)

【物理过程】
- 洛希瓣溢出: 伴星充满洛希瓣，物质通过内拉格朗日点转移
- 质量转移率: 10^-11 到 10^-8 M_sun/year
- 角动量损失: 磁制动(周期>2-3小时)或引力波辐射(周期<2小时)

【分类体系】
1. 非磁激变变星(B<1MG):
   - 矮新星(Dwarf Novae): 吸积盘不稳定导致爆发，增亮2-6星等
     * U Gem型: 规则爆发
     * SU UMa型: 超爆发+正常爆发，存在超涨(superhump)
     * Z Cam型: 静止态
   - 类新星(Nova-like): 持续高亮状态
   - 新星(Novae): 热核爆发，增亮6-19星等

2. 磁激变变星:
   - 高偏振星/Polar(AM Her型, B>10MG): 同步自转，无吸积盘
   - 中介偏振星/IP(DQ Her型, 1<B<10MG): 非同步，有截断吸积盘

【观测特征】
- 光谱: 强发射线(H I, He I, He II)，双峰轮廓来自吸积盘
- 光变: 爆发、闪烁(flickering)、轨道调制
- X射线: 来自边界层或吸积柱

【周期空缺】
2-3小时范围内的系统数量明显减少，这是CV演化的关键特征。
""",
            "polar": """
Polar(高偏振星/AM Her型星)是强磁场激变变星:

【物理特征】
- 磁场强度: 10-100 MG(兆高斯)，足以完全控制物质流
- 白矮星质量: 0.5-1.0 M_sun
- 轨道周期: 1-4小时(典型约2小时)
- 自转: 与轨道同步(P_spin = P_orb)

【吸积过程】
- 无吸积盘形成: 强磁场阻止盘的形成
- 物质沿磁力线直接落到磁极
- 吸积柱: 高温(10^8 K)等离子体柱
- 吸积区: 磁极附近的热点

【观测特征】
- 强偏振: 线性偏振可达10-30%，圆偏振可达10-20%(来自cyclotron辐射)
- X射线: 硬X射线来自吸积柱，软X射线来自白矮星表面
- 光学/红外: cyclotron辐射主导，呈现特征谐波结构
- 光变: 准周期振荡(QPO)、轨道调制、高低态转变

【光谱特征】
- 强发射线: Hα, Hβ, He II λ4686
- cyclotron谐波: 红外到光学波段，间隔与磁场强度相关
- 高激发线: N III/C III Bowen blend

【子类型】
- 硬X-ray Polar: 强X射线，弱光学
- 软X-ray Polar: 弱X射线，强光学  
- 异步Polar: 自转与轨道略微不同步

【著名源】
AM Her(第一个发现的), EF Eri, VV Pup, HU Aqr, AR UMa
""",
            "am_cvn": """
AM CVn型星是超紧密氦双星系统，是引力波天文的重要目标:

【系统组成】(关键!)
- 白矮星主星 + 氦星(或氦白矮星)伴星
- ⚠️ 绝对不含中子星! 这是最常见的错误
- 轨道周期: 5-65分钟(极短!)
- 是已知最紧凑的双星系统

【物理特征】
- 物质转移: 氦的转移(不含氢)
- 驱动机制: 引力波辐射主导的角动量损失
- 质量转移率: 与周期相关，Mdot ∝ P^(-10/3)

【光变特性】
- 超涨(Superhump): 来自吸积盘的进动，周期略长于轨道周期
- 爆发: 类似矮新星的不稳定性
- 直接撞击: 周期<10分钟时可能无吸积盘

【周期-光度关系】
- 短周期系统倾向于更亮(高质量转移率)
- 但⚠️ 不存在严格的P-L公式如ΔF/F = ...
- 是统计趋势而非严格关系

【观测特征】
- 光谱: 氦线主导(He I, He II)，无氢线
- 颜色: 偏蓝
- 光变: 超涨周期通常为轨道周期的1-5%

【引力波物理】
- LISA空间引力波探测器的核心目标
- 已知参数的"验证双星"
- 距离决定信号强度

【著名源】
- HM Cnc: P = 5.4分钟(最短已知)
- AM CVn本身: P = 17.1分钟
- GP Com, V396 Hya
""",
            "white_dwarf": """
白矮星(White Dwarf, WD)是恒星演化的最终产物之一:

【基本参数】
- 质量: 0.17-1.33 M_sun(钱德拉塞卡极限)
- 半径: ~0.01 R_sun(约地球大小)
- 密度: 10^6 g/cm³
- 表面重力: log g ≈ 7-9
- 光度: 10^-4 到 10^-1 L_sun

【物理支持】
- 电子简并压力: 抵抗引力坍缩
- 非热核反应: 余热辐射，缓慢冷却

【光谱分类】
- DA型: 氢巴尔末线，占~80%
- DB型: He I线，无氢
- DC型: 连续谱，无明显线
- DO型: He II线
- DZ型: 金属线(Ca, Mg, Fe)
- DQ型: 碳特征(C2 Swan带)
- 混合型: DA+DB, DBA等

【冷却轨迹】
- 热WD: Teff > 20,000 K，年龄<10^8年
- 温WD: 10,000 < Teff < 20,000 K
- 冷WD: Teff < 10,000 K，年龄>10^9年
- 冷却时标: 数十亿年，宇宙定年器

【质量-半径关系】
R ∝ M^(-1/3)，质量越大半径越小，相对论性修正在大质量时重要。

【形成途径】
1. 单星演化: 主序星→红巨星→AGB→行星状星云→WD
2. 双星演化: 洛希瓣溢出、公共包层、双WD并合

【脉动白矮星】
- DAV/ZZCeti: H大气，Teff=12,000-11,000K，周期100-1000s
- DBV/V777Her: He大气，Teff=24,000-19,000K
- GW Vir: 热脉动，Teff>100,000K
""",
            "period_analysis": """
光变周期分析是天体物理研究的重要手段:

【周期搜索方法】
1. 相位弥散最小化(PDM, Phase Dispersion Minimization):
   - 将数据按试验周期折叠
   - 计算各相位区间的方差
   - 最佳周期使方差最小
   - 适用于非正弦光变

2. Lomb-Scargle周期图:
   - 基于最小二乘拟合
   - 适用于非均匀采样数据
   - 给出频率空间功率谱

3. 弦长法(String Length):
   - 最小化折叠后曲线的总长度
   - 对尖锐特征敏感

4. 快速傅里叶变换(FFT):
   - 适用于均匀采样数据
   - 计算效率高

【周期类型】
- 轨道周期: 双星系统的基本周期
- 超涨周期: 比轨道周期长1-5%，来自盘进动
- 负超涨: 比轨道周期短，罕见
- 自转周期: 白矮星自转，通常短于轨道周期

【周期变化】
- 周期变化率 dP/dt 反映角动量演化
- 引力波辐射导致周期缩短
- 质量转移可能导致周期增加或减少

【不确定性估计】
- 考虑数据长度、采样率、信噪比
- bootstrap重采样估计误差
- 多波段周期一致性检验
""",
            "sed_analysis": """
光谱能量分布(SED, Spectral Energy Distribution)分析:

【SED构建】
- 收集多波段测光数据
- 覆盖从X射线到射电的全波段
- 流量校准到同一参考系

【组成部分】
1. 主星辐射: 黑体或大气模型
2. 吸积盘: 多温黑体谱
3. 边界层: 高温成分
4. 吸积柱: X射线辐射
5. 伴星: 晚型恒星谱
6. 星际消光: 红化效应

【拟合方法】
- 单温/多温黑体拟合
- 大气模型拟合
- 吸积盘模型
- 消光修正

【颜色-颜色图】
- 利用不同波段的流量比
- 区分不同天体类型
- 识别特殊/异常源

【红化与消光】
- E(B-V): 色余
- Rv = Av/E(B-V) ≈ 3.1 (银河系典型值)
- 影响: 使SED变红，需修正

【物理参数导出】
- 有效温度 Teff
- 角直径
- 光度 L
- 半径 R
- 距离 d
""",
        }


# ==================== 便捷函数 ====================

def get_enhanced_rag(knowledge_dir: str = None, 
                     use_vector_store: bool = True) -> EnhancedAstronomyRAG:
    """获取增强版RAG实例"""
    return EnhancedAstronomyRAG(
        knowledge_dir=knowledge_dir,
        use_vector_store=use_vector_store
    )


def quick_search(query: str, knowledge_dir: str = None, top_k: int = 5) -> str:
    """快速搜索知识库"""
    rag = get_enhanced_rag(knowledge_dir=knowledge_dir)
    return rag.search(query, top_k=top_k)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("增强版天文RAG系统测试")
    print("=" * 70)
    
    rag = get_enhanced_rag(use_vector_store=True)
    
    print("\n可用主题:")
    for topic in rag.list_topics()[:10]:
        print(f"  - {topic}")
    print("  ...")
    
    # 测试搜索
    print("\n" + "=" * 70)
    print("测试: 激变变星分类")
    print("=" * 70)
    result = rag.search("cataclysmic variable classification", top_k=3)
    print(result[:2000])
    
    # 测试天体类型搜索
    print("\n" + "=" * 70)
    print("测试: 按天体类型搜索")
    print("=" * 70)
    result = rag.search_by_object_type("polar")
    print(result[:1500])
    
    print("\n测试完成!")
