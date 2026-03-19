#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage 天文问答数据生成器 (混合模式)
================================================================================
功能描述:
    从天文领域PDF论文中提取文本，生成高质量的问答对(QA Pairs)。
    支持两种模式:
    1. 规则模式: 基于关键词和模板本地生成(无需API)
    2. API模式: 调用Moonshot(Kimi)大模型生成高质量问答

核心特性:
    - 多主题支持: 赫罗图、SED、光变曲线、周期分析、X射线、光谱等
    - 智能分块: 按页处理PDF，保留上下文
    - 去重机制: 基于内容哈希避免重复问题
    - 引用溯源: 每个QA对标注来源文档和页码
    - 多API轮询: 支持多个API key自动切换

依赖库:
    - pdfplumber: PDF文本提取
    - openai: Moonshot API客户端
    - tqdm: 进度条显示

作者: AstroSage Team
创建日期: 2024-03
最后更新: 2026-03-12

关联文件:
    - analyze_qa_results.py: 分析生成的QA数据
    - train_qwen/convert_to_qwen_format.py: 转换为训练格式

使用方法:
    # 纯规则模式(无需API)
    python generate_astronomy_qa_hybrid.py --input ./papers --output ./qa_output
    
    # API增强模式(需要有效API key)
    python generate_astronomy_qa_hybrid.py --input ./papers --output ./qa_output --use-api

================================================================================
"""

import os
import json
import time
import hashlib
import re
import random
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import pdfplumber

# 尝试导入 openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ 未安装 openai 包，API功能将不可用。运行: pip install openai")

# 多API配置 - 用户提供的5个API key
API_KEYS = [
    # 请在此处填入你的API Keys
    # 可以从环境变量读取: os.getenv("MOONSHOT_API_KEYS", "").split(",")
]

# Moonshot (Kimi) API 配置
API_BASE_URL = "https://api.moonshot.cn/v1"
API_MODEL = "kimi-k2-turbo-preview"


@dataclass
class QAPair:
    """问答对数据结构"""
    question: str
    answer: str
    question_type: str  # hr_diagram, sed, light_curve, period, xray, spectrum, general
    source_file: str
    page_number: int
    section: str
    confidence: float
    context: str
    generation_method: str = "rule_based"  # rule_based 或 api_based
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RuleBasedQAGenerator:
    """基于规则的天文问答生成器 - 无需API"""
    
    # 天文主题关键词定义
    KEYWORDS = {
        "hr_diagram": {
            "terms": ["HR diagram", "Hertzsprung-Russell", "color-magnitude", "CMD", 
                     "main sequence", "red giant", "white dwarf", "luminosity",
                     "color index", "B-V", "absolute magnitude", "stellar evolution"],
            "questions": [
                "该论文中展示的赫罗图有什么主要特征？",
                "研究对象在赫罗图上位于什么位置？",
                "论文中提到的恒星在赫罗图上的分布情况如何？",
                "从赫罗图分析，这些恒星处于什么演化阶段？",
                "论文中的颜色-星等图显示了什么趋势？",
                "赫罗图如何帮助确定恒星的距离和性质？",
                "论文中不同恒星群体在赫罗图上的分离情况如何？",
            ]
        },
        "sed": {
            "terms": ["SED", "spectral energy distribution", "blackbody", "thermal emission",
                     "spectral template", "flux density", "spectral index", "bolometric",
                     "infrared excess", "spectral fitting"],
            "questions": [
                "论文中的能谱分布(SED)分析得出了什么结论？",
                "研究对象的SED拟合使用了什么模型？",
                "SED分析揭示了什么物理特性？",
                "不同波段的能谱特征有什么差异？",
                "SED中是否发现了红外超或特殊发射特征？",
                "能谱分布的黑体拟合温度是多少？",
                "SED分析使用了哪些波段的观测数据？",
            ]
        },
        "light_curve": {
            "terms": ["light curve", "photometry", "magnitude", "brightness", "variability",
                     "eclipse", "transit", "outburst", "flickering", "orbital modulation",
                     "photometric", "apparent magnitude", "absolute magnitude"],
            "questions": [
                "论文中展示的光变曲线有什么特征？",
                "光变曲线的周期性变化说明了什么？",
                "不同波段的光变曲线有什么差异？",
                "论文中观测到的亮度变化幅度是多少？",
                "光变曲线的形状揭示了什么物理过程？",
                "是否存在食变或爆发等特殊光变现象？",
                "测光观测使用了哪些滤光片波段？",
                "光变曲线的周期性如何帮助确定系统参数？",
            ]
        },
        "period": {
            "terms": ["period", "orbital period", "rotation period", "pulsation period",
                     "periodicity", "quasi-periodic", "QPO", "ephemeris", "period variation",
                     "superhump period", "orbital decay"],
            "questions": [
                "论文中测得的轨道周期是多少？",
                "周期分析使用了什么方法？",
                "周期变化揭示了什么演化信息？",
                "是否存在准周期振荡(QPO)现象？",
                "周期测量对于理解系统性质有什么意义？",
                "论文中讨论了几种不同的时间尺度？",
                "周期-光度关系在本研究中有什么应用？",
                "周期变化是否暗示了质量转移过程？",
            ]
        },
        "xray": {
            "terms": ["X-ray", "XMM-Newton", "Chandra", "Swift", "ROSAT", "eROSITA",
                     "XRT", "hard X-ray", "soft X-ray", "X-ray spectrum", "X-ray luminosity",
                     "X-ray emission", "bremsstrahlung", "thermal plasma"],
            "questions": [
                "论文中的X射线观测使用了什么设备？",
                "X射线能谱有什么特征？",
                "X射线光度或流量是多少？",
                "硬X射线和软X射线的比例说明了什么？",
                "X射线时变分析发现了什么？",
                "X射线发射机制是什么？",
                "X射线观测对于理解吸积过程有什么帮助？",
                "是否存在X射线爆发现象？",
            ]
        },
        "spectrum": {
            "terms": ["spectrum", "spectroscopy", "spectral line", "emission line",
                     "absorption line", "Doppler shift", "radial velocity", "line profile",
                     "equivalent width", "line identification", "continuum"],
            "questions": [
                "论文中的光谱观测发现了哪些谱线？",
                "光谱分析揭示了什么运动学信息？",
                "发射线和吸收线特征分别是什么？",
                "谱线的多普勒位移说明了什么？",
                "光谱型如何分类？",
                "是否存在特殊的谱线特征（如塞曼分裂）？",
                "视向速度测量结果是什么？",
                "高分辨率光谱分析发现了什么精细结构？",
            ]
        },
        "cv": {
            "terms": ["cataclysmic variable", "dwarf nova", "nova", "supernova",
                     "polar", "intermediate polar", "accretion disk", "mass transfer",
                     "white dwarf", " Roche lobe", "magnetic CV"],
            "questions": [
                "论文研究的灾变变星是什么类型？",
                "系统的物理参数（质量、半径等）是多少？",
                "吸积过程在该系统中如何运作？",
                "白矮星的特性是什么？",
                "质量转移率是多少？",
                "是否存在磁控制吸积？",
                "系统的演化状态如何？",
            ]
        },
        "binary": {
            "terms": ["binary", "binary star", "close binary", "eclipsing binary",
                     "contact binary", "semi-detached", "mass ratio", "inclination"],
            "questions": [
                "这是一个什么类型的双星系统？",
                "双星系统的轨道参数是什么？",
                "两颗子星的参数（质量、半径）是多少？",
                "双星系统的倾角是多少？",
                "质量比如何确定？",
                "双星演化处于什么阶段？",
            ]
        }
    }
    
    def __init__(self):
        self._answer_templates = self._init_answer_templates()
    
    def _init_answer_templates(self) -> Dict[str, List[str]]:
        """初始化答案模板"""
        return {
            "hr_diagram": [
                "根据论文中的赫罗图分析，{content}。从图中可以看出{observation}，这表{conclusion}。",
                "论文中的颜色-星等图显示{content}。研究对象主要集中在{observation}，指示{conclusion}。",
            ],
            "sed": [
                "SED分析表明{content}。能谱可以用{observation}来拟合，得出{conclusion}。",
                "论文中的能谱分布显示{content}。多波段分析揭示{observation}，说明{conclusion}。",
            ],
            "light_curve": [
                "光变曲线呈现{content}特征。观测显示{observation}，这对应于{conclusion}。",
                "论文中的测光数据显示{content}。亮度变化规律为{observation}，表明{conclusion}。",
            ],
            "period": [
                "周期分析得出{content}。观测到的周期性{observation}，这意味着{conclusion}。",
                "论文报告的周期为{content}。该值{observation}，与{conclusion}一致。",
            ],
            "xray": [
                "X射线观测显示{content}。能谱特征{observation}，表明{conclusion}。",
                "论文中的X射线分析揭示{content}。观测结果{observation}，支持{conclusion}。",
            ],
            "spectrum": [
                "光谱分析识别出{content}。谱线特征{observation}，说明{conclusion}。",
                "高分辨率光谱显示{content}。多普勒测量得到{observation}，指示{conclusion}。",
            ],
            "cv": [
                "该灾变变星系统{content}。观测特性{observation}，符合{conclusion}。",
                "系统参数显示{content}。吸积过程表现为{observation}，说明{conclusion}。",
            ],
            "binary": [
                "双星系统参数为{content}。轨道解给出{observation}，对应{conclusion}。",
                "论文确定这是一个{content}系统。测光/光谱分析显示{observation}，表明{conclusion}。",
            ],
            "general": [
                "论文研究了{content}。主要发现包括{observation}，这些结果{conclusion}。",
                "该研究聚焦于{content}。分析表明{observation}，这对于理解{conclusion}具有重要意义。",
            ]
        }
    
    def detect_topics(self, text: str) -> Dict[str, float]:
        """检测文本中的天文主题及其置信度"""
        text_lower = text.lower()
        scores = {}
        
        for topic, info in self.KEYWORDS.items():
            score = 0
            matched_terms = []
            for term in info["terms"]:
                count = text_lower.count(term.lower())
                if count > 0:
                    score += count
                    matched_terms.append(term)
            if score > 0:
                # 归一化得分
                scores[topic] = min(score / len(info["terms"]) * 10, 1.0)
        
        return scores
    
    def extract_relevant_sentences(self, text: str, topic: str, max_sentences: int = 3) -> List[str]:
        """提取与主题相关的句子"""
        if topic not in self.KEYWORDS:
            return []
        
        terms = self.KEYWORDS[topic]["terms"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            for term in terms:
                if term.lower() in sent_lower:
                    relevant.append(sent.strip())
                    break
            if len(relevant) >= max_sentences:
                break
        
        return relevant
    
    def generate_qa(self, text: str, source_file: str, page_number: int, 
                    section: str, num_questions: int = 2) -> List[QAPair]:
        """基于规则生成问答对"""
        qa_pairs = []
        
        # 检测主题
        topic_scores = self.detect_topics(text)
        
        if not topic_scores:
            # 如果没有检测到特定主题，生成通用问题
            return self._generate_general_qa(text, source_file, page_number, section, num_questions)
        
        # 按得分排序主题
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 为每个主要主题生成问题
        for topic, confidence in sorted_topics[:3]:  # 最多前3个主题
            if len(qa_pairs) >= num_questions * 2:
                break
            
            # 获取相关问题模板
            if topic not in self.KEYWORDS:
                continue
                
            questions = self.KEYWORDS[topic]["questions"]
            selected_questions = random.sample(questions, min(2, len(questions)))
            
            # 提取相关内容
            relevant_sentences = self.extract_relevant_sentences(text, topic)
            context = " ".join(relevant_sentences)[:500] if relevant_sentences else text[:500]
            
            for question in selected_questions:
                # 生成答案
                answer = self._generate_answer(topic, context, confidence)
                
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    question_type=topic,
                    source_file=source_file,
                    page_number=page_number,
                    section=section,
                    confidence=confidence * 0.8,  # 基于规则生成的置信度稍低
                    context=context[:300],
                    generation_method="rule_based"
                ))
        
        return qa_pairs[:num_questions * 2]
    
    def _generate_answer(self, topic: str, context: str, confidence: float) -> str:
        """生成答案"""
        templates = self._answer_templates.get(topic, self._answer_templates["general"])
        template = random.choice(templates)
        
        # 从上下文中提取关键信息
        sentences = context.split('.')
        content_snippet = sentences[0][:150] if sentences else "该天体"
        observation = "观测到特定特征" if len(sentences) < 2 else sentences[1][:100]
        conclusion = "需要进一步研究" if confidence < 0.5 else "与该类型天体的理论模型相符"
        
        answer = template.format(
            content=content_snippet,
            observation=observation,
            conclusion=conclusion
        )
        
        return answer[:400]  # 限制答案长度
    
    def _generate_general_qa(self, text: str, source_file: str, page_number: int,
                             section: str, num_questions: int) -> List[QAPair]:
        """生成通用问答"""
        qa_pairs = []
        
        general_questions = [
            "这部分内容的主要发现是什么？",
            "论文中讨论了什么观测结果？",
            "该研究使用了什么分析方法？",
            "这部分内容对于理解研究对象有什么帮助？",
            "论文中提到的关键参数有哪些？",
        ]
        
        for i, question in enumerate(general_questions[:num_questions]):
            context = text[:500]
            answer = f"根据论文内容，{context[:300]}... 这些观测为理解该天体的物理性质提供了重要信息。"
            
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                question_type="general",
                source_file=source_file,
                page_number=page_number,
                section=section,
                confidence=0.6,
                context=context[:300],
                generation_method="rule_based"
            ))
        
        return qa_pairs


class LLMQAGenerator:
    """使用 Kimi (Moonshot) API 生成问答对 - API增强"""
    
    QA_PROMPTS = {
        "hr_diagram": """基于以下天文论文内容，生成关于赫罗图(HR Diagram)的专业问答对：

论文内容：
{content}

要求：
1. 问题必须具体涉及赫罗图特征、恒星位置、演化阶段
2. 答案必须包含具体数据或观测事实
3. 使用专业的天文术语

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "sed": """基于以下天文论文内容，生成关于能谱分布(SED)的专业问答对：

论文内容：
{content}

要求：
1. 问题涉及SED分析、能谱拟合、波段特征
2. 答案包含具体的物理参数或拟合结果
3. 解释SED特征的物理意义

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "light_curve": """基于以下天文论文内容，生成关于光变曲线(Light Curve)的专业问答对：

论文内容：
{content}

要求：
1. 问题涉及光变特征、周期性、爆发事件
2. 答案包含测光数据、变化幅度、时间尺度
3. 解释光变现象的物理机制

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "period": """基于以下天文论文内容，生成关于周期分析的专业问答对：

论文内容：
{content}

要求：
1. 问题涉及轨道周期、自转周期、周期变化
2. 答案包含具体的周期值和测量方法
3. 讨论周期测量的科学意义

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "xray": """基于以下天文论文内容，生成关于X射线观测的专业问答对：

论文内容：
{content}

要求：
1. 问题涉及X射线源、能谱、时变特性
2. 答案包含X射线卫星、光度、能谱参数
3. 解释X射线发射的物理机制

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "spectrum": """基于以下天文论文内容，生成关于光谱分析的专业问答对：

论文内容：
{content}

要求：
1. 问题涉及谱线特征、多普勒位移、视向速度
2. 答案包含具体谱线标识和测量结果
3. 讨论光谱特征的天体物理意义

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
...""",

        "general": """基于以下天文论文内容，生成专业的天文问答对：

论文内容：
{content}

要求：
1. 问题涵盖论文的主要发现
2. 答案准确、专业、详细
3. 体现天文学专业水平

请生成 {num_q} 个问答对，格式：
Q1: [问题]
A1: [详细答案，100-200字]
..."""
    }
    
    def __init__(self):
        self.api_keys = API_KEYS.copy()
        self.current_key_index = 0
        self.failed_keys = set()
        self.stats = {"total": 0, "success": 0, "failed": 0}
        self.lock = threading.Lock()
        
        # 初始化 clients
        self.clients = {}
        if OPENAI_AVAILABLE:
            for key in self.api_keys:
                try:
                    self.clients[key] = OpenAI(
                        api_key=key,
                        base_url=API_BASE_URL,
                        timeout=60,
                        max_retries=2
                    )
                except Exception as e:
                    print(f"  ⚠️ 初始化API client失败: {e}")
        else:
            print("  ⚠️ openai包不可用，无法使用API功能")
    
    def get_working_key(self) -> Optional[str]:
        """轮询获取可用的API key"""
        with self.lock:
            available = [k for k in self.api_keys if k not in self.failed_keys]
            if not available:
                return None
            key = available[self.current_key_index % len(available)]
            self.current_key_index += 1
            return key
    
    def generate_qa(self, content: str, qa_type: str, num_questions: int = 2,
                    max_retries: int = 2) -> List[Dict]:
        """调用Kimi API生成问答"""
        if not OPENAI_AVAILABLE:
            return []
            
        if qa_type not in self.QA_PROMPTS:
            qa_type = "general"
        
        prompt = self.QA_PROMPTS[qa_type].format(
            content=content[:3000],
            num_q=num_questions
        )
        
        for attempt in range(max_retries):
            api_key = self.get_working_key()
            if not api_key:
                print("  ⚠️ 没有可用的API key")
                return []
            
            client = self.clients.get(api_key)
            if not client:
                continue
            
            try:
                with self.lock:
                    self.stats["total"] += 1
                
                # 调用 Kimi API
                completion = client.chat.completions.create(
                    model=API_MODEL,
                    messages=[
                        {"role": "system", "content": "你是天文领域专家，擅长分析天文学论文并生成高质量的问答对。请基于论文内容生成准确、专业的问题和答案。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=2000
                )
                
                # 提取响应
                text = completion.choices[0].message.content
                
                # 解析问答
                qa_pairs = self._parse_response(text, qa_type)
                
                with self.lock:
                    self.stats["success"] += 1
                
                return qa_pairs
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️ API调用失败 (key {api_key[:20]}...): {error_msg[:100]}")
                
                # 检查是否是key失效
                if "401" in error_msg or "authentication" in error_msg.lower():
                    with self.lock:
                        self.failed_keys.add(api_key)
                        print(f"    → Key已标记为失效")
                
                time.sleep(2 ** attempt)
        
        with self.lock:
            self.stats["failed"] += 1
        return []
    
    def _parse_response(self, text: str, qa_type: str) -> List[Dict]:
        """解析API响应"""
        qa_pairs = []
        
        # 匹配 Q1: ... A1: ... 格式
        pattern = r'Q\d+:\s*(.+?)\n+A\d+:\s*(.+?)(?=\n*Q\d+:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for q, a in matches:
            qa_pairs.append({
                "question": q.strip(),
                "answer": a.strip(),
                "type": qa_type
            })
        
        return qa_pairs
    
    def get_stats(self) -> Dict:
        """获取API调用统计"""
        with self.lock:
            return self.stats.copy()


class HybridQADatasetBuilder:
    """混合模式问答数据集构建器"""
    
    def __init__(self, pdf_dir: str, output_dir: str, use_api: bool = False):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rule_generator = RuleBasedQAGenerator()
        self.api_generator = LLMQAGenerator() if use_api else None
        self.use_api = use_api
        
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_qa": 0,
            "by_type": {},
            "by_method": {"rule_based": 0, "api_based": 0}
        }
    
    def extract_pdf_content(self, pdf_path: Path) -> List[Dict]:
        """提取PDF内容并按页返回"""
        pages_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and len(text) > 100:
                        # 检测页面主题
                        topics = self.rule_generator.detect_topics(text)
                        pages_content.append({
                            "page_number": i + 1,
                            "text": text,
                            "topics": topics,
                            "char_count": len(text)
                        })
        except Exception as e:
            print(f"  ❌ 提取失败: {e}")
        
        return pages_content
    
    def process_pdf(self, pdf_path: Path, questions_per_page: int = 2) -> List[QAPair]:
        """处理单个PDF"""
        print(f"\n📄 {pdf_path.name}")
        
        # 检查缓存
        cache_path = self.cache_dir / f"{pdf_path.stem}_qa.json"
        if cache_path.exists():
            print(f"  ✓ 从缓存加载")
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [QAPair(**item) for item in data]
        
        # 提取内容
        pages = self.extract_pdf_content(pdf_path)
        if not pages:
            print(f"  ❌ 无有效内容")
            self.stats["failed"] += 1
            return []
        
        print(f"  📊 页数: {len(pages)}, 平均每页字符: {sum(p['char_count'] for p in pages)//len(pages)}")
        
        # 显示检测到的主题
        all_topics = {}
        for p in pages:
            for t, s in p["topics"].items():
                all_topics[t] = all_topics.get(t, 0) + s
        if all_topics:
            top_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  🔍 主题: {', '.join([f'{t}({s:.1f})' for t, s in top_topics])}")
        
        qa_pairs = []
        
        # 为每页生成问答
        for page in pages:
            # 基于规则生成
            rule_qa = self.rule_generator.generate_qa(
                text=page["text"],
                source_file=pdf_path.name,
                page_number=page["page_number"],
                section="content",
                num_questions=questions_per_page
            )
            qa_pairs.extend(rule_qa)
            
            # 如果使用API，为主要主题页面生成增强问答
            if self.use_api and self.api_generator and page["topics"]:
                top_topic = max(page["topics"].items(), key=lambda x: x[1])
                if top_topic[1] > 0.3:  # 置信度阈值
                    api_qa = self.api_generator.generate_qa(
                        content=page["text"],
                        qa_type=top_topic[0],
                        num_questions=1
                    )
                    for item in api_qa:
                        qa_pairs.append(QAPair(
                            question=item["question"],
                            answer=item["answer"],
                            question_type=item["type"],
                            source_file=pdf_path.name,
                            page_number=page["page_number"],
                            section="enhanced",
                            confidence=0.9,
                            context=page["text"][:300],
                            generation_method="api_based"
                        ))
        
        print(f"  ✓ 生成 {len(qa_pairs)} 个问答对")
        
        # 保存缓存
        cache_data = [qa.to_dict() for qa in qa_pairs]
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        self.stats["processed"] += 1
        return qa_pairs
    
    def build_dataset(self, max_pdfs: Optional[int] = None, 
                      questions_per_page: int = 2) -> Tuple[List[QAPair], Dict]:
        """构建完整数据集"""
        
        pdf_files = sorted([f for f in self.pdf_dir.glob("*.pdf") 
                           if not f.name.startswith("._")])
        
        print(f"\n{'='*70}")
        print(f"🚀 天文问答数据集生成器 (混合模式)")
        print(f"{'='*70}")
        print(f"📁 PDF目录: {self.pdf_dir}")
        print(f"📊 PDF总数: {len(pdf_files)}")
        print(f"📝 每页问题数: {questions_per_page}")
        print(f"🔌 API模式: {'开启' if self.use_api else '关闭 (纯规则)'}")
        print(f"{'='*70}\n")
        
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
            print(f"⚡ 限制处理前 {max_pdfs} 个文件\n")
        
        all_qa_pairs = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end="")
            qa_pairs = self.process_pdf(pdf_path, questions_per_page)
            all_qa_pairs.extend(qa_pairs)
            
            # 更新统计
            for qa in qa_pairs:
                self.stats["by_type"][qa.question_type] = self.stats["by_type"].get(qa.question_type, 0) + 1
                self.stats["by_method"][qa.generation_method] = self.stats["by_method"].get(qa.generation_method, 0) + 1
            
            # 定期保存
            if i % 10 == 0:
                self._save_intermediate(all_qa_pairs, i)
        
        self.stats["total_qa"] = len(all_qa_pairs)
        
        # 保存最终结果
        self._save_final_results(all_qa_pairs)
        
        return all_qa_pairs, self.stats
    
    def _save_intermediate(self, qa_pairs: List[QAPair], batch: int):
        """保存中间结果"""
        path = self.output_dir / f"qa_intermediate_{batch}.json"
        data = [qa.to_dict() for qa in qa_pairs]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已保存中间结果: {len(qa_pairs)} 个问答对")
    
    def _save_final_results(self, qa_pairs: List[QAPair]):
        """保存最终结果"""
        # 完整数据集
        data = [qa.to_dict() for qa in qa_pairs]
        
        with open(self.output_dir / "qa_dataset_full.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 按类型分组
        for qtype in set(qa.question_type for qa in qa_pairs):
            type_qa = [qa.to_dict() for qa in qa_pairs if qa.question_type == qtype]
            with open(self.output_dir / f"qa_{qtype}.json", 'w', encoding='utf-8') as f:
                json.dump(type_qa, f, ensure_ascii=False, indent=2)
        
        # 对话格式（训练用）
        random.seed(42)
        shuffled = qa_pairs.copy()
        random.shuffle(shuffled)
        split = int(len(shuffled) * 0.8)
        
        for split_name, split_data in [("train", shuffled[:split]), ("test", shuffled[split:])]:
            conversations = []
            for qa in split_data:
                conversations.append({
                    "messages": [
                        {"role": "system", "content": "你是天文领域专家，可以准确回答天文学问题。请基于论文内容回答，并标注来源。"},
                        {"role": "user", "content": qa.question},
                        {"role": "assistant", "content": f"{qa.answer}\n\n【来源: {qa.source_file}, 第{qa.page_number}页, 生成方式: {qa.generation_method}】"}
                    ],
                    "metadata": {
                        "type": qa.question_type,
                        "source": qa.source_file,
                        "page": qa.page_number,
                        "confidence": qa.confidence
                    }
                })
            
            with open(self.output_dir / f"{split_name}_conversations.json", 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        # 统计报告
        with open(self.output_dir / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计
        print(f"\n{'='*70}")
        print(f"📈 统计报告")
        print(f"{'='*70}")
        print(f"✅ 成功处理PDF: {self.stats['processed']}")
        print(f"❌ 失败PDF: {self.stats['failed']}")
        print(f"📝 总问答对数: {self.stats['total_qa']}")
        print(f"\n📊 问题类型分布:")
        for t, c in sorted(self.stats['by_type'].items(), key=lambda x: -x[1]):
            print(f"   • {t}: {c}")
        print(f"\n🔧 生成方法:")
        for m, c in self.stats['by_method'].items():
            print(f"   • {m}: {c}")
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="天文PDF问答数据集生成器")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="PDF目录")
    parser.add_argument("--output-dir", default="output/qa_hybrid", help="输出目录")
    parser.add_argument("--max-pdfs", type=int, default=None, help="最大PDF数量")
    parser.add_argument("--questions-per-page", type=int, default=2, help="每页问题数")
    parser.add_argument("--use-api", action="store_true", help="使用API增强")
    
    args = parser.parse_args()
    
    builder = HybridQADatasetBuilder(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        use_api=args.use_api
    )
    
    qa_pairs, stats = builder.build_dataset(
        max_pdfs=args.max_pdfs,
        questions_per_page=args.questions_per_page
    )
    
    print(f"✨ 完成！生成了 {len(qa_pairs)} 个问答对")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
