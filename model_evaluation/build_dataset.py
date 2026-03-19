#!/usr/bin/env python3
"""
天文领域数据集构建脚本

功能:
1. 从 arXiv 下载天文论文
2. 解析 Annual Review of Astronomy and Astrophysics 格式问题
3. 生成多种格式的训练数据 (ShareGPT, Alpaca, etc.)
4. 创建评估用的 MCQ 数据集

使用方法:
    # 下载 arXiv 论文
    python build_dataset.py --download-arxiv --categories astro-ph.SR astro-ph.EP --output ./data/arxiv_papers.json
    
    # 生成训练数据
    python build_dataset.py --build-training --input ./data/arxiv_papers.json --output ./data/training_data.json
    
    # 生成评估数据集
    python build_dataset.py --build-eval --output ./data/eval_dataset.json --n-samples 1000
"""

import argparse
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class AstronomyQA:
    """天文问答对"""
    question: str
    answer: str
    options: Dict[str, str] = None
    correct_option: str = None
    source: str = ""
    subfield: str = ""
    difficulty: str = "medium"
    question_type: str = "open"  # open, mcq, calculation


class ArxivDownloader:
    """arXiv 论文下载器"""
    
    CATEGORIES = {
        "astro-ph.SR": "Solar and Stellar Astrophysics",
        "astro-ph.EP": "Earth and Planetary Astrophysics",
        "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
        "astro-ph.GA": "Astrophysics of Galaxies",
        "astro-ph.HE": "High Energy Astrophysical Phenomena",
        "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    }
    
    def __init__(self, output_dir: str = "./data/arxiv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search_papers(self, categories: List[str], max_results: int = 1000, 
                      date_range: str = "2020-01-01 TO 2025-12-31") -> List[Dict]:
        """
        搜索 arXiv 论文
        
        Args:
            categories: 论文分类列表
            max_results: 最大结果数
            date_range: 日期范围
            
        Returns:
            论文元数据列表
        """
        try:
            import feedparser
        except ImportError:
            print("请安装 feedparser: pip install feedparser")
            raise
        
        papers = []
        
        for category in categories:
            print(f"搜索分类: {category}")
            
            # 构建查询 URL
            query = f"cat:{category} AND submittedDate:[{date_range}]"
            base_url = "http://export.arxiv.org/api/query"
            
            # 分页获取
            start = 0
            batch_size = 100
            
            while start < max_results:
                params = {
                    "search_query": query,
                    "start": start,
                    "max_results": min(batch_size, max_results - start),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
                
                import urllib.parse
                import urllib.request
                
                url = base_url + "?" + urllib.parse.urlencode(params)
                
                try:
                    response = urllib.request.urlopen(url, timeout=30)
                    feed = feedparser.parse(response.read())
                    
                    for entry in feed.entries:
                        paper = {
                            "id": entry.get("id", ""),
                            "title": entry.get("title", "").replace("\n", " "),
                            "abstract": entry.get("summary", "").replace("\n", " "),
                            "authors": [a.get("name", "") for a in entry.get("authors", [])],
                            "published": entry.get("published", ""),
                            "category": category,
                            "pdf_url": entry.get("link", "").replace("abs", "pdf") + ".pdf"
                        }
                        papers.append(paper)
                    
                    print(f"  获取 {len(feed.entries)} 篇论文")
                    
                    if len(feed.entries) < batch_size:
                        break
                    
                    start += batch_size
                    
                except Exception as e:
                    print(f"  错误: {e}")
                    break
        
        print(f"\n总计获取: {len(papers)} 篇论文")
        return papers
    
    def save_papers(self, papers: List[Dict], output_path: str):
        """保存论文数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"保存到: {output_path}")


class DatasetBuilder:
    """数据集构建器"""
    
    # 天文概念模板 - 用于生成训练数据
    CONCEPT_TEMPLATES = [
        {
            "concept": "造父变星",
            "templates": [
                "什么是造父变星的周期-光度关系？",
                "请解释造父变星如何用作标准烛光测量距离。",
                "造父变星和天琴座RR型变星有什么区别？",
            ],
            "answers": [
                "造父变星具有周期-光度关系，即周期越长，光度越高。这由Henrietta Leavitt于1912年发现。",
                "造父变星的光度可以通过其周期确定，因此可以比较视星等和绝对星等来测量距离。",
                "造父变星是年轻的大质量恒星，周期1-50天；天琴座RR型变星是老年低质量恒星，周期约0.2-1天。",
            ],
            "subfield": "stellar"
        },
        {
            "concept": "AM CVn",
            "templates": [
                "什么是AM CVn型双星系统？",
                "AM CVn系统的组成是什么？",
                "AM CVn系统的典型周期范围是多少？",
                "AM CVn系统的光变机制是什么？",
            ],
            "answers": [
                "AM CVn是一类由白矮星和氦供体星组成的超紧凑双星系统，是LISA重要的引力波源。",
                "AM CVn系统由一颗白矮星（吸积体）和一颗氦星或白矮星（供体）组成。注意：不含中子星！",
                "AM CVn系统的轨道周期非常短，典型范围是5-65分钟，是已知周期最短的双星系统之一。",
                "AM CVn的光变主要来自吸积盘不稳定性（DIM），而不是潮汐相互作用。",
            ],
            "subfield": "stellar"
        },
        {
            "concept": "赫罗图",
            "templates": [
                "什么是赫罗图（HR Diagram）？",
                "如何在赫罗图上区分主序星和巨星？",
                "赫罗图的主要序列代表什么物理意义？",
            ],
            "answers": [
                "赫罗图是天文学家 Hertzsprung 和 Russell 发明的，以恒星的光度（或绝对星等）对有效温度的图。",
                "主序星位于赫罗图的对角带上，巨星位于右上方（低温高光度），白矮星位于左下方。",
                "主序代表恒星处于核心氢燃烧的稳定阶段，质量和光度、温度之间存在特定关系。",
            ],
            "subfield": "stellar"
        },
        {
            "concept": "宇宙学",
            "templates": [
                "什么是宇宙微波背景辐射（CMB）？",
                "暗能量在宇宙演化中起什么作用？",
                "什么是宇宙学红移？",
            ],
            "answers": [
                "CMB是宇宙大爆炸后约38万年发出的辐射，现在温度约2.7K，是早期宇宙的重要遗迹。",
                "暗能量占宇宙总能量的约68%，导致宇宙加速膨胀，由1998年超新星观测发现。",
                "宇宙学红移是由于宇宙膨胀导致的光波波长拉伸，z = (λ_obs - λ_emit) / λ_emit。",
            ],
            "subfield": "cosmology"
        },
        {
            "concept": "系外行星",
            "templates": [
                "凌日法探测系外行星的原理是什么？",
                "什么是径向速度法（多普勒法）？",
                "直接成像法探测系外行星的主要挑战是什么？",
            ],
            "answers": [
                "当行星从恒星前方经过时，会遮挡部分星光，导致恒星亮度周期性下降约0.01%-1%。",
                "行星引力使恒星绕共同质心运动，产生周期性多普勒频移，通过光谱测量可以探测行星。",
                "直接成像面临恒星比行星亮10^6-10^10倍的巨大反差挑战，需要使用星冕仪和自适应光学。",
            ],
            "subfield": "exoplanet"
        },
    ]
    
    # 反幻觉训练数据
    ANTI_HALLUCINATION_DATA = [
        {
            "misconception": "AM CVn 包含中子星",
            "correction": "AM CVn 由白矮星和氦星组成，绝不含中子星！",
            "question": "AM CVn 系统包含中子星吗？",
            "answer": "不，AM CVn 系统不包含中子星。它由一颗白矮星（吸积体）和一颗氦供体星组成。这是AM CVn系统最基本的特征之一。",
            "subfield": "stellar"
        },
        {
            "misconception": "AM CVn 有标准的周期-光度关系公式",
            "correction": "AM CVn 只有统计相关性，没有严格的P-L公式如ΔF/F = ...",
            "question": "AM CVn 有像造父变星那样的周期-光度关系公式吗？",
            "answer": "AM CVn 没有严格的周期-光度公式。虽然有观测表明短周期系统倾向于更亮（统计相关性），但这不像造父变星那样有明确的物理公式如ΔF/F = (1+q)*sin(...)。",
            "subfield": "stellar"
        },
        {
            "misconception": "激变变星的光变是潮汐引起的",
            "correction": "CV/AM CVn 的光变来自吸积盘过程，不是潮汐！",
            "question": "激变变星的光变是由什么引起的？",
            "answer": "激变变星（包括AM CVn）的光变主要来自吸积盘的不稳定性（如DIM模型），而不是潮汐相互作用。潮汐会影响轨道演化，但不直接导致周期性光变。",
            "subfield": "stellar"
        },
        {
            "misconception": "潮汐直接导致周期性光变",
            "correction": "潮汐改变轨道动力学，不直接导致周期性光变！",
            "question": "潮汐相互作用会直接导致周期性光变吗？",
            "answer": "潮汐相互作用主要影响双星系统的轨道动力学（如轨道圆化、自转同步），不直接导致观测到的周期性光变。光变通常由掩食、脉动或吸积过程引起。",
            "subfield": "stellar"
        },
    ]
    
    def __init__(self):
        pass
    
    def generate_training_data(self, n_samples: int = 10000) -> List[Dict]:
        """
        生成训练数据集
        
        Args:
            n_samples: 目标样本数
            
        Returns:
            ShareGPT 格式的对话数据
        """
        data = []
        
        # 1. 基于模板生成
        samples_per_template = n_samples // (len(self.CONCEPT_TEMPLATES) * 3)
        
        for concept_data in self.CONCEPT_TEMPLATES:
            templates = concept_data["templates"]
            answers = concept_data["answers"]
            subfield = concept_data["subfield"]
            
            for i, (template, answer) in enumerate(zip(templates, answers)):
                for _ in range(samples_per_template // len(templates) + 1):
                    conversation = {
                        "messages": [
                            {
                                "role": "system",
                                "content": f"你是一位专业的天体物理学家，擅长{subfield}领域的研究。请用准确的天文术语回答问题，避免产生幻觉。"
                            },
                            {
                                "role": "user",
                                "content": template
                            },
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        ],
                        "metadata": {
                            "concept": concept_data["concept"],
                            "subfield": subfield,
                            "type": "concept_qa"
                        }
                    }
                    data.append(conversation)
        
        # 2. 反幻觉训练数据
        for item in self.ANTI_HALLUCINATION_DATA:
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的天体物理学家。请纠正以下错误概念，并给出准确的答案。"
                    },
                    {
                        "role": "user",
                        "content": f"常见误解: {item['misconception']}\n\n{item['question']}"
                    },
                    {
                        "role": "assistant",
                        "content": f"纠正: {item['correction']}\n\n{item['answer']}"
                    }
                ],
                "metadata": {
                    "subfield": item["subfield"],
                    "type": "anti_hallucination"
                }
            }
            # 重复多次以加强记忆
            data.extend([conversation] * 10)
        
        # 3. 随机打乱
        random.shuffle(data)
        
        # 截断到目标数量
        data = data[:n_samples]
        
        print(f"生成训练数据: {len(data)} 条")
        return data
    
    def generate_eval_dataset(self, n_samples: int = 1000) -> List[Dict]:
        """
        生成评估数据集 (MCQ格式)
        """
        eval_questions = []
        
        # AstroMLab-1 风格的 MCQ
        mcq_templates = [
            {
                "question": "造父变星的周期-光度关系表明：",
                "options": {
                    "A": "周期越长，光度越低",
                    "B": "周期越长，光度越高",
                    "C": "周期和光度无关",
                    "D": "周期和光度呈反比关系"
                },
                "correct": "B",
                "explanation": "造父变星具有正的周期-光度关系，这是它们作为标准烛光的基础。",
                "subfield": "stellar",
                "difficulty": "medium"
            },
            {
                "question": "AM CVn 型双星系统的组成是：",
                "options": {
                    "A": "中子星 + 主序星",
                    "B": "白矮星 + 氦星（或白矮星）",
                    "C": "黑洞 + 恒星",
                    "D": "两颗主序星"
                },
                "correct": "B",
                "explanation": "AM CVn 由白矮星（吸积体）和氦供体星组成，绝不含中子星。",
                "subfield": "stellar",
                "difficulty": "hard"
            },
            {
                "question": "AM CVn 系统的典型轨道周期范围是：",
                "options": {
                    "A": "几小时到几天",
                    "B": "几分钟到几小时",
                    "C": "几秒到几分钟",
                    "D": "几天到几周"
                },
                "correct": "B",
                "explanation": "AM CVn 是超紧凑双星，周期极短，典型范围是5-65分钟。",
                "subfield": "stellar",
                "difficulty": "hard"
            },
            {
                "question": "哈勃定律描述的是：",
                "options": {
                    "A": "恒星演化规律",
                    "B": "星系退行速度与距离的关系",
                    "C": "行星轨道运动",
                    "D": "恒星脉动周期"
                },
                "correct": "B",
                "explanation": "哈勃定律 v = H₀d 表明星系退行速度与其距离成正比。",
                "subfield": "cosmology",
                "difficulty": "easy"
            },
            {
                "question": "以下哪个不是探测系外行星的方法？",
                "options": {
                    "A": "凌日法",
                    "B": "径向速度法",
                    "C": "引力透镜法",
                    "D": "射电干涉法"
                },
                "correct": "D",
                "explanation": "射电干涉法主要用于射电源成像，不是常规系外行星探测方法。",
                "subfield": "exoplanet",
                "difficulty": "medium"
            },
            {
                "question": "在赫罗图上，红巨星位于：",
                "options": {
                    "A": "左上方（高温高光度）",
                    "B": "右上方（低温高光度）",
                    "C": "左下方（高温低光度）",
                    "D": "主序带上"
                },
                "correct": "B",
                "explanation": "红巨星表面温度低但光度高，因此在赫罗图右上方。",
                "subfield": "stellar",
                "difficulty": "easy"
            },
            {
                "question": "宇宙微波背景辐射的温度约为：",
                "options": {
                    "A": "2.7 K",
                    "B": "27 K",
                    "C": "270 K",
                    "D": "2700 K"
                },
                "correct": "A",
                "explanation": "CMB 温度约为 2.725 K，是大爆炸的重要遗迹。",
                "subfield": "cosmology",
                "difficulty": "easy"
            },
            {
                "question": "AM CVn 系统的光变主要由什么引起？",
                "options": {
                    "A": "潮汐相互作用",
                    "B": "吸积盘不稳定性",
                    "C": "恒星脉动",
                    "D": "掩食效应"
                },
                "correct": "B",
                "explanation": "AM CVn 的光变主要来自吸积盘不稳定性（DIM模型），而非潮汐。",
                "subfield": "stellar",
                "difficulty": "hard"
            },
        ]
        
        # 复制模板以达到目标数量
        n_copies = n_samples // len(mcq_templates) + 1
        for _ in range(n_copies):
            for template in mcq_templates:
                q = template.copy()
                q["id"] = f"eval_{len(eval_questions):05d}"
                eval_questions.append(q)
                if len(eval_questions) >= n_samples:
                    break
            if len(eval_questions) >= n_samples:
                break
        
        print(f"生成评估数据: {len(eval_questions)} 题")
        return eval_questions[:n_samples]
    
    def convert_to_alpaca(self, sharegpt_data: List[Dict]) -> List[Dict]:
        """转换为 Alpaca 格式"""
        alpaca_data = []
        
        for item in sharegpt_data:
            messages = item["messages"]
            instruction = ""
            input_text = ""
            output = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    instruction = msg["content"]
                elif msg["role"] == "assistant":
                    output = msg["content"]
            
            alpaca_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "metadata": item.get("metadata", {})
            })
        
        return alpaca_data
    
    def convert_to_llama_factory(self, sharegpt_data: List[Dict]) -> List[Dict]:
        """转换为 LLaMA-Factory 格式"""
        # LLaMA-Factory 也使用 ShareGPT 格式
        return sharegpt_data
    
    def save_dataset(self, data: List[Dict], output_path: str, format: str = "json"):
        """保存数据集"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"保存数据集: {output_path} ({len(data)} 条)")


def main():
    parser = argparse.ArgumentParser(description="天文领域数据集构建工具")
    
    # 下载模式
    parser.add_argument("--download-arxiv", action="store_true", help="下载 arXiv 论文")
    parser.add_argument("--categories", type=str, nargs="+", 
                       default=["astro-ph.SR", "astro-ph.CO"],
                       help="arXiv 分类")
    parser.add_argument("--max-papers", type=int, default=1000, help="最大论文数")
    
    # 构建模式
    parser.add_argument("--build-training", action="store_true", help="构建训练数据")
    parser.add_argument("--build-eval", action="store_true", help="构建评估数据")
    parser.add_argument("--n-samples", type=int, default=10000, help="样本数量")
    
    # 输入输出
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--format", type=str, default="sharegpt", 
                       choices=["sharegpt", "alpaca", "jsonl"],
                       help="输出格式")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder()
    
    # 下载 arXiv 论文
    if args.download_arxiv:
        downloader = ArxivDownloader()
        papers = downloader.search_papers(args.categories, args.max_papers)
        downloader.save_papers(papers, args.output)
        return
    
    # 构建训练数据
    if args.build_training:
        data = builder.generate_training_data(args.n_samples)
        
        if args.format == "alpaca":
            data = builder.convert_to_alpaca(data)
        elif args.format == "jsonl":
            builder.save_dataset(data, args.output, format="jsonl")
            return
        
        builder.save_dataset(data, args.output)
        return
    
    # 构建评估数据
    if args.build_eval:
        data = builder.generate_eval_dataset(args.n_samples)
        builder.save_dataset(data, args.output)
        return
    
    print("请指定操作模式: --download-arxiv, --build-training, 或 --build-eval")


if __name__ == "__main__":
    main()
