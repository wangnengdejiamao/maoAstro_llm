#!/usr/bin/env python3
"""
白矮星领域文献下载器

功能:
1. 从 arXiv 下载白矮星相关论文
2. 使用 Kimi API 生成论文摘要和问答对
3. 构建专门的训练数据集
4. 支持用户上传文献处理

侧重领域:
- 单白矮星 (WD) 演化与冷却
- 双白矮星 (DWD) 系统
- 磁性白矮星 (MWD)
- 吸积白矮星与激变变星 (CV)
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# 尝试导入 feedparser
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("警告: feedparser 未安装，将使用备用方法")
    print("建议安装: pip install feedparser")


@dataclass
class Paper:
    """论文数据结构"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    categories: List[str]
    pdf_url: str
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class WhiteDwarfPaperDownloader:
    """
    白矮星论文下载器
    """
    
    # 白矮星相关搜索关键词
    WD_KEYWORDS = [
        # 基础白矮星
        "white dwarf",
        "white dwarfs",
        "degenerate star",
        "degenerate dwarf",
        
        # 双白矮星
        "double white dwarf",
        "double degenerate",
        "DWD",
        "white dwarf binary",
        "white dwarf merger",
        
        # 磁性白矮星
        "magnetic white dwarf",
        "MWD",
        "high field magnetic",
        "polar",
        "intermediate polar",
        "DQ Herculis",
        "AM Herculis",
        
        # 吸积白矮星
        "accreting white dwarf",
        "accretion onto white dwarf",
        "cataclysmic variable",
        "CV",
        "nova",
        "dwarf nova",
        "recurrent nova",
        "supersoft X-ray source",
        "SSS",
        
        # 特殊类型
        "AM CVn",
        " helium white dwarf",
        "carbon-oxygen white dwarf",
        "oxygen-neon white dwarf",
        "DB white dwarf",
        "DA white dwarf",
        "DC white dwarf",
        
        # 演化与冷却
        "white dwarf cooling",
        "luminosity function",
        "white dwarf mass",
        "initial-final mass relation",
    ]
    
    # arXiv 天体物理分类
    ARXIV_CATEGORIES = [
        "astro-ph.SR",  # Solar and Stellar Astrophysics
        "astro-ph.HE",  # High Energy Astrophysical Phenomena
    ]
    
    def __init__(self, output_dir: str = "./data/white_dwarf_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.papers: List[Paper] = []
        
    def search_arxiv(self, query: str, max_results: int = 100, 
                     start_date: str = "2020-01-01",
                     end_date: str = None) -> List[Paper]:
        """
        搜索 arXiv
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (默认今天)
            
        Returns:
            论文列表
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        papers = []
        
        if HAS_FEEDPARSER:
            papers = self._search_with_feedparser(query, max_results, start_date, end_date)
        else:
            papers = self._search_with_api(query, max_results, start_date, end_date)
        
        return papers
    
    def _search_with_feedparser(self, query: str, max_results: int,
                                start_date: str, end_date: str) -> List[Paper]:
        """使用 feedparser 搜索"""
        papers = []
        
        date_range = f"{start_date} TO {end_date}"
        full_query = f"({query}) AND submittedDate:[{date_range}]"
        
        base_url = "http://export.arxiv.org/api/query"
        
        batch_size = 100
        start = 0
        
        print(f"搜索: {query[:50]}...")
        
        while start < max_results:
            params = {
                "search_query": full_query,
                "start": start,
                "max_results": min(batch_size, max_results - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            url = base_url + "?" + urllib.parse.urlencode(params)
            
            try:
                response = urllib.request.urlopen(url, timeout=30)
                feed = feedparser.parse(response.read())
                
                for entry in feed.entries:
                    paper = Paper(
                        arxiv_id=entry.get("id", "").split("/")[-1].split("v")[0],
                        title=entry.get("title", "").replace("\n", " "),
                        abstract=entry.get("summary", "").replace("\n", " "),
                        authors=[a.get("name", "") for a in entry.get("authors", [])],
                        published=entry.get("published", ""),
                        categories=[t.get("term", "") for t in entry.get("tags", [])],
                        pdf_url=entry.get("link", "").replace("abs", "pdf") + ".pdf"
                    )
                    papers.append(paper)
                
                print(f"  获取 {len(feed.entries)} 篇")
                
                if len(feed.entries) < batch_size:
                    break
                
                start += batch_size
                time.sleep(3)  # 避免请求过快
                
            except Exception as e:
                print(f"  错误: {e}")
                break
        
        return papers
    
    def _search_with_api(self, query: str, max_results: int,
                         start_date: str, end_date: str) -> List[Paper]:
        """备用：直接解析 XML"""
        import xml.etree.ElementTree as ET
        
        papers = []
        
        date_range = f"{start_date} TO {end_date}"
        full_query = f"({query}) AND submittedDate:[{date_range}]"
        
        base_url = "http://export.arxiv.org/api/query"
        
        batch_size = 100
        start = 0
        
        print(f"搜索: {query[:50]}... (备用方法)")
        
        while start < max_results:
            params = {
                "search_query": full_query,
                "start": start,
                "max_results": min(batch_size, max_results - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            url = base_url + "?" + urllib.parse.urlencode(params)
            
            try:
                response = urllib.request.urlopen(url, timeout=30)
                root = ET.fromstring(response.read())
                
                # 命名空间
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                for entry in root.findall("atom:entry", ns):
                    paper = Paper(
                        arxiv_id=entry.find("atom:id", ns).text.split("/")[-1].split("v")[0],
                        title=entry.find("atom:title", ns).text.replace("\n", " "),
                        abstract=entry.find("atom:summary", ns).text.replace("\n", " "),
                        authors=[author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
                        published=entry.find("atom:published", ns).text,
                        categories=[cat.get("term") for cat in entry.findall("atom:category", ns)],
                        pdf_url=entry.find("atom:id", ns).text.replace("abs", "pdf") + ".pdf"
                    )
                    papers.append(paper)
                
                entries_count = len(root.findall("atom:entry", ns))
                print(f"  获取 {entries_count} 篇")
                
                if entries_count < batch_size:
                    break
                
                start += batch_size
                time.sleep(3)
                
            except Exception as e:
                print(f"  错误: {e}")
                break
        
        return papers
    
    def search_comprehensive(self, max_results_per_topic: int = 100,
                            start_date: str = "2020-01-01") -> List[Paper]:
        """
        全面搜索白矮星相关论文
        
        按不同子领域搜索并合并结果
        """
        all_papers = {}
        
        # 定义子领域搜索
        searches = {
            "general_wd": "white dwarf",
            "double_degenerate": "double white dwarf OR double degenerate OR DWD",
            "magnetic": "magnetic white dwarf OR MWD OR polar OR intermediate polar",
            "accretion": "accreting white dwarf OR cataclysmic variable OR CV",
            "am_cvn": "AM CVn",
            "cooling": "white dwarf cooling OR luminosity function",
            "type_ia": "type Ia supernova AND white dwarf",
        }
        
        for topic, query in searches.items():
            print(f"\n[{topic}]")
            papers = self.search_arxiv(query, max_results_per_topic, start_date)
            
            for paper in papers:
                # 去重（基于 arXiv ID）
                if paper.arxiv_id not in all_papers:
                    paper.keywords = [topic]
                    all_papers[paper.arxiv_id] = paper
                else:
                    all_papers[paper.arxiv_id].keywords.append(topic)
        
        self.papers = list(all_papers.values())
        
        print(f"\n{'='*60}")
        print(f"总计获取: {len(self.papers)} 篇唯一论文")
        print(f"{'='*60}")
        
        return self.papers
    
    def save_papers(self, filename: str = "wd_papers.json"):
        """保存论文数据"""
        output_path = self.output_dir / filename
        
        data = []
        for paper in self.papers:
            data.append({
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "published": paper.published,
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "keywords": paper.keywords
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 保存论文数据: {output_path}")
        return output_path
    
    def generate_training_data_with_kimi(self, max_papers: int = 50) -> List[Dict]:
        """
        使用 Kimi API 生成训练数据
        
        Args:
            max_papers: 处理的最大论文数
            
        Returns:
            ShareGPT 格式的训练数据
        """
        try:
            from kimi_interface import KimiInterface, WhiteDwarfDataGenerator
        except ImportError:
            print("错误: 无法导入 kimi_interface，请确保文件存在")
            return []
        
        kimi = KimiInterface()
        generator = WhiteDwarfDataGenerator(kimi)
        
        training_data = []
        
        papers_to_process = self.papers[:max_papers]
        
        print(f"\n使用 Kimi 生成训练数据 ({len(papers_to_process)} 篇论文)...")
        print("=" * 60)
        
        for i, paper in enumerate(papers_to_process, 1):
            print(f"\n[{i}/{len(papers_to_process)}] {paper.title[:60]}...")
            
            # 生成论文摘要问答
            result = generator.generate_paper_summary(paper.title, paper.abstract)
            
            if result and 'qa_pairs' in result:
                for qa in result['qa_pairs']:
                    conversation = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是白矮星天体物理学专家。请基于专业知识回答问题。"
                            },
                            {
                                "role": "user",
                                "content": qa.get('question', '')
                            },
                            {
                                "role": "assistant",
                                "content": qa.get('answer', '')
                            }
                        ],
                        "metadata": {
                            "source": f"arxiv:{paper.arxiv_id}",
                            "topic": paper.keywords,
                            "type": "paper_qa"
                        }
                    }
                    training_data.append(conversation)
                
                print(f"  生成 {len(result['qa_pairs'])} 个 QA")
            
            # 避免请求过快
            time.sleep(1)
        
        print(f"\n✓ 总共生成 {len(training_data)} 条训练数据")
        
        return training_data
    
    def generate_synthetic_qa(self, use_kimi: bool = True) -> List[Dict]:
        """
        生成合成问答数据
        
        Args:
            use_kimi: 是否使用 Kimi API 生成
            
        Returns:
            问答数据列表
        """
        if use_kimi:
            try:
                from kimi_interface import KimiInterface, WhiteDwarfDataGenerator
                
                kimi = KimiInterface()
                generator = WhiteDwarfDataGenerator(kimi)
                
                training_data = []
                
                # 定义白矮星子领域主题
                topics = [
                    "单白矮星冷却序列和年龄测定",
                    "双白矮星系统形成和演化",
                    "双白矮星引力波波源",
                    "磁性白矮星的磁场产生机制",
                    "吸积白矮星的热核爆发（新星）",
                    "激变变星的光变机制",
                    "AM CVn 型系统的吸积物理",
                    "白矮星的质量-半径关系",
                    "白矮星的光谱分类 (DA, DB, DC, DQ, DZ)",
                    "超软 X 射线源的物理机制",
                    "Ia 型超新星的前身星模型",
                    "白矮星结晶化过程",
                ]
                
                print(f"\n生成合成 QA 数据 ({len(topics)} 个主题)...")
                print("=" * 60)
                
                for topic in topics:
                    print(f"\n主题: {topic}")
                    
                    # 生成问答对
                    qa_pairs = generator.generate_qa_pairs(topic, n_pairs=5)
                    
                    for qa in qa_pairs:
                        conversation = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "你是白矮星天体物理学专家。请准确、详细地回答专业问题。"
                                },
                                {
                                    "role": "user",
                                    "content": qa.get('question', '')
                                },
                                {
                                    "role": "assistant",
                                    "content": qa.get('answer', '')
                                }
                            ],
                            "metadata": {
                                "topic": topic,
                                "type": "synthetic_qa",
                                "difficulty": qa.get('difficulty', 'medium'),
                                "key_points": qa.get('key_points', [])
                            }
                        }
                        training_data.append(conversation)
                    
                    print(f"  生成 {len(qa_pairs)} 个 QA")
                    time.sleep(0.5)
                
                # 生成反幻觉数据
                print("\n生成反幻觉训练数据...")
                anti_hall_data = generator.generate_anti_hallucination_data()
                
                for item in anti_hall_data:
                    conversation = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是白矮星天体物理学专家。请纠正错误概念并给出准确答案。"
                            },
                            {
                                "role": "user",
                                "content": item.get('question', '')
                            },
                            {
                                "role": "assistant",
                                "content": item.get('correct_answer', '')
                            }
                        ],
                        "metadata": {
                            "topic": item.get('topic', ''),
                            "type": "anti_hallucination",
                            "misconception": item.get('misconception', '')
                        }
                    }
                    training_data.append(conversation)
                
                print(f"  生成 {len(anti_hall_data)} 条反幻觉数据")
                
                # 生成计算题
                print("\n生成计算题...")
                calc_topics = [
                    "白矮星冷却时标计算",
                    "吸积盘温度分布",
                    "双白矮星并合参数",
                ]
                
                for topic in calc_topics:
                    calc_problems = generator.generate_calculation_problems(topic, n_problems=2)
                    
                    for prob in calc_problems:
                        problem_text = f"{prob.get('problem', '')}\n\n"
                        problem_text += "已知:\n" + "\n".join([f"- {g}" for g in prob.get('given', [])])
                        problem_text += f"\n\n求: {prob.get('find', '')}"
                        
                        solution_text = "解题步骤:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(prob.get('solution_steps', []))])
                        solution_text += f"\n\n公式: {prob.get('formula', '')}"
                        solution_text += f"\n答案: {prob.get('answer', '')}"
                        
                        conversation = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "你是白矮星天体物理学专家。请逐步解决计算问题。"
                                },
                                {
                                    "role": "user",
                                    "content": problem_text
                                },
                                {
                                    "role": "assistant",
                                    "content": solution_text
                                }
                            ],
                            "metadata": {
                                "topic": topic,
                                "type": "calculation",
                                "difficulty": prob.get('difficulty', 'medium')
                            }
                        }
                        training_data.append(conversation)
                    
                    print(f"  {topic}: {len(calc_problems)} 题")
                
                print(f"\n✓ 总共生成 {len(training_data)} 条训练数据")
                
                return training_data
                
            except Exception as e:
                print(f"使用 Kimi 生成失败: {e}")
                print("使用备用模板生成...")
                return self._generate_template_qa()
        else:
            return self._generate_template_qa()
    
    def _generate_template_qa(self) -> List[Dict]:
        """使用模板生成问答（备用方法）"""
        
        template_qa = [
            {
                "question": "什么是白矮星的钱德拉塞卡极限？",
                "answer": "钱德拉塞卡极限是指白矮星能够通过电子简并压支撑的最大质量，约为 1.4 倍太阳质量（精确值为 1.38 M☉）。超过这个质量，电子简并压无法抵抗引力，恒星将坍缩为中子星或发生 Ia 型超新星爆发。",
                "topic": "基本性质"
            },
            {
                "question": "双白矮星系统如何产生引力波？",
                "answer": "双白矮星是紧密双星系统，轨道周期通常为数小时到数分钟。根据广义相对论，这样的系统会辐射引力波，导致轨道能量损失和轨道收缩。频率在 mHz 范围内的双白矮星是空间引力波探测器（如 LISA）的主要目标源。",
                "topic": "双白矮星"
            },
            {
                "question": "磁性白矮星的磁场是如何产生的？",
                "answer": "磁性白矮星的强磁场（10^3 - 10^9 G）可能来源于：1) 原恒星磁场的冻结和放大；2) 双星并合过程中发电机效应；3) 氦核闪期间的湍流发电机。约 10-20% 的白矮星表现出可探测的磁场。",
                "topic": "磁性白矮星"
            },
            {
                "question": "吸积白矮星的新星爆发机制是什么？",
                "answer": "当白矮星从伴星吸积物质时，氢层在表面积累。当压力和温度达到临界值时，发生失控的热核聚变（CNO 循环），产生亮度增加 10^4 - 10^6 倍的新星爆发。根据吸积率不同，可能表现为经典新星、再发新星或矮新星。",
                "topic": "吸积白矮星"
            },
        ]
        
        training_data = []
        for qa in template_qa:
            conversation = {
                "messages": [
                    {"role": "system", "content": "你是白矮星天体物理学专家。"},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "metadata": {"topic": qa["topic"], "type": "template_qa"}
            }
            training_data.append(conversation)
        
        return training_data
    
    def process_user_papers(self, pdf_dir: str) -> List[Dict]:
        """
        处理用户上传的论文 PDF
        
        Args:
            pdf_dir: PDF 文件目录
            
        Returns:
            训练数据
        """
        import glob
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"在 {pdf_dir} 中未找到 PDF 文件")
            return []
        
        print(f"\n处理用户论文: {len(pdf_files)} 篇")
        print("=" * 60)
        
        # 这里可以集成 PDF 解析和 Kimi 处理
        # 简化版本：提示用户提供已解析的文本
        
        print("注意: 请确保 PDF 文件可以被解析")
        print("如果需要，我可以帮你提取文本并生成训练数据")
        
        return []


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="白矮星论文下载器")
    
    parser.add_argument("--search", action="store_true", help="搜索 arXiv 论文")
    parser.add_argument("--max-results", type=int, default=100, help="每主题最大结果数")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--generate-data", action="store_true", help="生成训练数据")
    parser.add_argument("--use-kimi", action="store_true", help="使用 Kimi API")
    parser.add_argument("--output-dir", type=str, default="./data/white_dwarf_papers")
    parser.add_argument("--user-papers", type=str, help="用户论文目录")
    
    args = parser.parse_args()
    
    downloader = WhiteDwarfPaperDownloader(args.output_dir)
    
    if args.search:
        # 搜索论文
        print("=" * 60)
        print("白矮星论文搜索")
        print("=" * 60)
        
        downloader.search_comprehensive(
            max_results_per_topic=args.max_results,
            start_date=args.start_date
        )
        
        # 保存论文数据
        downloader.save_papers("wd_papers.json")
    
    if args.generate_data:
        # 生成训练数据
        if not downloader.papers:
            # 尝试加载已保存的论文
            papers_path = Path(args.output_dir) / "wd_papers.json"
            if papers_path.exists():
                with open(papers_path, 'r') as f:
                    papers_data = json.load(f)
                    downloader.papers = [Paper(**p) for p in papers_data]
                print(f"加载 {len(downloader.papers)} 篇已保存论文")
        
        if args.use_kimi:
            # 使用 Kimi 生成
            training_data = downloader.generate_training_data_with_kimi(max_papers=30)
            
            # 同时生成合成数据
            synthetic_data = downloader.generate_synthetic_qa(use_kimi=True)
            training_data.extend(synthetic_data)
        else:
            # 使用模板
            training_data = downloader.generate_synthetic_qa(use_kimi=False)
        
        # 保存训练数据
        output_path = Path(args.output_dir) / "wd_training_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 训练数据已保存: {output_path}")
    
    if args.user_papers:
        # 处理用户论文
        downloader.process_user_papers(args.user_papers)
    
    if not any([args.search, args.generate_data, args.user_papers]):
        parser.print_help()


if __name__ == "__main__":
    main()
