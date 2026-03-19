#!/usr/bin/env python3
"""
PDF数据集生成器 - 提取PDF文本并创建训练数据集
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import pdfplumber
import random

class PDFDatasetBuilder:
    def __init__(self, pdf_dir, output_dir):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path, max_pages=15):
        """从PDF提取文本内容"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            return None
        return text.strip()
    
    def parse_paper_info(self, text, filename):
        """从文本解析论文信息"""
        lines = text.split('\n')
        
        # 提取标题（通常是前3行中非空的最长行）
        title = ""
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and len(line) > len(title):
                title = line
        
        # 提取摘要（查找Abstract后面的内容）
        abstract = ""
        abstract_patterns = [
            r'Abstract[\s:]*(.+?)(?=\n\s*\n|Introduction|1\s+Introduction)',
            r'ABSTRACT[\s:]*(.+?)(?=\n\s*\n|Introduction|1\s+Introduction)',
        ]
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()[:2000]
                break
        
        # 提取关键词
        keywords = []
        keyword_patterns = [
            r'Keywords[\s:]*(.+?)(?=\n)',
            r'Key words[\s:]*(.+?)(?=\n)',
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                kw_text = match.group(1)
                keywords = [k.strip() for k in re.split(r'[,;]', kw_text) if len(k.strip()) > 2]
                break
        
        # 提取年份
        year = ""
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            year = year_match.group()
        
        # 提取作者（通常在标题附近）
        authors = []
        author_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*)(?=\n|$)'
        for line in lines[1:5]:
            if 'and' in line.lower() or ',' in line:
                potential_authors = re.split(r',|\sand\s', line)
                if len(potential_authors) > 1:
                    authors = [a.strip() for a in potential_authors if len(a.strip()) > 2]
                    break
        
        return {
            "title": title[:500],
            "abstract": abstract,
            "keywords": keywords[:10],
            "year": year,
            "authors": authors[:5],
            "source_file": filename,
            "full_text": text[:8000]
        }
    
    def generate_qa_pairs(self, paper_info):
        """基于论文信息生成QA对"""
        qa_pairs = []
        
        if paper_info["title"]:
            qa_pairs.append({
                "question": f"这篇论文的标题是什么？",
                "answer": paper_info["title"],
                "context": "title",
                "source": paper_info["source_file"]
            })
        
        if paper_info["abstract"]:
            # 摘要相关问题
            qa_pairs.append({
                "question": f"请简述这篇关于论文的主要内容。",
                "answer": paper_info["abstract"][:500],
                "context": "abstract",
                "source": paper_info["source_file"]
            })
            
            # 研究主题问题
            qa_pairs.append({
                "question": f"这篇论文的研究主题是什么？",
                "answer": paper_info["abstract"][:300] if len(paper_info["abstract"]) > 300 else paper_info["abstract"],
                "context": "abstract",
                "source": paper_info["source_file"]
            })
        
        if paper_info["keywords"]:
            qa_pairs.append({
                "question": f"这篇论文涉及哪些关键词？",
                "answer": ", ".join(paper_info["keywords"]),
                "context": "keywords",
                "source": paper_info["source_file"]
            })
        
        if paper_info["year"]:
            qa_pairs.append({
                "question": f"这篇论文发表于哪一年？",
                "answer": paper_info["year"],
                "context": "year",
                "source": paper_info["source_file"]
            })
        
        # 天体物理特定问题
        text_lower = paper_info.get("full_text", "").lower()
        
        # 灾变变星相关问题
        if "cataclysmic" in text_lower or "variable" in text_lower:
            qa_pairs.append({
                "question": "这篇论文研究的灾变变星有什么特性？",
                "answer": paper_info["abstract"][:400] if paper_info["abstract"] else "研究了灾变变星的特性",
                "context": "specific",
                "source": paper_info["source_file"]
            })
        
        # 白矮星相关问题
        if "white dwarf" in text_lower:
            qa_pairs.append({
                "question": "这篇论文关于白矮星的主要发现是什么？",
                "answer": paper_info["abstract"][:400] if paper_info["abstract"] else "研究了白矮星的特性",
                "context": "specific",
                "source": paper_info["source_file"]
            })
        
        # 磁星相关问题
        if "magnetic" in text_lower or "polar" in text_lower:
            qa_pairs.append({
                "question": "这篇论文中提到的磁星或极星有什么特点？",
                "answer": paper_info["abstract"][:400] if paper_info["abstract"] else "研究了磁星的特性",
                "context": "specific",
                "source": paper_info["source_file"]
            })
        
        return qa_pairs
    
    def build_dataset(self, max_pdfs=None):
        """构建数据集"""
        pdf_files = [f for f in self.pdf_dir.glob("*.pdf") if not f.name.startswith("._")]
        print(f"找到 {len(pdf_files)} 个有效PDF文件")
        
        if max_pdfs:
            pdf_files = pdf_files[:max_pdfs]
            print(f"限制处理前 {max_pdfs} 个文件")
        
        all_papers = []
        all_qa_pairs = []
        failed_files = []
        
        for pdf_path in tqdm(pdf_files, desc="处理PDF"):
            text = self.extract_text_from_pdf(pdf_path)
            if text and len(text) > 200:
                paper_info = self.parse_paper_info(text, pdf_path.name)
                all_papers.append(paper_info)
                
                qa_pairs = self.generate_qa_pairs(paper_info)
                all_qa_pairs.extend(qa_pairs)
            else:
                failed_files.append(pdf_path.name)
        
        # 保存论文信息
        papers_file = self.output_dir / "papers_info.json"
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(all_papers, f, ensure_ascii=False, indent=2)
        
        # 保存QA数据集
        qa_file = self.output_dir / "qa_dataset.json"
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 生成训练集和测试集
        random.shuffle(all_qa_pairs)
        split_idx = int(len(all_qa_pairs) * 0.8)
        train_data = all_qa_pairs[:split_idx]
        test_data = all_qa_pairs[split_idx:]
        
        with open(self.output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(self.output_dir / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成!")
        print(f"- 成功处理: {len(all_papers)} 篇论文")
        print(f"- 生成QA对: {len(all_qa_pairs)} 个")
        print(f"- 训练集: {len(train_data)} 个")
        print(f"- 测试集: {len(test_data)} 个")
        print(f"- 失败文件: {len(failed_files)} 个")
        
        return all_papers, all_qa_pairs

if __name__ == "__main__":
    builder = PDFDatasetBuilder(
        pdf_dir="data/pdfs",
        output_dir="output"
    )
    builder.build_dataset(max_pdfs=None)
