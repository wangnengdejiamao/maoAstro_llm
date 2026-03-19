#!/usr/bin/env python3
"""
PDF 处理器
功能：
1. 扫描 F:\storage 目录下的所有 PDF
2. 去重（基于文件名、文件大小、内容哈希）
3. 提取 PDF 元数据和文本
4. 使用 Kimi 提取结构化信息
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import shutil


@dataclass
class PDFInfo:
    """PDF 文件信息"""
    filepath: str
    filename: str
    file_size: int
    md5_hash: str
    content_hash: str  # 前 10 页内容的哈希
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    arxiv_id: str = ""
    year: int = 0
    category: str = ""  # white_dwarf, binary, cv, etc.
    processed: bool = False
    extracted_text: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PDFScanner:
    """
    PDF 文件扫描器
    扫描指定目录下的所有 PDF 文件
    """
    
    def __init__(self, root_dir: str = r"F:\storage"):
        self.root_dir = Path(root_dir)
        self.pdf_files: List[Path] = []
        
    def scan(self, extensions: List[str] = [".pdf"]) -> List[Path]:
        """
        递归扫描所有 PDF 文件
        
        Args:
            extensions: 文件扩展名列表
            
        Returns:
            PDF 文件路径列表
        """
        print(f"扫描目录: {self.root_dir}")
        
        if not self.root_dir.exists():
            print(f"⚠ 目录不存在: {self.root_dir}")
            return []
        
        self.pdf_files = []
        
        for ext in extensions:
            pattern = f"**/*{ext}"
            files = list(self.root_dir.glob(pattern))
            self.pdf_files.extend(files)
        
        # 去重（同一文件的不同路径）
        seen_paths = set()
        unique_files = []
        for f in self.pdf_files:
            real_path = f.resolve()
            if real_path not in seen_paths:
                seen_paths.add(real_path)
                unique_files.append(f)
        
        self.pdf_files = unique_files
        
        print(f"✓ 找到 {len(self.pdf_files)} 个唯一 PDF 文件")
        return self.pdf_files
    
    def get_directory_stats(self) -> Dict:
        """获取目录统计信息"""
        if not self.pdf_files:
            self.scan()
        
        stats = {
            "total_files": len(self.pdf_files),
            "total_size_mb": sum(f.stat().st_size for f in self.pdf_files) / (1024 * 1024),
            "directories": len(set(f.parent for f in self.pdf_files)),
            "by_directory": {}
        }
        
        for f in self.pdf_files:
            dir_name = str(f.parent.relative_to(self.root_dir))
            if dir_name not in stats["by_directory"]:
                stats["by_directory"][dir_name] = 0
            stats["by_directory"][dir_name] += 1
        
        return stats


class PDFDeduplicator:
    """
    PDF 去重器
    基于多种策略去重
    """
    
    def __init__(self):
        self.pdf_infos: List[PDFInfo] = []
        
    def compute_file_hash(self, filepath: Path) -> str:
        """计算文件 MD5 哈希"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def compute_content_hash(self, filepath: Path, max_pages: int = 10) -> str:
        """
        计算内容哈希（基于前几页文本）
        用于检测内容相同但格式不同的 PDF
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(filepath)
            text = ""
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text += page.get_text()
            doc.close()
            
            # 清理文本（去除空格、换行、转为小写）
            text = re.sub(r'\s+', '', text.lower())
            return hashlib.md5(text[:5000].encode()).hexdigest()
        except Exception as e:
            print(f"  内容哈希计算失败: {e}")
            return ""
    
    def extract_pdf_metadata(self, filepath: Path) -> Dict:
        """提取 PDF 元数据"""
        metadata = {
            "title": "",
            "authors": [],
            "abstract": "",
            "doi": "",
            "arxiv_id": ""
        }
        
        try:
            import fitz
            
            doc = fitz.open(filepath)
            
            # 从元数据提取
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata["title"] = pdf_metadata.get("title", "")
            
            # 尝试从第一页提取更多信息
            if len(doc) > 0:
                first_page_text = doc[0].get_text()
                
                # 提取 arXiv ID
                arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', first_page_text)
                if arxiv_match:
                    metadata["arxiv_id"] = arxiv_match.group(1)
                
                # 提取 DOI
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', first_page_text)
                if doi_match:
                    metadata["doi"] = doi_match.group(0)
            
            doc.close()
            
        except Exception as e:
            print(f"  元数据提取失败: {e}")
        
        return metadata
    
    def deduplicate(self, pdf_files: List[Path], 
                   use_content_hash: bool = True) -> Tuple[List[PDFInfo], List[PDFInfo]]:
        """
        去重处理
        
        Args:
            pdf_files: PDF 文件路径列表
            use_content_hash: 是否使用内容哈希去重
            
        Returns:
            (唯一文件列表, 重复文件列表)
        """
        print("\n开始去重...")
        print("=" * 60)
        
        seen_hashes: Set[str] = set()
        seen_content_hashes: Set[str] = set()
        
        unique_files = []
        duplicate_files = []
        
        for i, filepath in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {filepath.name[:50]}")
            
            # 计算文件哈希
            file_hash = self.compute_file_hash(filepath)
            
            if file_hash in seen_hashes:
                print(f"  ⚠ 重复文件（相同文件）")
                info = PDFInfo(
                    filepath=str(filepath),
                    filename=filepath.name,
                    file_size=filepath.stat().st_size,
                    md5_hash=file_hash,
                    content_hash=""
                )
                duplicate_files.append(info)
                continue
            
            seen_hashes.add(file_hash)
            
            # 计算内容哈希
            content_hash = ""
            if use_content_hash:
                content_hash = self.compute_content_hash(filepath)
                if content_hash and content_hash in seen_content_hashes:
                    print(f"  ⚠ 重复文件（内容相同）")
                    info = PDFInfo(
                        filepath=str(filepath),
                        filename=filepath.name,
                        file_size=filepath.stat().st_size,
                        md5_hash=file_hash,
                        content_hash=content_hash
                    )
                    duplicate_files.append(info)
                    continue
                
                if content_hash:
                    seen_content_hashes.add(content_hash)
            
            # 提取元数据
            metadata = self.extract_pdf_metadata(filepath)
            
            info = PDFInfo(
                filepath=str(filepath),
                filename=filepath.name,
                file_size=filepath.stat().st_size,
                md5_hash=file_hash,
                content_hash=content_hash,
                title=metadata.get("title", ""),
                doi=metadata.get("doi", ""),
                arxiv_id=metadata.get("arxiv_id", "")
            )
            
            unique_files.append(info)
            print(f"  ✓ 唯一文件")
        
        print(f"\n{'='*60}")
        print(f"去重结果:")
        print(f"  唯一文件: {len(unique_files)}")
        print(f"  重复文件: {len(duplicate_files)}")
        
        return unique_files, duplicate_files
    
    def save_index(self, pdf_infos: List[PDFInfo], output_path: str):
        """保存 PDF 索引"""
        data = [info.to_dict() for info in pdf_infos]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 索引已保存: {output_path}")
    
    def load_index(self, index_path: str) -> List[PDFInfo]:
        """加载 PDF 索引"""
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [PDFInfo(**info) for info in data]


class PDFTextExtractor:
    """
    PDF 文本提取器
    提取 PDF 文本内容用于后续处理
    """
    
    def __init__(self):
        pass
    
    def extract_text(self, filepath: str, max_pages: Optional[int] = None) -> str:
        """
        提取 PDF 文本
        
        Args:
            filepath: PDF 文件路径
            max_pages: 最大提取页数（None 表示全部）
            
        Returns:
            提取的文本
        """
        try:
            import fitz
            
            doc = fitz.open(filepath)
            text_parts = []
            
            total_pages = len(doc)
            pages_to_extract = min(total_pages, max_pages) if max_pages else total_pages
            
            for i in range(pages_to_extract):
                page = doc[i]
                text_parts.append(page.get_text())
            
            doc.close()
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"文本提取失败 {filepath}: {e}")
            return ""
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        提取论文各部分
        
        识别：Abstract, Introduction, Methods, Results, Discussion, Conclusion
        """
        sections = {}
        
        # 定义章节模式
        section_patterns = [
            ("abstract", r'(?i)(abstract|摘要)\s*[\n\r]'),
            ("introduction", r'(?i)(1\.?\s+)?(introduction|简介|引言)\s*[\n\r]'),
            ("methods", r'(?i)(2\.?\s+)?(methods?|methodology|方法)\s*[\n\r]'),
            ("results", r'(?i)(3\.?\s+)?(results?|结果)\s*[\n\r]'),
            ("discussion", r'(?i)(4\.?\s+)?(discussion|讨论)\s*[\n\r]'),
            ("conclusion", r'(?i)(5\.?\s+)?(conclusion|conclusions|结论)\s*[\n\r]'),
        ]
        
        # 找到所有章节位置
        section_positions = []
        for section_name, pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                section_positions.append((match.start(), section_name))
        
        # 排序并提取
        section_positions.sort()
        
        for i, (pos, name) in enumerate(section_positions):
            start = pos
            end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
            sections[name] = text[start:end].strip()
        
        return sections
    
    def extract_tables(self, filepath: str) -> List[Dict]:
        """
        提取 PDF 中的表格
        特别关注天体源数据表
        """
        tables = []
        
        try:
            import fitz
            import pandas as pd
            
            doc = fitz.open(filepath)
            
            for page_num, page in enumerate(doc):
                # 查找表格
                tabs = page.find_tables()
                
                if tabs.tables:
                    for tab in tabs.tables:
                        df = tab.to_pandas()
                        
                        # 检查是否包含天体源信息
                        cols = ' '.join(df.columns.astype(str)).lower()
                        if any(kw in cols for kw in ['ra', 'dec', 'period', 'magnitude', 'source', 'name']):
                            tables.append({
                                "page": page_num + 1,
                                "data": df.to_dict(orient='records'),
                                "columns": list(df.columns)
                            })
            
            doc.close()
            
        except Exception as e:
            print(f"表格提取失败: {e}")
        
        return tables


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF 处理器")
    parser.add_argument("--scan", action="store_true", help="扫描 PDF 文件")
    parser.add_argument("--dedup", action="store_true", help="去重")
    parser.add_argument("--extract", action="store_true", help="提取文本")
    parser.add_argument("--input-dir", type=str, default=r"F:\storage", help="输入目录")
    parser.add_argument("--output-dir", type=str, default="./pdf_library", help="输出目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDF 处理器")
    print("=" * 60)
    
    # 扫描
    scanner = PDFScanner(args.input_dir)
    pdf_files = scanner.scan()
    
    if args.scan:
        stats = scanner.get_directory_stats()
        print(f"\n统计信息:")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  总大小: {stats['total_size_mb']:.1f} MB")
        print(f"  目录数: {stats['directories']}")
        print(f"\n按目录分布:")
        for dir_name, count in sorted(stats['by_directory'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {dir_name}: {count} 个文件")
    
    if args.dedup:
        # 去重
        deduplicator = PDFDeduplicator()
        unique_files, duplicate_files = deduplicator.deduplicate(pdf_files)
        
        # 保存索引
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        deduplicator.save_index(unique_files, output_dir / "unique_pdfs.json")
        deduplicator.save_index(duplicate_files, output_dir / "duplicate_pdfs.json")
    
    if args.extract:
        # 提取文本
        extractor = PDFTextExtractor()
        
        output_dir = Path(args.output_dir) / "extracted_text"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载去重后的索引
        index_path = Path(args.output_dir) / "unique_pdfs.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                unique_files = [PDFInfo(**info) for info in json.load(f)]
        else:
            unique_files = [PDFInfo(str(p), p.name, p.stat().st_size, "") for p in pdf_files]
        
        for i, info in enumerate(unique_files[:5], 1):  # 只处理前 5 个用于测试
            print(f"\n[{i}/5] 提取: {info.filename[:40]}")
            
            text = extractor.extract_text(info.filepath, max_pages=20)
            
            # 保存文本
            text_file = output_dir / f"{info.filename}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"  ✓ 提取 {len(text)} 字符")


if __name__ == "__main__":
    main()
