#!/usr/bin/env python3
"""
提取PDF论文的文本数据，保存为结构化JSON
等API可用时再批量生成问答对
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pdfplumber
from concurrent.futures import ThreadPoolExecutor

# 路径配置
PAPERS_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_from_storage")
OUTPUT_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_extracted_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_pdf_full(pdf_path):
    """完整提取PDF信息"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            # 提取所有页面文本
            pages_content = []
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:  # 过滤空页
                        pages_content.append({
                            "page_num": i + 1,
                            "text": text,
                            "length": len(text)
                        })
                except Exception as e:
                    print(f"    第{i+1}页提取失败: {e}")
            
            # 提取元数据
            first_page_text = pages_content[0]["text"] if pages_content else ""
            
            # 尝试从第一行提取标题
            title = "Unknown"
            if first_page_text:
                lines = first_page_text.split('\n')
                for line in lines[:5]:
                    if len(line) > 10 and not line.startswith('arXiv'):
                        title = line.strip()
                        break
            
            return {
                "filename": pdf_path.name,
                "title": title,
                "num_pages": num_pages,
                "extracted_pages": len(pages_content),
                "first_page": first_page_text[:2000],
                "pages": pages_content[:20],  # 只保存前20页避免文件过大
                "full_text_length": sum(p["length"] for p in pages_content),
                "extraction_time": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"[错误] 无法处理 {pdf_path.name}: {e}")
        return None

def main():
    """主函数"""
    print("="*80)
    print("PDF论文数据提取系统")
    print("="*80)
    
    # 获取所有PDF
    pdf_files = [f for f in PAPERS_DIR.glob("*.pdf") if f.name != "EXTRACTION_REPORT.txt"]
    total = len(pdf_files)
    
    print(f"发现PDF文件: {total} 个\n")
    
    # 检查已处理的文件
    processed = set()
    if (OUTPUT_DIR / "index.json").exists():
        with open(OUTPUT_DIR / "index.json", 'r', encoding='utf-8') as f:
            index = json.load(f)
            processed = set(index.get("processed", []))
    
    remaining = [f for f in pdf_files if f.name not in processed]
    print(f"已处理: {len(processed)}")
    print(f"待处理: {len(remaining)}\n")
    
    if not remaining:
        print("✓ 所有文件已提取完成!")
        return
    
    # 分批处理
    batch_size = 50  # 每批处理50个
    current_batch = remaining[:batch_size]
    
    print(f"本批次处理: {len(current_batch)} 个PDF\n")
    
    results = []
    for i, pdf_path in enumerate(current_batch, 1):
        print(f"[{i}/{len(current_batch)}] 提取: {pdf_path.name[:60]}...")
        
        data = extract_pdf_full(pdf_path)
        if data:
            # 保存单个文件
            output_file = OUTPUT_DIR / f"{pdf_path.name.replace('.pdf', '').replace(' ', '_')[:50]}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 成功! 页数: {data['num_pages']}, 提取页数: {data['extracted_pages']}")
            results.append({
                "filename": pdf_path.name,
                "title": data['title'],
                "num_pages": data['num_pages'],
                "output_file": output_file.name
            })
            processed.add(pdf_path.name)
        else:
            print(f"  ✗ 失败")
    
    # 保存索引
    index = {
        "total_pdfs": total,
        "processed": list(processed),
        "results": results,
        "last_update": datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / "index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("提取完成!")
    print(f"本批次成功: {len(results)}/{len(current_batch)}")
    print(f"总计已处理: {len(processed)}/{total}")
    print(f"数据保存在: {OUTPUT_DIR}")
    print("="*80)
    
    if len(remaining) > batch_size:
        print(f"\n还有 {len(remaining) - batch_size} 个文件待处理")
        print("可以再次运行此脚本继续处理")

if __name__ == "__main__":
    main()
