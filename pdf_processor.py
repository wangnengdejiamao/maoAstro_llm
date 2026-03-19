#!/usr/bin/env python3
"""
PDF处理器 - 提取PDF内容并使用API处理
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import pdfplumber
import re

# API配置
API_KEY = "19cccf25-f532-861b-8000-000042a859dc"
API_BASE_URL = "https://api.moonshot.cn/v1"  # Kimi API

class PDFProcessor:
    def __init__(self, pdf_dir, output_dir, api_key):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path):
        """从PDF提取文本内容"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 只提取前20页，避免太长
                for i, page in enumerate(pdf.pages[:20]):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"提取PDF失败 {pdf_path}: {e}")
            return None
        return text
    
    def get_cache_path(self, pdf_path):
        """获取缓存文件路径"""
        file_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.json"
    
    def call_api(self, text, max_retries=3):
        """调用API处理文本"""
        # 截取文本，避免超过API限制
        truncated_text = text[:8000] if len(text) > 8000 else text
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""请分析以下天文学论文的内容，提取以下信息并以JSON格式返回：
1. title: 论文标题
2. authors: 作者列表
3. abstract: 摘要（如果没有找到，总结前3段内容）
4. keywords: 关键词（3-5个）
5. main_findings: 主要发现（2-3句话）
6. objects_studied: 研究的天体对象（如白矮星、灾变变星等）
7. methods: 使用的观测或分析方法
8. year: 发表年份（如果可以从文本推断）

论文内容：
{truncated_text}

请只返回JSON格式，不要其他解释：
{{
    "title": "...",
    "authors": ["..."],
    "abstract": "...",
    "keywords": ["..."],
    "main_findings": "...",
    "objects_studied": ["..."],
    "methods": ["..."],
    "year": "..."
}}"""

        payload = {
            "model": "moonshot-v1-8k",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 提取JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return None
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        return None
    
    def process_pdf(self, pdf_path):
        """处理单个PDF"""
        cache_path = self.get_cache_path(pdf_path)
        
        # 检查缓存
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            return None
        
        # 调用API
        result = self.call_api(text)
        if result:
            result["source_file"] = str(pdf_path.name)
            result["full_text"] = text[:5000]  # 保存部分原文用于训练
            
            # 保存缓存
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result
        return None
    
    def process_all_pdfs(self, limit=None):
        """处理所有PDF"""
        pdf_files = [f for f in self.pdf_dir.glob("*.pdf") if not f.name.startswith("._")]
        print(f"找到 {len(pdf_files)} 个有效PDF文件 (已跳过隐藏文件)")
        if limit:
            pdf_files = pdf_files[:limit]
            print(f"限制处理前 {limit} 个文件")
        
        results = []
        for pdf_path in tqdm(pdf_files, desc="处理PDF"):
            result = self.process_pdf(pdf_path)
            if result:
                results.append(result)
            time.sleep(0.5)  # 避免API限流
        
        # 保存结果
        output_file = self.output_dir / "processed_pdfs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n成功处理 {len(results)} 个PDF，结果保存到 {output_file}")
        return results

if __name__ == "__main__":
    import sys
    # 可以设置处理数量限制，例如: python pdf_processor.py 50
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    processor = PDFProcessor(
        pdf_dir="data/pdfs",
        output_dir="output",
        api_key=API_KEY
    )
    processor.process_all_pdfs(limit=limit)
