#!/usr/bin/env python3
"""
使用本地Ollama处理PDF论文
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import pdfplumber
import requests

# Ollama配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "kimi-k2-07132-preview:latest"  # 或其他可用模型如 llama2, qwen等

# 路径配置
PAPERS_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_from_storage")
OUTPUT_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_qa_output_ollama")
OUTPUT_DIR.mkdir(exist_ok=True)

def check_ollama():
    """检查Ollama是否运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✓ Ollama运行中")
            print(f"可用模型: {[m['name'] for m in models]}")
            return True
    except:
        pass
    print("✗ Ollama未运行或未安装")
    print("请运行: ollama serve")
    return False

def extract_pdf_info(pdf_path):
    """提取PDF信息"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            # 提取文本
            sample_text = ""
            for i in range(min(8, num_pages)):
                try:
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        sample_text += f"\n--- Page {i+1} ---\n{text}\n"
                except:
                    continue
            
            return {
                "filename": pdf_path.name,
                "num_pages": num_pages,
                "sample_text": sample_text[:6000]
            }
    except Exception as e:
        print(f"[错误] 无法读取PDF: {e}")
        return None

def generate_qa_with_ollama(pdf_info, num_questions=20):
    """使用Ollama生成问答对"""
    
    prompt = f"""你是一位天文学专家。请基于以下论文内容生成{num_questions}个问答对。

论文: {pdf_info['filename']}
页数: {pdf_info['num_pages']}

论文内容摘要:
{pdf_info['sample_text'][:4000]}

请生成{num_questions}个问答对，涵盖以下方面:
1. 赫罗图位置特征
2. SED/光谱能量分布
3. 光变曲线类型
4. 周期特征
5. X射线辐射
6. 光谱特征
7. 物理机制
8. 观测特征

每个问答对必须包含:
- 问题（专业详细）
- 答案（至少150字，包含物理机制和定量数据）
- 分类标签

输出格式（JSON数组）:
[
  {{"question": "...", "answer": "...", "category": "...", "key_points": ["..."]}},
  ...
]

只输出JSON格式，不要有其他说明。"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            
            # 提取JSON
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()
                
                qa_list = json.loads(json_str)
                return qa_list
            except:
                return None
        else:
            print(f"  错误: {response.status_code}")
            return None
    except Exception as e:
        print(f"  异常: {e}")
        return None

def main():
    """主函数"""
    print("="*80)
    print("PDF论文处理系统 (Ollama版本)")
    print("="*80)
    
    # 检查Ollama
    if not check_ollama():
        return
    
    # 获取PDF文件
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))[:5]  # 先处理5个测试
    print(f"\n待处理PDF: {len(pdf_files)} 个")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 处理: {pdf_path.name}")
        
        # 提取信息
        pdf_info = extract_pdf_info(pdf_path)
        if not pdf_info:
            continue
        
        print(f"  页数: {pdf_info['num_pages']}")
        
        # 生成问答对
        qa_list = generate_qa_with_ollama(pdf_info, num_questions=20)
        
        if qa_list:
            # 添加元数据
            for qa in qa_list:
                qa["source_pdf"] = pdf_path.name
                qa["num_pages"] = pdf_info['num_pages']
            
            # 保存
            output_file = OUTPUT_DIR / f"{pdf_path.name.replace('.pdf', '')}_qa.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 成功! 生成 {len(qa_list)} 个问答对")
        else:
            print(f"  ✗ 失败")
        
        time.sleep(2)

if __name__ == "__main__":
    main()
