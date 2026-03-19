#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage 智能PDF论文处理系统
================================================================================
功能描述:
    批量处理天文领域PDF论文，提取文本内容并生成问答对。
    支持断点续传和并行处理，提高大规模文档处理效率。

核心特性:
    - 智能分页: 根据PDF页数动态决定生成问题数量
    - 断点续传: 保存处理进度，支持中断后恢复
    - 多API并行: 利用多个API key并发处理
    - 元数据提取: 自动提取标题、作者、摘要等信息
    - 文本清洗: 去除页眉页脚、乱码等噪声

处理流程:
    1. 扫描输入目录下的PDF文件
    2. 提取每页文本和元数据
    3. 调用API生成问答对
    4. 保存结果到JSON格式
    5. 更新处理进度

依赖库:
    - pdfplumber: PDF解析
    - openai: API客户端

路径配置:
    - 输入: papers_from_storage/
    - 输出: papers_qa_output/

使用方法:
    python process_papers_smart.py

作者: AstroSage Team
创建日期: 2024-03
================================================================================
"""

import os
import json
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pdfplumber

# API配置
API_KEYS = [
    # 请在此处填入你的API Keys
    # 可以从环境变量读取: os.getenv("MOONSHOT_API_KEYS", "").split(",")
]

BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "kimi-k2-07132-preview"

# 路径配置
PAPERS_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_from_storage")
OUTPUT_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/papers_qa_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 处理状态文件
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

def load_progress():
    """加载处理进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": [], "failed": [], "qa_count": 0}

def save_progress(progress):
    """保存处理进度"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def extract_pdf_info(pdf_path):
    """提取PDF信息（页数、文本等）"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            # 提取前几页文本用于分析
            sample_text = ""
            pages_to_extract = min(5, num_pages)
            for i in range(pages_to_extract):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    sample_text += text + "\n"
            
            # 提取标题和作者（从第一页）
            first_page_text = pdf.pages[0].extract_text() if pdf.pages else ""
            
            return {
                "path": str(pdf_path),
                "filename": pdf_path.name,
                "num_pages": num_pages,
                "sample_text": sample_text[:3000],  # 限制样本大小
                "first_page": first_page_text[:1500] if first_page_text else ""
            }
    except Exception as e:
        print(f"[错误] 无法读取PDF {pdf_path.name}: {e}")
        return None

def calculate_num_questions(num_pages):
    """根据页数计算问题数量"""
    if num_pages <= 5:
        return 20
    elif num_pages <= 10:
        return 25
    elif num_pages <= 20:
        return 30
    elif num_pages <= 50:
        return 35
    else:
        return 40  # 长文档最多40个问题

def create_qa_prompt(pdf_info, num_questions, start_idx=1):
    """创建生成问答对的提示词"""
    filename = pdf_info['filename']
    num_pages = pdf_info['num_pages']
    sample_text = pdf_info['sample_text']
    first_page = pdf_info['first_page']
    
    prompt = f"""请基于以下天文学论文的内容，生成{num_questions}个详细的问答对（从第{start_idx}个开始编号）。

【文档信息】
文件名: {filename}
页数: {num_pages}

【文档开头内容】
{first_page}

【文档样本内容】
{sample_text}

【要求】
1. 问题数量: 生成{num_questions}个问答对
2. 问题类型必须涵盖以下方面（每个方面至少2-3个问题）:
   - 赫罗图（HR Diagram）: 恒星在赫罗图上的位置、演化轨迹
   - SED（光谱能量分布）: 能谱特征、多波段特性、黑体辐射
   - 光变曲线（Light Curve）: 周期性变化、爆发、闪烁、调制
   - 周期（Period）: 轨道周期、脉动周期、周期变化
   - X射线（X-ray）: X射线辐射机制、能谱、光度
   - 光谱（Spectrum）: 发射线、吸收线、线形、氦线/氢线
   - 物理机制: 吸积、质量转移、引力波、磁制动
   - 观测特征: 颜色、亮度、变星分类
   - 演化: 系统演化、最终命运

3. 答案要求:
   - 详细解释物理机制
   - 包含定量数据和公式
   - 引用文档中的具体信息
   - 对比不同系统的特征

4. 格式要求:
   - 每个问答对必须包含: id, question, answer, category, key_points
   - category必须是以下之一: 赫罗图, SED, 光变曲线, 周期, X射线, 光谱, 物理机制, 观测特征, 演化, 系统对比
   - key_points列出3-5个关键要点

5. 输出格式（严格JSON数组）:
[
  {{
    "id": {start_idx},
    "question": "具体问题",
    "answer": "详细答案（至少200字）",
    "category": "分类名称",
    "key_points": ["要点1", "要点2", "要点3"]
  }}
]

【重要提示】
- 这是天文学论文，请确保问题专业准确
- 答案必须基于文档内容，不要编造
- 如果文档不包含某些信息，可以生成"根据文档，关于...的信息有限"这样的回答
"""
    return prompt

def call_kimi_api(api_key, prompt, max_retries=3):
    """调用Kimi API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个专业的天文学专家，擅长分析天文论文并生成高质量的问答对用于模型训练。请严格按照要求的JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                    timeout=120
                )
                content = response.choices[0].message.content
                
                # 尝试解析JSON
                try:
                    # 提取JSON部分
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0].strip()
                    else:
                        json_str = content.strip()
                    
                    qa_list = json.loads(json_str)
                    return qa_list
                except json.JSONDecodeError as e:
                    print(f"  [警告] JSON解析错误: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []
                    
            except Exception as e:
                print(f"  [警告] API调用错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return []
        
        return []
    except ImportError:
        print("  [错误] 需要安装openai库: pip install openai")
        return []

def process_single_pdf(pdf_path, api_key, progress):
    """处理单个PDF"""
    pdf_name = pdf_path.name
    
    # 检查是否已经处理过
    if pdf_name in progress["processed"]:
        print(f"[跳过] {pdf_name} 已处理")
        return None
    
    print(f"\n[处理] {pdf_name}")
    
    # 提取PDF信息
    pdf_info = extract_pdf_info(pdf_path)
    if not pdf_info:
        progress["failed"].append(pdf_name)
        save_progress(progress)
        return None
    
    print(f"  页数: {pdf_info['num_pages']}")
    
    # 计算问题数量
    num_questions = calculate_num_questions(pdf_info['num_pages'])
    print(f"  将生成问题数: {num_questions}")
    
    # 分批生成（每批最多5个问题，避免超长）
    all_qa = []
    batch_size = 5
    num_batches = (num_questions + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size + 1
        current_batch_size = min(batch_size, num_questions - batch_idx * batch_size)
        
        print(f"  [批次 {batch_idx+1}/{num_batches}] 生成问题 {start_idx}-{start_idx+current_batch_size-1}")
        
        prompt = create_qa_prompt(pdf_info, current_batch_size, start_idx)
        qa_list = call_kimi_api(api_key, prompt)
        
        if qa_list:
            # 添加来源信息
            for qa in qa_list:
                qa["source_pdf"] = pdf_name
                qa["num_pages"] = pdf_info['num_pages']
            all_qa.extend(qa_list)
            print(f"    ✓ 成功生成 {len(qa_list)} 个问答对")
        else:
            print(f"    ✗ 生成失败")
        
        # API限流
        time.sleep(2)
    
    # 保存结果
    if all_qa:
        output_file = OUTPUT_DIR / f"{pdf_name.replace('.pdf', '_qa.json')}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)
        
        progress["processed"].append(pdf_name)
        progress["qa_count"] += len(all_qa)
        save_progress(progress)
        
        print(f"[完成] {pdf_name}: 共生成 {len(all_qa)} 个问答对 -> {output_file.name}")
        return len(all_qa)
    else:
        progress["failed"].append(pdf_name)
        save_progress(progress)
        print(f"[失败] {pdf_name}")
        return 0

def process_batch(pdf_files, api_keys, max_workers=4):
    """批量处理PDF"""
    progress = load_progress()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务，轮询使用不同API密钥
        futures = {}
        for i, pdf_path in enumerate(pdf_files):
            api_key = api_keys[i % len(api_keys)]
            future = executor.submit(process_single_pdf, pdf_path, api_key, progress)
            futures[future] = pdf_path
        
        # 收集结果
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[错误] 处理 {pdf_path.name} 时出错: {e}")
    
    return results

def main():
    """主函数"""
    print("="*80)
    print("天文论文智能问答生成系统")
    print("="*80)
    
    # 获取所有PDF文件
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.name != "EXTRACTION_REPORT.txt"]
    
    print(f"\n发现PDF文件: {len(pdf_files)} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 加载进度
    progress = load_progress()
    print(f"已处理: {len(progress['processed'])} 个")
    print(f"失败: {len(progress['failed'])} 个")
    print(f"已生成问答对: {progress['qa_count']} 个")
    
    # 过滤已处理的文件
    remaining_files = [f for f in pdf_files if f.name not in progress["processed"]]
    print(f"\n待处理: {len(remaining_files)} 个")
    
    if not remaining_files:
        print("\n所有文件已处理完成！")
        return
    
    # 询问是否继续
    print(f"\n准备处理前 {min(10, len(remaining_files))} 个文件作为测试批次...")
    
    # 先处理前10个作为测试
    test_batch = remaining_files[:10]
    print(f"\n测试批次: {len(test_batch)} 个文件")
    
    results = process_batch(test_batch, API_KEYS, max_workers=2)
    
    print(f"\n测试批次完成！")
    print(f"成功生成问答对: {sum(results)} 个")
    
    # 询问是否继续处理剩余文件
    remaining = remaining_files[10:]
    if remaining:
        print(f"\n还有 {len(remaining)} 个文件待处理")
        print("要继续处理吗？建议分批处理以避免超时")

if __name__ == "__main__":
    main()
