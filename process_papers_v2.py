#!/usr/bin/env python3
"""
PDF论文智能问答生成系统 V2
使用用户提供的5个Kimi API Key
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pdfplumber
from openai import OpenAI

# API配置 - 用户提供的5个API Key
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
PROGRESS_FILE = OUTPUT_DIR / "progress_v2.json"

def load_progress():
    """加载处理进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": [], "failed": [], "qa_count": 0, "current_batch": 0}

def save_progress(progress):
    """保存处理进度"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def extract_pdf_info(pdf_path):
    """提取PDF信息"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            # 提取多页文本用于分析
            sample_text = ""
            pages_to_extract = min(10, num_pages)  # 提取前10页
            for i in range(pages_to_extract):
                try:
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        sample_text += f"\n--- Page {i+1} ---\n{text}\n"
                except:
                    continue
            
            # 提取标题（从第一页）
            first_page_text = ""
            try:
                first_page_text = pdf.pages[0].extract_text()[:2000] if pdf.pages else ""
            except:
                pass
            
            return {
                "path": str(pdf_path),
                "filename": pdf_path.name,
                "num_pages": num_pages,
                "sample_text": sample_text[:8000],  # 限制样本大小
                "first_page": first_page_text
            }
    except Exception as e:
        print(f"[错误] 无法读取PDF {pdf_path.name}: {e}")
        return None

def calculate_num_questions(num_pages):
    """根据页数计算问题数量 - 至少20个"""
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

def create_qa_prompt_batch(pdf_info, batch_num, batch_size, total_questions):
    """创建批量生成问答对的提示词"""
    filename = pdf_info['filename']
    num_pages = pdf_info['num_pages']
    sample_text = pdf_info['sample_text']
    first_page = pdf_info['first_page']
    
    start_idx = (batch_num - 1) * batch_size + 1
    end_idx = min(batch_num * batch_size, total_questions)
    actual_batch_size = end_idx - start_idx + 1
    
    # 根据批次分配不同的主题重点
    topic_focus = {
        1: "赫罗图位置、SED/能谱特征、颜色特性",
        2: "光变曲线类型（爆发、调制、闪烁）、周期特征",
        3: "X射线辐射机制、光谱特征（发射线、氦线/氢线）",
        4: "物理机制（吸积、质量转移、引力波）、演化过程",
        5: "观测特征、系统对比、其他重要信息"
    }.get(batch_num, "综合信息")
    
    prompt = f"""请基于以下天文学论文，生成{actual_batch_size}个详细的问答对（编号{start_idx}到{end_idx}）。

【文档信息】
文件名: {filename}
页数: {num_pages}
本批次主题重点: {topic_focus}

【文档开头内容】
{first_page}

【文档样本内容】
{sample_text[:6000]}

【生成要求】

1. **问题数量**: 严格生成{actual_batch_size}个问答对

2. **本批次主题覆盖**:
   - {topic_focus}
   - 每个问题都要有深度，不是简单的事实查询

3. **答案要求**:
   - 每个答案至少200字
   - 包含物理机制解释
   - 包含定量数据、公式或数值
   - 引用文档中的具体信息
   - 如果文档没有相关信息，说明"根据本文档，该信息有限"

4. **分类标签** (必须选择最相关的):
   - 赫罗图 / SED / 光变曲线 / 周期 / X射线 / 光谱 / 
   - 物理机制 / 观测特征 / 演化 / 系统对比

5. **输出格式** (严格JSON数组，不要任何其他内容):
[
  {{
    "id": {start_idx},
    "question": "具体问题",
    "answer": "详细答案（200字以上）",
    "category": "分类名称",
    "key_points": ["要点1", "要点2", "要点3", "要点4"]
  }},
  ...
]

【重要】
- 答案必须基于上述文档内容
- 问题要专业，体现天文学专业知识
- 输出必须是纯JSON格式，可以被json.loads解析
"""
    return prompt, actual_batch_size

def call_kimi_api(api_key, prompt, max_retries=3):
    """调用Kimi API"""
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的天文学专家，擅长分析天文论文并生成高质量的问答对。请严格按照JSON格式输出，不要添加任何额外说明。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=8000,
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
                
                # 清理可能的BOM
                json_str = json_str.lstrip('\ufeff')
                
                qa_list = json.loads(json_str)
                
                # 验证格式
                if isinstance(qa_list, list) and len(qa_list) > 0:
                    return qa_list
                else:
                    print(f"    [警告] 返回格式不正确")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"    [警告] JSON解析错误: {str(e)[:50]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return []
                
        except Exception as e:
            print(f"    [警告] API错误 (尝试 {attempt+1}/{max_retries}): {str(e)[:50]}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return []
    
    return []

def process_pdf_with_api(pdf_path, api_key):
    """使用指定API处理单个PDF"""
    pdf_name = pdf_path.name
    
    print(f"\n[开始处理] {pdf_name}")
    
    # 提取PDF信息
    pdf_info = extract_pdf_info(pdf_path)
    if not pdf_info:
        return pdf_name, 0, "提取失败"
    
    num_pages = pdf_info['num_pages']
    num_questions = calculate_num_questions(num_pages)
    
    print(f"  页数: {num_pages}")
    print(f"  计划生成问题数: {num_questions}")
    
    # 分批生成（每批5-8个问题）
    all_qa = []
    batch_size = 6
    num_batches = (num_questions + batch_size - 1) // batch_size
    
    for batch_num in range(1, num_batches + 1):
        print(f"  [批次 {batch_num}/{num_batches}] ", end="", flush=True)
        
        prompt, actual_size = create_qa_prompt_batch(pdf_info, batch_num, batch_size, num_questions)
        qa_list = call_kimi_api(api_key, prompt)
        
        if qa_list:
            # 添加来源信息
            for qa in qa_list:
                qa["source_pdf"] = pdf_name
                qa["num_pages"] = num_pages
                qa["batch"] = batch_num
            all_qa.extend(qa_list)
            print(f"✓ 生成 {len(qa_list)} 个")
        else:
            print(f"✗ 失败")
        
        # API限流 - 避免过快
        time.sleep(3)
    
    # 保存结果
    if all_qa:
        output_file = OUTPUT_DIR / f"{pdf_name.replace('.pdf', '').replace(' ', '_')[:50]}_qa.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)
        
        print(f"[完成] {pdf_name}: 共 {len(all_qa)} 个问答对")
        return pdf_name, len(all_qa), "成功"
    else:
        print(f"[失败] {pdf_name}")
        return pdf_name, 0, "生成失败"

def main():
    """主函数"""
    print("="*80)
    print("天文论文智能问答生成系统 V2")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API数量: {len(API_KEYS)}")
    print(f"论文目录: {PAPERS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    # 获取所有PDF文件
    pdf_files = sorted([f for f in PAPERS_DIR.glob("*.pdf") if f.name != "EXTRACTION_REPORT.txt"])
    total_pdfs = len(pdf_files)
    
    print(f"\n发现PDF文件: {total_pdfs} 个")
    
    # 加载进度
    progress = load_progress()
    already_processed = set(progress.get("processed", []))
    failed_files = set(progress.get("failed", []))
    
    print(f"已处理: {len(already_processed)} 个")
    print(f"之前失败: {len(failed_files)} 个")
    print(f"已生成问答对: {progress.get('qa_count', 0)} 个")
    
    # 过滤待处理文件
    remaining_files = [f for f in pdf_files if f.name not in already_processed]
    print(f"\n待处理: {len(remaining_files)} 个")
    
    if not remaining_files:
        print("\n✓ 所有文件已处理完成！")
        return
    
    # 选择处理数量
    # 先处理20个作为第一批
    batch_size = min(20, len(remaining_files))
    current_batch = remaining_files[:batch_size]
    
    print(f"\n{'='*80}")
    print(f"本批次处理: {batch_size} 个PDF")
    print(f"{'='*80}\n")
    
    # 分配API Key（轮流使用）
    results = []
    for i, pdf_path in enumerate(current_batch):
        api_key = API_KEYS[i % len(API_KEYS)]
        pdf_name, qa_count, status = process_pdf_with_api(pdf_path, api_key)
        
        results.append({
            "pdf": pdf_name,
            "qa_count": qa_count,
            "status": status
        })
        
        # 更新进度
        if status == "成功":
            progress["processed"].append(pdf_name)
            progress["qa_count"] = progress.get("qa_count", 0) + qa_count
        else:
            progress["failed"].append(pdf_name)
        
        progress["current_batch"] = progress.get("current_batch", 0) + 1
        save_progress(progress)
        
        # 每处理5个暂停一下
        if (i + 1) % 5 == 0:
            print(f"\n--- 已处理 {i+1}/{batch_size} 个，暂停10秒 ---\n")
            time.sleep(10)
    
    # 生成报告
    print("\n" + "="*80)
    print("批次处理完成报告")
    print("="*80)
    
    total_qa = sum(r["qa_count"] for r in results)
    success_count = sum(1 for r in results if r["status"] == "成功")
    
    print(f"成功: {success_count}/{batch_size}")
    print(f"生成问答对: {total_qa} 个")
    print(f"\n详细结果:")
    for r in results:
        status_icon = "✓" if r["status"] == "成功" else "✗"
        print(f"  {status_icon} {r['pdf'][:60]}... - {r['qa_count']}个问答")
    
    # 总体进度
    remaining_count = len(remaining_files) - batch_size
    print(f"\n总体进度:")
    print(f"  已处理: {len(progress['processed'])}/{total_pdfs}")
    print(f"  剩余: {remaining_count}")
    print(f"  总问答对: {progress['qa_count']}")
    
    if remaining_count > 0:
        print(f"\n还有 {remaining_count} 个文件待处理")
        print("建议: 可以继续运行脚本处理下一批")

if __name__ == "__main__":
    main()
