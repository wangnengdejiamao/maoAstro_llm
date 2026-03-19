#!/usr/bin/env python3
"""
天文知识文档问答对生成器
为每个文档生成至少20个详细问答对
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# API配置 - 使用用户提供的4个API密钥
API_KEYS = [
    # 请在此处填入你的API Keys
    # 可以从环境变量读取: os.getenv("MOONSHOT_API_KEYS", "").split(",")
]

BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "kimi-k2-07132-preview"

# 文档路径
DOCS = {
    "amcvn": "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/amcvn.txt",
    "binary_systems": "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/binary_systems.txt",
    "cataclysmic_variables": "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/cataclysmic_variables.txt",
    "period_luminosity": "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/period_luminosity_relations.txt",
}

OUTPUT_DIR = Path("/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge/qa_output")


def read_doc(doc_path):
    """读取文档内容"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_qa_prompt(doc_name, doc_content, start_idx=1):
    """创建生成问答对的提示词"""
    prompts = {
        "amcvn": f"""请基于以下AM CVn型星文档内容，生成第{start_idx}到第{start_idx+4}个详细的问答对（共5个）。

文档内容：
{doc_content}

要求：
1. 每个问答对必须包含：问题、详细答案、相关文档引用
2. 问题要涵盖以下方面：
   - 赫罗图上的位置和特点
   - SED（光谱能量分布）特征
   - 光变曲线类型和特点
   - 轨道周期范围及物理意义
   - X射线辐射机制
   - 光谱特征（发射线、连续谱）
   - 与其他双星系统的区别
   - 引力波辐射特性
   - 吸积盘物理
   - 爆发和静默态
3. 答案必须详细，包含物理机制和观测特征
4. 明确引用文档中的具体信息
5. 输出格式为JSON数组

输出格式示例：
[
  {{
    "id": {start_idx},
    "question": "问题内容",
    "answer": "详细答案",
    "category": "分类（如：光变曲线/X射线/光谱等）",
    "source_doc": "amcvn.txt",
    "key_points": ["要点1", "要点2"]
  }}
]
""",
        "binary_systems": f"""请基于以下双星系统文档内容，生成第{start_idx}到第{start_idx+4}个详细的问答对（共5个）。

文档内容：
{doc_content}

要求：
1. 每个问答对必须包含：问题、详细答案、相关文档引用
2. 问题要涵盖以下方面：
   - 双星系统的分类（观测和物理）
   - 不同致密双星的组成（CV、AM CVn、X射线双星）
   - 光变的原因（正确vs错误解释）
   - 潮汐相互作用的真实作用
   - 吸积物理过程
   - 赫罗图上的演化
   - 质量转移机制
   - 轨道周期变化
   - 常见误解辨析
3. 答案必须详细，包含物理机制和观测特征
4. 明确引用文档中的具体信息
5. 输出格式为JSON数组

输出格式示例：
[
  {{
    "id": {start_idx},
    "question": "问题内容",
    "answer": "详细答案",
    "category": "分类",
    "source_doc": "binary_systems.txt",
    "key_points": ["要点1", "要点2"]
  }}
]
""",
        "cataclysmic_variables": f"""请基于以下激变变星文档内容，生成第{start_idx}到第{start_idx+4}个详细的问答对（共5个）。

文档内容：
{doc_content}

要求：
1. 每个问答对必须包含：问题、详细答案、相关文档引用
2. 问题要涵盖以下方面：
   - 系统组成和参数
   - 质量转移机制（磁制动、引力波辐射）
   - 分类（非磁性、磁性CV）
   - 吸积盘物理和不稳定性
   - 光变曲线特征（爆发、闪烁）
   - 光谱特征（发射线、双峰值）
   - 周期隙和演化
   - X射线辐射
   - 与其他系统的区别
3. 答案必须详细，包含物理机制和观测特征
4. 明确引用文档中的具体信息
5. 输出格式为JSON数组

输出格式示例：
[
  {{
    "id": {start_idx},
    "question": "问题内容",
    "answer": "详细答案",
    "category": "分类",
    "source_doc": "cataclysmic_variables.txt",
    "key_points": ["要点1", "要点2"]
  }}
]
""",
        "period_luminosity": f"""请基于以下周光关系文档内容，生成第{start_idx}到第{start_idx+4}个详细的问答对（共5个）。

文档内容：
{doc_content}

要求：
1. 每个问答对必须包含：问题、详细答案、相关文档引用
2. 问题要涵盖以下方面：
   - 造父变星的周光关系及应用
   - RR Lyrae变星的特点
   - 米拉变星的周光关系
   - δ Scuti变星
   - AM CVn系统的统计关系
   - 食双星为什么没有内在周光关系
   - 周光关系在距离测量中的应用
   - 常见错误辨析
   - 不同波段的关系差异
3. 答案必须详细，包含公式、数值和物理意义
4. 明确引用文档中的具体信息
5. 输出格式为JSON数组

输出格式示例：
[
  {{
    "id": {start_idx},
    "question": "问题内容",
    "answer": "详细答案",
    "category": "分类",
    "source_doc": "period_luminosity_relations.txt",
    "key_points": ["要点1", "要点2"]
  }}
]
""",
    }
    return prompts.get(doc_name, "")


def call_api(api_key, prompt, max_retries=3):
    """调用API生成问答对"""
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的天文学专家，擅长生成高质量的问答对用于模型训练。请严格按照要求的JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000,
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
                print(f"JSON解析错误: {e}")
                print(f"原始内容: {content[:500]}...")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return []
                
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []
    
    return []


def process_doc_batch(args):
    """处理文档的一个批次"""
    doc_name, doc_path, api_key, batch_idx = args
    
    print(f"[处理] {doc_name} - 批次 {batch_idx} (问题 {batch_idx*5+1}-{(batch_idx+1)*5})")
    
    doc_content = read_doc(doc_path)
    prompt = create_qa_prompt(doc_name, doc_content, batch_idx * 5 + 1)
    
    qa_list = call_api(api_key, prompt)
    
    if qa_list:
        print(f"[完成] {doc_name} - 批次 {batch_idx}: 生成 {len(qa_list)} 个问答对")
    else:
        print(f"[失败] {doc_name} - 批次 {batch_idx}")
    
    return doc_name, batch_idx, qa_list


def generate_qa_for_doc(doc_name, doc_path, num_questions=20):
    """为单个文档生成问答对"""
    print(f"\n{'='*60}")
    print(f"开始处理文档: {doc_name}")
    print(f"目标: {num_questions} 个问答对")
    print(f"{'='*60}")
    
    all_qa = []
    num_batches = (num_questions + 4) // 5  # 每批5个问题
    
    # 准备任务列表
    tasks = []
    for i in range(num_batches):
        api_key = API_KEYS[i % len(API_KEYS)]
        tasks.append((doc_name, doc_path, api_key, i))
    
    # 并行处理，最多4个并发
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_doc_batch, task) for task in tasks]
        
        for future in as_completed(futures):
            doc, batch_idx, qa_list = future.result()
            if qa_list:
                all_qa.extend(qa_list)
    
    # 按id排序
    all_qa.sort(key=lambda x: x.get('id', 0))
    
    print(f"[汇总] {doc_name}: 共生成 {len(all_qa)} 个问答对")
    
    return all_qa


def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for doc_name, doc_path in DOCS.items():
        qa_list = generate_qa_for_doc(doc_name, doc_path, num_questions=20)
        all_results[doc_name] = qa_list
        
        # 保存单个文档的结果
        output_file = OUTPUT_DIR / f"{doc_name}_qa.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)
        print(f"[保存] {output_file}")
    
    # 保存汇总结果
    summary_file = OUTPUT_DIR / "all_qa_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 生成训练格式
    train_data = []
    for doc_name, qa_list in all_results.items():
        for qa in qa_list:
            train_data.append({
                "instruction": qa.get("question", ""),
                "input": "",
                "output": qa.get("answer", ""),
                "source_doc": qa.get("source_doc", doc_name),
                "category": qa.get("category", ""),
                "key_points": qa.get("key_points", [])
            })
    
    train_file = OUTPUT_DIR / "training_dataset.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("处理完成统计:")
    print(f"{'='*60}")
    total = 0
    for doc_name, qa_list in all_results.items():
        count = len(qa_list)
        total += count
        print(f"  {doc_name}: {count} 个问答对")
    print(f"  总计: {total} 个问答对")
    print(f"{'='*60}")
    print(f"输出文件:")
    print(f"  - {summary_file}")
    print(f"  - {train_file}")
    for doc_name in DOCS.keys():
        print(f"  - {OUTPUT_DIR / f'{doc_name}_qa.json'}")


if __name__ == "__main__":
    main()
