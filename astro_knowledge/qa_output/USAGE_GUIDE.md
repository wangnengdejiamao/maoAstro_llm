# 天文知识问答对数据集 - 使用指南

## 快速开始

### 1. 文件列表

本数据集包含以下文件：

#### JSON格式（推荐用于程序处理）

| 文件名 | 用途 | 行数/记录数 |
|--------|------|-------------|
| `all_qa_summary.json` | 所有问答对汇总 | 100个问答对 |
| `amcvn_qa.json` | AM CVn系统问答对 | 25个问答对 |
| `binary_systems_qa.json` | 双星系统问答对 | 25个问答对 |
| `cataclysmic_variables_qa.json` | 激变变星问答对 | 25个问答对 |
| `period_luminosity_qa.json` | 周光关系问答对 | 25个问答对 |
| `training_dataset.json` | 训练格式（标准） | 100条记录 |
| `training_alpaca.json` | 训练格式（Alpaca） | 100条记录 |

#### CSV格式（推荐用于数据分析）

| 文件名 | 用途 | 行数 |
|--------|------|------|
| `all_qa_dataset.csv` | 完整数据集 | 101行（含表头） |
| `training_dataset.csv` | 简化训练格式 | 101行（含表头） |
| `category_statistics.csv` | 分类统计 | 77行（含表头） |
| `keywords_dataset.csv` | 关键词数据 | 451行（含表头） |

#### 文档

| 文件名 | 说明 |
|--------|------|
| `README.md` | 数据集说明文档 |
| `USAGE_GUIDE.md` | 本使用指南 |
| `analysis_report.txt` | 数据分析报告 |

## 使用示例

### 示例1: 加载JSON数据

```python
import json

# 加载单个文档的问答对
with open('amcvn_qa.json', 'r', encoding='utf-8') as f:
    amcvn_qa = json.load(f)

print(f"AM CVn问答对数量: {len(amcvn_qa)}")
print(f"第一个问题: {amcvn_qa[0]['question']}")
print(f"第一个答案摘要: {amcvn_qa[0]['answer'][:100]}...")

# 加载所有问答对
with open('all_qa_summary.json', 'r', encoding='utf-8') as f:
    all_qa = json.load(f)

# 遍历所有文档
for doc_name, qa_list in all_qa.items():
    print(f"{doc_name}: {len(qa_list)} 个问答对")
```

### 示例2: 加载CSV数据

```python
import pandas as pd

# 加载完整数据集
df = pd.read_csv('all_qa_dataset.csv')

# 查看数据统计
print(f"总行数: {len(df)}")
print(f"文档分布:\n{df['doc_name'].value_counts()}")
print(f"分类分布:\n{df['category'].value_counts()}")

# 筛选特定文档的数据
amcvn_df = df[df['doc_name'] == 'amcvn']
print(f"AM CVn数据: {len(amcvn_df)} 行")

# 筛选特定分类的数据
hr_diagram_df = df[df['category'] == '赫罗图']
print(f"赫罗图相关问题: {len(hr_diagram_df)} 个")
```

### 示例3: 转换为训练格式

```python
import json

# 加载标准格式
with open('training_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 转换为OpenAI格式
openai_format = []
for item in train_data:
    openai_format.append({
        "messages": [
            {"role": "system", "content": "你是一个天文学专家。"},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]}
        ]
    })

# 保存
with open('openai_format.jsonl', 'w', encoding='utf-8') as f:
    for item in openai_format:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"已转换 {len(openai_format)} 条记录到OpenAI格式")
```

### 示例4: 构建简单的问答系统

```python
import json
from difflib import SequenceMatcher

def load_qa_data():
    with open('training_dataset.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def find_similar_question(query, qa_data, top_k=3):
    """基于相似度找到最相关的问题"""
    scores = []
    for item in qa_data:
        score = SequenceMatcher(None, query, item['instruction']).ratio()
        scores.append((score, item))
    
    # 排序并返回前k个
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def answer_question(query):
    qa_data = load_qa_data()
    similar = find_similar_question(query, qa_data, top_k=1)
    
    if similar[0][0] > 0.5:  # 相似度阈值
        best_match = similar[0][1]
        return {
            'answer': best_match['output'],
            'source_doc': best_match['source_doc'],
            'category': best_match['category'],
            'similarity': similar[0][0]
        }
    else:
        return {'answer': '抱歉，没有找到相关问题的答案。'}

# 测试
result = answer_question("AM CVn的周期是多少？")
print(f"答案: {result['answer'][:200]}...")
print(f"来源: {result['source_doc']}")
print(f"分类: {result['category']}")
```

### 示例5: 使用Embedding构建向量检索

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载数据
with open('training_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 加载embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 构建corpus
corpus_texts = [item['instruction'] + " " + item['output'] for item in train_data]
corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True)

# 保存embedding
np.save('corpus_embeddings.npy', corpus_embeddings)
with open('corpus_texts.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)

print(f"已生成 {len(corpus_embeddings)} 个embedding向量")

# 检索函数
def search(query, top_k=3):
    query_embedding = model.encode([query])
    
    # 计算相似度
    similarities = np.dot(corpus_embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'question': train_data[idx]['instruction'],
            'answer': train_data[idx]['output'][:200],
            'source': train_data[idx]['source_doc'],
            'score': float(similarities[idx])
        })
    return results

# 测试
results = search("什么是超驼峰？")
for r in results:
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}...")
    print(f"来源: {r['source']}, 相似度: {r['score']:.3f}\n")
```

## 常见问题

### Q1: 如何扩展数据集？

您可以使用提供的脚本框架添加更多问答对：

1. 编辑 `generate_qa_local.py` 中的问答对列表
2. 添加新的文档和问答对
3. 重新运行脚本生成更新后的数据集

### Q2: 数据质量如何保证？

- 所有问答对基于专业天文文档编写
- 包含详细的技术解释和关键点
- 经过结构化处理，字段完整率100%
- 每个问题都有明确的来源文档引用

### Q3: 可以用于商业用途吗？

本数据集仅供学术研究和教育使用。如需商业使用，请联系相关文档的版权持有者。

### Q4: 如何贡献新的问答对？

欢迎贡献！请确保：
1. 内容基于可靠的天文来源
2. 问题涵盖文档的多个方面（赫罗图、光变曲线、光谱等）
3. 答案详细且包含关键点
4. 遵循现有的数据格式

## 数据字段说明

### JSON格式字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 问题编号 |
| `question` | string | 问题内容 |
| `answer` | string | 详细答案 |
| `category` | string | 问题分类 |
| `source_doc` | string | 来源文档 |
| `key_points` | list | 关键点列表 |

### CSV格式字段

| 字段 | 说明 |
|------|------|
| `doc_name` | 文档名称 |
| `qa_id` | 问题编号 |
| `question` | 问题内容 |
| `answer` | 详细答案 |
| `category` | 问题分类 |
| `source_doc` | 来源文档 |
| `key_points` | 关键点（竖线分隔） |
| `q_length` | 问题长度（字符） |
| `a_length` | 答案长度（字符） |

## 联系和支持

如有问题或建议，请通过以下方式联系：
- 检查README.md获取更多信息
- 查看analysis_report.txt了解数据分析详情

---

*最后更新: 2026-03-11*
