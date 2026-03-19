# 天文知识问答对数据集

## 概述

本数据集包含100个高质量的天文知识问答对，涵盖双星系统、激变变星、AM CVn系统和周光关系等主题。每个问答对都基于原始文档内容，包含详细的技术解释、关键点和文档引用。

## 数据统计

- **文档数量**: 4个
- **问答对总数**: 100个
- **每文档问答对数**: 25个
- **平均答案长度**: 301字符
- **字段完整性**: 100%

## 文档列表

| 文档 | 文件名 | 主题 | 问答对数 |
|------|--------|------|----------|
| AM CVn系统 | amcvn.txt | 超致密双星系统 | 25 |
| 双星系统 | binary_systems.txt | 双星分类和物理 | 25 |
| 激变变星 | cataclysmic_variables.txt | CV分类和观测 | 25 |
| 周光关系 | period_luminosity_relations.txt | 距离测量 | 25 |

## 问题分类覆盖

### 核心天文概念
- **赫罗图**: 恒星在赫罗图上的位置、演化轨迹
- **SED/能谱**: 光谱能量分布、多波段特性
- **光变曲线**: 爆发、周期调制、闪烁等
- **周期**: 轨道周期、脉动周期
- **X射线**: X射线辐射机制、观测特性
- **光谱**: 发射线、吸收线、线形分析

### 物理机制
- 吸积物理（盘形成、质量转移）
- 潮汐相互作用
- 引力波辐射
- 磁制动
- 演化机制

### 应用和对比
- 系统类型对比（CV vs AM CVn vs X射线双星）
- 距离测量方法
- 常见误解辨析
- 观测技术和巡天

## 数据格式

### 标准格式 (training_dataset.json)

```json
{
  "instruction": "问题内容",
  "input": "",
  "output": "详细答案",
  "source_doc": "来源文档名称",
  "category": "问题分类",
  "key_points": ["要点1", "要点2", "要点3"]
}
```

### Alpaca格式 (training_alpaca.json)

```json
{
  "instruction": "基于文档xxx，请回答：问题",
  "input": "",
  "output": "详细答案",
  "system": "你是一个天文学专家。请基于提供的文档内容回答问题，确保答案准确且引用文档信息。",
  "history": []
}
```

## 文件说明

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `all_qa_summary.json` | 所有问答对的汇总 | 112 KB |
| `amcvn_qa.json` | AM CVn文档问答对 | 25 KB |
| `binary_systems_qa.json` | 双星系统文档问答对 | 28 KB |
| `cataclysmic_variables_qa.json` | 激变变星文档问答对 | 28 KB |
| `period_luminosity_qa.json` | 周光关系文档问答对 | 29 KB |
| `training_dataset.json` | 训练格式（标准） | 110 KB |
| `training_alpaca.json` | 训练格式（Alpaca） | 109 KB |
| `analysis_report.txt` | 数据分析报告 | - |

## 使用建议

### 1. 模型训练

用于微调天文领域的大语言模型：

```python
import json

# 加载训练数据
with open('training_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 转换为模型训练格式
training_samples = []
for item in train_data:
    training_samples.append({
        "messages": [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]}
        ]
    })
```

### 2. 知识检索 (RAG)

构建向量数据库用于检索增强生成：

```python
from sentence_transformers import SentenceTransformer

# 加载embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 构建向量库
corpus = [item["instruction"] + " " + item["output"] for item in train_data]
corpus_embeddings = model.encode(corpus)
```

### 3. 教育应用

- 作为天文学课程的练习题
- 构建自适应学习系统
- 知识测验和评估

## 特色内容

### 详细技术解释
每个答案都包含：
- 物理机制解释
- 定量关系和公式
- 观测特征描述
- 与其他系统的对比
- 常见误解澄清

### 多维分类
每个问答对包含：
- **分类标签**: 主题类别（如"光变曲线"、"X射线"）
- **来源文档**: 原始知识文档引用
- **关键点**: 答案的核心要点列表

### 数据质量保证
- 所有字段完整（100%）
- 答案详细（99% > 200字符）
- 信息准确（基于权威文档）
- 覆盖全面（多维度分类）

## 主题示例

### 赫罗图相关
- AM CVn在赫罗图上的极蓝位置
- 双星系统的分布特征
- CV的演化轨迹

### 光变曲线
- 矮新星的爆发循环
- 超驼峰的形成机制
- 食双星的掩食形状

### X射线
- AM CVn的边界层辐射
- CV的硬X射线来源
- X射线双星的吸积柱

### 周光关系
- 造父变星的P-L关系
- RR Lyrae的标准烛光应用
- 距离尺度的构建

## 贡献和反馈

本数据集基于专业天文文档构建，旨在：
1. 提供高质量的天文知识问答对
2. 支持天文领域的AI应用开发
3. 促进天文学教育和科普

## 许可证

本数据集仅供学术研究和教育使用。

---

*数据集生成时间: 2026-03-11*  
*文档版本: 1.0*
