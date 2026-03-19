#!/usr/bin/env python3
"""
模型评估系统 - 生成题目并评估模型性能
"""

import json
import random
from pathlib import Path
from train_model import AstroQAModel

# 扩展评估题库
EVALUATION_QUESTIONS = [
    # 灾变变星相关问题
    {
        "question": "什么是灾变变星？",
        "expected_keywords": ["白矮星", "双星", "吸积", "光度变化"],
        "topic": "灾变变星"
    },
    {
        "question": "灾变变星的周期空隙是多少小时？",
        "expected_keywords": ["2-3", "小时", "蜕变"],
        "topic": "灾变变星"
    },
    {
        "question": "极星和中介极星的主要区别是什么？",
        "expected_keywords": ["磁场", "吸积盘", "偏振"],
        "topic": "灾变变星"
    },
    {
        "question": "矮新星为什么会爆发？",
        "expected_keywords": ["吸积盘", "不稳定", "电离"],
        "topic": "灾变变星"
    },
    {
        "question": "什么是周期跳跃者？",
        "expected_keywords": ["周期", "最小值", "褐矮星"],
        "topic": "灾变变星"
    },
    # 白矮星相关问题
    {
        "question": "什么是白矮星？",
        "expected_keywords": ["恒星演化", "简并", "致密", "残骸"],
        "topic": "白矮星"
    },
    {
        "question": "查德拉塞卡极限是多少？",
        "expected_keywords": ["1.4", "太阳质量"],
        "topic": "白矮星"
    },
    {
        "question": "磁白矮星的磁场强度是多少？",
        "expected_keywords": ["磁场", "高斯", "10^5", "10^9"],
        "topic": "白矮星"
    },
    {
        "question": "DA型白矮星有什么特征？",
        "expected_keywords": ["氢", "巴尔末线", "光谱"],
        "topic": "白矮星"
    },
    {
        "question": "白矮星是如何冷却的？",
        "expected_keywords": ["辐射", "冷却", "热", "温度"],
        "topic": "白矮星"
    },
    # 观测方法
    {
        "question": "ZTF是什么？",
        "expected_keywords": ["Zwicky", "Transient", "Facility", "巡天", "帕洛玛"],
        "topic": "观测方法"
    },
    {
        "question": "LAMOST有多少根光纤？",
        "expected_keywords": ["4000", "光纤"],
        "topic": "观测方法"
    },
    {
        "question": "什么是多普勒层析成像？",
        "expected_keywords": ["速度空间", "吸积盘", "谱线"],
        "topic": "观测方法"
    },
    {
        "question": "TESS的主要任务是什么？",
        "expected_keywords": ["系外行星", "凌星", "变星"],
        "topic": "观测方法"
    },
    {
        "question": "什么是测光红移？",
        "expected_keywords": ["光度", "红移", "多波段"],
        "topic": "观测方法"
    },
    # 双星演化
    {
        "question": "什么是公共包层演化？",
        "expected_keywords": ["包层", "质量转移", "洛希瓣", "角动量"],
        "topic": "双星演化"
    },
    {
        "question": "磁制动如何影响双星系统？",
        "expected_keywords": ["角动量", "损失", "磁风", "轨道"],
        "topic": "双星演化"
    },
    {
        "question": "什么是洛希瓣？",
        "expected_keywords": ["等位面", "质量转移", "拉格朗日"],
        "topic": "双星演化"
    },
    {
        "question": "引力波辐射对双星有什么影响？",
        "expected_keywords": ["轨道", "能量", "角动量", "并合"],
        "topic": "双星演化"
    },
    {
        "question": "Ia型超新星的前身星是什么？",
        "expected_keywords": ["白矮星", "双星", "吸积", "查德拉塞卡"],
        "topic": "双星演化"
    },
    # 吸积物理
    {
        "question": "什么是吸积盘？",
        "expected_keywords": ["盘状", "角动量", "粘滞", "致密天体"],
        "topic": "吸积物理"
    },
    {
        "question": "吸积盘中角动量如何传递？",
        "expected_keywords": ["粘滞", "MRI", "磁转动", "不稳定性"],
        "topic": "吸积物理"
    },
    {
        "question": "什么是吸积激波？",
        "expected_keywords": ["激波", "动能", "热能", "X射线"],
        "topic": "吸积物理"
    },
    {
        "question": "什么是吸积柱？",
        "expected_keywords": ["磁", "磁力线", "白矮星", "柱状"],
        "topic": "吸积物理"
    },
    {
        "question": "什么是盘风？",
        "expected_keywords": ["外流", "辐射压", "吸收线", "角动量"],
        "topic": "吸积物理"
    },
    # 恒星演化
    {
        "question": "什么是赫罗图？",
        "expected_keywords": ["光度", "温度", "演化", "主序"],
        "topic": "恒星演化"
    },
    {
        "question": "太阳最终会变成什么？",
        "expected_keywords": ["白矮星", "红巨星", "行星状星云"],
        "topic": "恒星演化"
    },
    {
        "question": "什么是行星状星云？",
        "expected_keywords": ["气体壳层", "电离", "紫外", "白矮星"],
        "topic": "恒星演化"
    },
    {
        "question": "什么是褐矮星？",
        "expected_keywords": ["氢", "聚变", "木星", "质量"],
        "topic": "恒星演化"
    },
    {
        "question": "什么是蓝离散星？",
        "expected_keywords": ["并合", "质量转移", "球状星团", "主序"],
        "topic": "恒星演化"
    },
    # X射线天文学
    {
        "question": "灾变变星的X射线从何而来？",
        "expected_keywords": ["边界层", "吸积柱", "激波", "高温"],
        "topic": "X射线天文学"
    },
    {
        "question": "ROSAT是什么？",
        "expected_keywords": ["X射线", "巡天", "卫星", "伦琴"],
        "topic": "X射线天文学"
    },
    {
        "question": "什么是eROSITA？",
        "expected_keywords": ["X射线", "望远镜", "Spektr-RG", "巡天"],
        "topic": "X射线天文学"
    },
    {
        "question": "什么是X射线双星？",
        "expected_keywords": ["致密天体", "中子星", "黑洞", "吸积"],
        "topic": "X射线天文学"
    },
    {
        "question": "什么是超软X射线源？",
        "expected_keywords": ["白矮星", "核燃烧", "温度", "双星"],
        "topic": "X射线天文学"
    },
]

def evaluate_model(model, questions, top_k=1):
    """评估模型性能"""
    results = {
        "total": len(questions),
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
        "details": []
    }
    
    for item in questions:
        query = item["question"]
        expected_keywords = item["expected_keywords"]
        topic = item["topic"]
        
        # 获取模型预测
        predictions = model.predict(query, top_k=top_k)
        
        if not predictions:
            results["incorrect"] += 1
            results["details"].append({
                "question": query,
                "topic": topic,
                "status": "no_prediction",
                "predicted": None,
                "similarity": 0
            })
            continue
        
        best_pred = predictions[0]
        answer = best_pred["answer"]
        similarity = best_pred["similarity"]
        
        # 检查关键词匹配
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        match_ratio = len(matched_keywords) / len(expected_keywords)
        
        # 判断结果
        if match_ratio >= 0.6 or similarity > 0.3:
            status = "correct"
            results["correct"] += 1
        elif match_ratio >= 0.3 or similarity > 0.1:
            status = "partial"
            results["partial"] += 1
        else:
            status = "incorrect"
            results["incorrect"] += 1
        
        results["details"].append({
            "question": query,
            "topic": topic,
            "status": status,
            "predicted_answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "similarity": round(similarity, 3),
            "matched_keywords": matched_keywords,
            "match_ratio": round(match_ratio, 2)
        })
    
    return results

def print_report(results):
    """打印评估报告"""
    print("\n" + "="*60)
    print("模型评估报告")
    print("="*60)
    
    total = results["total"]
    correct = results["correct"]
    partial = results["partial"]
    incorrect = results["incorrect"]
    
    print(f"\n总体性能:")
    print(f"  总题目数: {total}")
    print(f"  完全正确: {correct} ({correct/total*100:.1f}%)")
    print(f"  部分正确: {partial} ({partial/total*100:.1f}%)")
    print(f"  错误/未答: {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"  综合得分: {(correct + 0.5*partial)/total*100:.1f}%")
    
    # 按主题统计
    topic_stats = {}
    for detail in results["details"]:
        topic = detail["topic"]
        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "correct": 0, "partial": 0}
        topic_stats[topic]["total"] += 1
        if detail["status"] == "correct":
            topic_stats[topic]["correct"] += 1
        elif detail["status"] == "partial":
            topic_stats[topic]["partial"] += 1
    
    print(f"\n各主题表现:")
    for topic, stats in sorted(topic_stats.items()):
        score = (stats["correct"] + 0.5*stats["partial"]) / stats["total"] * 100
        print(f"  {topic}: {score:.0f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n详细结果:")
    print("-"*60)
    for i, detail in enumerate(results["details"], 1):
        status_icon = "✓" if detail["status"] == "correct" else ("~" if detail["status"] == "partial" else "✗")
        print(f"{i:2d}. [{status_icon}] {detail['question'][:50]}...")
        print(f"    主题: {detail['topic']} | 相似度: {detail['similarity']} | 匹配: {detail['match_ratio']}")
        if detail["status"] != "correct":
            print(f"    预测: {detail['predicted_answer'][:100]}...")
        print()

def run_evaluation():
    """运行完整评估"""
    # 加载模型
    model = AstroQAModel()
    model.load('models/astro_qa_model.pkl')
    
    # 评估
    print("开始评估模型...")
    results = evaluate_model(model, EVALUATION_QUESTIONS)
    
    # 打印报告
    print_report(results)
    
    # 保存结果
    with open('output/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 output/evaluation_results.json")
    
    return results

if __name__ == "__main__":
    run_evaluation()
