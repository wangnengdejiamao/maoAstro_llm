#!/usr/bin/env python3
"""
生成项目总结报告
"""

import json
from pathlib import Path
from datetime import datetime

def generate_report():
    report = []
    report.append("="*70)
    report.append("  天体物理PDF知识提取与QA模型训练项目报告")
    report.append("="*70)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. PDF文件处理
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    pdf_files = [f for f in pdf_files if not f.name.startswith("._")]
    
    report.append("\n" + "-"*70)
    report.append("1. PDF文件处理")
    report.append("-"*70)
    report.append(f"   - PDF文件总数: {len(pdf_files)}")
    report.append(f"   - PDF目录: {pdf_dir.absolute()}")
    if pdf_files:
        total_size = sum(f.stat().st_size for f in pdf_files) / (1024*1024)
        report.append(f"   - 总大小: {total_size:.1f} MB")
        report.append(f"   - 示例文件:")
        for f in list(pdf_files)[:5]:
            report.append(f"     • {f.name[:60]}")
    
    # 2. 数据集构建
    report.append("\n" + "-"*70)
    report.append("2. 知识数据集构建")
    report.append("-"*70)
    
    dataset_file = Path("output/astro_full_dataset.json")
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 统计主题
        topics = {}
        for item in dataset:
            topic = item.get('topic', 'Unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        report.append(f"   - 总QA对数: {len(dataset)}")
        report.append(f"   - 知识主题数: {len(topics)}")
        report.append(f"   - 各主题分布:")
        for topic, count in sorted(topics.items()):
            report.append(f"     • {topic}: {count} 个QA对")
        
        # 训练/测试集
        train_file = Path("output/astro_train.json")
        test_file = Path("output/astro_test.json")
        if train_file.exists() and test_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            report.append(f"   - 训练集: {len(train_data)} 个样本 ({len(train_data)/len(dataset)*100:.0f}%)")
            report.append(f"   - 测试集: {len(test_data)} 个样本 ({len(test_data)/len(dataset)*100:.0f}%)")
    
    # 3. 模型训练
    report.append("\n" + "-"*70)
    report.append("3. 模型训练")
    report.append("-"*70)
    
    model_file = Path("models/astro_qa_model.pkl")
    if model_file.exists():
        import pickle
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        report.append(f"   - 模型类型: TF-IDF + 余弦相似度")
        report.append(f"   - 词汇表大小: {len(model_data['vectorizer'].vocabulary_)}")
        report.append(f"   - 训练样本数: {len(model_data['questions'])}")
        report.append(f"   - 模型文件: {model_file}")
    
    # 4. 模型评估
    report.append("\n" + "-"*70)
    report.append("4. 模型评估结果")
    report.append("-"*70)
    
    eval_file = Path("output/evaluation_results.json")
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        total = eval_results['total']
        correct = eval_results['correct']
        partial = eval_results['partial']
        score = (correct + 0.5*partial) / total * 100
        
        report.append(f"   - 评估题目数: {total}")
        report.append(f"   - 完全正确: {correct} ({correct/total*100:.1f}%)")
        report.append(f"   - 部分正确: {partial} ({partial/total*100:.1f}%)")
        report.append(f"   - 错误/未答: {eval_results['incorrect']} ({eval_results['incorrect']/total*100:.1f}%)")
        report.append(f"   - 综合得分: {score:.1f}%")
    
    # 5. 使用说明
    report.append("\n" + "-"*70)
    report.append("5. 使用说明")
    report.append("-"*70)
    report.append("   运行交互式问答:")
    report.append("     python demo_qa.py")
    report.append("\n   运行模型评估:")
    report.append("     python evaluate_model.py")
    report.append("\n   文件说明:")
    report.append("     - data/pdfs/          : PDF文件目录")
    report.append("     - output/             : 输出文件目录")
    report.append("     - models/             : 模型文件目录")
    report.append("     - astro_qa_dataset.py : 数据集生成脚本")
    report.append("     - train_model.py      : 模型训练脚本")
    report.append("     - evaluate_model.py   : 模型评估脚本")
    
    # 6. 示例问答
    report.append("\n" + "-"*70)
    report.append("6. 示例问答")
    report.append("-"*70)
    
    if model_file.exists():
        from train_model import AstroQAModel
        model = AstroQAModel()
        model.load('models/astro_qa_model.pkl')
        
        examples = [
            "什么是灾变变星？",
            "什么是白矮星？",
            "什么是吸积盘？",
            "什么是赫罗图？"
        ]
        
        for q in examples:
            results = model.predict(q, top_k=1)
            if results:
                answer = results[0]['answer'][:150]
                report.append(f"\n   Q: {q}")
                report.append(f"   A: {answer}...")
                report.append(f"      [置信度: {results[0]['similarity']:.2f}]")
    
    report.append("\n" + "="*70)
    report.append("报告生成完成")
    report.append("="*70)
    
    return "\n".join(report)

if __name__ == "__main__":
    report = generate_report()
    print(report)
    
    # 保存报告
    report_file = Path("output/project_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_file}")
