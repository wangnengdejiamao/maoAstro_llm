#!/usr/bin/env python3
"""
白矮星模型训练流程测试

测试整个流程的各个组件是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def test_kimi_api():
    """测试 Kimi API"""
    print("\n" + "=" * 60)
    print("测试 1: Kimi API 连接")
    print("=" * 60)
    
    try:
        from kimi_interface import KimiInterface, WhiteDwarfDataGenerator
        
        # 检查 API Key
        api_key = os.getenv("KIMI_API_KEY", "9cccf25-f532-861b-8000-000042a859dc")
        if not api_key:
            print("⚠ 未设置 KIMI_API_KEY")
            return False
        
        kimi = KimiInterface()
        
        # 简单测试
        response = kimi.generate("什么是白矮星？简要回答。")
        if response and "[错误]" not in response and "[生成失败]" not in response:
            print(f"✓ Kimi API 正常")
            print(f"  响应: {response[:80]}...")
            return True
        else:
            print(f"✗ Kimi API 异常: {response}")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_data_generator():
    """测试数据生成器"""
    print("\n" + "=" * 60)
    print("测试 2: 白矮星数据生成器")
    print("=" * 60)
    
    try:
        from kimi_interface import KimiInterface, WhiteDwarfDataGenerator
        
        kimi = KimiInterface()
        generator = WhiteDwarfDataGenerator(kimi)
        
        # 测试 QA 生成
        print("生成 QA 对...")
        qa_pairs = generator.generate_qa_pairs("单白矮星冷却", n_pairs=2)
        
        if qa_pairs:
            print(f"✓ 生成 {len(qa_pairs)} 个 QA 对")
            print(f"  样例: {qa_pairs[0].get('question', 'N/A')[:50]}...")
            return True
        else:
            print("⚠ 未生成 QA 对")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arxiv_downloader():
    """测试 arXiv 下载器"""
    print("\n" + "=" * 60)
    print("测试 3: arXiv 下载器")
    print("=" * 60)
    
    try:
        from download_wd_papers import WhiteDwarfPaperDownloader
        
        downloader = WhiteDwarfPaperDownloader(output_dir="./test_data")
        
        # 只搜索 5 篇用于测试
        print("搜索白矮星论文 (测试模式，最多 5 篇)...")
        papers = downloader.search_arxiv(
            query="white dwarf cooling",
            max_results=5,
            start_date="2023-01-01"
        )
        
        if papers:
            print(f"✓ 找到 {len(papers)} 篇论文")
            print(f"  样例: {papers[0].title[:60]}...")
            return True
        else:
            print("⚠ 未找到论文")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wd_evaluator():
    """测试白矮星评估器"""
    print("\n" + "=" * 60)
    print("测试 4: 白矮星评估器")
    print("=" * 60)
    
    try:
        from wd_evaluator import WhiteDwarfEvaluator
        
        evaluator = WhiteDwarfEvaluator()
        
        # 加载基准题库
        benchmark = evaluator.load_wd_benchmark()
        
        if benchmark:
            print(f"✓ 加载 {len(benchmark)} 题白矮星基准题库")
            
            # 显示题目分布
            subfields = {}
            for q in benchmark:
                sf = q.get('subfield', 'unknown')
                subfields[sf] = subfields.get(sf, 0) + 1
            
            print("  题目分布:")
            for sf, count in sorted(subfields.items()):
                print(f"    {sf}: {count} 题")
            
            return True
        else:
            print("✗ 未加载到题目")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("测试 5: 文件结构检查")
    print("=" * 60)
    
    required_files = [
        "kimi_interface.py",
        "download_wd_papers.py",
        "wd_evaluator.py",
        "train_whitewarf.sh",
        "configs/qwen8b_whitewarf_finetune.yaml",
        "Modelfile.whitewarf",
        "WHITEWARF_QUICKSTART.md",
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(__file__).parent / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
            all_exist = False
    
    return all_exist


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 WhiteWarf 训练流程测试")
    print("=" * 60)
    
    results = {
        "Kimi API": test_kimi_api(),
        "数据生成器": test_data_generator(),
        "arXiv 下载器": test_arxiv_downloader(),
        "白矮星评估器": test_wd_evaluator(),
        "文件结构": test_file_structure(),
    }
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n总计: {passed}/{total} 项通过 ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以开始训练了。")
        print("\n运行: bash train_whitewarf.sh")
    else:
        print("\n⚠ 部分测试失败，请检查上述错误。")
        print("\n常见问题:")
        print("  1. Kimi API 失败: 检查网络连接和 API Key")
        print("  2. arXiv 失败: 可能需要代理或 feedparser 未安装")
        print("  3. 依赖问题: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
