#!/usr/bin/env python3
"""
增强版天体分析演示
==================
展示整合RAG知识库和AI大模型的智能天体分析系统

功能:
1. 增强版RAG知识检索
2. 智能天文分析器
3. 高级RAG分析 (领域知识+文献索引)
4. 完整的分析流程演示

使用:
    python demo_enhanced_analysis.py --target EV_UMa --ra 196.9744 --dec 53.8585
"""

import os
import sys
import json
import argparse
from pathlib import Path


def demo_enhanced_rag():
    """演示增强版RAG系统"""
    print("\n" + "="*70)
    print("演示1: 增强版RAG知识检索")
    print("="*70)
    
    from src.enhanced_rag_system import get_enhanced_rag
    
    rag = get_enhanced_rag(use_vector_store=True)
    
    # 测试查询
    queries = [
        "cataclysmic variable classification",
        "AM CVn period luminosity",
        "polar magnetic field",
        "white dwarf cooling"
    ]
    
    for query in queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 50)
        result = rag.search(query, top_k=2)
        print(result[:800] + "..." if len(result) > 800 else result)


def demo_intelligent_analyzer(target_data: dict, target_name: str):
    """演示智能分析器"""
    print("\n" + "="*70)
    print("演示2: 智能天文分析器")
    print("="*70)
    
    from src.intelligent_astro_analyzer import IntelligentAstroAnalyzer
    
    analyzer = IntelligentAstroAnalyzer(
        ollama_model="astrosage-local:latest",
        use_rag=True,
        use_vision=False
    )
    
    result = analyzer.analyze_target(target_data, target_name=target_name)
    
    return result


def demo_advanced_rag(target_data: dict, target_name: str):
    """演示高级RAG分析"""
    print("\n" + "="*70)
    print("演示3: 高级RAG分析 (领域知识 + 文献)")
    print("="*70)
    
    from src.advanced_rag_analyzer import create_advanced_analyzer
    
    # 创建分析器
    analyzer = create_advanced_analyzer(use_literature=False)
    
    # 执行RAG检索
    rag_result = analyzer.analyze_with_rag(target_data, target_name)
    
    # 生成增强提示词
    enhanced_prompt = analyzer.generate_enhanced_prompt(rag_result)
    
    print("\n📋 生成的增强提示词 (前3000字符):")
    print("-" * 70)
    print(enhanced_prompt[:3000])
    if len(enhanced_prompt) > 3000:
        print(f"\n... (总共 {len(enhanced_prompt)} 字符)")
    
    return rag_result, enhanced_prompt


def demo_literature_index():
    """演示文献索引"""
    print("\n" + "="*70)
    print("演示4: 文献索引系统")
    print("="*70)
    
    try:
        from src.literature_indexer import create_sample_literature_db, get_literature_index
        
        # 创建示例数据库
        papers = create_sample_literature_db()
        
        # 初始化索引
        index = get_literature_index("simple")
        
        from src.literature_indexer import LiteratureEntry
        entries = [LiteratureEntry(**p) for p in papers]
        index.add_entries(entries)
        
        # 搜索
        print("\n🔍 搜索 'gravitational waves':")
        results = index.search("gravitational waves", top_k=3)
        for r in results:
            print(f"  • {r.title[:70]}...")
            print(f"    作者: {', '.join(r.authors[:2])}")
            print(f"    被引: {r.citations} 次\n")
        
    except Exception as e:
        print(f"⚠ 文献索引演示失败: {e}")


def demo_full_analysis(ra: float, dec: float, name: str):
    """
    执行完整的增强版分析流程
    
    Args:
        ra: 赤经
        dec: 赤纬  
        name: 目标名称
    """
    print("\n" + "="*70)
    print("🚀 完整增强版分析流程")
    print("="*70)
    print(f"目标: {name}")
    print(f"坐标: RA={ra}, DEC={dec}")
    
    # 检查是否存在已有分析结果
    result_file = Path(f"./output/{name}_analysis.json")
    
    if result_file.exists():
        print(f"\n📂 加载已有分析结果: {result_file}")
        with open(result_file, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
    else:
        print("\n⚠ 未找到已有分析结果，使用示例数据")
        # 创建示例数据
        target_data = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'simbad': {
                'matched': True,
                'main_id': f'V* {name}',
                'otype': 'CataclyV*',
                'distance': 656.4,
                'parallax': 1.5234
            },
            'periods': {
                'TESS': {'period': 0.05534, 'theta_min': 0.665}
            },
            'extinction': {
                'success': True,
                'A_V': 0.060,
                'E_B_V': 0.020
            },
            'lightcurves': {'TESS': {'n_points': 13683}},
            'spectrum': {'success': True, 'spectra': []},
            'sed': {'success': True, 'n_points': 15}
        }
    
    # 步骤1: RAG知识检索
    print("\n" + "-"*70)
    print("步骤1: RAG知识检索")
    print("-"*70)
    
    from src.advanced_rag_analyzer import create_advanced_analyzer
    rag_analyzer = create_advanced_analyzer(use_literature=False)
    rag_result = rag_analyzer.analyze_with_rag(target_data, name)
    
    # 步骤2: 生成增强提示词
    print("\n" + "-"*70)
    print("步骤2: 生成增强提示词")
    print("-"*70)
    
    enhanced_prompt = rag_analyzer.generate_enhanced_prompt(rag_result)
    print(f"✓ 提示词长度: {len(enhanced_prompt)} 字符")
    print(f"✓ 知识片段数: {len(rag_result['fragments'])}")
    
    # 步骤3: AI分析 (如果Ollama可用)
    print("\n" + "-"*70)
    print("步骤3: AI智能分析")
    print("-"*70)
    
    try:
        from src.intelligent_astro_analyzer import IntelligentAstroAnalyzer
        
        ai_analyzer = IntelligentAstroAnalyzer(
            ollama_model="astrosage-local:latest",
            use_rag=False,  # 已经通过rag_analyzer获取了知识
            use_vision=False
        )
        
        # 手动执行分析
        from src.ollama_qwen_interface import OllamaQwenInterface
        ollama = OllamaQwenInterface(model_name="astrosage-local:latest")
        
        system_prompt = """你是一位资深天体物理学家，擅长变星和双星系统研究。
请基于提供的观测数据和背景知识进行分析，避免编造信息。"""
        
        print("🤖 正在调用Ollama AI...")
        response = ollama.analyze_text(enhanced_prompt, system_prompt=system_prompt, max_retries=1)
        
        if response and not response.startswith('['):
            print("\n" + "="*70)
            print("📝 AI分析报告")
            print("="*70)
            print(response)
            print("="*70)
            
            # 保存结果
            output = {
                'target': name,
                'ra': ra,
                'dec': dec,
                'rag_knowledge': rag_result['knowledge'][:5000],
                'knowledge_sources': rag_result['knowledge_sources'],
                'ai_analysis': response
            }
            
            output_file = f"./output/{name}_enhanced_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 分析结果已保存: {output_file}")
        else:
            print(f"⚠ AI分析失败: {response[:200] if response else '空响应'}")
            
    except Exception as e:
        print(f"⚠ AI分析不可用: {e}")
        print("  提示词已生成，可以手动复制到Ollama进行测试")


def main():
    parser = argparse.ArgumentParser(description='增强版天体分析演示')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['rag', 'analyzer', 'advanced', 'literature', 'full', 'all'],
                       help='演示模式')
    parser.add_argument('--target', type=str, default='EV_UMa',
                       help='目标名称')
    parser.add_argument('--ra', type=float, default=196.9744,
                       help='赤经 (度)')
    parser.add_argument('--dec', type=float, default=53.8585,
                       help='赤纬 (度)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🌟 增强版天体分析系统演示")
    print("="*70)
    print("\n本演示展示以下功能:")
    print("  1. 增强版RAG知识检索 (向量+关键词混合)")
    print("  2. 智能天文分析器 (RAG+AI整合)")
    print("  3. 高级RAG分析 (领域知识+文献索引)")
    print("  4. 文献索引系统 (可扩展至50万+文献)")
    print("="*70)
    
    # 准备测试数据
    test_data = {
        'name': args.target,
        'ra': args.ra,
        'dec': args.dec,
        'simbad': {
            'matched': True,
            'main_id': f'V* {args.target}',
            'otype': 'CataclyV*',
            'distance': 656.4,
            'parallax': 1.5234
        },
        'periods': {
            'TESS': {'period': 0.05534, 'theta_min': 0.665}
        },
        'extinction': {
            'success': True,
            'A_V': 0.060,
            'E_B_V': 0.020
        }
    }
    
    if args.mode == 'rag' or args.mode == 'all':
        demo_enhanced_rag()
    
    if args.mode == 'analyzer' or args.mode == 'all':
        try:
            demo_intelligent_analyzer(test_data, args.target)
        except Exception as e:
            print(f"\n⚠ 智能分析器演示失败: {e}")
    
    if args.mode == 'advanced' or args.mode == 'all':
        try:
            demo_advanced_rag(test_data, args.target)
        except Exception as e:
            print(f"\n⚠ 高级RAG演示失败: {e}")
    
    if args.mode == 'literature' or args.mode == 'all':
        demo_literature_index()
    
    if args.mode == 'full':
        demo_full_analysis(args.ra, args.dec, args.target)
    
    print("\n" + "="*70)
    print("✅ 演示完成!")
    print("="*70)
    print("\n提示:")
    print("  • 运行 'python demo_enhanced_analysis.py --mode all' 查看所有演示")
    print("  • 运行 'python demo_enhanced_analysis.py --mode full --target <name> --ra <ra> --dec <dec>'")
    print("    对指定目标执行完整分析")
    print("  • 查看 output/ 目录获取分析结果")


if __name__ == "__main__":
    main()
