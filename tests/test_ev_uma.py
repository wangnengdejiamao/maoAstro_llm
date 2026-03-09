#!/usr/bin/env python3
"""
EV UMa 测试脚本
===============
测试 AstroSage + VSP 系统
"""

import sys
sys.path.insert(0, '.')

from astrosage_with_vsp import VSPAnalyzer, OllamaInterface

def test_ev_uma():
    print("=" * 70)
    print("🌟 EV UMa 测试")
    print("=" * 70)
    print()
    print("目标信息:")
    print("  名称: EV UMa (EV Ursae Majoris)")
    print("  坐标: RA=13.1316°, DEC=53.8585°")
    print("  类型: 激变变星 (Cataclysmic Variable)")
    print("  周期: ~0.1 天")
    print()
    
    # 1. 查询 VSP
    print("=" * 70)
    print("步骤 1: VSP 数据查询")
    print("=" * 70)
    
    vsp = VSPAnalyzer()
    results = vsp.query_source(ra=13.1316, dec=53.8585, radius=5.0)
    
    # 生成摘要
    summary = vsp.generate_summary(results)
    print(summary)
    
    # 2. AI 分析
    print()
    print("=" * 70)
    print("步骤 2: AI 分析")
    print("=" * 70)
    
    prompt = vsp.generate_analysis_prompt(results)
    
    print("生成提示词...")
    print("-" * 70)
    print(prompt[:500] + "...")
    print("-" * 70)
    
    # 3. 调用 Ollama
    print()
    print("=" * 70)
    print("步骤 3: 调用 AstroSage 模型")
    print("=" * 70)
    
    ollama = OllamaInterface(model_name="astrosage-local")
    
    system_prompt = """你是 AstroSage，一位专业的天体物理学家。
基于观测数据给出简洁的天体分析。"""
    
    print("正在请求 AI 分析...")
    response = ollama.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=1024
    )
    
    print()
    print("🤖 AstroSage 分析结果:")
    print("=" * 70)
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    test_ev_uma()
