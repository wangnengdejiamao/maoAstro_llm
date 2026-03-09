#!/usr/bin/env python3
"""
AstroSage + VSP 集成测试
======================

测试各个组件是否正常工作。

用法:
    python test_astrosage_vsp.py [测试项]

测试项:
    all       - 运行所有测试 (默认)
    vsp       - 测试 VSP 模块
    ollama    - 测试 Ollama 连接
    coord     - 测试坐标解析
    query     - 测试完整查询流程

示例:
    python test_astrosage_vsp.py
    python test_astrosage_vsp.py vsp
    python test_astrosage_vsp.py query
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_vsp_module():
    """测试 VSP 模块加载"""
    print("\n" + "=" * 70)
    print("测试 1: VSP 模块加载")
    print("=" * 70)
    
    try:
        from astrosage_with_vsp import VSPAnalyzer
        vsp = VSPAnalyzer()
        if vsp.vsp is not None:
            print("✓ VSP 模块加载成功")
            return True
        else:
            print("✗ VSP 模块未正确初始化")
            return False
    except Exception as e:
        print(f"✗ VSP 模块加载失败: {e}")
        return False


def test_ollama_connection():
    """测试 Ollama 连接"""
    print("\n" + "=" * 70)
    print("测试 2: Ollama 连接")
    print("=" * 70)
    
    try:
        from astrosage_with_vsp import OllamaInterface
        ollama = OllamaInterface(model_name="astrosage-local")
        
        if ollama.is_available():
            print("✓ Ollama 服务连接成功")
            
            # 测试简单生成
            print("\n测试简单文本生成...")
            response = ollama.generate("你好", max_tokens=50)
            if response and not response.startswith("["):
                print(f"✓ 模型响应正常")
                print(f"  响应预览: {response[:100]}...")
                return True
            else:
                print(f"✗ 模型响应异常: {response}")
                return False
        else:
            print("✗ Ollama 服务不可用")
            print("  请确保 Ollama 正在运行: ollama serve")
            return False
            
    except Exception as e:
        print(f"✗ Ollama 连接失败: {e}")
        return False


def test_coordinate_parsing():
    """测试坐标解析"""
    print("\n" + "=" * 70)
    print("测试 3: 坐标解析")
    print("=" * 70)
    
    try:
        from astrosage_with_vsp import AstroSageAssistant
        assistant = AstroSageAssistant()
        
        test_cases = [
            ("分析 13.1316, 53.8585", (13.1316, 53.8585)),
            ("查询 150.5 23.8", (150.5, 23.8)),
            ("RA=150.5 DEC=23.8", (150.5, 23.8)),
            ("200.5, -30.2", (200.5, -30.2)),
            ("看看这个天体 100.0 50.0", (100.0, 50.0)),
            ("普通问题，没有坐标", None),
        ]
        
        passed = 0
        for input_text, expected in test_cases:
            result = assistant.parse_coordinates(input_text)
            if expected is None:
                if result is None:
                    print(f"  ✓ '{input_text}' -> 正确识别为无坐标")
                    passed += 1
                else:
                    print(f"  ✗ '{input_text}' -> 错误识别为 {result}")
            else:
                if result and abs(result[0] - expected[0]) < 0.0001 and abs(result[1] - expected[1]) < 0.0001:
                    print(f"  ✓ '{input_text}' -> RA={result[0]}, DEC={result[1]}")
                    passed += 1
                else:
                    print(f"  ✗ '{input_text}' -> 期望 {expected}, 得到 {result}")
        
        print(f"\n坐标解析测试: {passed}/{len(test_cases)} 通过")
        return passed == len(test_cases)
        
    except Exception as e:
        print(f"✗ 坐标解析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_workflow():
    """测试完整查询流程"""
    print("\n" + "=" * 70)
    print("测试 4: 完整查询流程")
    print("=" * 70)
    
    try:
        from astrosage_with_vsp import VSPAnalyzer
        
        vsp = VSPAnalyzer()
        if vsp.vsp is None:
            print("✗ VSP 未初始化，跳过测试")
            return False
        
        # 使用 EV UMa 作为测试目标
        test_ra, test_dec = 13.1316, 53.8585
        print(f"测试目标: RA={test_ra}, DEC={test_dec} (EV UMa)")
        print("-" * 70)
        
        # 查询数据
        print("正在查询 VSP 数据库...")
        start_time = time.time()
        results = vsp.query_source(test_ra, test_dec, radius=5.0)
        query_time = time.time() - start_time
        
        print(f"查询耗时: {query_time:.2f} 秒")
        
        if "error" in results:
            print(f"✗ 查询失败: {results['error']}")
            return False
        
        # 检查各项数据
        print("\n数据检查结果:")
        checks = {
            'VSX': results.get('vsx') is not None and results.get('vsx') is not False,
            'Gaia': results.get('gaia') is not None and results.get('gaia') is not False,
            '消光': results.get('ebv_sfd') is not None,
            'SIMBAD': results.get('simbad') is not None,
        }
        
        all_passed = True
        for name, status in checks.items():
            symbol = "✓" if status else "✗"
            print(f"  {symbol} {name}: {'可用' if status else '不可用'}")
            if not status:
                all_passed = False
        
        # 生成摘要
        print("\n生成数据摘要...")
        summary = vsp.generate_summary(results)
        if summary:
            print("✓ 摘要生成成功")
        else:
            print("✗ 摘要生成失败")
            all_passed = False
        
        # 生成分析提示词
        print("生成 AI 分析提示词...")
        prompt = vsp.generate_analysis_prompt(results)
        if prompt and len(prompt) > 100:
            print("✓ 提示词生成成功")
            print(f"  提示词长度: {len(prompt)} 字符")
        else:
            print("✗ 提示词生成失败")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ 查询流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AstroSage + VSP 集成测试')
    parser.add_argument('test', nargs='?', default='all',
                        choices=['all', 'vsp', 'ollama', 'coord', 'query'],
                        help='要运行的测试项')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AstroSage + VSP 集成测试")
    print("=" * 70)
    
    results = {}
    
    if args.test in ['all', 'vsp']:
        results['vsp'] = test_vsp_module()
    
    if args.test in ['all', 'ollama']:
        results['ollama'] = test_ollama_connection()
    
    if args.test in ['all', 'coord']:
        results['coord'] = test_coordinate_parsing()
    
    if args.test in ['all', 'query']:
        results['query'] = test_query_workflow()
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name.upper()}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统就绪。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
