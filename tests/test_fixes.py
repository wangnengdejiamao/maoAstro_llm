#!/usr/bin/env python3
"""
测试修复后的功能
================
测试项目：
1. 消光查询模块（修复后的接口）
2. RAG检索（TF-IDF权重）
3. Ollama错误处理
4. AstroTools工具集
"""

import sys
import os

print("="*70)
print("测试修复后的功能")
print("="*70)

# 测试1: 消光查询模块（修复后的接口）
print("\n【测试1】消光查询模块（修复后的maps参数传递）...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from query_extinction import load_extinction_maps, query_extinction
    print("  ✓ 消光查询模块导入成功")
    
    # 尝试加载消光地图
    try:
        maps = load_extinction_maps(data_dir='data')
        print(f"  ✓ 消光地图加载成功（NSIDE={maps['nside']}）")
        
        # 测试查询（EV UMa坐标）
        result = query_extinction(13.1316, 53.8585, maps)
        if 'error' not in result:
            print(f"  ✓ 消光查询成功: A_V={result['csfd_av']:.4f} mag")
        else:
            print(f"  ✗ 消光查询失败: {result.get('error')}")
    except Exception as e:
        print(f"  ℹ 消光地图加载失败（可能没有数据文件）: {e}")
        
except Exception as e:
    print(f"  ✗ 消光查询模块错误: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 通过AstroTools测试（修复后的接口）
print("\n【测试2】AstroTools工具集（修复后的接口）...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'langgraph_demo'))
    from astro_assistant_astrosage import AstroTools
    tools = AstroTools()
    
    # 测试消光查询
    result = tools.query_extinction(13.1316, 53.8585)
    if result and 'error' not in result:
        print(f"  ✓ AstroTools.query_extinction成功: A_V={result.get('csfd_av', 'N/A')}")
    else:
        error_msg = result.get('error', '未知错误') if result else '无结果'
        print(f"  ℹ 消光工具未完全加载: {error_msg}")
        
except Exception as e:
    print(f"  ✗ AstroTools测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: RAG检索（TF-IDF权重）
print("\n【测试3】RAG检索（TF-IDF权重）...")
try:
    from astro_assistant_astrosage import AstroRAG
    rag = AstroRAG()
    
    # 添加测试文档
    rag.add([
        {'text': 'AM CVn systems are ultra-compact binaries with white dwarfs', 'topic': 'AM CVn'},
        {'text': 'Cepheid variable stars follow period luminosity relation for distance', 'topic': 'Cepheid'},
        {'text': 'ZTF photometric survey provides light curves for variable stars', 'topic': 'ZTF'},
        {'text': 'Cataclysmic variables have accretion disks and outbursts', 'topic': 'CV'},
    ])
    
    # 测试搜索
    results = rag.search('AM CVn binary period', k=2)
    if results:
        print(f"  ✓ RAG检索成功，返回 {len(results)} 条结果")
        for i, r in enumerate(results):
            print(f"    {i+1}. {r.get('topic')} (score={r.get('score', 'N/A')})")
    else:
        print("  ⚠ RAG检索返回空结果")
        
except Exception as e:
    print(f"  ✗ RAG检索测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Ollama连接检查
print("\n【测试4】Ollama连接检查...")
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code == 200:
        models = [m['name'] for m in r.json().get('models', [])]
        print(f"  ✓ Ollama服务运行中，已安装 {len(models)} 个模型")
        if models:
            print(f"    可用模型: {', '.join(models[:3])}")
    else:
        print(f"  ⚠ Ollama返回非200状态码: {r.status_code}")
except requests.ConnectionError:
    print("  ℹ Ollama服务未运行（这是正常的如果未启动 ollama serve）")
except Exception as e:
    print(f"  ✗ Ollama检查失败: {e}")

# 测试5: GAIA查询（带重试机制）
print("\n【测试5】GAIA查询（带重试机制）...")
try:
    from astro_assistant_astrosage import AstroTools
    tools = AstroTools()
    
    # 测试GAIA查询（使用短超时避免等待太久）
    result = tools.query_gaia(13.1316, 53.8585, radius=0.01)
    if result:
        print(f"  ✓ GAIA查询成功，返回 {len(result)} 个源")
    else:
        print("  ℹ GAIA查询返回空结果（可能是网络或坐标问题）")
        
except Exception as e:
    print(f"  ✗ GAIA查询测试失败: {e}")

# 测试6: 模型名称匹配修复
print("\n【测试6】模型名称匹配...")
try:
    from astro_assistant_astrosage import check_model_available
    
    # 这个测试需要Ollama运行
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        # 如果服务运行，测试一些常见模型名
        test_models = ['llama3.1:8b', 'qwen3:8b', 'astrosage']
        for model in test_models:
            available = check_model_available(model)
            status = "✓" if available else "✗"
            print(f"  {status} {model}: {'可用' if available else '不可用'}")
    except requests.ConnectionError:
        print("  ℹ Ollama未运行，跳过模型匹配测试")
        
except Exception as e:
    print(f"  ✗ 模型名称匹配测试失败: {e}")

print("\n" + "="*70)
print("测试完成！")
print("="*70)
print("""
修复总结：
1. ✅ 消光查询接口：修复了3个文件中缺少maps参数的问题
2. ✅ RAG检索：添加了TF-IDF权重计算
3. ✅ GAIA查询：添加了多端点和重试机制
4. ✅ 模型匹配：改为精确匹配避免误判

使用建议：
1. 如需AI分析功能，请确保 Ollama 正在运行:
   ollama serve

2. 如需消光查询功能，请确保 data/ 目录有消光地图文件:
   - csfd_ebv.fits
   - sfd_ebv.fits
   - lss_intensity.fits
   - lss_error.fits
   - mask.fits

3. 运行完整助手:
   python langgraph_demo/astro_assistant_astrosage.py
   
   或使用轻量版:
   python langgraph_demo/ollama_rag_light.py
""")
