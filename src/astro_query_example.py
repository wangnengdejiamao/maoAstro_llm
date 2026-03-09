#!/usr/bin/env python3
"""
统一天文数据查询接口 - 简化使用示例
==================================
只需要提供 RA 和 DEC，即可查询多个天文数据库
"""

import warnings
warnings.filterwarnings('ignore')

from unified_astro_query import AstroSourceQuerier


def simple_query_example():
    """
    最简单的使用示例
    """
    print("="*70)
    print("统一天文数据查询接口 - 简化示例")
    print("="*70)
    
    # 第一步：初始化查询器
    print("\n1. 初始化查询器...")
    querier = AstroSourceQuerier(cache_dir="./cache")
    
    # 第二步：提供坐标查询
    # 以天狼星 (Sirius) 为例
    ra = 101.2875    # 赤经 (度)
    dec = -16.7161   # 赤纬 (度)
    
    print(f"\n2. 查询坐标: RA={ra}°, DEC={dec}° (Sirius)")
    
    # 执行全源查询
    results = querier.query_all(ra, dec, radius=10.0)
    
    # 第三步：查看结果
    print("\n3. 查询结果:")
    print("-"*70)
    
    # SIMBAD 结果
    simbad = results.get('simbad')
    if simbad and simbad.success and simbad.data and simbad.data.get('matched'):
        print(f"✓ SIMBAD: {simbad.data['main_id']}")
        print(f"  类型: {simbad.data.get('otype', 'N/A')}")
        print(f"  光谱型: {simbad.data.get('sp_type', 'N/A')}")
        if simbad.data.get('magnitudes'):
            print(f"  星等: V={simbad.data['magnitudes'].get('V', 'N/A')}")
    else:
        print("✗ SIMBAD: 未找到")
    
    # Gaia 结果
    gaia = results.get('gaia')
    if gaia and gaia.success and gaia.data and gaia.data.get('found'):
        print(f"\n✓ Gaia DR3:")
        print(f"  Source ID: {gaia.data['source_id'][:20]}...")
        print(f"  G = {gaia.data['g_mag']:.2f}")
        if gaia.data.get('parallax'):
            print(f"  视差: {gaia.data['parallax']:.2f} mas")
        if gaia.data.get('distance'):
            print(f"  距离: {gaia.data['distance']:.1f} pc")
        if gaia.data.get('teff'):
            print(f"  温度: {gaia.data['teff']:.0f} K")
    else:
        print("\n✗ Gaia: 未找到")
    
    # SDSS 结果
    sdss = results.get('sdss')
    if sdss and sdss.success and sdss.data and sdss.data.get('has_spectra'):
        print(f"\n✓ SDSS: 有光谱数据")
        spec = sdss.data.get('spectra_info', {})
        print(f"  红移: z={spec.get('z', 'N/A')}")
        print(f"  分类: {spec.get('class', 'N/A')}")
    else:
        print(f"\n✗ SDSS: 无数据")
    
    # 第四步：生成完整报告
    print("\n4. 生成完整报告...")
    report = querier.generate_summary_report(results, ra, dec)
    print(report)
    
    # 第五步：保存结果
    print("\n5. 保存结果到文件...")
    filename = querier.save_results(results, ra, dec, output_dir="./output")
    print(f"   已保存: {filename}")
    
    print("\n" + "="*70)
    print("示例完成!")
    print("="*70)


def batch_query_example():
    """
    批量查询示例
    """
    print("\n\n" + "="*70)
    print("批量查询示例")
    print("="*70)
    
    querier = AstroSourceQuerier(cache_dir="./cache")
    
    # 多个目标
    targets = [
        ("Sirius", 101.2875, -16.7161),
        ("M31", 10.6847, 41.2687),
        ("Vega", 279.2347, 38.7837),
    ]
    
    for name, ra, dec in targets:
        print(f"\n查询: {name}")
        results = querier.query_all(ra, dec, radius=10.0)
        
        # 快速结果统计
        simbad_ok = results.get('simbad') and results['simbad'].data and results['simbad'].data.get('matched')
        gaia_ok = results.get('gaia') and results['gaia'].data and results['gaia'].data.get('found')
        
        print(f"  SIMBAD: {'✓' if simbad_ok else '✗'}, Gaia: {'✓' if gaia_ok else '✗'}")


def single_service_example():
    """
    单独查询某个服务的示例
    """
    print("\n\n" + "="*70)
    print("单独查询服务示例")
    print("="*70)
    
    querier = AstroSourceQuerier(cache_dir="./cache")
    ra, dec = 101.2875, -16.7161
    
    # 只查询 SIMBAD
    print("\n只查询 SIMBAD:")
    result = querier.query_simbad(ra, dec, radius=5.0)
    if result.success and result.data:
        print(f"  成功: {result.data}")
    
    # 只查询 Gaia
    print("\n只查询 Gaia:")
    result = querier.query_gaia(ra, dec, radius=5.0)
    if result.success and result.data:
        print(f"  成功: {result.data.get('source_id', 'N/A')}")


if __name__ == "__main__":
    # 运行示例
    simple_query_example()
    
    # 可选：批量查询
    # batch_query_example()
    
    # 可选：单独服务查询
    # single_service_example()
