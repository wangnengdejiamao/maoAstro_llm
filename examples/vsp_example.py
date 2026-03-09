#!/usr/bin/env python3
"""
VSP 工具包使用示例
================
演示如何使用 VSP 进行天体数据查询

目标: AM Her (一个著名的激变变星)
坐标: RA=274.0554, DEC=49.8679
"""

import os
import sys

# 添加 VSP 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lib'))

# 导入 VSP
from vsp import VSP, query_target, query_vsx_local, find_LM_DR10LRS, generate_ztf_url, get_tess_lc
from astropy.coordinates import SkyCoord
from astropy import units as u


def example_1_basic_query():
    """示例1: 基础完整查询"""
    print("=" * 60)
    print("示例 1: VSP 完整查询")
    print("=" * 60)
    
    # 创建 VSP 实例
    vsp = VSP()
    
    # AM Her 坐标
    ra, dec = 274.0554, 49.8679
    
    # 执行完整查询
    result = vsp.query_all(ra, dec, radius=2.0, name="AM_Her")
    
    print("\n查询结果摘要:")
    print(f"  VSX: {'✓' if result.get('vsx') is not False else '✗'}")
    print(f"  SIMBAD: {'✓' if result.get('simbad') else '✗'}")
    print(f"  Gaia: {'✓' if result.get('gaia') is not False else '✗'}")
    print(f"  LAMOST: {'✓' if result.get('lamost') is not False else '✗'}")
    print(f"  ZTF: {'✓' if result.get('ztf') else '✗'}")
    
    return result


def example_2_quick_query():
    """示例2: 快速查询函数"""
    print("\n" + "=" * 60)
    print("示例 2: 快速查询")
    print("=" * 60)
    
    # EV UMa 坐标
    ra, dec = 13.1316273124, 53.8584719271
    
    # 使用便捷函数
    result = query_target(ra, dec, radius=2.0, name="EV_UMa")
    
    return result


def example_3_individual_modules():
    """示例3: 单独使用模块"""
    print("\n" + "=" * 60)
    print("示例 3: 单独模块查询")
    print("=" * 60)
    
    # 坐标
    ra, dec = 274.0554, 49.8679
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    
    # 1. 查询 VSX
    print("\n1. 查询 VSX...")
    try:
        vsx_data = query_vsx_local(coord, radius=2.0)
        if vsx_data is not False:
            print(f"   找到变星: {vsx_data['Name'][0]}")
            print(f"   类型: {vsx_data['Type'][0]}")
    except Exception as e:
        print(f"   查询失败: {e}")
    
    # 2. 查询 LAMOST
    print("\n2. 查询 LAMOST DR10...")
    try:
        lamost_data = find_LM_DR10LRS(coord, radius=2.0)
        if lamost_data is not False:
            idx_tbl, data_tbl, obsid_list = lamost_data
            print(f"   找到 {len(obsid_list)} 次观测")
            print(f"   OBSID 列表: {obsid_list[:5]}...")  # 只显示前5个
    except Exception as e:
        print(f"   查询失败: {e}")
    
    # 3. 查询 ZTF
    print("\n3. 查询 ZTF...")
    try:
        ztf_data = generate_ztf_url(coord, radius=2.0)
        if ztf_data:
            print(f"   找到 ZTF 数据")
    except Exception as e:
        print(f"   查询失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("VSP (Variable Star Pipeline) 工具包示例")
    print("=" * 60)
    
    try:
        # 运行示例
        example_1_basic_query()
    except Exception as e:
        print(f"示例 1 出错: {e}")
    
    try:
        example_2_quick_query()
    except Exception as e:
        print(f"示例 2 出错: {e}")
    
    try:
        example_3_individual_modules()
    except Exception as e:
        print(f"示例 3 出错: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
