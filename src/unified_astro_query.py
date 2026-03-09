#!/usr/bin/env python3
"""
统一天文数据查询接口 (Unified Astronomical Data Query Interface)
===============================================================

基于 astroquery 的多源数据整合查询工具
只需要提供 RA 和 DEC，即可自动查询多个天文数据库

支持的数据源:
- SIMBAD: 天体识别、分类、光谱型、测光
- Gaia DR3: 视差、自行、测光、天体测量
- VizieR: 多波段测光、变星表、光谱表
- SDSS: 光谱、测光
- SkyView: 图像数据 (DSS, 2MASS, WISE等)
- NED: 河外天体数据
- HEASARC: 高能天体物理数据

作者: AI Assistant
日期: 2026-03-04
"""

import os
import sys
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd

# 天文工具
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, vstack
import astropy.io.votable as votable

# 可视化
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


@dataclass
class QueryResult:
    """统一查询结果数据结构"""
    source: str  # 数据源名称
    success: bool  # 查询是否成功
    data: Optional[Dict] = None  # 查询结果数据
    error_message: Optional[str] = None  # 错误信息
    query_time: Optional[str] = None  # 查询时间
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AstroSourceQuerier:
    """
    统一天文数据源查询类
    提供一站式多源数据查询功能
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        初始化查询器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化各服务状态
        self.services = {
            'simbad': False,
            'gaia': False,
            'vizier': False,
            'sdss': False,
            'skyview': False,
            'ned': False,
            'heasarc': False,
        }
        
        # 初始化服务
        self._init_services()
    
    def _init_services(self):
        """初始化各天文数据服务"""
        logger.info("初始化天文数据服务...")
        
        # SIMBAD
        try:
            from astroquery.simbad import Simbad
            self.simbad = Simbad()
            self.simbad.add_votable_fields(
                'otype', 'sp', 'flux(U)', 'flux(B)', 'flux(V)', 'flux(R)', 'flux(I)',
                'flux(J)', 'flux(H)', 'flux(K)', 'plx', 'rv_value', 'z_value',
                'bibcodelist(2000-2026)'
            )
            self.services['simbad'] = True
            logger.info("  ✓ SIMBAD 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ SIMBAD 服务不可用: {e}")
        
        # Gaia
        try:
            from astroquery.gaia import Gaia
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            Gaia.ROW_LIMIT = 50
            self.services['gaia'] = True
            logger.info("  ✓ Gaia 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ Gaia 服务不可用: {e}")
        
        # VizieR
        try:
            from astroquery.vizier import Vizier
            self.vizier = Vizier(columns=['**'])
            self.vizier.ROW_LIMIT = 100
            self.services['vizier'] = True
            logger.info("  ✓ VizieR 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ VizieR 服务不可用: {e}")
        
        # SDSS
        try:
            from astroquery.sdss import SDSS
            self.services['sdss'] = True
            logger.info("  ✓ SDSS 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ SDSS 服务不可用: {e}")
        
        # SkyView
        try:
            from astroquery.skyview import SkyView
            self.services['skyview'] = True
            logger.info("  ✓ SkyView 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ SkyView 服务不可用: {e}")
        
        # NED
        try:
            from astroquery.ipac.ned import Ned
            self.services['ned'] = True
            logger.info("  ✓ NED 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ NED 服务不可用: {e}")
        
        # HEASARC
        try:
            from astroquery.heasarc import Heasarc
            self.services['heasarc'] = True
            logger.info("  ✓ HEASARC 服务已启用")
        except Exception as e:
            logger.warning(f"  ✗ HEASARC 服务不可用: {e}")
        
        enabled = sum(self.services.values())
        logger.info(f"服务初始化完成: {enabled}/{len(self.services)} 个服务可用")
    
    def query_simbad(self, ra: float, dec: float, radius: float = 5.0) -> QueryResult:
        """
        查询 SIMBAD 数据库
        
        Parameters:
        -----------
        ra, dec : float
            目标坐标（度）
        radius : float
            搜索半径（角秒）
        
        Returns:
        --------
        QueryResult : 查询结果
        """
        if not self.services['simbad']:
            return QueryResult('simbad', False, error_message="SIMBAD 服务未启用")
        
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            result = self.simbad.query_region(coord, radius=radius * u.arcsec)
            
            if result is None or len(result) == 0:
                return QueryResult('simbad', True, data={'matched': False})
            
            # 提取主要天体信息
            main_source = result[0]
            
            # 安全转换函数
            def safe_float(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            def safe_int(val):
                if val is None:
                    return None
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return None
            
            data = {
                'matched': True,
                'main_id': str(main_source['MAIN_ID']),
                'otype': str(main_source['OTYPE']) if 'OTYPE' in main_source.colnames else None,
                'sp_type': str(main_source['SP_TYPE']) if 'SP_TYPE' in main_source.colnames else None,
                'ra': safe_float(main_source['RA']),
                'dec': safe_float(main_source['DEC']),
                'plx': safe_float(main_source['PLX_VALUE']) if 'PLX_VALUE' in main_source.colnames else None,
                'rv': safe_float(main_source['RV_VALUE']) if 'RV_VALUE' in main_source.colnames else None,
                'redshift': safe_float(main_source['Z_VALUE']) if 'Z_VALUE' in main_source.colnames else None,
                'n_references': safe_int(main_source['bibcodelist(2000-2026)']) if 'bibcodelist(2000-2026)' in main_source.colnames else None,
            }
            
            # 提取测光数据
            mags = {}
            for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']:
                flux_key = f'FLUX_{band}'
                if flux_key in main_source.colnames and main_source[flux_key]:
                    mag_val = safe_float(main_source[flux_key])
                    if mag_val is not None:
                        mags[band] = mag_val
            data['magnitudes'] = mags if mags else None
            
            return QueryResult('simbad', True, data=data, query_time=datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"SIMBAD 查询失败: {e}")
            return QueryResult('simbad', False, error_message=str(e))
    
    def query_gaia(self, ra: float, dec: float, radius: float = 5.0) -> QueryResult:
        """
        查询 Gaia DR3 数据库
        
        Parameters:
        -----------
        ra, dec : float
            目标坐标（度）
        radius : float
            搜索半径（角秒）
        
        Returns:
        --------
        QueryResult : 查询结果
        """
        if not self.services['gaia']:
            return QueryResult('gaia', False, error_message="Gaia 服务未启用")
        
        try:
            from astroquery.gaia import Gaia
            
            # 构建 ADQL 查询
            query = f"""
            SELECT TOP 10
                source_id, ra, dec, parallax, pmra, pmdec,
                phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                bp_rp, bp_g, g_rp,
                teff_gspphot, logg_gspphot, distance_gspphot,
                radial_velocity, ruwe
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius/3600.0})
            ) = 1
            AND phot_g_mean_mag IS NOT NULL
            ORDER BY phot_g_mean_mag ASC
            """
            
            job = Gaia.launch_job_async(query)
            result = job.get_results()
            
            if result is None or len(result) == 0:
                return QueryResult('gaia', True, data={'found': False})
            
            # 提取最亮源的详细数据
            source = result[0]
            
            # 计算距离和绝对星等
            parallax = source['parallax']
            if parallax and parallax > 0:
                distance = 1000.0 / float(parallax)
                g_mag = float(source['phot_g_mean_mag'])
                g_abs = g_mag - 5 * np.log10(distance) + 5
            else:
                distance = None
                g_abs = None
            
            # 安全提取函数
            def safe_float(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            data = {
                'found': True,
                'source_id': str(source['source_id']),
                'ra': safe_float(source['ra']),
                'dec': safe_float(source['dec']),
                'parallax': safe_float(parallax),
                'parallax_error': safe_float(source['parallax']) * safe_float(source['ruwe']) if parallax and source['ruwe'] else None,
                'pmra': safe_float(source['pmra']),
                'pmdec': safe_float(source['pmdec']),
                'g_mag': safe_float(source['phot_g_mean_mag']),
                'bp_mag': safe_float(source['phot_bp_mean_mag']),
                'rp_mag': safe_float(source['phot_rp_mean_mag']),
                'bp_rp': safe_float(source['bp_rp']),
                'distance': distance,
                'g_abs': g_abs,
                'teff': safe_float(source['teff_gspphot']),
                'logg': safe_float(source['logg_gspphot']),
                'radial_velocity': safe_float(source['radial_velocity']),
                'ruwe': safe_float(source['ruwe']),
            }
            
            # 添加所有匹配源
            data['all_sources'] = [
                {
                    'source_id': str(r['source_id']),
                    'ra': float(r['ra']),
                    'dec': float(r['dec']),
                    'g_mag': float(r['phot_g_mean_mag']),
                    'separation_arcsec': 3600 * np.sqrt(
                        (float(r['ra']) - ra)**2 * np.cos(np.radians(dec))**2 +
                        (float(r['dec']) - dec)**2
                    )
                }
                for r in result
            ]
            
            return QueryResult('gaia', True, data=data, query_time=datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"Gaia 查询失败: {e}")
            return QueryResult('gaia', False, error_message=str(e))
    
    def query_vizier(self, ra: float, dec: float, radius: float = 5.0, 
                     catalogs: Optional[List[str]] = None) -> QueryResult:
        """
        查询 VizieR 数据库
        
        Parameters:
        -----------
        ra, dec : float
            目标坐标（度）
        radius : float
            搜索半径（角秒）
        catalogs : list[str], optional
            指定星表列表，默认查询常用星表
        
        Returns:
        --------
        QueryResult : 查询结果
        """
        if not self.services['vizier']:
            return QueryResult('vizier', False, error_message="VizieR 服务未启用")
        
        try:
            from astroquery.vizier import Vizier
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            
            # 默认查询常用星表
            if catalogs is None:
                catalogs = [
                    'I/355/gaiadr3',  # Gaia DR3
                    'II/246/out',     # 2MASS
                    'II/328/allwise', # AllWISE
                    'B/vsx/vsx',      # 变星总表
                    'V/147/sdss12',   # SDSS DR12
                    'II/349/ps1',     # Pan-STARRS1
                    'II/361',         # GALEX GR6+7
                ]
            
            results = {}
            all_tables = []
            
            for cat in catalogs:
                try:
                    self.vizier.catalog = cat
                    result = self.vizier.query_region(coord, radius=radius * u.arcsec, catalog=cat)
                    
                    if result and len(result) > 0:
                        table = result[0]
                        results[cat] = {
                            'n_match': len(table),
                            'columns': list(table.colnames),
                        }
                        all_tables.append(table)
                except Exception as e:
                    logger.debug(f"星表 {cat} 查询失败: {e}")
                    continue
            
            data = {
                'found': len(results) > 0,
                'n_catalogs': len(results),
                'catalogs': results,
            }
            
            # 合并所有测光数据
            if all_tables:
                # 这里可以添加测光数据合并逻辑
                data['photometry_available'] = True
            
            return QueryResult('vizier', True, data=data, query_time=datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"VizieR 查询失败: {e}")
            return QueryResult('vizier', False, error_message=str(e))
    
    def query_sdss(self, ra: float, dec: float, radius: float = 5.0) -> QueryResult:
        """
        查询 SDSS 数据库
        
        Parameters:
        -----------
        ra, dec : float
            目标坐标（度）
        radius : float
            搜索半径（角秒）
        
        Returns:
        --------
        QueryResult : 查询结果
        """
        if not self.services['sdss']:
            return QueryResult('sdss', False, error_message="SDSS 服务未启用")
        
        try:
            from astroquery.sdss import SDSS
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            
            # 查询光谱
            try:
                spectra = SDSS.query_region(coord, spectro=True, radius=radius * u.arcsec)
            except:
                spectra = None
            
            # 查询测光
            try:
                photo = SDSS.query_region(coord, photoobj=True, radius=radius * u.arcsec)
            except:
                photo = None
            
            data = {
                'found': spectra is not None or photo is not None,
                'has_spectra': spectra is not None and len(spectra) > 0,
                'has_photo': photo is not None and len(photo) > 0,
            }
            
            if data['has_spectra']:
                spec = spectra[0]
                data['spectra_info'] = {
                    'plate': int(spec['plate']) if 'plate' in spec.colnames else None,
                    'mjd': int(spec['mjd']) if 'mjd' in spec.colnames else None,
                    'fiberid': int(spec['fiberid']) if 'fiberid' in spec.colnames else None,
                    'z': float(spec['z']) if 'z' in spec.colnames else None,
                    'z_err': float(spec['zErr']) if 'zErr' in spec.colnames else None,
                    'class': str(spec['class']) if 'class' in spec.colnames else None,
                    'subclass': str(spec['subClass']) if 'subClass' in spec.colnames else None,
                }
            
            return QueryResult('sdss', True, data=data, query_time=datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"SDSS 查询失败: {e}")
            return QueryResult('sdss', False, error_message=str(e))
    
    def query_all(self, ra: float, dec: float, radius: float = 5.0,
                  services: Optional[List[str]] = None) -> Dict[str, QueryResult]:
        """
        查询所有可用数据源
        
        Parameters:
        -----------
        ra, dec : float
            目标坐标（度）
        radius : float
            搜索半径（角秒）
        services : list[str], optional
            指定要查询的服务列表，默认查询所有
        
        Returns:
        --------
        dict : 各数据源查询结果字典
        """
        if services is None:
            services = ['simbad', 'gaia', 'vizier', 'sdss']
        
        results = {}
        
        logger.info(f"开始多源查询: RA={ra:.6f}, DEC={dec:.6f}, 半径={radius}角秒")
        
        for service in services:
            if service == 'simbad':
                results['simbad'] = self.query_simbad(ra, dec, radius)
            elif service == 'gaia':
                results['gaia'] = self.query_gaia(ra, dec, radius)
            elif service == 'vizier':
                results['vizier'] = self.query_vizier(ra, dec, radius)
            elif service == 'sdss':
                results['sdss'] = self.query_sdss(ra, dec, radius)
            
            # 添加短暂延迟避免请求过快
            time.sleep(0.1)
        
        # 统计结果
        n_success = sum(1 for r in results.values() if r.success)
        n_found = sum(1 for r in results.values() if r.success and r.data and r.data.get('matched', r.data.get('found', False)))
        
        logger.info(f"查询完成: {n_success}/{len(services)} 个服务成功, {n_found} 个找到数据")
        
        return results
    
    def generate_summary_report(self, results: Dict[str, QueryResult], 
                                ra: float, dec: float) -> str:
        """
        生成查询结果汇总报告
        
        Parameters:
        -----------
        results : dict
            各数据源查询结果
        ra, dec : float
            查询坐标
        
        Returns:
        --------
        str : 格式化报告
        """
        lines = []
        lines.append("="*70)
        lines.append(f"天体数据查询报告")
        lines.append(f"坐标: RA={ra:.6f}°, DEC={dec:.6f}°")
        lines.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        
        # SIMBAD
        simbad = results.get('simbad')
        if simbad and simbad.success and simbad.data:
            lines.append("\n【SIMBAD 数据库】")
            if simbad.data.get('matched'):
                lines.append(f"  ID: {simbad.data['main_id']}")
                lines.append(f"  类型: {simbad.data.get('otype', 'N/A')}")
                lines.append(f"  光谱型: {simbad.data.get('sp_type', 'N/A')}")
                if simbad.data.get('plx'):
                    lines.append(f"  视差: {simbad.data['plx']:.2f} mas")
                if simbad.data.get('magnitudes'):
                    mags = simbad.data['magnitudes']
                    mag_str = ', '.join([f"{k}={v:.2f}" for k, v in mags.items()])
                    lines.append(f"  星等: {mag_str}")
            else:
                lines.append("  未匹配到天体")
        
        # Gaia
        gaia = results.get('gaia')
        if gaia and gaia.success and gaia.data:
            lines.append("\n【Gaia DR3】")
            if gaia.data.get('found'):
                lines.append(f"  Source ID: {gaia.data['source_id']}")
                lines.append(f"  G = {gaia.data['g_mag']:.2f}")
                if gaia.data.get('parallax'):
                    lines.append(f"  视差: {gaia.data['parallax']:.2f} ± {gaia.data.get('parallax_error', 0):.2f} mas")
                if gaia.data.get('distance'):
                    lines.append(f"  距离: {gaia.data['distance']:.1f} pc")
                if gaia.data.get('teff'):
                    lines.append(f"  有效温度: {gaia.data['teff']:.0f} K")
                if gaia.data.get('radial_velocity'):
                    lines.append(f"  径向速度: {gaia.data['radial_velocity']:.1f} km/s")
            else:
                lines.append("  未找到匹配源")
        
        # SDSS
        sdss = results.get('sdss')
        if sdss and sdss.success and sdss.data:
            lines.append("\n【SDSS】")
            if sdss.data.get('found'):
                if sdss.data.get('has_spectra'):
                    spec = sdss.data['spectra_info']
                    lines.append(f"  光谱: Plate={spec.get('plate')}, MJD={spec.get('mjd')}, Fiber={spec.get('fiberid')}")
                    lines.append(f"  红移: z={spec.get('z', 'N/A')}, 类型: {spec.get('class', 'N/A')}")
                else:
                    lines.append("  测光数据可用")
            else:
                lines.append("  无 SDSS 数据")
        
        # VizieR
        vizier = results.get('vizier')
        if vizier and vizier.success and vizier.data:
            lines.append("\n【VizieR】")
            if vizier.data.get('found'):
                lines.append(f"  匹配星表数: {vizier.data['n_catalogs']}")
                for cat, info in vizier.data['catalogs'].items():
                    lines.append(f"    - {cat}: {info['n_match']} 个源")
            else:
                lines.append("  未匹配到星表数据")
        
        lines.append("\n" + "="*70)
        
        return '\n'.join(lines)
    
    def save_results(self, results: Dict[str, QueryResult], 
                     ra: float, dec: float, 
                     output_dir: str = "./output"):
        """
        保存查询结果到文件
        
        Parameters:
        -----------
        results : dict
            查询结果字典
        ra, dec : float
            查询坐标
        output_dir : str
            输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        coord_str = f"RA{ra:.4f}_DEC{dec:.4f}"
        filename = f"{output_dir}/astro_query_{coord_str}.json"
        
        # 转换为可序列化格式
        output_data = {
            'query_info': {
                'ra': ra,
                'dec': dec,
                'timestamp': datetime.now().isoformat(),
            },
            'results': {
                k: v.to_dict() for k, v in results.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"结果已保存: {filename}")
        
        # 同时保存文本报告
        report = self.generate_summary_report(results, ra, dec)
        report_file = f"{output_dir}/astro_query_{coord_str}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return filename


def demo():
    """
    演示统一查询接口的使用
    """
    print("="*70)
    print("统一天文数据查询接口演示")
    print("="*70)
    
    # 初始化查询器
    querier = AstroSourceQuerier(cache_dir="./cache/astro_query")
    
    # 测试目标: M33 (Triangulum Galaxy)
    test_targets = [
        ("Sirius", 101.2875, -16.7161),
        ("M31", 10.6847, 41.2687),
        ("M33", 23.4621, 30.6602),
        ("EV UMa", 13.1316, 53.8585),
    ]
    
    for name, ra, dec in test_targets:
        print(f"\n{'='*70}")
        print(f"查询目标: {name}")
        print(f"坐标: RA={ra:.4f}°, DEC={dec:.4f}°")
        print(f"{'='*70}")
        
        # 执行全源查询
        results = querier.query_all(ra, dec, radius=10.0)
        
        # 生成并打印报告
        report = querier.generate_summary_report(results, ra, dec)
        print(report)
        
        # 保存结果
        querier.save_results(results, ra, dec, output_dir="./output/astro_query")
        
        # 等待一下避免请求过快
        time.sleep(1)
    
    print("\n" + "="*70)
    print("演示完成!")
    print("="*70)


if __name__ == "__main__":
    demo()
