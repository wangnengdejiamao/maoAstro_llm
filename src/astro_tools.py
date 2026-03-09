#!/usr/bin/env python3
"""
天文工具集
==========
消光查询、ZTF光变曲线、多波段测光、光谱查询
"""

import os
import sys
import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

warnings.filterwarnings('ignore')

# 尝试导入healpy
try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False
    warnings.warn("healpy未安装，消光查询功能将不可用")


@dataclass
class Config:
    """配置"""
    EXTINCTION_FILE: str = "./data/sfd_ebv.fits"
    EXTINCTION_FILE_ALT: str = "../Astro_qwen/csfd_ebv.fits"
    OUTPUT_DIR: str = "./output"
    ZTF_RADIUS: float = 3.0  # arcsec
    VIZIER_RADIUS: float = 5.0


CONFIG = Config()

# 自动检测消光文件路径
if not os.path.exists(CONFIG.EXTINCTION_FILE):
    if os.path.exists(CONFIG.EXTINCTION_FILE_ALT):
        CONFIG.EXTINCTION_FILE = CONFIG.EXTINCTION_FILE_ALT
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)


class ExtinctionTool:
    """消光查询工具 - 使用HEALPix CSFD/SFD地图"""
    
    _csfd_map = None
    _sfd_map = None
    _nside = None
    _loaded = False
    
    @classmethod
    def load(cls):
        """加载消光数据"""
        if cls._loaded:
            return
        try:
            import healpy as hp
            
            # 加载CSFD和SFD地图
            csfd_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'csfd_ebv.fits')
            sfd_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sfd_ebv.fits')
            
            # 尝试不同路径
            if not os.path.exists(csfd_file):
                csfd_file = './data/csfd_ebv.fits'
                sfd_file = './data/sfd_ebv.fits'
            
            cls._csfd_map = hp.read_map(csfd_file, dtype=np.float64)
            cls._sfd_map = hp.read_map(sfd_file, dtype=np.float64)
            cls._nside = hp.get_nside(cls._csfd_map)
            cls._loaded = True
        except Exception as e:
            print(f"消光数据加载失败: {e}")
    
    @classmethod
    def query(cls, ra: float, dec: float, use_csfd: bool = True) -> Dict:
        """
        查询消光
        
        Args:
            ra: 赤经 (度)
            dec: 赤纬 (度)
            use_csfd: 使用CSFD修正地图 (推荐), False则使用原始SFD
        
        Returns:
            消光值字典
        """
        cls.load()
        
        if cls._csfd_map is None:
            return {'success': False, 'error': '消光数据未加载'}
        
        try:
            import healpy as hp
            
            # 转换为HEALPix坐标
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            
            # 查询像素索引
            pixel = hp.ang2pix(cls._nside, theta, phi)
            
            # 获取消光值
            if use_csfd:
                ebv = float(cls._csfd_map[pixel])
                source = 'CSFD'
            else:
                ebv = float(cls._sfd_map[pixel])
                source = 'SFD'
            
            # 计算银道坐标
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            gal = coord.galactic
            
            return {
                'success': True,
                'E_B_V': round(ebv, 6),
                'A_V': round(3.1 * ebv, 6),
                'source': source,
                'nside': cls._nside,
                'pixel': int(pixel),
                'l': round(gal.l.deg, 4),
                'b': round(gal.b.deg, 4)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class ZTFTool:
    """ZTF光变曲线工具"""
    
    @staticmethod
    def query(ra: float, dec: float, timeout: int = 30) -> Dict:
        """查询ZTF数据"""
        try:
            import requests
            
            cache_file = os.path.join(CONFIG.OUTPUT_DIR, f"ztf_{ra:.4f}_{dec:.4f}.csv")
            
            # 检查缓存
            if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
                df = pd.read_csv(cache_file)
            else:
                radius_deg = CONFIG.ZTF_RADIUS / 3600.0
                url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?" \
                      f"POS=CIRCLE+{ra}+{dec}+{radius_deg}&COLLECTION=&FORMAT=csv"
                
                response = requests.get(url, timeout=timeout)
                if response.status_code != 200:
                    return {'success': False, 'error': f'HTTP {response.status_code}'}
                
                if len(response.content) < 1000:
                    return {'success': False, 'error': 'No data available'}
                
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                df = pd.read_csv(cache_file)
            
            # 处理数据
            df.columns = [c.lower() for c in df.columns]
            if 'catflags' in df.columns:
                df = df[df['catflags'] < 32768]
            
            return {
                'success': True,
                'n_points': len(df),
                'mean_mag': float(df['mag'].mean()) if len(df) > 0 else None,
                'time_span_days': float(df['mjd'].max() - df['mjd'].min()) if len(df) > 1 else 0,
                'bands': df['filtercode'].unique().tolist() if 'filtercode' in df.columns else []
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)[:50]}


class PhotometryTool:
    """多波段测光查询工具"""
    
    CATALOGS = {
        'Gaia_DR3': 'I/355/gaiadr3',
        '2MASS': 'II/246/out',
        'AllWISE': 'II/328/allwise',
        'Pan-STARRS': 'II/349/ps1',
        'GALEX': 'II/335/galex_ais',
    }
    
    @staticmethod
    def query(ra: float, dec: float) -> Dict:
        """查询多波段测光"""
        try:
            from astroquery.vizier import Vizier
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            v = Vizier(row_limit=3, columns=['**'])
            
            results = {}
            for name, cat_id in PhotometryTool.CATALOGS.items():
                try:
                    result = v.query_region(coord, radius=CONFIG.VIZIER_RADIUS * u.arcsec,
                                           catalog=cat_id)
                    if result and len(result) > 0:
                        results[name] = len(result[0])
                    else:
                        results[name] = 0
                except:
                    results[name] = -1
            
            return {
                'success': True,
                'catalogs': results,
                'total_matches': sum(1 for v in results.values() if v > 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class SpectrumTool:
    """光谱查询工具"""
    
    @staticmethod
    def query_sdss(ra: float, dec: float) -> Dict:
        """查询SDSS光谱"""
        try:
            from astroquery.sdss import SDSS
            
            pos = SkyCoord(ra=ra, dec=dec, unit='deg')
            xid = SDSS.query_region(pos, radius=3.0 * u.arcsec, spectro=True)
            
            if xid is None or len(xid) == 0:
                return {'success': True, 'matched': False}
            
            row = xid[0]
            return {
                'success': True,
                'matched': True,
                'class': str(row.get('class', 'Unknown')),
                'subclass': str(row.get('subclass', '')),
                'z': float(row.get('z')) if row.get('z') else None,
                'plate': int(row.get('plate', 0)),
                'mjd': int(row.get('mjd', 0)),
                'fiberid': int(row.get('fiberID', 0))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class AstroTools:
    """天文工具集统一接口"""
    
    def __init__(self):
        self.extinction = ExtinctionTool()
        self.ztf = ZTFTool()
        self.photometry = PhotometryTool()
        self.spectrum = SpectrumTool()
    
    def query_extinction(self, ra: float, dec: float) -> Dict:
        return self.extinction.query(ra, dec)
    
    def query_ztf(self, ra: float, dec: float) -> Dict:
        return self.ztf.query(ra, dec)
    
    def query_photometry(self, ra: float, dec: float) -> Dict:
        return self.photometry.query(ra, dec)
    
    def query_sdss(self, ra: float, dec: float) -> Dict:
        return self.spectrum.query_sdss(ra, dec)


if __name__ == "__main__":
    # 测试工具集
    print("=" * 60)
    print("天文工具集测试")
    print("=" * 60)
    
    tools = AstroTools()
    
    # 测试AM Her坐标
    ra, dec = 274.0554, 49.8679
    print(f"\n测试坐标: RA={ra}, DEC={dec}")
    
    print("\n1. 消光查询:")
    ext = tools.query_extinction(ra, dec)
    print(f"   {ext}")
    
    print("\n2. 测光目录查询:")
    phot = tools.query_photometry(ra, dec)
    print(f"   {phot}")
