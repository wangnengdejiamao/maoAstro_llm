#!/usr/bin/env python3
"""
完整天文数据下载接口 (Complete Astronomical Data Download Interface)
====================================================================

整合 astroquery 所有光谱、图像、数据下载功能
支持下载光谱、图像、时序数据、星表等

作者: AI Assistant
日期: 2026-03-04
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """下载结果"""
    service: str
    success: bool
    local_path: Optional[str] = None
    data_type: Optional[str] = None  # 'spectrum', 'image', 'table', 'lightcurve'
    metadata: Optional[Dict] = None
    error: Optional[str] = None


class SpectrumDownloader:
    """
    光谱下载器
    支持多个光谱数据库
    """
    
    def __init__(self, output_dir: str = "./downloads/spectra"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_sdss_spectrum(self, ra: float, dec: float, radius: float = 5.0) -> DownloadResult:
        """
        下载 SDSS 光谱
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        radius : float
            搜索半径（角秒）
        
        Returns:
        --------
        DownloadResult : 下载结果
        """
        try:
            from astroquery.sdss import SDSS
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # 查询光谱
            spectra = SDSS.query_region(coord, spectro=True, radius=radius * u.arcsec)
            
            if spectra is None or len(spectra) == 0:
                return DownloadResult('sdss', False, error="未找到光谱数据")
            
            # 获取第一个匹配的光谱
            spec = spectra[0]
            plate = int(spec['plate'])
            mjd = int(spec['mjd'])
            fiberid = int(spec['fiberid'])
            
            # 下载光谱文件
            try:
                sp_files = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)
                if sp_files:
                    # 保存文件
                    filename = f"sdss_spec_{plate}_{mjd}_{fiberid}.fits"
                    local_path = self.output_dir / filename
                    
                    # 复制到目标位置
                    import shutil
                    shutil.copy(sp_files[0], local_path)
                    
                    return DownloadResult(
                        'sdss', True, 
                        local_path=str(local_path),
                        data_type='spectrum',
                        metadata={
                            'plate': plate, 'mjd': mjd, 'fiberid': fiberid,
                            'z': float(spec['z']) if 'z' in spec.colnames else None,
                            'class': str(spec['class']) if 'class' in spec.colnames else None,
                        }
                    )
            except Exception as e:
                return DownloadResult('sdss', False, error=f"下载失败: {e}")
            
        except Exception as e:
            return DownloadResult('sdss', False, error=str(e))
    
    def download_heasarc_spectrum(self, object_name: str, mission: str = 'xmm') -> DownloadResult:
        """
        下载 HEASARC 高能光谱 (X射线等)
        
        Parameters:
        -----------
        object_name : str
            天体名称
        mission : str
            任务名称 ('xmm', 'chandra', 'suzaku', 'nustar')
        
        Returns:
        --------
        DownloadResult : 下载结果
        """
        try:
            from astroquery.heasarc import Heasarc
            
            heasarc = Heasarc()
            
            # 查询可用数据
            mission_tables = {
                'xmm': 'xmmmaster',
                'chandra': 'chanmaster',
                'suzaku': 'suzamaster',
                'nustar': 'numaster',
            }
            
            if mission not in mission_tables:
                return DownloadResult('heasarc', False, error=f"不支持的任务: {mission}")
            
            # 查询数据
            table = heasarc.query_object(object_name, mission=mission_tables[mission])
            
            if table is None or len(table) == 0:
                return DownloadResult('heasarc', False, error="未找到数据")
            
            # 返回查询结果（HEASARC通常需要进一步通过FTP下载）
            return DownloadResult(
                'heasarc', True,
                data_type='spectrum',
                metadata={
                    'mission': mission,
                    'n_observations': len(table),
                    'observations': [
                        {
                            'obsid': str(row.get('OBSID', 'N/A')),
                            'date': str(row.get('START_DATE', 'N/A')),
                            'exposure': float(row.get('EXPOSURE', 0)) if row.get('EXPOSURE') else None,
                        }
                        for row in table[:5]  # 只返回前5个
                    ]
                }
            )
            
        except Exception as e:
            return DownloadResult('heasarc', False, error=str(e))
    
    def query_splatalogue(self, frequency_range: Tuple[float, float], 
                          species: Optional[str] = None) -> DownloadResult:
        """
        查询 Splatalogue 分子光谱线数据库
        
        Parameters:
        -----------
        frequency_range : tuple(float, float)
            频率范围 (GHz)
        species : str, optional
            分子种类
        
        Returns:
        --------
        DownloadResult : 查询结果
        """
        try:
            from astroquery.splatalogue import Splatalogue
            
            result = Splatalogue.query_lines(
                min_frequency=frequency_range[0] * u.GHz,
                max_frequency=frequency_range[1] * u.GHz,
                chemical_name=species if species else None
            )
            
            if result is None or len(result) == 0:
                return DownloadResult('splatalogue', False, error="未找到光谱线")
            
            # 保存为文件
            filename = f"splatalogue_{frequency_range[0]}_{frequency_range[1]}GHz.csv"
            local_path = self.output_dir / filename
            result.write(str(local_path), format='csv', overwrite=True)
            
            return DownloadResult(
                'splatalogue', True,
                local_path=str(local_path),
                data_type='table',
                metadata={
                    'n_lines': len(result),
                    'frequency_range': frequency_range,
                    'species': species,
                }
            )
            
        except Exception as e:
            return DownloadResult('splatalogue', False, error=str(e))
    
    def query_nist_lines(self, element: str, wavelength_range: Tuple[float, float],
                         ion_state: Optional[int] = None) -> DownloadResult:
        """
        查询 NIST 原子光谱线数据库
        
        Parameters:
        -----------
        element : str
            元素名称 (如 'H', 'He', 'Fe')
        wavelength_range : tuple(float, float)
            波长范围 (Angstrom)
        ion_state : int, optional
            电离态 (0=中性, 1=一次电离...)
        
        Returns:
        --------
        DownloadResult : 查询结果
        """
        try:
            from astroquery.nist import Nist
            
            result = Nist.query(
                minwav=wavelength_range[0],
                maxwav=wavelength_range[1],
                wavelength_unit='Angstrom',
                element=element,
                ion_state=ion_state
            )
            
            if result is None or len(result) == 0:
                return DownloadResult('nist', False, error="未找到光谱线")
            
            # 保存为文件
            ion_str = f"_{ion_state}" if ion_state is not None else ""
            filename = f"nist_{element}{ion_str}_{wavelength_range[0]}_{wavelength_range[1]}A.csv"
            local_path = self.output_dir / filename
            result.write(str(local_path), format='csv', overwrite=True)
            
            return DownloadResult(
                'nist', True,
                local_path=str(local_path),
                data_type='table',
                metadata={
                    'element': element,
                    'ion_state': ion_state,
                    'n_lines': len(result),
                    'wavelength_range': wavelength_range,
                }
            )
            
        except Exception as e:
            return DownloadResult('nist', False, error=str(e))


class ImageDownloader:
    """
    图像下载器
    支持 DSS、2MASS、WISE、GALEX 等巡天图像
    """
    
    def __init__(self, output_dir: str = "./downloads/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_skyview_image(self, ra: float, dec: float, 
                               surveys: Optional[List[str]] = None,
                               radius: float = 0.2) -> List[DownloadResult]:
        """
        下载 SkyView 图像
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        surveys : list[str], optional
            巡天项目列表
        radius : float
            图像半径（度）
        
        Returns:
        --------
        list[DownloadResult] : 下载结果列表
        """
        if surveys is None:
            surveys = ['DSS2 Red', '2MASS-J', 'WISE 3.4', 'GALEX Near UV']
        
        try:
            from astroquery.skyview import SkyView
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            results = []
            for survey in surveys:
                try:
                    paths = SkyView.get_images(
                        position=coord,
                        survey=survey,
                        radius=radius * u.deg,
                        pixels=500
                    )
                    
                    if paths:
                        # 保存文件
                        safe_name = survey.replace(' ', '_').replace('.', '')
                        filename = f"skyview_{safe_name}_RA{ra:.3f}_DEC{dec:.3f}.fits"
                        local_path = self.output_dir / filename
                        
                        # 保存 HDU
                        hdu = paths[0][0]
                        hdu.writeto(local_path, overwrite=True)
                        
                        results.append(DownloadResult(
                            'skyview', True,
                            local_path=str(local_path),
                            data_type='image',
                            metadata={'survey': survey, 'radius': radius}
                        ))
                    else:
                        results.append(DownloadResult(
                            'skyview', False,
                            error=f"{survey}: 无数据"
                        ))
                        
                except Exception as e:
                    results.append(DownloadResult(
                        'skyview', False,
                        error=f"{survey}: {str(e)}"
                    ))
            
            return results
            
        except Exception as e:
            return [DownloadResult('skyview', False, error=str(e))]
    
    def download_esasky_image(self, ra: float, dec: float, 
                              mission: str = 'XMM',
                              radius: float = 0.1) -> DownloadResult:
        """
        下载 ESASky 图像 (ESA多波段服务)
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        mission : str
            任务名称 ('XMM', 'Herschel', 'Integral')
        radius : float
            搜索半径（度）
        
        Returns:
        --------
        DownloadResult : 下载结果
        """
        try:
            from astroquery.esasky import ESASky
            
            # 查询可用图像
            images = ESASky.query_region_maps(
                position=SkyCoord(ra=ra, dec=dec, unit=u.deg),
                radius=radius * u.deg,
                missions=mission
            )
            
            if images is None or len(images) == 0:
                return DownloadResult('esasky', False, error="未找到图像")
            
            # 返回查询结果（ESASky需要进一步下载）
            return DownloadResult(
                'esasky', True,
                data_type='image',
                metadata={
                    'mission': mission,
                    'n_images': len(images),
                    'images': [
                        {
                            'obsid': str(img.get('observation_id', 'N/A')),
                            'filter': str(img.get('filter', 'N/A')),
                        }
                        for img in images[:5]
                    ]
                }
            )
            
        except Exception as e:
            return DownloadResult('esasky', False, error=str(e))
    
    def download_hips_image(self, ra: float, dec: float, 
                            hips_survey: str = 'CDS/P/DSS2/color',
                            fov: float = 0.5, 
                            width: int = 500) -> DownloadResult:
        """
        下载 HiPS 图像
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        hips_survey : str
            HiPS 巡天名称
        fov : float
            视场（度）
        width : int
            图像宽度（像素）
        
        Returns:
        --------
        DownloadResult : 下载结果
        """
        try:
            from astroquery.hips2fits import hips2fits
            
            result = hips2fits.query(
                hips=hips_survey,
                width=width,
                height=int(width * 0.8),  # 保持比例
                ra=ra,
                dec=dec,
                fov=fov
            )
            
            if result is None:
                return DownloadResult('hips2fits', False, error="下载失败")
            
            # 保存文件
            safe_survey = hips_survey.replace('/', '_')
            filename = f"hips_{safe_survey}_RA{ra:.3f}_DEC{dec:.3f}.fits"
            local_path = self.output_dir / filename
            result.writeto(local_path, overwrite=True)
            
            return DownloadResult(
                'hips2fits', True,
                local_path=str(local_path),
                data_type='image',
                metadata={
                    'hips_survey': hips_survey,
                    'fov': fov,
                    'width': width
                }
            )
            
        except Exception as e:
            return DownloadResult('hips2fits', False, error=str(e))


class LightCurveDownloader:
    """
    光变曲线下载器
    支持 Kepler、TESS、OGLE 等时序数据
    """
    
    def __init__(self, output_dir: str = "./downloads/lightcurves"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def query_mast_lightcurve(self, target_name: str, 
                              mission: str = 'TESS') -> DownloadResult:
        """
        查询 MAST 光变曲线 (Kepler/TESS)
        
        Parameters:
        -----------
        target_name : str
            目标名称
        mission : str
            任务名称 ('Kepler', 'TESS', 'K2')
        
        Returns:
        --------
        DownloadResult : 查询结果
        """
        try:
            from astroquery.mast import Mast
            
            # 查询观测数据
            service = f"Mast.Catalogs.Filtered.{mission}.LongCadence"
            params = {
                'columns': '*',
                'filters': [
                    {'paramName': 'target_name', 'values': [target_name]}
                ]
            }
            
            result = Mast.service_request(service, params)
            
            if result is None or len(result) == 0:
                return DownloadResult('mast', False, error="未找到光变曲线数据")
            
            # 返回查询结果（MAST需要进一步下载LC文件）
            return DownloadResult(
                'mast', True,
                data_type='lightcurve',
                metadata={
                    'mission': mission,
                    'target': target_name,
                    'n_observations': len(result),
                    'observations': [
                        {
                            'sector': str(row.get('sequence_number', 'N/A')),
                            'camera': str(row.get('camera', 'N/A')),
                            'ccd': str(row.get('ccd', 'N/A')),
                        }
                        for row in result[:5]
                    ]
                }
            )
            
        except Exception as e:
            return DownloadResult('mast', False, error=str(e))
    
    def query_ogle_lightcurve(self, ra: float, dec: float) -> DownloadResult:
        """
        查询 OGLE 光变曲线
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        
        Returns:
        --------
        DownloadResult : 查询结果
        """
        try:
            from astroquery.ogle import Ogle
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # 查询星际消光（OGLE的主要功能）
            result = Ogle.query_region(coord=coord, width=0.1 * u.deg)
            
            if result is None or len(result) == 0:
                return DownloadResult('ogle', False, error="未找到数据")
            
            # 保存为文件
            filename = f"ogle_RA{ra:.3f}_DEC{dec:.3f}.csv"
            local_path = self.output_dir / filename
            result.write(str(local_path), format='csv', overwrite=True)
            
            return DownloadResult(
                'ogle', True,
                local_path=str(local_path),
                data_type='table',
                metadata={
                    'n_stars': len(result),
                }
            )
            
        except Exception as e:
            return DownloadResult('ogle', False, error=str(e))


class CatalogDownloader:
    """
    星表下载器
    """
    
    def __init__(self, output_dir: str = "./downloads/catalogs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_vizier_catalog(self, ra: float, dec: float, 
                                catalog: str,
                                radius: float = 1.0) -> DownloadResult:
        """
        下载 VizieR 星表数据
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        catalog : str
            星表名称 (如 'I/355/gaiadr3', 'B/vsx/vsx')
        radius : float
            搜索半径（角分）
        
        Returns:
        --------
        DownloadResult : 下载结果
        """
        try:
            from astroquery.vizier import Vizier
            
            vizier = Vizier(columns=['**'])
            vizier.ROW_LIMIT = 1000
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            result = vizier.query_region(
                coord,
                radius=radius * u.arcmin,
                catalog=catalog
            )
            
            if result is None or len(result) == 0:
                return DownloadResult('vizier', False, error="未找到数据")
            
            # 保存为文件
            table = result[0]
            safe_cat = catalog.replace('/', '_')
            filename = f"vizier_{safe_cat}_RA{ra:.3f}_DEC{dec:.3f}.csv"
            local_path = self.output_dir / filename
            table.write(str(local_path), format='csv', overwrite=True)
            
            return DownloadResult(
                'vizier', True,
                local_path=str(local_path),
                data_type='table',
                metadata={
                    'catalog': catalog,
                    'n_rows': len(table),
                    'columns': list(table.colnames),
                }
            )
            
        except Exception as e:
            return DownloadResult('vizier', False, error=str(e))


class CompleteAstroDataInterface:
    """
    完整天文数据接口
    整合所有下载功能
    """
    
    def __init__(self, base_output_dir: str = "./downloads"):
        self.spectrum_downloader = SpectrumDownloader(f"{base_output_dir}/spectra")
        self.image_downloader = ImageDownloader(f"{base_output_dir}/images")
        self.lightcurve_downloader = LightCurveDownloader(f"{base_output_dir}/lightcurves")
        self.catalog_downloader = CatalogDownloader(f"{base_output_dir}/catalogs")
    
    def download_all(self, ra: float, dec: float, 
                     target_name: Optional[str] = None) -> Dict[str, List[DownloadResult]]:
        """
        下载所有可用数据
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称（用于某些服务的查询）
        
        Returns:
        --------
        dict : 各类数据的下载结果
        """
        results = {
            'spectra': [],
            'images': [],
            'lightcurves': [],
            'catalogs': [],
        }
        
        logger.info(f"开始完整数据下载: RA={ra}, DEC={dec}")
        
        # 1. 下载光谱
        logger.info("尝试下载 SDSS 光谱...")
        sdss_spec = self.spectrum_downloader.download_sdss_spectrum(ra, dec)
        results['spectra'].append(sdss_spec)
        
        # 2. 下载图像
        logger.info("尝试下载 SkyView 图像...")
        skyview_imgs = self.image_downloader.download_skyview_image(
            ra, dec, 
            surveys=['DSS2 Red', '2MASS-J', 'WISE 3.4']
        )
        results['images'].extend(skyview_imgs)
        
        # 3. 下载 HiPS 图像
        logger.info("尝试下载 HiPS 图像...")
        hips_img = self.image_downloader.download_hips_image(ra, dec)
        results['images'].append(hips_img)
        
        # 4. 下载星表
        logger.info("尝试下载 VizieR 星表...")
        for catalog in ['I/355/gaiadr3', 'II/246/out', 'II/328/allwise']:
            cat_result = self.catalog_downloader.download_vizier_catalog(ra, dec, catalog)
            results['catalogs'].append(cat_result)
        
        # 5. 如果有目标名称，查询光变曲线
        if target_name:
            logger.info("尝试查询光变曲线...")
            lc_result = self.lightcurve_downloader.query_mast_lightcurve(target_name, 'TESS')
            results['lightcurves'].append(lc_result)
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """生成下载结果汇总"""
        lines = []
        lines.append("="*70)
        lines.append("天文数据下载汇总")
        lines.append("="*70)
        
        # 光谱
        lines.append("\n【光谱】")
        for r in results['spectra']:
            status = "✓" if r.success else "✗"
            lines.append(f"  {status} {r.service}: {r.local_path if r.local_path else r.error}")
        
        # 图像
        lines.append("\n【图像】")
        for r in results['images']:
            status = "✓" if r.success else "✗"
            if r.success and r.metadata:
                survey = r.metadata.get('survey', r.service)
                lines.append(f"  {status} {survey}: {r.local_path}")
            else:
                lines.append(f"  {status} {r.service}: {r.error}")
        
        # 星表
        lines.append("\n【星表】")
        for r in results['catalogs']:
            status = "✓" if r.success else "✗"
            if r.success and r.metadata:
                lines.append(f"  {status} {r.metadata['catalog']}: {r.metadata['n_rows']} 行")
            else:
                lines.append(f"  {status} {r.service}: {r.error}")
        
        lines.append("\n" + "="*70)
        return '\n'.join(lines)


def demo():
    """演示完整数据下载"""
    print("="*70)
    print("完整天文数据下载演示")
    print("="*70)
    
    # 创建接口
    interface = CompleteAstroDataInterface()
    
    # 测试目标：M31 中心附近
    ra, dec = 10.6847, 41.2687
    
    print(f"\n目标坐标: RA={ra}, DEC={dec} (M31)")
    print("\n开始下载数据...\n")
    
    # 下载所有数据
    results = interface.download_all(ra, dec, target_name="M31")
    
    # 生成汇总
    summary = interface.generate_summary(results)
    print(summary)
    
    print("\n演示完成!")


if __name__ == "__main__":
    demo()
