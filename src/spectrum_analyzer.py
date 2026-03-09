#!/usr/bin/env python3
"""
光谱分析模块
============
获取和绘制LAMOST/SDSS光谱
使用 astroquery.sdss 进行光谱查询和下载
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')


class SpectrumAnalyzer:
    """光谱分析器"""
    
    @staticmethod
    def query_sdss_spectrum(ra, dec, radius=3.0):
        """
        使用astroquery查询SDSS光谱
        
        Args:
            ra: 赤经 (度)
            dec: 赤纬 (度)
            radius: 搜索半径 (角秒)
            
        Returns:
            光谱信息字典
        """
        try:
            from astroquery.sdss import SDSS
            
            # 定义坐标
            pos = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
            
            # 查询SDSS (spectro=True 仅搜索有光谱的数据)
            xid = SDSS.query_region(
                pos, 
                radius=radius * u.arcsec, 
                spectro=True
            )
            
            if xid is None or len(xid) == 0:
                return {
                    'success': False, 
                    'message': 'No SDSS spectrum found at this position'
                }
            
            return {
                'success': True,
                'matches': xid,
                'count': len(xid),
                'message': f'Found {len(xid)} spectrum(s)'
            }
            
        except ImportError:
            return {
                'success': False, 
                'message': 'astroquery not installed'
            }
        except Exception as e:
            return {
                'success': False, 
                'message': f'Query error: {str(e)}'
            }
    
    @staticmethod
    def download_sdss_spectra(query_result, name, output_dir='./output/data'):
        """
        下载SDSS光谱
        
        Args:
            query_result: SDSS.query_region返回的结果
            name: 目标名称
            output_dir: 输出目录
            
        Returns:
            下载的光谱文件路径列表
        """
        from astroquery.sdss import SDSS
        
        downloaded_files = []
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 下载光谱
            spectra = SDSS.get_spectra(matches=query_result)
            
            if spectra is None:
                return downloaded_files
            
            # 清理文件名
            safe_name = str(name).replace(' ', '_').replace('/', '-').replace(':', '').strip()
            
            # 遍历查询结果和下载的光谱
            for i, sp in enumerate(spectra):
                try:
                    # 获取SDSS标识符
                    plate = query_result[i]['plate']
                    mjd = query_result[i]['mjd']
                    fiber = query_result[i]['fiberID']
                    
                    # 构造文件名
                    filename = f"{safe_name}_spec-{plate}-{mjd}-{fiber:04d}.fits"
                    save_path = os.path.join(output_dir, filename)
                    
                    # 保存文件
                    sp.writeto(save_path, overwrite=True)
                    downloaded_files.append({
                        'path': save_path,
                        'filename': filename,
                        'plate': plate,
                        'mjd': mjd,
                        'fiber': fiber
                    })
                    
                except Exception as e:
                    print(f"  保存光谱 {i+1} 失败: {e}")
                    continue
            
            return downloaded_files
            
        except Exception as e:
            print(f"  下载光谱失败: {e}")
            return downloaded_files
    
    @staticmethod
    def read_and_plot_spectrum(fits_file, output_path=None, title="Spectrum"):
        """
        读取并绘制光谱
        
        Args:
            fits_file: FITS文件路径
            output_path: 输出图像路径
            title: 图表标题
            
        Returns:
            光谱数据字典
        """
        try:
            with fits.open(fits_file) as hdul:
                # 读取数据表
                data = hdul[1].data
                
                # 获取波长和流量
                # SDSS格式: loglam需要转换为线性波长
                if 'loglam' in data.columns.names:
                    loglam = data['loglam']
                    wavelength = 10**loglam
                    flux = data['flux']
                elif 'wavelength' in data.columns.names:
                    wavelength = data['wavelength']
                    flux = data['flux']
                else:
                    # 尝试其他常见列名
                    wavelength = data[data.columns.names[0]]
                    flux = data[data.columns.names[1]]
                
                # 获取表头信息
                header = hdul[0].header
                obj_ra = header.get('RA', 0)
                obj_dec = header.get('DEC', 0)
                plate = header.get('PLATEID', 0)
                mjd = header.get('MJD', 0)
                fiber = header.get('FIBERID', 0)
                
                # 绘制光谱
                if output_path:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    
                    ax.plot(wavelength, flux, 'k-', linewidth=0.8, alpha=0.8)
                    
                    # 标记重要发射线
                    emission_lines = {
                        'Hα': 6563,
                        'Hβ': 4861,
                        'Hγ': 4340,
                        'Hδ': 4102,
                        '[OIII]': 5007,
                        '[NII]': 6584,
                        '[SII]': 6717,
                    }
                    
                    yrange = np.nanmax(flux) - np.nanmin(flux)
                    ypos = np.nanmax(flux) - 0.1 * yrange
                    
                    for name, wl in emission_lines.items():
                        if np.nanmin(wavelength) < wl < np.nanmax(wavelength):
                            ax.axvline(wl, color='r', linestyle='--', alpha=0.3)
                            ax.text(wl, ypos, name, rotation=90, 
                                   fontsize=8, color='red', alpha=0.7)
                    
                    ax.set_xlabel('Wavelength (Å)', fontsize=12)
                    ax.set_ylabel('Flux', fontsize=12)
                    ax.set_title(f'{title}\nRA={obj_ra:.4f}°, Dec={obj_dec:.4f}° | Plate={plate} MJD={mjd} Fiber={fiber}',
                                fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                
                return {
                    'success': True,
                    'wavelength': wavelength,
                    'flux': flux,
                    'ra': obj_ra,
                    'dec': obj_dec,
                    'plot_path': output_path
                }
                
        except Exception as e:
            print(f"  读取光谱失败: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def analyze_and_plot(ra, dec, name, output_dir='./output', search_radius=3.0, 
                         download=True, plot=True):
        """
        综合分析光谱并绘图 - 使用astroquery自动下载
        
        Args:
            ra: 赤经 (度)
            dec: 赤纬 (度)
            name: 目标名称
            output_dir: 输出目录
            search_radius: 搜索半径 (角秒)
            download: 是否自动下载光谱
            plot: 是否绘制光谱图
            
        Returns:
            分析结果字典
        """
        result = {
            'success': False,
            'name': name,
            'ra': ra,
            'dec': dec,
            'spectra': [],
            'downloaded': [],
            'plots': []
        }
        
        print(f"\n【光谱分析】{name}")
        print(f"  坐标: RA={ra}, DEC={dec}")
        
        # 1. 使用astroquery查询SDSS光谱
        print(f"  查询SDSS光谱 (半径={search_radius}角秒)...")
        query_result = SpectrumAnalyzer.query_sdss_spectrum(ra, dec, search_radius)
        
        if not query_result.get('success'):
            print(f"  {query_result.get('message', 'No data')}")
            
            # 如果没有在线数据，检查本地文件
            print(f"  检查本地光谱文件...")
            local_result = SpectrumAnalyzer._check_local_spectra(
                ra, dec, name, output_dir, search_radius
            )
            if local_result.get('success'):
                result.update(local_result)
            
            return result
        
        print(f"  ✓ 发现 {query_result['count']} 条SDSS光谱!")
        result['matches'] = query_result['matches']
        
        # 2. 下载光谱
        if download:
            print(f"  正在下载光谱...")
            data_dir = os.path.join(output_dir, 'data')
            downloaded = SpectrumAnalyzer.download_sdss_spectra(
                query_result['matches'], name, data_dir
            )
            
            if downloaded:
                print(f"  ✓ 下载完成: {len(downloaded)} 个文件")
                result['downloaded'] = downloaded
                result['success'] = True
                
                # 3. 绘制光谱
                if plot:
                    print(f"  绘制光谱图...")
                    fig_dir = os.path.join(output_dir, 'figures')
                    os.makedirs(fig_dir, exist_ok=True)
                    
                    for i, spec_info in enumerate(downloaded):
                        output_path = os.path.join(
                            fig_dir, 
                            f"{name}_spectrum_{i+1}.png"
                        )
                        
                        spec_result = SpectrumAnalyzer.read_and_plot_spectrum(
                            spec_info['path'],
                            output_path=output_path,
                            title=f"{name} - Spectrum {i+1}"
                        )
                        
                        if spec_result['success']:
                            result['plots'].append(output_path)
                            result['spectra'].append({
                                'type': 'sdss_online',
                                'file': spec_info['path'],
                                'plot': output_path,
                                'plate': spec_info['plate'],
                                'mjd': spec_info['mjd'],
                                'fiber': spec_info['fiber']
                            })
                    
                    print(f"  ✓ 生成 {len(result['plots'])} 个光谱图")
            else:
                print(f"  ✗ 下载失败")
        
        return result
    
    @staticmethod
    def _check_local_spectra(ra, dec, name, output_dir, search_radius):
        """检查本地光谱文件"""
        result = {
            'success': False,
            'spectra': []
        }
        
        # 检查本地SDSS目录
        sdss_dir = os.path.join(os.path.dirname(__file__), '..', 'SDSS_Spectra_Downloads')
        if not os.path.exists(sdss_dir):
            sdss_dir = './SDSS_Spectra_Downloads'
        
        if not os.path.exists(sdss_dir):
            return result
        
        import glob
        fits_files = glob.glob(os.path.join(sdss_dir, '*.fits'))
        
        local_spectra = []
        for fits_file in fits_files:
            try:
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    file_ra = header.get('RA', 0)
                    file_dec = header.get('DEC', 0)
                    
                    # 计算角距离
                    coord1 = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                    coord2 = SkyCoord(ra=file_ra*u.deg, dec=file_dec*u.deg)
                    sep = coord1.separation(coord2).arcsecond
                    
                    if sep < search_radius:
                        local_spectra.append({
                            'file': fits_file,
                            'separation': sep,
                            'ra': file_ra,
                            'dec': file_dec
                        })
            except:
                continue
        
        # 绘制本地光谱
        if local_spectra:
            print(f"  ✓ 发现 {len(local_spectra)} 个本地光谱")
            
            fig_dir = os.path.join(output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            
            for i, spec_info in enumerate(local_spectra[:3]):  # 最多3个
                output_path = os.path.join(fig_dir, f"{name}_spectrum_{i+1}.png")
                
                spec_result = SpectrumAnalyzer.read_and_plot_spectrum(
                    spec_info['file'],
                    output_path=output_path,
                    title=f"{name} - Local Spectrum {i+1}"
                )
                
                if spec_result['success']:
                    result['spectra'].append({
                        'type': 'local',
                        'file': spec_info['file'],
                        'plot': output_path,
                        'separation': spec_info['separation']
                    })
            
            result['success'] = len(result['spectra']) > 0
        
        return result


# 便捷函数
def analyze_spectrum(ra, dec, name, output_dir='./output', **kwargs):
    """便捷函数：分析光谱"""
    analyzer = SpectrumAnalyzer()
    return analyzer.analyze_and_plot(ra, dec, name, output_dir, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 测试EV UMa的光谱查询
    print("="*60)
    print("测试SDSS光谱查询")
    print("="*60)
    
    # EV UMa的坐标
    ra, dec = 207.1084, 42.7841
    
    result = SpectrumAnalyzer.analyze_and_plot(
        ra, dec, 
        name="EV_UMa_Test",
        output_dir='./output',
        search_radius=3.0,
        download=True,
        plot=True
    )
    
    print("\n结果:")
    print(f"  成功: {result['success']}")
    print(f"  下载文件数: {len(result.get('downloaded', []))}")
    print(f"  生成图数: {len(result.get('plots', []))}")
