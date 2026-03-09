#!/usr/bin/env python3
"""
扩展天文工具集
==============
包含SIMBAD、TESS、Gaia赫罗图、SED构建等高级功能
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')


# ==================== 1. SIMBAD查询 ====================

class SIMBADQuery:
    """SIMBAD天体数据库查询"""
    
    @staticmethod
    def query(ra: float, dec: float, radius: float = 5.0):
        """查询SIMBAD数据库"""
        try:
            from astroquery.simbad import Simbad
            
            # 配置SIMBAD返回字段
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('otype', 'sp', 'flux(U)', 'flux(B)', 
                                              'flux(V)', 'flux(R)', 'flux(I)',
                                              'flux(J)', 'flux(H)', 'flux(K)',
                                              'plx', 'pmra', 'pmdec', 'rv_value')
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            result = custom_simbad.query_region(coord, radius=radius * u.arcsec)
            
            if result is None or len(result) == 0:
                return {'success': True, 'matched': False}
            
            # 提取第一条记录
            row = result[0]
            
            # 构建返回数据
            data = {
                'success': True,
                'matched': True,
                'main_id': str(row.get('MAIN_ID', 'Unknown')).strip(),
                'otype': str(row.get('OTYPE', 'Unknown')).strip(),
                'sp_type': str(row.get('SP_TYPE', 'N/A')).strip() if row.get('SP_TYPE') else 'N/A',
            }
            
            # 安全获取坐标（处理字符串格式如 "18 16 13.2546"）
            try:
                ra_val = row.get('RA')
                dec_val = row.get('DEC')
                
                # 尝试直接转换
                data['ra'] = float(ra_val)
                data['dec'] = float(dec_val)
            except (ValueError, TypeError):
                # 如果是字符串格式（如 "18 16 13.2546"），用SkyCoord解析
                try:
                    coord_str = f"{ra_val} {dec_val}"
                    parsed = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
                    data['ra'] = float(parsed.ra.degree)
                    data['dec'] = float(parsed.dec.degree)
                except:
                    data['ra'] = 0.0
                    data['dec'] = 0.0
            
            # 添加测光数据
            mags = {}
            for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']:
                key = f'FLUX_{band}'
                if key in row.colnames and row[key] is not None:
                    try:
                        mags[band] = float(row[key])
                    except:
                        pass
            if mags:
                data['magnitudes'] = mags
            
            # 添加视差
            if 'PLX_VALUE' in row.colnames and row['PLX_VALUE'] is not None:
                try:
                    data['parallax'] = float(row['PLX_VALUE'])
                    data['distance'] = 1000.0 / float(row['PLX_VALUE']) if float(row['PLX_VALUE']) > 0 else None
                except:
                    pass
            
            # 添加自行
            if 'PMRA' in row.colnames and row['PMRA'] is not None:
                try:
                    data['pm_ra'] = float(row['PMRA'])
                    data['pm_dec'] = float(row['PMDEC'])
                except:
                    pass
            
            return data
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ==================== 2. TESS数据获取 ====================

class TESSQuery:
    """TESS光变曲线查询"""
    
    @staticmethod
    def query(ra: float, dec: float, radius: float = 10.0):
        """查询TESS数据"""
        try:
            import lightkurve as lk
            
            search_str = f"{ra} {dec}"
            search_result = lk.search_lightcurve(search_str, radius=radius * u.arcsec)
            
            if len(search_result) == 0:
                return {'success': True, 'found': False, 'message': 'No TESS data found'}
            
            # 获取搜索结果摘要
            missions = []
            for i in range(len(search_result)):
                mission = search_result[i].mission[0] if len(search_result[i].mission) > 0 else 'Unknown'
                author = search_result[i].author[0] if len(search_result[i].author) > 0 else 'Unknown'
                year = search_result[i].year[0] if len(search_result[i].year) > 0 else 'Unknown'
                exptime = int(search_result[i].exptime[0].value) if len(search_result[i].exptime) > 0 else 0
                
                missions.append({
                    'mission': mission,
                    'author': author,
                    'year': year,
                    'exptime': exptime
                })
            
            return {
                'success': True,
                'found': True,
                'n_sectors': len(search_result),
                'missions': missions
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def download_and_plot(ra: float, dec: float, output_dir: str, prefix: str = ""):
        """下载TESS数据并绘图"""
        try:
            import lightkurve as lk
            
            search_str = f"{ra} {dec}"
            search_result = lk.search_lightcurve(search_str, radius=10 * u.arcsec)
            
            if len(search_result) == 0:
                return None
            
            plots = []
            for i in range(min(3, len(search_result))):  # 最多下载3个
                try:
                    lc = search_result[i].download()
                    if lc is None:
                        continue
                    
                    # 保存数据
                    mission = search_result[i].mission[0].replace(' ', '_')
                    author = search_result[i].author[0]
                    filename = f"{prefix}_TESS_{mission}_{author}.csv"
                    filepath = os.path.join(output_dir, filename)
                    lc.to_csv(filepath, overwrite=True)
                    
                    # 绘图
                    fig, ax = plt.subplots(figsize=(12, 4))
                    lc.plot(ax=ax, c='k', linewidth=0.5)
                    ax.set_title(f'TESS {mission} - {author}')
                    plt.tight_layout()
                    
                    plot_path = os.path.join(output_dir, f"{prefix}_TESS_{mission}_{author}.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    
                    plots.append(plot_path)
                except Exception as e:
                    print(f"  TESS下载失败: {e}")
                    continue
            
            return plots
            
        except Exception as e:
            print(f"  TESS查询失败: {e}")
            return None


# ==================== 3. Gaia赫罗图 ====================

class GaiaHRDiagram:
    """Gaia赫罗图绘制"""
    
    @staticmethod
    def query_gaia_data(ra: float, dec: float, radius: float = 60.0):
        """查询Gaia DR3数据"""
        try:
            from astroquery.gaia import Gaia
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            
            # 查询中心星
            job = Gaia.cone_search_async(coord, radius=radius * u.arcsec)
            result = job.get_results()
            
            if len(result) == 0:
                return None
            
            return result
            
        except Exception as e:
            print(f"  Gaia查询失败: {e}")
            return None
    
    @staticmethod
    def plot_hr_diagram(ra: float, dec: float, output_path: str, radius: float = 300.0):
        """
        绘制赫罗图
        
        Args:
            ra, dec: 中心坐标
            output_path: 输出图像路径
            radius: 搜索半径（角秒）
        """
        try:
            from astroquery.gaia import Gaia
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            
            # 查询区域星（背景）
            job_bg = Gaia.cone_search_async(coord, radius=radius * u.arcsec)
            bg_stars = job_bg.get_results()
            
            # 查询中心星
            job_center = Gaia.cone_search_async(coord, radius=5 * u.arcsec)
            center_stars = job_center.get_results()
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制背景星
            if bg_stars is not None and len(bg_stars) > 0:
                bp_rp_bg = bg_stars['bp_rp']
                g_abs_bg = bg_stars['phot_g_mean_mag'] - 5 * np.log10(1000.0 / bg_stars['parallax']) + 5
                
                # 过滤有效数据
                mask = (bp_rp_bg > -1) & (bp_rp_bg < 4) & (g_abs_bg > -5) & (g_abs_bg < 15) & (bg_stars['parallax'] > 0)
                ax.scatter(bp_rp_bg[mask], g_abs_bg[mask], c='gray', s=1, alpha=0.3, label='Field stars')
            
            # 绘制中心星
            if center_stars is not None and len(center_stars) > 0:
                star = center_stars[0]
                if star['parallax'] > 0:
                    bp_rp = star['bp_rp']
                    g_abs = star['phot_g_mean_mag'] - 5 * np.log10(1000.0 / star['parallax']) + 5
                    ax.scatter([bp_rp], [g_abs], c='red', s=200, marker='*', 
                              edgecolors='black', linewidth=2, label='Target star', zorder=10)
                    ax.annotate('Target', (bp_rp, g_abs), xytext=(10, 10), 
                               textcoords='offset points', fontsize=12, color='red')
            
            ax.set_xlabel('BP - RP (mag)', fontsize=14)
            ax.set_ylabel('M_G (mag)', fontsize=14)
            ax.set_title(f'Hertzsprung-Russell Diagram\n(RA={ra:.4f}, DEC={dec:.4f})', fontsize=16)
            ax.invert_yaxis()
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"  赫罗图绘制失败: {e}")
            return None


# ==================== 4. SED构建 ====================

class SEDBuilder:
    """光谱能量分布(SED)构建"""
    
    # 波段中心波长（埃）
    WAVELENGTHS = {
        'FUV': 1516, 'NUV': 2267,
        'U': 3600, 'B': 4450, 'V': 5510, 'R': 6580, 'I': 8060,
        'g': 4686, 'r': 6165, 'i': 7481, 'z': 8931,
        'J': 12350, 'H': 16620, 'K': 21590, 'Ks': 21590,
        'W1': 33526, 'W2': 46028, 'W3': 115608, 'W4': 220883,
        'G': 6730, 'BP': 5320, 'RP': 7970,
    }
    
    @staticmethod
    def calculate_flux(magnitude: float, band: str):
        """
        将星等转换为流量
        
        Returns:
            flux_nu (Jy), flux_lambda (erg/s/cm^2/Angstrom)
        """
        # 零 flux (Jy)
        ZP = {
            'U': 1790, 'B': 4063, 'V': 3636, 'R': 3064, 'I': 2416,
            'J': 1595, 'H': 1024, 'K': 667, 'Ks': 667,
            'g': 3562, 'r': 2564, 'i': 1996, 'z': 909,
            'W1': 310, 'W2': 172, 'W3': 31, 'W4': 8,
            'G': 3231, 'BP': 3552, 'RP': 2623,
        }
        
        if band not in ZP:
            return None, None
        
        zp = ZP[band]
        flux_nu = zp * 10**(-0.4 * magnitude)  # Jy
        
        # 转换为波长单位
        wavelength = SEDBuilder.WAVELENGTHS.get(band, 5500)  # Angstrom
        c = 2.998e18  # cm/s
        flux_lambda = flux_nu * 1e-23 * c / (wavelength**2)  # erg/s/cm^2/A
        
        return flux_nu, flux_lambda
    
    @staticmethod
    def build_sed_from_simbad(simbad_data: dict):
        """从SIMBAD数据构建SED"""
        if not simbad_data.get('matched') or 'magnitudes' not in simbad_data:
            return None
        
        sed_data = []
        mags = simbad_data['magnitudes']
        
        for band, mag in mags.items():
            if band in SEDBuilder.WAVELENGTHS:
                flux_nu, flux_lambda = SEDBuilder.calculate_flux(mag, band)
                if flux_lambda:
                    sed_data.append({
                        'band': band,
                        'wavelength': SEDBuilder.WAVELENGTHS[band],
                        'magnitude': mag,
                        'flux_nu': flux_nu,
                        'flux_lambda': flux_lambda
                    })
        
        return pd.DataFrame(sed_data)
    
    @staticmethod
    def plot_sed(sed_df: pd.DataFrame, output_path: str, name: str = "Target"):
        """绘制SED图"""
        if sed_df is None or len(sed_df) == 0:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: 波长 vs 流量
        ax1.scatter(sed_df['wavelength'], sed_df['flux_lambda'], 
                   s=100, c='red', zorder=10, edgecolors='black')
        
        for i, row in sed_df.iterrows():
            ax1.annotate(row['band'], (row['wavelength'], row['flux_lambda']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        ax1.set_xlabel('Wavelength (Angstrom)', fontsize=12)
        ax1.set_ylabel('Flux (erg/s/cm^2/A)', fontsize=12)
        ax1.set_title(f'{name} - Spectral Energy Distribution', fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 右图: 波段 vs 星等
        ax2.scatter(sed_df['wavelength'], sed_df['magnitude'],
                   s=100, c='blue', zorder=10, edgecolors='black')
        
        for i, row in sed_df.iterrows():
            ax2.annotate(row['band'], (row['wavelength'], row['magnitude']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        ax2.set_xlabel('Wavelength (Angstrom)', fontsize=12)
        ax2.set_ylabel('Magnitude', fontsize=12)
        ax2.set_title(f'{name} - Magnitude Distribution', fontsize=14)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path


# ==================== 5. ZTF折叠光变曲线 ====================

class ZTFAnalyzer:
    """ZTF光变曲线分析（含折叠）"""
    
    @staticmethod
    def fold_lightcurve(times, magnitudes, errors, period):
        """
        折叠光变曲线
        
        Returns:
            phase, folded_mag, folded_err
        """
        phase = ((times - times[0]) / period) % 1.0
        return phase, magnitudes, errors
    
    @staticmethod
    def find_period(times, magnitudes, errors, 
                   freq_min=0.1, freq_max=50.0, samples_per_peak=15):
        """
        使用Lomb-Scargle寻找周期
        
        Returns:
            best_period, best_frequency, power_spectrum
        """
        ls = LombScargle(times, magnitudes, errors)
        frequency, power = ls.autopower(
            minimum_frequency=freq_min,
            maximum_frequency=freq_max,
            samples_per_peak=samples_per_peak
        )
        
        best_idx = np.argmax(power)
        best_freq = frequency[best_idx]
        best_period = 1.0 / best_freq
        
        return best_period, best_freq, frequency, power
    
    @staticmethod
    def plot_folded_lc(ztf_data: pd.DataFrame, period: float, 
                      output_path: str, name: str = "Target"):
        """
        绘制折叠光变曲线
        
        Args:
            ztf_data: DataFrame with hjd, mag, magerr, filter columns
            period: 周期（天）
            output_path: 输出路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 原始光变曲线
        ax = axes[0, 0]
        for band, color in [('zg', 'green'), ('zr', 'red'), ('zi', 'blue')]:
            band_data = ztf_data[ztf_data['filter'] == band]
            if len(band_data) > 0:
                ax.errorbar(band_data['hjd'], band_data['mag'], 
                           yerr=band_data['magerr'], fmt='o', 
                           c=color, label=band, markersize=3, alpha=0.6)
        ax.set_xlabel('HJD')
        ax.set_ylabel('Magnitude')
        ax.set_title('Original Light Curve')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # 周期图
        try:
            times = ztf_data['hjd'].values
            mags = ztf_data['mag'].values
            errs = ztf_data['magerr'].values
            
            period_est, freq, frequencies, power = ZTFAnalyzer.find_period(
                times, mags, errs
            )
            
            ax = axes[0, 1]
            ax.plot(frequencies, power, 'k-', linewidth=0.5)
            ax.axvline(1.0/period, color='r', linestyle='--', 
                      label=f'P={period:.4f}d')
            ax.set_xlabel('Frequency (1/day)')
            ax.set_ylabel('Power')
            ax.set_title('Lomb-Scargle Periodogram')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except:
            axes[0, 1].text(0.5, 0.5, 'Period analysis failed', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 折叠光变曲线
        ax = axes[1, 0]
        for band, color in [('zg', 'green'), ('zr', 'red'), ('zi', 'blue')]:
            band_data = ztf_data[ztf_data['filter'] == band]
            if len(band_data) > 0:
                phase, mag, err = ZTFAnalyzer.fold_lightcurve(
                    band_data['hjd'].values,
                    band_data['mag'].values,
                    band_data['magerr'].values,
                    period
                )
                # 绘制两个周期
                ax.errorbar(np.concatenate([phase, phase+1]), 
                           np.concatenate([mag, mag]),
                           yerr=np.concatenate([err, err]),
                           fmt='o', c=color, label=band, markersize=3, alpha=0.6)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Folded Light Curve (P={period:.4f}d)')
        ax.set_xlim(0, 2)
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # 统计信息
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
{name} - ZTF Statistics

Total points: {len(ztf_data)}
Period: {period:.6f} days
Period: {period*24:.4f} hours

By band:
"""
        for band in ['zg', 'zr', 'zi']:
            n = len(ztf_data[ztf_data['filter'] == band])
            stats_text += f"  {band}: {n} points\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'ZTF Light Curve Analysis - {name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        return output_path


# ==================== 测试 ====================

if __name__ == "__main__":
    print("="*60)
    print("扩展工具集测试")
    print("="*60)
    
    # 测试坐标
    ra, dec = 274.0554, 49.8679  # AM Her
    
    print(f"\n测试坐标: RA={ra}, DEC={dec}")
    
    # 测试SIMBAD
    print("\n1. SIMBAD查询...")
    simbad = SIMBADQuery.query(ra, dec)
    print(f"   {simbad}")
    
    # 测试TESS
    print("\n2. TESS查询...")
    tess = TESSQuery.query(ra, dec)
    print(f"   {tess}")
    
    print("\n测试完成!")
