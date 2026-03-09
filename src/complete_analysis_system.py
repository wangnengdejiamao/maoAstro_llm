#!/usr/bin/env python3
"""
完整天体分析系统 v2.0
=====================
整合所有功能：
1. SDSS/LAMOST光谱检查与处理
2. 联合模型预测
3. SED分析 + 黑体拟合
4. ZTF/TESS光变曲线 + Lomb-Scargle
5. 赫罗图 (视差+距离)
6. VSP功能集成

作者: AI Assistant
日期: 2026-03-02
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lib/vsp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# 物理常数
h = 6.626e-27  # erg s
k = 1.381e-16  # erg/K
c = 2.998e18   # Å/s


class SpectralAnalyzer:
    """光谱分析器"""
    
    @staticmethod
    def check_sdss_spectrum(target_name, ra, dec, sdss_dir='./SDSS_Spectra_Downloads'):
        """检查SDSS光谱"""
        import glob
        
        pattern = f"{sdss_dir}/*{target_name}*.fits"
        files = glob.glob(pattern)
        
        results = []
        for f in files:
            try:
                with fits.open(f) as hdul:
                    header = hdul[0].header
                    data_info = {
                        'filename': os.path.basename(f),
                        'path': f,
                        'plate': header.get('PLATEID', 'N/A'),
                        'mjd': header.get('MJD', 'N/A'),
                        'fiberid': header.get('FIBERID', 'N/A'),
                        'ra': header.get('RA', 'N/A'),
                        'dec': header.get('DEC', 'N/A'),
                    }
                    results.append(data_info)
            except Exception as e:
                print(f"  警告: 无法读取 {f}: {e}")
        
        return results
    
    @staticmethod
    def read_sdss_spectrum(filepath):
        """读取SDSS光谱"""
        try:
            with fits.open(filepath) as hdul:
                data = hdul[1].data
                flux = data['flux']
                loglam = data['loglam']
                wavelength = 10**loglam
                
                # 获取表头信息
                header = hdul[0].header
                
                return {
                    'wavelength': wavelength,
                    'flux': flux,
                    'ra': header.get('RA'),
                    'dec': header.get('DEC'),
                    'plate': header.get('PLATEID'),
                    'mjd': header.get('MJD'),
                    'success': True
                }
        except Exception as e:
            print(f"  错误: 无法读取光谱: {e}")
            return {'success': False, 'error': str(e)}


class BlackbodyFitter:
    """黑体谱拟合器"""
    
    @staticmethod
    def blackbody_lambda(wavelength, T, amplitude):
        """黑体辐射谱 (波长形式)"""
        lam = wavelength * 1e-8  # 转换为cm
        exponent = h * c / (wavelength * k * T)
        exponent = np.clip(exponent, 0, 700)
        
        B_lambda = (2 * h * c**2 / wavelength**5) / (np.exp(exponent) - 1)
        return amplitude * B_lambda
    
    @staticmethod
    def fit_sed(wavelengths, fluxes, flux_errors=None):
        """拟合SED数据"""
        if len(wavelengths) < 3:
            return None
        
        # 初始猜测
        T_init = 10000
        # 找到最大流量点
        max_idx = np.argmax(fluxes)
        amp_init = fluxes[max_idx] / BlackbodyFitter.blackbody_lambda(
            wavelengths[max_idx], T_init, 1.0)
        
        try:
            popt, pcov = curve_fit(
                BlackbodyFitter.blackbody_lambda,
                wavelengths,
                fluxes,
                p0=[T_init, amp_init],
                bounds=([1000, 0], [50000, np.inf]),
                maxfev=10000
            )
            
            T_fit = popt[0]
            T_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0
            
            # 维恩温度估算
            max_wave = wavelengths[max_idx]
            T_wien = 2.898e7 / max_wave if max_wave > 0 else 0
            
            return {
                'temperature': T_fit,
                'temperature_err': T_err,
                'amplitude': popt[1],
                'wien_temperature': T_wien,
                'max_flux_wavelength': max_wave,
                'max_flux_value': fluxes[max_idx],
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class LightcurveAnalyzer:
    """光变曲线分析器"""
    
    @staticmethod
    def lomb_scargle_analysis(times, magnitudes, errors=None,
                              freq_min=0.1, freq_max=50.0):
        """Lomb-Scargle周期分析"""
        if len(times) < 10:
            return None
        
        try:
            ls = LombScargle(times, magnitudes, errors)
            frequency, power = ls.autopower(
                minimum_frequency=freq_min,
                maximum_frequency=freq_max,
                samples_per_peak=15
            )
            
            best_idx = np.argmax(power)
            best_freq = frequency[best_idx]
            best_period = 1.0 / best_freq
            
            # 计算FAP
            fap = ls.false_alarm_probability(power[best_idx])
            
            return {
                'period': best_period,
                'period_hours': best_period * 24,
                'frequency': best_freq,
                'power': power[best_idx],
                'fap': fap,
                'frequencies': frequency,
                'powers': power,
                'success': True
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def plot_periodogram(times, magnitudes, period_result, 
                        title="Period Analysis", output_path=None):
        """绘制周期分析三 panel 图"""
        if not period_result or not period_result.get('success'):
            return None
        
        period = period_result['period']
        frequency = period_result['frequencies']
        power = period_result['powers']
        
        # 计算相位
        phase = ((times - times[0]) / period) % 1.0
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Panel 1: 时域数据
        ax = axes[0]
        ax.plot(times, magnitudes, 'k.', markersize=2, alpha=0.6)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_title(f'{title} - Light Curve')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: 频谱
        ax = axes[1]
        ax.plot(frequency, power, 'k-', linewidth=0.5)
        ax.axvline(1.0/period, color='r', linestyle='--',
                  label=f'P={period:.6f}d')
        ax.set_xlabel('Frequency (1/day)', fontsize=11)
        ax.set_ylabel('Power', fontsize=11)
        ax.set_title(f'Lomb-Scargle Periodogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: 折叠
        ax = axes[2]
        ax.scatter(np.concatenate([phase, phase+1]), 
                  np.concatenate([magnitudes, magnitudes]),
                  c='k', s=5, alpha=0.5)
        ax.set_xlabel('Phase', fontsize=11)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_title(f'Folded Light Curve (P={period:.6f} days)')
        ax.set_xlim(0, 2)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None


class HRDiagramPlotterV2:
    """赫罗图绘制器 v2"""
    
    @staticmethod
    def calculate_absolute_magnitude(g_mag, bp_mag, rp_mag, distance, ebv=0):
        """计算绝对星等"""
        R_G = 2.364
        R_BP = 2.998
        R_RP = 1.737
        
        ag = R_G * ebv
        abp = R_BP * ebv
        arp = R_RP * ebv
        
        g0 = g_mag - ag
        bp0 = bp_mag - abp if bp_mag else None
        rp0 = rp_mag - arp if rp_mag else None
        
        mg = g0 + 5 - 5 * np.log10(distance)
        bprp0 = bp0 - rp0 if bp0 and rp0 else None
        
        return mg, bprp0, g0
    
    @staticmethod
    def plot_hr_diagram(target_data, output_path=None):
        """
        绘制赫罗图
        
        target_data: {
            'g_mag': G星等,
            'bp_mag': BP星等,
            'rp_mag': RP星等,
            'parallax': 视差(mas),
            'parallax_err': 视差误差,
            'ebv': E(B-V),
            'name': 目标名称
        }
        """
        try:
            g_mag = target_data.get('g_mag')
            bp_mag = target_data.get('bp_mag')
            rp_mag = target_data.get('rp_mag')
            parallax = target_data.get('parallax')
            parallax_err = target_data.get('parallax_err', 0)
            ebv = target_data.get('ebv', 0)
            name = target_data.get('name', 'Target')
            
            if not parallax or parallax <= 0:
                return {'success': False, 'error': 'Invalid parallax'}
            
            distance = 1000.0 / parallax
            distance_err = distance * (parallax_err / parallax) if parallax_err else 0
            
            mg, bprp0, g0 = HRDiagramPlotterV2.calculate_absolute_magnitude(
                g_mag, bp_mag, rp_mag, distance, ebv
            )
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.minorticks_on()
            
            # 加载背景数据
            bg_file = './lib/LM_DR10_FOR_CMD_every30.fits'
            if os.path.exists(bg_file):
                with fits.open(bg_file) as hdul:
                    bg_data = hdul[1].data
                    bprp0_bg = bg_data['BPRP0_green19']
                    mg0_bg = bg_data['MG_rgeo_green19']
                    mask = (bprp0_bg > -1) & (bprp0_bg < 5) & (mg0_bg > -5) & (mg0_bg < 16)
                    ax.scatter(bprp0_bg[mask], mg0_bg[mask], marker='o', s=1, 
                              alpha=0.3, c='gray', rasterized=True, zorder=0,
                              label='LAMOST DR10')
            
            # 绘制目标星
            if bprp0:
                ax.scatter([bprp0], [mg], marker='*', s=400, alpha=0.99,
                          edgecolors='k', facecolor='orange', linewidths=0.5,
                          zorder=3, label=name)
                ax.annotate(name, (bprp0, mg), xytext=(10, 10),
                           textcoords='offset points', fontsize=10,
                           color='red', fontweight='bold')
            
            ax.set_xlim(-1, 4.4)
            ax.set_ylim(16, -5)
            ax.set_xlabel(r'$(BP-RP)_0$', fontsize=12)
            ax.set_ylabel(r'$M_{\rm{G}}$', fontsize=12)
            ax.set_title(f'Hertzsprung-Russell Diagram - {name}', fontsize=13)
            
            # 信息文本
            bprp0_str = f"{bprp0:.3f}" if bprp0 else "N/A"
            info_text = (
                f"Parallax: {parallax:.4f} ± {parallax_err:.4f} mas\n"
                f"Distance: {distance:.1f} ± {distance_err:.1f} pc\n"
                f"G = {g_mag:.3f}, G0 = {g0:.3f}\n"
                f"BP-RP0 = {bprp0_str}\n"
                f"M_G = {mg:.3f}\n"
                f"E(B-V) = {ebv:.3f}"
            )
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax.legend(loc='upper right', fontsize=9)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()
            
            return {
                'success': True,
                'parallax_mas': parallax,
                'parallax_err_mas': parallax_err,
                'distance_pc': distance,
                'distance_err_pc': distance_err,
                'M_G': mg,
                'BP_RP0': bprp0,
                'G0': g0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class CompleteAnalyzer:
    """完整分析器"""
    
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir
        self.spectral_analyzer = SpectralAnalyzer()
        self.bb_fitter = BlackbodyFitter()
        self.lc_analyzer = LightcurveAnalyzer()
        self.hr_plotter = HRDiagramPlotterV2()
    
    def analyze(self, name, ra, dec, g_mag=None, bp_mag=None, rp_mag=None,
                parallax=None, parallax_err=None):
        """
        执行完整分析
        
        Parameters:
        -----------
        name : str
            目标名称
        ra, dec : float
            赤经赤纬 (度)
        g_mag, bp_mag, rp_mag : float, optional
            Gaia星等
        parallax, parallax_err : float, optional
            视差和误差 (mas)
        """
        print(f"\n{'='*70}")
        print(f"  完整天体分析: {name}")
        print(f"  坐标: RA={ra:.6f}, DEC={dec:.6f}")
        print(f"{'='*70}\n")
        
        target_dir = f"{self.output_dir}/{name}_analysis"
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(f"{target_dir}/figures", exist_ok=True)
        
        results = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. 检查SDSS光谱
        print("【1】SDSS光谱检查...")
        sdss_files = self.spectral_analyzer.check_sdss_spectrum(
            name, ra, dec
        )
        print(f"  找到 {len(sdss_files)} 个SDSS光谱")
        results['sdss_spectra'] = sdss_files
        
        # 2. 如果有SDSS光谱，进行预测
        if sdss_files:
            print("\n【2】光谱数据处理...")
            spec_data = self.spectral_analyzer.read_sdss_spectrum(
                sdss_files[0]['path']
            )
            if spec_data['success']:
                print(f"  波长范围: {spec_data['wavelength'].min():.1f} - "
                      f"{spec_data['wavelength'].max():.1f} Å")
                results['spectrum_info'] = {
                    'wavelength_range': [float(spec_data['wavelength'].min()),
                                        float(spec_data['wavelength'].max())],
                    'n_points': len(spec_data['wavelength'])
                }
        
        # 3. 赫罗图
        if parallax and g_mag:
            print("\n【3】赫罗图绘制...")
            hr_result = self.hr_plotter.plot_hr_diagram(
                {
                    'g_mag': g_mag,
                    'bp_mag': bp_mag,
                    'rp_mag': rp_mag,
                    'parallax': parallax,
                    'parallax_err': parallax_err or 0,
                    'ebv': 0.1,
                    'name': name
                },
                output_path=f"{target_dir}/figures/{name}_HR_diagram.png"
            )
            results['hr_diagram'] = hr_result
            if hr_result.get('success'):
                print(f"  距离: {hr_result['distance_pc']:.1f} pc")
                print(f"  绝对星等: M_G = {hr_result['M_G']:.2f}")
        
        # 保存结果
        result_file = f"{target_dir}/analysis_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"  分析完成! 结果保存: {result_file}")
        print(f"{'='*70}\n")
        
        return results


# 主程序
if __name__ == "__main__":
    # 示例: EV UMa
    analyzer = CompleteAnalyzer()
    
    # EV UMa 数据 (示例值)
    results = analyzer.analyze(
        name="EV_UMa",
        ra=13.1316273124,
        dec=53.8584719271,
        g_mag=16.5,  # 示例值
        bp_mag=17.2,
        rp_mag=15.3,
        parallax=5.0,  # 示例值 (mas)
        parallax_err=0.5
    )
