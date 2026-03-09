#!/usr/bin/env python3
"""
完整天体分析系统 - 修复版
==========================
包含：SIMBAD、TESS周期折叠、SED、赫罗图、ZTF等所有功能
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.timeseries import LombScargle
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

# 确保输出目录存在
OUTPUT_DIR = "./output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ==================== 1. SIMBAD查询（修复版）====================

class SIMBADTool:
    """修复的SIMBAD查询工具"""
    
    @staticmethod
    def query(ra, dec, radius=5):
        """查询SIMBAD数据库"""
        try:
            from astroquery.simbad import Simbad
            
            # 配置SIMBAD
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('otype', 'sp', 'fluxdata', 'plx', 'rv_value')
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            result = custom_simbad.query_region(coord, radius=radius * u.arcsec)
            
            if result is None or len(result) == 0:
                return {'success': True, 'matched': False, 'message': 'No match found'}
            
            row = result[0]
            
            # 解析坐标（处理字符串格式）
            ra_str = str(row['RA'])
            dec_str = str(row['DEC'])
            
            # 使用SkyCoord解析字符串坐标
            try:
                parsed_coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra_deg = float(parsed_coord.ra.degree)
                dec_deg = float(parsed_coord.dec.degree)
            except:
                ra_deg = ra
                dec_deg = dec
            
            # 构建结果
            data = {
                'success': True,
                'matched': True,
                'main_id': str(row['MAIN_ID']).strip() if row['MAIN_ID'] else 'Unknown',
                'otype': str(row['OTYPE']).strip() if row['OTYPE'] else 'Unknown',
                'ra': ra_deg,
                'dec': dec_deg,
                'ra_str': ra_str,
                'dec_str': dec_str,
            }
            
            # 光谱型
            if 'SP_TYPE' in row.colnames and row['SP_TYPE']:
                data['sp_type'] = str(row['SP_TYPE']).strip()
            
            # 视差和距离
            if 'PLX_VALUE' in row.colnames and row['PLX_VALUE']:
                try:
                    plx = float(row['PLX_VALUE'])
                    data['parallax'] = plx
                    if plx > 0:
                        data['distance'] = 1000.0 / plx
                except:
                    pass
            
            # 自行
            if 'PMRA' in row.colnames and row['PMRA']:
                try:
                    data['pm_ra'] = float(row['PMRA'])
                    data['pm_dec'] = float(row['PMDEC'])
                except:
                    pass
            
            # 测光数据
            mags = {}
            for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']:
                try:
                    val = row.get(f'FLUX_{band}')
                    if val is not None:
                        mags[band] = float(val)
                except:
                    pass
            
            if mags:
                data['magnitudes'] = mags
            
            return data
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'matched': False}


# ==================== 2. 周期分析工具 ====================

class PeriodAnalyzer:
    """光变曲线周期分析"""
    
    @staticmethod
    def find_period(times, mags, errors, fmin=0.1, fmax=50):
        """
        Lomb-Scargle周期分析
        
        Returns:
            period, frequency, power, significance
        """
        # 清理数据
        mask = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errors)
        t = times[mask]
        y = mags[mask]
        dy = errors[mask]
        
        if len(t) < 20:
            return None, None, None, None
        
        # Lomb-Scargle周期图
        ls = LombScargle(t, y, dy)
        frequency, power = ls.autopower(
            minimum_frequency=fmin,
            maximum_frequency=fmax,
            samples_per_peak=15
        )
        
        # 最佳频率
        best_idx = np.argmax(power)
        best_freq = frequency[best_idx]
        best_period = 1.0 / best_freq
        
        # 计算显著性（简化版）
        false_alarm = ls.false_alarm_probability(power[best_idx])
        significance = 1.0 - false_alarm if false_alarm < 1 else 0
        
        return best_period, frequency, power, significance
    
    @staticmethod
    def fold_lightcurve(times, mags, errors, period):
        """折叠光变曲线"""
        phase = ((times - times[0]) / period) % 1.0
        return phase, mags, errors
    
    @staticmethod
    def analyze_tess_period(csv_file, period_guess=None):
        """分析TESS数据周期"""
        try:
            df = pd.read_csv(csv_file)
            
            # 查找时间列
            time_col = None
            for col in ['time', 'TIME', 'bjd', 'BJD', 'btjd', 'BTJD']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                return None
            
            # 查找流量列
            flux_col = None
            for col in ['pdcsap_flux', 'SAP_FLUX', 'flux', 'FLUX', 'sap_flux']:
                if col in df.columns:
                    flux_col = col
                    break
            
            if flux_col is None:
                return None
            
            # 数据清理
            df = df[df[flux_col] > 0]  # 去除无效值
            df = df[np.isfinite(df[flux_col])]
            
            times = df[time_col].values
            fluxes = df[flux_col].values
            
            # 转换为星等（用于周期性分析）
            mags = -2.5 * np.log10(fluxes)
            mags = mags - np.median(mags)  # 归零化
            errors = np.ones_like(mags) * 0.01  # 假设误差
            
            # 周期分析（针对AM Her的轨道周期范围）
            # AM Her周期约3.09小时 = 0.129天
            # 搜索范围：2-5小时
            fmin = 1.0 / 0.25  # 4天周期
            fmax = 1.0 / 0.08  # 2小时周期
            
            period, freq, power, sig = PeriodAnalyzer.find_period(
                times, mags, errors, fmin=fmin, fmax=fmax
            )
            
            return {
                'period': period,
                'frequency': freq,
                'power': power,
                'significance': sig,
                'times': times,
                'mags': mags,
                'errors': errors
            }
            
        except Exception as e:
            print(f"  周期分析错误: {e}")
            return None


# ==================== 3. 可视化工具 ====================

class VisualizationTool:
    """天文数据可视化"""
    
    @staticmethod
    def plot_folded_lc(times, mags, errors, period, output_path, name="Target", 
                       title_suffix=""):
        """绘制折叠光变曲线"""
        
        # 折叠
        phase = ((times - times[0]) / period) % 1.0
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 原始光变曲线
        ax = axes[0]
        ax.scatter(times, mags, c='black', s=1, alpha=0.5)
        ax.set_xlabel('Time (BTJD)', fontsize=12)
        ax.set_ylabel('Relative Magnitude', fontsize=12)
        ax.set_title('Original Light Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 折叠光变曲线（显示2个周期）
        ax = axes[1]
        phase_double = np.concatenate([phase, phase + 1])
        mags_double = np.concatenate([mags, mags])
        
        ax.scatter(phase_double, mags_double, c='red', s=5, alpha=0.5, label='Data')
        
        # 分箱平均
        bin_edges = np.linspace(0, 2, 41)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_mags = []
        for i in range(len(bin_edges) - 1):
            mask = (phase_double >= bin_edges[i]) & (phase_double < bin_edges[i+1])
            if np.sum(mask) > 0:
                binned_mags.append(np.median(mags_double[mask]))
            else:
                binned_mags.append(np.nan)
        
        ax.plot(bin_centers, binned_mags, 'b-', linewidth=2, label='Binned', alpha=0.7)
        
        ax.set_xlabel('Phase', fontsize=12)
        ax.set_ylabel('Relative Magnitude', fontsize=12)
        ax.set_title(f'Folded Light Curve\nPeriod = {period*24:.4f} hours', fontsize=14)
        ax.set_xlim(0, 2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{name} {title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    @staticmethod
    def plot_periodogram(frequency, power, best_period, output_path, name="Target"):
        """绘制周期图"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(frequency, power, 'k-', linewidth=0.5)
        ax.axvline(1.0/best_period, color='r', linestyle='--', 
                  label=f'Best: P={best_period*24:.4f}h')
        
        ax.set_xlabel('Frequency (1/day)', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.set_title('Lomb-Scargle Periodogram', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    @staticmethod
    def plot_hr_diagram(ra, dec, output_path, name="Target", radius=300):
        """绘制Gaia赫罗图"""
        try:
            from astroquery.gaia import Gaia
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            
            # 查询背景星
            job = Gaia.cone_search_async(coord, radius=radius * u.arcsec)
            stars = job.get_results()
            
            if stars is None or len(stars) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 筛选有效数据
            bp_rp = stars['bp_rp']
            g_mag = stars['phot_g_mean_mag']
            parallax = stars['parallax']
            
            # 计算绝对星等
            mask = (bp_rp > -1) & (bp_rp < 4) & (parallax > 0) & (parallax < 100)
            
            if np.sum(mask) > 0:
                g_abs = g_mag[mask] - 5 * np.log10(1000.0 / parallax[mask]) + 5
                bp_rp_clean = bp_rp[mask]
                
                # 绘制背景星
                ax.scatter(bp_rp_clean, g_abs, c='gray', s=1, alpha=0.3, label='Field stars')
            
            # 标记目标星
            job_center = Gaia.cone_search_async(coord, radius=5 * u.arcsec)
            center_stars = job_center.get_results()
            
            if center_stars is not None and len(center_stars) > 0:
                star = center_stars[0]
                if star['parallax'] > 0:
                    bp_rp_target = star['bp_rp']
                    g_abs_target = star['phot_g_mean_mag'] - 5 * np.log10(1000.0 / star['parallax']) + 5
                    
                    ax.scatter([bp_rp_target], [g_abs_target], 
                              c='red', s=200, marker='*', 
                              edgecolors='black', linewidth=2, 
                              label=f'{name}', zorder=10)
                    ax.annotate(name, (bp_rp_target, g_abs_target), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=12, color='red')
            
            ax.set_xlabel('BP - RP (mag)', fontsize=14)
            ax.set_ylabel('M_G (mag)', fontsize=14)
            ax.set_title(f'Hertzsprung-Russell Diagram\n{center_stars[0]["source_id"] if center_stars else ""}', 
                        fontsize=16)
            ax.invert_yaxis()
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"  赫罗图绘制失败: {e}")
            return None
    
    @staticmethod
    def plot_sed(simbad_data, output_path, name="Target"):
        """绘制SED图"""
        if not simbad_data.get('matched') or 'magnitudes' not in simbad_data:
            return None
        
        # 波段波长（Angstrom）
        wavelengths = {
            'U': 3600, 'B': 4450, 'V': 5510, 'R': 6580, 'I': 8060,
            'J': 12350, 'H': 16620, 'K': 21590
        }
        
        mags = simbad_data['magnitudes']
        
        # 收集数据
        waves = []
        mags_list = []
        labels = []
        
        for band, mag in mags.items():
            if band in wavelengths:
                waves.append(wavelengths[band])
                mags_list.append(mag)
                labels.append(band)
        
        if len(waves) == 0:
            return None
        
        # 绘制
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 星等 vs 波长
        ax1.scatter(waves, mags_list, s=100, c='red', edgecolors='black', zorder=10)
        for w, m, l in zip(waves, mags_list, labels):
            ax1.annotate(l, (w, m), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax1.set_xlabel('Wavelength (Angstrom)', fontsize=12)
        ax1.set_ylabel('Magnitude', fontsize=12)
        ax1.set_title(f'{name} - Magnitudes', fontsize=14)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 流量 vs 波长（简化计算）
        # F_nu = 10^(-0.4 * (m + 48.6)) erg/s/cm^2/Hz
        fluxes = [10**(-0.4 * (m + 48.6)) for m in mags_list]
        
        ax2.scatter(waves, fluxes, s=100, c='blue', edgecolors='black', zorder=10)
        for w, f, l in zip(waves, fluxes, labels):
            ax2.annotate(l, (w, f), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax2.set_xlabel('Wavelength (Angstrom)', fontsize=12)
        ax2.set_ylabel('Flux (erg/s/cm^2/Hz)', fontsize=12)
        ax2.set_title(f'{name} - SED', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Spectral Energy Distribution - {name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


# ==================== 4. 主分析流程 ====================

def analyze_am_her():
    """分析AM Her的完整流程"""
    
    ra, dec = 274.0554, 49.8679
    name = "AM_Her"
    
    print("="*70)
    print(f"🔭 完整天体分析: {name}")
    print(f"   坐标: RA={ra}°, DEC={dec}°")
    print("="*70)
    
    result = {
        'name': name,
        'ra': ra,
        'dec': dec,
        'timestamp': datetime.now().isoformat(),
        'figures': {}
    }
    
    # 1. SIMBAD查询
    print("\n【1/5】SIMBAD数据库查询...")
    simbad = SIMBADTool.query(ra, dec)
    result['simbad'] = simbad
    
    if simbad.get('matched'):
        print(f"   ✅ 匹配: {simbad['main_id']}")
        print(f"   ✅ 类型: {simbad.get('otype', 'N/A')}")
        print(f"   ✅ 光谱型: {simbad.get('sp_type', 'N/A')}")
        
        # 绘制SED
        if 'magnitudes' in simbad:
            print("   绘制SED...")
            sed_path = os.path.join(FIG_DIR, f"{name}_sed.png")
            vis_result = VisualizationTool.plot_sed(simbad, sed_path, name)
            if vis_result:
                result['figures']['sed'] = sed_path
                print(f"   ✅ SED图: {sed_path}")
    else:
        print(f"   ⚠️ {simbad.get('message', '未匹配')}")
    
    # 2. TESS周期分析
    print("\n【2/5】TESS光变曲线周期分析...")
    
    # 查找已下载的TESS文件
    tess_files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith(name) and f.endswith('.csv') and 'TESS' in f:
            tess_files.append(os.path.join(DATA_DIR, f))
    
    if tess_files:
        print(f"   找到 {len(tess_files)} 个TESS文件")
        
        # 分析第一个文件
        csv_file = tess_files[0]
        print(f"   分析: {os.path.basename(csv_file)}")
        
        period_result = PeriodAnalyzer.analyze_tess_period(csv_file)
        
        if period_result and period_result['period']:
            period = period_result['period']
            print(f"   ✅ 发现周期: {period*24:.4f} hours ({period:.6f} days)")
            result['period'] = period
            
            # 绘制折叠光变曲线
            print("   绘制折叠光变曲线...")
            folded_path = os.path.join(FIG_DIR, f"{name}_folded_lc.png")
            vis_result = VisualizationTool.plot_folded_lc(
                period_result['times'],
                period_result['mags'],
                period_result['errors'],
                period,
                folded_path,
                name,
                "(TESS)"
            )
            if vis_result:
                result['figures']['folded_lc'] = folded_path
                print(f"   ✅ 折叠图: {folded_path}")
            
            # 绘制周期图
            print("   绘制周期图...")
            periodogram_path = os.path.join(FIG_DIR, f"{name}_periodogram.png")
            vis_result = VisualizationTool.plot_periodogram(
                period_result['frequency'],
                period_result['power'],
                period,
                periodogram_path,
                name
            )
            if vis_result:
                result['figures']['periodogram'] = periodogram_path
                print(f"   ✅ 周期图: {periodogram_path}")
        else:
            print("   ⚠️ 周期分析失败")
    else:
        print("   ⚠️ 未找到TESS文件")
    
    # 3. Gaia赫罗图
    print("\n【3/5】Gaia赫罗图...")
    hr_path = os.path.join(FIG_DIR, f"{name}_hr_diagram.png")
    hr_result = VisualizationTool.plot_hr_diagram(ra, dec, hr_path, name)
    if hr_result:
        result['figures']['hr_diagram'] = hr_result
        print(f"   ✅ 赫罗图: {hr_result}")
    else:
        print("   ⚠️ 赫罗图绘制失败")
    
    # 4. 消光查询
    print("\n【4/5】银河消光...")
    try:
        sys.path.insert(0, './src')
        from astro_tools import AstroTools
        tools = AstroTools()
        ext = tools.query_extinction(ra, dec)
        result['extinction'] = ext
        if ext.get('success'):
            print(f"   ✅ A_V = {ext['A_V']}, E(B-V) = {ext['E_B_V']}")
    except Exception as e:
        print(f"   ⚠️ 消光查询失败: {e}")
    
    # 5. 保存结果
    print("\n【5/5】保存结果...")
    result_file = os.path.join(OUTPUT_DIR, f"{name}_full_analysis.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✅ 结果保存: {result_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 分析结果汇总")
    print("="*70)
    print(f"\n目标: {name}")
    print(f"坐标: RA={ra}, DEC={dec}")
    
    if simbad.get('matched'):
        print(f"\nSIMBAD:")
        print(f"  ID: {simbad['main_id']}")
        print(f"  类型: {simbad['otype']}")
    
    if 'period' in result:
        print(f"\n周期:")
        print(f"  P = {result['period']*24:.4f} hours")
    
    print(f"\n生成的图像:")
    for fig_type, fig_path in result['figures'].items():
        print(f"  ✅ {fig_type}: {os.path.basename(fig_path)}")
    
    print("\n" + "="*70)
    
    return result


if __name__ == "__main__":
    analyze_am_her()
