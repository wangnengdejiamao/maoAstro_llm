#!/usr/bin/env python3
"""
完整天体分析系统 V2
==================
修复SIMBAD和Gaia问题，添加所有可视化功能
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
from astropy.timeseries import LombScargle
from datetime import datetime

warnings.filterwarnings('ignore')

OUTPUT_DIR = "./output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ==================== SIMBAD查询（修复版）====================

def query_simbad_fixed(ra, dec, radius=5):
    """修复的SIMBAD查询 - 正确处理结果"""
    try:
        from astroquery.simbad import Simbad
        
        custom_simbad = Simbad()
        # 使用简单的字段查询
        custom_simbad.add_votable_fields('otype', 'sp', 'fluxdata(U)', 'fluxdata(B)', 'fluxdata(V)', 'fluxdata(R)', 'fluxdata(I)')
        
        coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        result = custom_simbad.query_region(coord, radius=radius * u.arcsec)
        
        if result is None or len(result) == 0:
            return {'success': True, 'matched': False, 'message': 'No match found'}
        
        row = result[0]
        
        # 构建基础信息
        data = {
            'success': True,
            'matched': True,
            'main_id': str(row['MAIN_ID']).strip() if row['MAIN_ID'] else 'Unknown',
            'otype': str(row['OTYPE']).strip() if row['OTYPE'] else 'Unknown',
            'sp_type': str(row['SP_TYPE']).strip() if row['SP_TYPE'] else None,
            'ra_input': ra,
            'dec_input': dec,
            'radius_arcsec': radius
        }
        
        # 视差和距离
        if 'PLX_VALUE' in row.colnames and row['PLX_VALUE'] is not None:
            try:
                plx = float(row['PLX_VALUE'])
                data['parallax'] = plx
                if plx > 0:
                    data['distance'] = 1000.0 / plx
            except:
                pass
        
        # 测光数据
        mags = {}
        for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'G', 'BP', 'RP']:
            try:
                key = f'FLUX_{band}'
                if key in row.colnames and row[key] is not None:
                    val = float(row[key])
                    mags[band] = val
            except:
                pass
        
        if mags:
            data['magnitudes'] = mags
            # 计算色指数
            if 'BP' in mags and 'RP' in mags:
                data['bp_rp'] = mags['BP'] - mags['RP']
            if 'B' in mags and 'V' in mags:
                data['bv'] = mags['B'] - mags['V']
        
        return data
        
    except Exception as e:
        return {'success': False, 'matched': False, 'error': str(e)}


# ==================== 周期分析工具 ====================

def analyze_period_lombscargle(times, mags, errors, fmin=0.1, fmax=50):
    """Lomb-Scargle周期分析"""
    mask = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errors)
    t, y, dy = times[mask], mags[mask], errors[mask]
    
    if len(t) < 20:
        return None, None, None, None
    
    ls = LombScargle(t, y, dy)
    frequency, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=15)
    
    best_idx = np.argmax(power)
    best_period = 1.0 / frequency[best_idx]
    
    # 计算显著性
    try:
        false_alarm = ls.false_alarm_probability(power[best_idx])
        significance = 1.0 - false_alarm
    except:
        significance = power[best_idx] / np.mean(power)
    
    return best_period, frequency, power, significance


def analyze_tess_csv(csv_file):
    """分析TESS CSV文件的周期"""
    try:
        df = pd.read_csv(csv_file)
        
        # 查找列
        time_col = None
        for col in ['time', 'TIME', 'bjd', 'BJD', 'btjd', 'BTJD']:
            if col in df.columns:
                time_col = col
                break
        
        flux_col = None
        for col in ['pdcsap_flux', 'SAP_FLUX', 'flux', 'FLUX', 'sap_flux', 'kspsap_flux']:
            if col in df.columns:
                flux_col = col
                break
        
        if time_col is None or flux_col is None:
            return None
        
        # 清理数据
        df = df[df[flux_col] > 0]
        df = df[np.isfinite(df[flux_col])]
        
        times = df[time_col].values
        fluxes = df[flux_col].values
        
        # 转换为星等
        mags = -2.5 * np.log10(fluxes)
        mags = mags - np.median(mags)
        errors = np.ones_like(mags) * 0.001
        
        # 周期搜索范围 (0.08-0.3天 = 2-7小时)
        period, freq, power, sig = analyze_period_lombscargle(
            times, mags, errors, fmin=1/0.3, fmax=1/0.08
        )
        
        if period is None:
            return None
        
        return {
            'period': period,
            'significance': sig,
            'frequency': freq,
            'power': power,
            'times': times,
            'mags': mags,
            'errors': errors
        }
        
    except Exception as e:
        print(f"  周期分析错误: {e}")
        return None


# ==================== 可视化工具 ====================

def plot_folded_lightcurve(times, mags, period, output_path, name="Target", subtitle=""):
    """绘制折叠光变曲线（双周期显示）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始光变曲线
    ax = axes[0]
    ax.scatter(times, mags, c='black', s=1, alpha=0.5, rasterized=True)
    ax.set_xlabel('Time (BTJD)', fontsize=12)
    ax.set_ylabel('Relative Magnitude', fontsize=12)
    ax.set_title('Original Light Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 折叠光变曲线
    phase = ((times - times[0]) / period) % 1.0
    phase_double = np.concatenate([phase, phase + 1])
    mags_double = np.concatenate([mags, mags])
    
    ax = axes[1]
    ax.scatter(phase_double, mags_double, c='red', s=3, alpha=0.5, label='Data', rasterized=True)
    
    # 分箱平均
    bin_edges = np.linspace(0, 2, 51)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_mags = []
    for i in range(len(bin_edges) - 1):
        mask = (phase_double >= bin_edges[i]) & (phase_double < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_mags.append(np.median(mags_double[mask]))
        else:
            binned_mags.append(np.nan)
    
    ax.plot(bin_centers, binned_mags, 'b-', linewidth=2, label='Binned', alpha=0.8)
    
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Relative Magnitude', fontsize=12)
    ax.set_title(f'Folded Light Curve\nPeriod = {period*24:.4f} h', fontsize=14)
    ax.set_xlim(0, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{name} {subtitle}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_periodogram(freq, power, period, output_path, name="Target"):
    """绘制周期图"""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # 周期图
    ax1.plot(freq, power, 'k-', linewidth=0.5)
    ax1.axvline(1.0/period, color='r', linestyle='--', linewidth=2,
                label=f'Best: P={period*24:.4f}h ({period:.6f}d)')
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title(f'{name} - Lomb-Scargle Periodogram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 周期 vs 功率（更直观）
    periods = 1.0 / freq
    ax2.plot(periods * 24, power, 'b-', linewidth=0.5)
    ax2.axvline(period * 24, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Period (hours)', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_xlim(0, 24)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_hr_diagram_gaia(ra, dec, output_path, name="Target"):
    """绘制Gaia赫罗图（修复版，带重试）"""
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = 5000
        
        coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        
        # 查询周围恒星
        query = f"""
        SELECT * FROM gaiadr3.gaia_source 
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.1))=1
        AND parallax > 0 AND parallax_over_error > 5
        AND phot_g_mean_mag IS NOT NULL
        AND bp_rp IS NOT NULL
        """
        
        job = Gaia.launch_job(query)
        stars = job.get_results()
        
        if stars is None or len(stars) == 0:
            return None
        
        # 计算绝对星等
        g_mag = stars['phot_g_mean_mag']
        bp_rp = stars['bp_rp']
        parallax = stars['parallax']
        
        g_abs = g_mag - 5 * np.log10(1000.0 / parallax) + 5
        
        # 创建图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制背景星（密度图）
        mask = (bp_rp > -1) & (bp_rp < 4) & (g_abs > -5) & (g_abs < 15)
        ax.scatter(bp_rp[mask], g_abs[mask], c='lightgray', s=1, alpha=0.5, label='Field stars')
        
        # 查询目标星
        query_target = f"""
        SELECT * FROM gaiadr3.gaia_source 
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01))=1
        """
        job_target = Gaia.launch_job(query_target)
        target_stars = job_target.get_results()
        
        if target_stars is not None and len(target_stars) > 0:
            star = target_stars[0]
            if star['parallax'] > 0:
                bp_rp_t = star['bp_rp']
                g_abs_t = star['phot_g_mean_mag'] - 5 * np.log10(1000.0 / star['parallax']) + 5
                
                ax.scatter([bp_rp_t], [g_abs_t], c='red', s=300, marker='*',
                          edgecolors='black', linewidth=2, label=f'{name}', zorder=10)
                ax.annotate(name, (bp_rp_t, g_abs_t), 
                           xytext=(15, 5), textcoords='offset points',
                           fontsize=14, color='red', fontweight='bold')
        
        ax.set_xlabel('BP - RP (mag)', fontsize=14)
        ax.set_ylabel('Absolute G Magnitude (mag)', fontsize=14)
        ax.set_title(f'Hertzsprung-Russell Diagram - {name}\n(Gaia DR3)', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 4)
        ax.set_ylim(15, -5)
        
        # 添加恒星类型标注
        ax.axhline(4, color='orange', linestyle=':', alpha=0.5)
        ax.text(3.5, 4.5, 'Main Sequence', fontsize=10, color='orange')
        ax.axhline(0, color='red', linestyle=':', alpha=0.5)
        ax.text(3.5, 0.5, 'Giants', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"  Gaia查询失败: {e}")
        return None


def plot_sed(simbad_data, output_path, name="Target"):
    """绘制SED图"""
    if not simbad_data.get('matched') or 'magnitudes' not in simbad_data:
        return None
    
    # 波长定义 (Angstrom)
    wavelengths = {
        'U': 3600, 'B': 4450, 'V': 5510, 'R': 6580, 'I': 8060,
        'J': 12350, 'H': 16620, 'K': 21590,
        'G': 6730, 'BP': 5320, 'RP': 7970
    }
    
    mags = simbad_data['magnitudes']
    
    waves, mags_list, labels = [], [], []
    for band, mag in mags.items():
        if band in wavelengths:
            waves.append(wavelengths[band])
            mags_list.append(mag)
            labels.append(band)
    
    if len(waves) < 2:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 星等图
    colors = plt.cm.rainbow(np.linspace(0, 1, len(waves)))
    ax1.scatter(waves, mags_list, c=colors, s=150, edgecolors='black', linewidth=2, zorder=10)
    for w, m, l in zip(waves, mags_list, labels):
        ax1.annotate(l, (w, m), textcoords="offset points", xytext=(0,12), 
                    ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Wavelength (Å)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_title(f'{name} - Photometric Magnitudes', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 流量密度图
    # F_nu ~ 10^(-0.4*m)
    fluxes = np.array([10**(-0.4 * m) for m in mags_list])
    
    ax2.scatter(waves, fluxes * 1e23, c=colors, s=150, edgecolors='black', linewidth=2, zorder=10)
    for w, f, l in zip(waves, fluxes * 1e23, labels):
        ax2.annotate(l, (w, f), textcoords="offset points", xytext=(0,12), 
                    ha='center', fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Wavelength (Å)', fontsize=12)
    ax2.set_ylabel('Flux Density (Jy)', fontsize=12)
    ax2.set_title(f'{name} - SED', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Spectral Energy Distribution - {name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_figure(result, output_path):
    """创建综合分析汇总图"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    name = result.get('name', 'Target')
    
    # 标题
    fig.suptitle(f'Astronomical Analysis Summary - {name}', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 基本信息
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    
    info_text = f"""
    Target: {name}
    
    RA: {result.get('ra', 'N/A')}°
    Dec: {result.get('dec', 'N/A')}°
    
    SIMBAD: {result.get('simbad', {}).get('main_id', 'N/A')}
    Type: {result.get('simbad', {}).get('otype', 'N/A')}
    Sp: {result.get('simbad', {}).get('sp_type', 'N/A')}
    
    Period: {result.get('period', 'N/A')}
    """
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 消光信息
    ax_ext = fig.add_subplot(gs[0, 1])
    ext = result.get('extinction', {})
    if ext.get('success'):
        ax_ext.bar(['E(B-V)', 'A_V'], [ext['E_B_V'], ext['A_V']], 
                  color=['blue', 'red'], alpha=0.7)
        ax_ext.set_ylabel('Mag')
        ax_ext.set_title('Extinction')
    else:
        ax_ext.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_ext.set_title('Extinction')
    
    # 测光数据
    ax_phot = fig.add_subplot(gs[0, 2])
    simbad = result.get('simbad', {})
    if simbad.get('matched') and 'magnitudes' in simbad:
        mags = simbad['magnitudes']
        bands = list(mags.keys())[:8]
        values = [mags[b] for b in bands]
        ax_phot.barh(bands, values, color='green', alpha=0.7)
        ax_phot.set_xlabel('Mag')
        ax_phot.set_title('Photometry')
        ax_phot.invert_xaxis()
    else:
        ax_phot.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_phot.set_title('Photometry')
    
    # 周期图（如果有）
    ax_period = fig.add_subplot(gs[1, :2])
    if 'period_analysis' in result and result['period_analysis']:
        pa = result['period_analysis']
        if 'frequency' in pa and 'power' in pa:
            ax_period.plot(pa['frequency'], pa['power'], 'k-', linewidth=0.5)
            if 'period' in result:
                ax_period.axvline(1.0/result['period'], color='r', linestyle='--')
            ax_period.set_xlabel('Frequency (1/day)')
            ax_period.set_ylabel('Power')
            ax_period.set_title('Periodogram')
            ax_period.grid(True, alpha=0.3)
    else:
        ax_period.text(0.5, 0.5, 'No Period Analysis', ha='center', va='center')
        ax_period.set_title('Periodogram')
    
    # 周期显示
    ax_per_text = fig.add_subplot(gs[1, 2])
    ax_per_text.axis('off')
    if 'period' in result:
        period_text = f"""
        Period Analysis
        
        P = {result['period']*24:.4f} hours
          = {result['period']:.6f} days
        
        Significance: {result.get('period_significance', 'N/A'):.3f}
        """
        ax_per_text.text(0.1, 0.9, period_text, transform=ax_per_text.transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 折叠光变曲线
    ax_folded = fig.add_subplot(gs[2, :])
    if 'period_analysis' in result and result['period_analysis']:
        pa = result['period_analysis']
        if 'times' in pa and 'mags' in pa and 'period' in result:
            times = np.array(pa['times'])
            mags = np.array(pa['mags'])
            period = result['period']
            
            phase = ((times - times[0]) / period) % 1.0
            phase_double = np.concatenate([phase, phase + 1])
            mags_double = np.concatenate([mags, mags])
            
            ax_folded.scatter(phase_double, mags_double, c='red', s=2, alpha=0.5)
            ax_folded.set_xlabel('Phase')
            ax_folded.set_ylabel('Relative Magnitude')
            ax_folded.set_title(f'Folded Light Curve (P={period*24:.4f}h)')
            ax_folded.set_xlim(0, 2)
            ax_folded.grid(True, alpha=0.3)
    else:
        ax_folded.text(0.5, 0.5, 'No Light Curve Data', ha='center', va='center')
        ax_folded.set_title('Folded Light Curve')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


# ==================== 主分析流程 ====================

def full_analysis(ra, dec, name):
    """完整分析流程"""
    print("="*70)
    print(f"🔭 完整天体分析: {name}")
    print(f"   坐标: RA={ra}°, DEC={dec}°")
    print("="*70)
    
    result = {
        'name': name, 'ra': ra, 'dec': dec,
        'timestamp': datetime.now().isoformat(),
        'figures': {}
    }
    
    # 1. SIMBAD查询
    print("\n【1/6】SIMBAD数据库查询...")
    simbad = query_simbad_fixed(ra, dec)
    result['simbad'] = simbad
    
    if simbad.get('matched'):
        print(f"   ✅ 匹配: {simbad['main_id']}")
        print(f"   ✅ 类型: {simbad.get('otype', 'N/A')}")
        print(f"   ✅ 光谱型: {simbad.get('sp_type', 'N/A')}")
        if 'magnitudes' in simbad:
            print(f"   ✅ 测光: {list(simbad['magnitudes'].keys())}")
        
        # 绘制SED
        print("   绘制SED...")
        sed_path = os.path.join(FIG_DIR, f"{name}_sed.png")
        if plot_sed(simbad, sed_path, name):
            result['figures']['sed'] = sed_path
            print(f"   ✅ SED图: {sed_path}")
    else:
        print(f"   ⚠️ {simbad.get('message', simbad.get('error', '未匹配'))}")
    
    # 2. 周期分析
    print("\n【2/6】TESS周期分析...")
    tess_files = [f for f in os.listdir(DATA_DIR) 
                  if f.startswith(name) and f.endswith('.csv') and 'TESS' in f]
    
    if tess_files:
        print(f"   找到 {len(tess_files)} 个TESS文件")
        
        # 选择最小的文件（SPOC通常是高质量的）
        csv_file = min([os.path.join(DATA_DIR, f) for f in tess_files], key=os.path.getsize)
        print(f"   分析: {os.path.basename(csv_file)}")
        
        period_result = analyze_tess_csv(csv_file)
        
        if period_result and period_result.get('period'):
            period = period_result['period']
            result['period'] = period
            result['period_significance'] = period_result['significance']
            result['period_analysis'] = {
                'times': period_result['times'].tolist(),
                'mags': period_result['mags'].tolist(),
                'frequency': period_result['frequency'].tolist(),
                'power': period_result['power'].tolist()
            }
            
            print(f"   ✅ 周期: {period*24:.4f} hours ({period:.6f} days)")
            print(f"   ✅ 显著性: {period_result['significance']:.4f}")
            
            # 绘制周期图
            print("   绘制周期图...")
            per_path = os.path.join(FIG_DIR, f"{name}_periodogram.png")
            plot_periodogram(
                np.array(period_result['frequency']), 
                np.array(period_result['power']), 
                period, per_path, name)
            result['figures']['periodogram'] = per_path
            
            # 绘制折叠光变曲线
            print("   绘制折叠光变曲线...")
            fold_path = os.path.join(FIG_DIR, f"{name}_folded_lc.png")
            plot_folded_lightcurve(
                np.array(period_result['times']), 
                np.array(period_result['mags']),
                period, fold_path, name, "(TESS)")
            result['figures']['folded_lc'] = fold_path
        else:
            print("   ⚠️ 周期分析失败")
    else:
        print("   ⚠️ 未找到TESS文件")
    
    # 3. Gaia赫罗图
    print("\n【3/6】Gaia赫罗图...")
    hr_path = os.path.join(FIG_DIR, f"{name}_hr_diagram.png")
    if plot_hr_diagram_gaia(ra, dec, hr_path, name):
        result['figures']['hr_diagram'] = hr_path
        print(f"   ✅ 赫罗图: {hr_path}")
    else:
        print("   ⚠️ 赫罗图失败")
    
    # 4. 消光
    print("\n【4/6】银河消光...")
    try:
        sys.path.insert(0, './src')
        from astro_tools import AstroTools
        tools = AstroTools()
        ext = tools.query_extinction(ra, dec)
        result['extinction'] = ext
        if ext.get('success'):
            print(f"   ✅ E(B-V) = {ext['E_B_V']}, A_V = {ext['A_V']}")
    except Exception as e:
        print(f"   ⚠️ 消光失败: {e}")
    
    # 5. 生成汇总图
    print("\n【5/6】生成汇总图...")
    summary_path = os.path.join(FIG_DIR, f"{name}_summary.png")
    create_summary_figure(result, summary_path)
    result['figures']['summary'] = summary_path
    print(f"   ✅ 汇总图: {summary_path}")
    
    # 6. 保存结果
    print("\n【6/6】保存结果...")
    result_file = os.path.join(OUTPUT_DIR, f"{name}_full_analysis.json")
    
    # 移除numpy数组以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    result_serializable = convert_to_serializable(result)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✅ 结果保存: {result_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 分析完成")
    print("="*70)
    print(f"\n目标: {name}")
    print(f"坐标: RA={ra}°, DEC={dec}°")
    
    if simbad.get('matched'):
        print(f"\n天体信息:")
        print(f"  ID: {simbad['main_id']}")
        print(f"  类型: {simbad['otype']}")
        print(f"  光谱型: {simbad.get('sp_type', 'N/A')}")
    
    if 'period' in result:
        print(f"\n周期分析:")
        print(f"  P = {result['period']*24:.4f} hours")
        print(f"  显著性 = {result['period_significance']:.4f}")
    
    print(f"\n生成的图像 ({len(result['figures'])}):")
    for fig_type, fig_path in sorted(result['figures'].items()):
        print(f"  ✅ {fig_type}: {os.path.basename(fig_path)}")
    
    print("\n" + "="*70)
    
    return result


if __name__ == "__main__":
    # 分析AM Her
    full_analysis(ra=274.0554, dec=49.8679, name="AM_Her")
