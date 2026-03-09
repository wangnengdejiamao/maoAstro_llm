#!/usr/bin/env python3
"""
天体分析系统 v2.0
================
修复版：完整SED标签、Gaia HR图、LAMOST光谱查询
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.timeseries import LombScargle
from datetime import datetime
from typing import Dict, Optional

warnings.filterwarnings('ignore')

OUTPUT_DIR = "./output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def ensure_dirs():
    for d in [OUTPUT_DIR, DATA_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)


# ==================== 数据查询模块 ====================

class SIMBADQuerier:
    """SIMBAD查询 - 含IDS信息用于光谱查询"""
    
    @staticmethod
    def query(ra: float, dec: float, radius: float = 5) -> Dict:
        try:
            from astroquery.simbad import Simbad
            
            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('otype', 'sp', 'ids', 'fluxdata')
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            result = custom_simbad.query_region(coord, radius=radius * u.arcsec)
            
            if result is None or len(result) == 0:
                return {'success': True, 'matched': False, 'message': 'No match'}
            
            row = result[0]
            
            data = {
                'success': True,
                'matched': True,
                'main_id': str(row['MAIN_ID']).strip() if row['MAIN_ID'] else 'Unknown',
                'otype': str(row['OTYPE']).strip() if row['OTYPE'] else 'Unknown',
                'sp_type': str(row['SP_TYPE']).strip() if row['SP_TYPE'] else None,
            }
            
            # 获取所有标识符
            if 'IDS' in row.colnames and row['IDS']:
                ids = str(row['IDS'])
                data['ids'] = ids
                # 检查光谱数据
                data['has_lamost'] = 'LAMOST' in ids
                data['has_sdss'] = 'SDSS' in ids
            
            # 测光数据 - 更多波段
            mags = {}
            for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'G', 'BP', 'RP']:
                try:
                    key = f'FLUX_{band}'
                    if key in row.colnames and row[key] is not None:
                        val = float(row[key])
                        if np.isfinite(val) and val > 0 and val < 30:
                            mags[band] = val
                except:
                    pass
            
            if mags:
                data['magnitudes'] = mags
                if 'B' in mags and 'V' in mags:
                    data['bv'] = mags['B'] - mags['V']
                if 'BP' in mags and 'RP' in mags:
                    data['bp_rp'] = mags['BP'] - mags['RP']
            
            return data
            
        except Exception as e:
            return {'success': False, 'matched': False, 'error': str(e)}


class LAMOSTQuerier:
    """LAMOST光谱查询"""
    
    @staticmethod
    def query_spectrum(ra: float, dec: float, radius: float = 3):
        """查询LAMOST光谱数据"""
        try:
            from astroquery.lamost import Lamost
            
            print(f"   查询LAMOST: RA={ra}, DEC={dec}")
            
            # 使用cone search查询LAMOST
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            
            # 查询数据发布DR8
            r = Lamost.query_region(coord, radius=radius * u.arcsec, dr=8)
            
            if r is None or len(r) == 0:
                # 尝试DR10
                r = Lamost.query_region(coord, radius=radius * u.arcsec, dr=10)
            
            if r is None or len(r) == 0:
                return {'success': True, 'found': False, 'message': 'No LAMOST data'}
            
            # 获取最佳匹配
            row = r[0]
            
            result = {
                'success': True,
                'found': True,
                'obsid': str(row['obsid']) if 'obsid' in row.colnames else 'N/A',
                'snr_u': float(row['snru']) if 'snru' in row.colnames else None,
                'snr_g': float(row['snrg']) if 'snrg' in row.colnames else None,
                'snr_r': float(row['snrr']) if 'snrr' in row.colnames else None,
                'snr_i': float(row['snri']) if 'snri' in row.colnames else None,
                'snr_z': float(row['snrz']) if 'snrz' in row.colnames else None,
                'mag_u': float(row['mag_u']) if 'mag_u' in row.colnames else None,
                'mag_g': float(row['mag_g']) if 'mag_g' in row.colnames else None,
                'mag_r': float(row['mag_r']) if 'mag_r' in row.colnames else None,
                'teff': float(row['teff']) if 'teff' in row.colnames else None,
                'logg': float(row['logg']) if 'logg' in row.colnames else None,
                'feh': float(row['feh']) if 'feh' in row.colnames else None,
            }
            
            return result
            
        except Exception as e:
            return {'success': False, 'found': False, 'error': str(e)}


class GaiaQuerier:
    """Gaia查询 - 用于HR图"""
    
    @staticmethod
    def query_star(ra: float, dec: float, radius: float = 3):
        """查询Gaia数据"""
        try:
            from astroquery.gaia import Gaia
            
            query = f"""
            SELECT source_id, ra, dec, parallax, parallax_over_error,
                   phot_g_mean_mag, bp_rp, pmra, pmdec, radial_velocity
            FROM gaiadr3.gaia_source 
            WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius/3600}))=1
            AND parallax > 0 AND parallax_over_error > 3
            """
            
            job = Gaia.launch_job(query)
            result = job.get_results()
            
            if result is None or len(result) == 0:
                return {'success': True, 'found': False}
            
            row = result[0]
            
            g_mag = float(row['phot_g_mean_mag'])
            parallax = float(row['parallax'])
            distance = 1000.0 / parallax  # pc
            g_abs = g_mag - 5 * np.log10(distance) + 5
            
            return {
                'success': True,
                'found': True,
                'source_id': str(row['source_id']),
                'g_mag': g_mag,
                'bp_rp': float(row['bp_rp']) if row['bp_rp'] is not None else None,
                'parallax': parallax,
                'distance': distance,
                'g_abs': g_abs,
                'pmra': float(row['pmra']) if row['pmra'] is not None else None,
                'pmdec': float(row['pmdec']) if row['pmdec'] is not None else None,
            }
            
        except Exception as e:
            return {'success': False, 'found': False, 'error': str(e)}
    
    @staticmethod
    def query_field_stars(ra: float, dec: float, radius: float = 0.1):
        """查询视场星用于背景"""
        try:
            from astroquery.gaia import Gaia
            
            query = f"""
            SELECT phot_g_mean_mag, bp_rp, parallax
            FROM gaiadr3.gaia_source 
            WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))=1
            AND parallax > 0 AND parallax_over_error > 5
            AND phot_g_mean_mag IS NOT NULL
            AND bp_rp IS NOT NULL
            LIMIT 5000
            """
            
            job = Gaia.launch_job(query)
            result = job.get_results()
            
            if result is None or len(result) == 0:
                return None
            
            g_mag = result['phot_g_mean_mag']
            bp_rp = result['bp_rp']
            parallax = result['parallax']
            
            g_abs = g_mag - 5 * np.log10(1000.0 / parallax) + 5
            
            return {
                'bp_rp': bp_rp.data,
                'g_abs': g_abs.data
            }
            
        except Exception as e:
            print(f"   Gaia field query error: {e}")
            return None


class ExtinctionQuerier:
    """消光查询"""
    
    @staticmethod
    def query(ra: float, dec: float) -> Dict:
        try:
            sys.path.insert(0, './src')
            from astro_tools import AstroTools
            tools = AstroTools()
            return tools.query_extinction(ra, dec)
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ==================== 周期分析 ====================

class PeriodAnalyzer:
    """周期分析"""
    
    @staticmethod
    def find_period(times, mags, errors, fmin=0.1, fmax=50):
        mask = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errors)
        t, y, dy = times[mask], mags[mask], errors[mask]
        
        if len(t) < 20:
            return None, None, None, None
        
        ls = LombScargle(t, y, dy)
        frequency, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=15)
        
        best_idx = np.argmax(power)
        best_period = 1.0 / frequency[best_idx]
        
        try:
            false_alarm = ls.false_alarm_probability(power[best_idx])
            significance = 1.0 - false_alarm
        except:
            significance = power[best_idx] / np.mean(power)
        
        return best_period, frequency, power, significance
    
    @staticmethod
    def analyze_csv(csv_file: str, period_range=(0.08, 0.3)):
        try:
            df = pd.read_csv(csv_file)
            
            time_col = next((c for c in ['time', 'TIME', 'bjd', 'btjd'] if c in df.columns), None)
            flux_col = next((c for c in ['pdcsap_flux', 'SAP_FLUX', 'flux', 'kspsap_flux'] if c in df.columns), None)
            
            if time_col is None or flux_col is None:
                return None
            
            df = df[(df[flux_col] > 0) & np.isfinite(df[flux_col])]
            
            times = df[time_col].values
            fluxes = df[flux_col].values
            
            mags = -2.5 * np.log10(fluxes)
            mags = mags - np.median(mags)
            errors = np.ones_like(mags) * 0.001
            
            fmin, fmax = 1/period_range[1], 1/period_range[0]
            period, freq, power, sig = PeriodAnalyzer.find_period(times, mags, errors, fmin, fmax)
            
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
            print(f"  Period analysis error: {e}")
            return None


# ==================== 可视化模块 ====================

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_periodogram(freq, power, period, output_path, name="Target"):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(freq, power, 'k-', linewidth=0.5)
        ax1.axvline(1.0/period, color='r', linestyle='--', linewidth=2,
                    label=f'P={period*24:.4f}h')
        ax1.set_ylabel('Power', fontsize=12)
        ax1.set_title(f'{name} - Lomb-Scargle Periodogram', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1])
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
    
    @staticmethod
    def plot_folded_lc(times, mags, period, output_path, name="Target", subtitle=""):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.scatter(times, mags, c='black', s=1, alpha=0.5, rasterized=True)
        ax.set_xlabel('Time (BTJD)', fontsize=12)
        ax.set_ylabel('Relative Magnitude', fontsize=12)
        ax.set_title('Original Light Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        
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
            binned_mags.append(np.median(mags_double[mask]) if np.sum(mask) > 0 else np.nan)
        
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
    
    @staticmethod
    def plot_sed(simbad_data: Dict, output_path: str, name: str = "Target"):
        """绘制SED图 - 确保所有标签显示"""
        if not simbad_data.get('matched') or 'magnitudes' not in simbad_data:
            return None
        
        # 波长定义 (Angstrom)
        wavelengths = {
            'U': 3600, 'B': 4450, 'V': 5510, 'R': 6580, 'I': 8060,
            'J': 12350, 'H': 16620, 'K': 21590,
            'G': 6730, 'BP': 5320, 'RP': 7970
        }
        
        mags = simbad_data['magnitudes']
        
        # 准备数据
        waves, mags_list, labels = [], [], []
        for band in ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'G', 'BP', 'RP']:
            if band in mags and np.isfinite(mags[band]):
                waves.append(wavelengths[band])
                mags_list.append(mags[band])
                labels.append(band)
        
        if len(waves) < 1:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, len(waves)))
        
        # ===== 左图：星等 vs 波长 =====
        ax1.scatter(waves, mags_list, c=colors, s=200, edgecolors='black', linewidth=2, zorder=10)
        
        # 添加标签 - 确保所有都有
        for i, (w, m, l, c) in enumerate(zip(waves, mags_list, labels, colors)):
            # 根据位置调整标签偏移
            offset_y = 15 if i % 2 == 0 else -25
            ax1.annotate(f'{l}\n{m:.2f}', (w, m), 
                        textcoords="offset points", xytext=(0, offset_y),
                        ha='center', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.7, edgecolor='black'))
        
        ax1.set_xlabel('Wavelength (Å)', fontsize=13)
        ax1.set_ylabel('Magnitude', fontsize=13)
        ax1.set_title(f'{name} - Photometric Magnitudes', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_xscale('log')
        ax1.set_xlim(2000, 30000)
        ax1.grid(True, alpha=0.3)
        
        # ===== 右图：流量密度 vs 波长 =====
        # 转换为流量密度 (Jy) - 简化的零星等流量
        f0 = {
            'U': 1810, 'B': 4260, 'V': 3640, 'R': 3080, 'I': 2550,
            'J': 1600, 'H': 1080, 'K': 670,
            'G': 3300, 'BP': 4100, 'RP': 2940
        }  # Jy at m=0
        
        fluxes = [f0.get(l, 1000) * 10**(-0.4 * m) for l, m in zip(labels, mags_list)]
        
        ax2.scatter(waves, fluxes, c=colors, s=200, edgecolors='black', linewidth=2, zorder=10)
        
        # 添加标签
        for i, (w, f, l, c) in enumerate(zip(waves, fluxes, labels, colors)):
            offset_y = 15 if i % 2 == 0 else -25
            ax2.annotate(l, (w, f), 
                        textcoords="offset points", xytext=(0, offset_y),
                        ha='center', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.7, edgecolor='black'))
        
        ax2.set_xlabel('Wavelength (Å)', fontsize=13)
        ax2.set_ylabel('Flux Density (Jy)', fontsize=13)
        ax2.set_title(f'{name} - SED', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xlim(2000, 30000)
        ax2.grid(True, alpha=0.3)
        
        # 添加表数据
        phot_text = "Photometry Data:\n"
        for l, m in zip(labels, mags_list):
            phot_text += f"{l}: {m:.2f}\n"
        
        fig.text(0.02, 0.02, phot_text, fontsize=10, 
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Spectral Energy Distribution - {name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    @staticmethod
    def plot_hr_diagram(gaia_data: Dict, field_data, output_path: str, name: str = "Target"):
        """绘制Gaia HR图 - 标记目标星位置"""
        fig, ax = plt.subplots(figsize=(11, 9))
        
        # 绘制背景星
        if field_data is not None:
            bp_rp = field_data['bp_rp']
            g_abs = field_data['g_abs']
            
            # 数据清理
            mask = (bp_rp > -1) & (bp_rp < 4) & (g_abs > -5) & (g_abs < 15) & np.isfinite(bp_rp) & np.isfinite(g_abs)
            
            if np.sum(mask) > 0:
                ax.scatter(bp_rp[mask], g_abs[mask], c='lightgray', s=3, alpha=0.5, 
                          label=f'Field stars ({np.sum(mask)})', rasterized=True)
        
        # 绘制目标星
        if gaia_data.get('found'):
            bp_rp_star = gaia_data.get('bp_rp')
            g_abs_star = gaia_data.get('g_abs')
            
            if bp_rp_star is not None and g_abs_star is not None:
                ax.scatter([bp_rp_star], [g_abs_star], c='red', s=400, marker='*',
                          edgecolors='black', linewidth=2, label=f'{name}', zorder=10)
                
                # 添加标注
                annot_text = f'{name}\n'
                annot_text += f'G={gaia_data["g_mag"]:.2f}\n'
                annot_text += f'd={gaia_data["distance"]:.0f}pc\n'
                if gaia_data.get('teff'):
                    annot_text += f'Teff={gaia_data["teff"]:.0f}K'
                
                ax.annotate(annot_text, 
                           xy=(bp_rp_star, g_abs_star),
                           xytext=(20, 20), textcoords='offset points',
                           fontsize=11, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
                
                print(f"   标记目标: BP-RP={bp_rp_star:.3f}, M_G={g_abs_star:.3f}")
        
        # 添加理论等龄线（简化）
        # 主序带大致位置
        bp_ms = np.linspace(-0.3, 2.5, 100)
        mg_ms = 5.5 + 2.0*bp_ms - 0.2*bp_ms**2
        ax.plot(bp_ms, mg_ms, 'b--', alpha=0.3, linewidth=1, label='Main sequence (approx)')
        
        # 白矮星位置
        ax.scatter([0.0], [11.0], c='cyan', s=150, marker='D',
                  edgecolors='black', linewidth=2, label='WD (typical)', zorder=5)
        
        ax.set_xlabel('BP - RP (mag)', fontsize=14)
        ax.set_ylabel('Absolute G Magnitude (mag)', fontsize=14)
        ax.set_title(f'Hertzsprung-Russell Diagram - {name}\n(Gaia DR3)', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 4)
        ax.set_ylim(16, -5)
        
        # 添加恒星演化阶段标注
        ax.axhline(4, color='orange', linestyle=':', alpha=0.3)
        ax.text(3.5, 3.5, 'Giants', fontsize=10, color='orange', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    @staticmethod
    def create_summary(result: Dict, output_path: str):
        """创建综合分析汇总图"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        name = result.get('name', 'Target')
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
        
        Gaia: {result.get('gaia', {}).get('source_id', 'N/A')[:20]}...
        Dist: {result.get('gaia', {}).get('distance', 'N/A'):.0f} pc
        """
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # 消光
        ax_ext = fig.add_subplot(gs[0, 1])
        ext = result.get('extinction', {})
        if ext.get('success'):
            ax_ext.bar(['E(B-V)', 'A_V'], [ext.get('E_B-V', 0), ext.get('A_V', 0)],
                      color=['blue', 'red'], alpha=0.7)
            ax_ext.set_ylabel('Mag')
            ax_ext.set_title('Extinction')
        
        # 测光
        ax_phot = fig.add_subplot(gs[0, 2])
        simbad = result.get('simbad', {})
        if simbad.get('matched') and 'magnitudes' in simbad:
            mags = simbad['magnitudes']
            bands = list(mags.keys())[:8]
            values = [mags[b] for b in bands]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(bands)))
            ax_phot.barh(bands, values, color=colors, alpha=0.8, edgecolor='black')
            ax_phot.set_xlabel('Mag')
            ax_phot.set_title('Photometry')
            ax_phot.invert_xaxis()
        
        # 周期图
        ax_period = fig.add_subplot(gs[1, :2])
        if 'period_analysis' in result and result['period_analysis']:
            pa = result['period_analysis']
            if 'frequency' in pa and 'power' in pa:
                ax_period.plot(pa['frequency'], pa['power'], 'k-', linewidth=0.5)
                if 'period' in result:
                    ax_period.axvline(1.0/result['period'], color='r', linestyle='--', linewidth=2)
                ax_period.set_xlabel('Frequency (1/day)')
                ax_period.set_ylabel('Power')
                ax_period.set_title('Periodogram')
                ax_period.grid(True, alpha=0.3)
        
        # 周期信息
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
                
                ax_folded.scatter(phase_double, mags_double, c='red', s=2, alpha=0.5, rasterized=True)
                ax_folded.set_xlabel('Phase')
                ax_folded.set_ylabel('Relative Magnitude')
                ax_folded.set_title(f'Folded Light Curve (P={period*24:.4f}h)')
                ax_folded.set_xlim(0, 2)
                ax_folded.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path


# ==================== 主分析类 ====================

class AstroAnalyzer:
    """天体分析主类"""
    
    def __init__(self):
        ensure_dirs()
        self.visualizer = Visualizer()
        self.period_analyzer = PeriodAnalyzer()
    
    def analyze(self, ra: float, dec: float, name: str = "Target") -> Dict:
        print("="*70)
        print(f"🔭 AstroAnalyzer v2: {name}")
        print(f"   Coordinates: RA={ra}°, DEC={dec}°")
        print("="*70)
        
        result = {
            'name': name, 'ra': ra, 'dec': dec,
            'timestamp': datetime.now().isoformat(),
            'figures': {}
        }
        
        # 1. SIMBAD查询
        print("\n【1/6】SIMBAD Query...")
        simbad = SIMBADQuerier.query(ra, dec)
        result['simbad'] = simbad
        
        if simbad.get('matched'):
            print(f"   ✅ ID: {simbad['main_id']}")
            print(f"   ✅ Type: {simbad.get('otype', 'N/A')}")
            print(f"   ✅ SpType: {simbad.get('sp_type', 'N/A')}")
            print(f"   ✅ Magnitudes: {list(simbad.get('magnitudes', {}).keys())}")
            if simbad.get('has_lamost'):
                print(f"   ✅ LAMOST data available")
            
            # 绘制SED
            sed_path = os.path.join(FIG_DIR, f"{name}_sed.png")
            if self.visualizer.plot_sed(simbad, sed_path, name):
                result['figures']['sed'] = sed_path
                print(f"   ✅ SED plot saved")
        
        # 2. Gaia查询（用于HR图）
        print("\n【2/6】Gaia Query...")
        gaia = GaiaQuerier.query_star(ra, dec)
        result['gaia'] = gaia
        
        if gaia.get('found'):
            print(f"   ✅ Source ID: {gaia['source_id']}")
            print(f"   ✅ Distance: {gaia['distance']:.1f} pc")
            print(f"   ✅ G = {gaia['g_mag']:.2f}, M_G = {gaia['g_abs']:.2f}")
            print(f"   ✅ BP-RP = {gaia.get('bp_rp', 'N/A')}")
            
            # 查询视场星
            field_data = GaiaQuerier.query_field_stars(ra, dec)
            
            # 绘制HR图
            hr_path = os.path.join(FIG_DIR, f"{name}_hr_diagram.png")
            self.visualizer.plot_hr_diagram(gaia, field_data, hr_path, name)
            result['figures']['hr_diagram'] = hr_path
            print(f"   ✅ HR diagram saved")
        else:
            print(f"   ⚠️ Gaia data not found")
        
        # 3. LAMOST光谱查询
        print("\n【3/6】LAMOST Spectrum Query...")
        if simbad.get('has_lamost'):
            lamost = LAMOSTQuerier.query_spectrum(ra, dec)
            result['lamost'] = lamost
            
            if lamost.get('found'):
                print(f"   ✅ LAMOST ID: {lamost['obsid']}")
                print(f"   ✅ SNR: u={lamost.get('snr_u', 'N/A')}, g={lamost.get('snr_g', 'N/A')}, r={lamost.get('snr_r', 'N/A')}")
                if lamost.get('teff'):
                    print(f"   ✅ Teff={lamost['teff']:.0f}K, logg={lamost['logg']:.2f}, [Fe/H]={lamost['feh']:.2f}")
            else:
                print(f"   ⚠️ LAMOST query failed: {lamost.get('error', 'Not found')}")
        else:
            print(f"   ℹ️ No LAMOST data in SIMBAD")
        
        # 4. 周期分析
        print("\n【4/6】Period Analysis...")
        tess_files = [f for f in os.listdir(DATA_DIR)
                     if f.startswith(name) and f.endswith('.csv') and 'TESS' in f]
        
        if tess_files:
            csv_file = min([os.path.join(DATA_DIR, f) for f in tess_files], key=os.path.getsize)
            print(f"   Analyzing: {os.path.basename(csv_file)}")
            
            period_result = self.period_analyzer.analyze_csv(csv_file)
            
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
                
                print(f"   ✅ Period: {period*24:.4f} hours")
                
                # 绘制周期图
                per_path = os.path.join(FIG_DIR, f"{name}_periodogram.png")
                self.visualizer.plot_periodogram(
                    period_result['frequency'], period_result['power'],
                    period, per_path, name)
                result['figures']['periodogram'] = per_path
                
                # 绘制折叠光变曲线
                fold_path = os.path.join(FIG_DIR, f"{name}_folded_lc.png")
                self.visualizer.plot_folded_lc(
                    period_result['times'], period_result['mags'],
                    period, fold_path, name, "(TESS)")
                result['figures']['folded_lc'] = fold_path
        
        # 5. 消光查询
        print("\n【5/6】Extinction Query...")
        ext = ExtinctionQuerier.query(ra, dec)
        result['extinction'] = ext
        if ext.get('success'):
            print(f"   ✅ E(B-V) = {ext.get('E_B-V', 'N/A')}, A_V = {ext.get('A_V', 'N/A')}")
        
        # 6. 汇总图
        print("\n【6/6】Generating Summary...")
        summary_path = os.path.join(FIG_DIR, f"{name}_summary.png")
        self.visualizer.create_summary(result, summary_path)
        result['figures']['summary'] = summary_path
        print(f"   ✅ Summary plot saved")
        
        # 保存结果
        result_file = os.path.join(OUTPUT_DIR, f"{name}_analysis_v2.json")
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(convert(result), f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✅ Results: {result_file}")
        
        # 打印汇总
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: Dict):
        print("\n" + "="*70)
        print("📊 ANALYSIS COMPLETE")
        print("="*70)
        
        print(f"\nTarget: {result['name']}")
        print(f"Coordinates: RA={result['ra']}°, DEC={result['dec']}°")
        
        simbad = result.get('simbad', {})
        if simbad.get('matched'):
            print(f"\nObject Information:")
            print(f"  ID: {simbad['main_id']}")
            print(f"  Type: {simbad['otype']}")
            print(f"  Spectral Type: {simbad.get('sp_type', 'N/A')}")
        
        gaia = result.get('gaia', {})
        if gaia.get('found'):
            print(f"\nGaia Data:")
            print(f"  Source ID: {gaia['source_id']}")
            print(f"  Distance: {gaia['distance']:.1f} pc")
            print(f"  M_G = {gaia['g_abs']:.2f}")
        
        lamost = result.get('lamost', {})
        if lamost and lamost.get('found'):
            print(f"\nLAMOST Spectrum:")
            print(f"  obsid: {lamost['obsid']}")
            if lamost.get('teff'):
                print(f"  Teff = {lamost['teff']:.0f} K")
        
        if 'period' in result:
            print(f"\nPeriod Analysis:")
            print(f"  P = {result['period']*24:.4f} hours")
        
        print(f"\nGenerated Figures ({len(result['figures'])}):")
        for fig_type, fig_path in sorted(result['figures'].items()):
            print(f"  ✅ {fig_type}: {os.path.basename(fig_path)}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Astronomical Analysis System v2')
    parser.add_argument('--ra', type=float, help='Right Ascension (deg)')
    parser.add_argument('--dec', type=float, help='Declination (deg)')
    parser.add_argument('--name', type=str, default=None, help='Target name')
    parser.add_argument('--demo', action='store_true', help='Run AM Her demo')
    
    args = parser.parse_args()
    
    analyzer = AstroAnalyzer()
    
    if args.demo or (args.ra is None and args.dec is None):
        print("Running AM Her demo...")
        analyzer.analyze(ra=274.0554, dec=49.8679, name="AM_Her")
    else:
        if args.ra is None or args.dec is None:
            print("Error: Both --ra and --dec are required")
            sys.exit(1)
        name = args.name or f"Target_{args.ra:.2f}_{args.dec:.2f}"
        analyzer.analyze(ra=args.ra, dec=args.dec, name=name)


if __name__ == "__main__":
    main()
