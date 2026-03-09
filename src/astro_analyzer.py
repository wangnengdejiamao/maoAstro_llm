#!/usr/bin/env python3
"""
天体分析系统 v1.0
================
完整的天文数据分析工具，支持：
- SIMBAD 数据库查询
- TESS 光变曲线下载与周期分析
- Gaia 赫罗图（或估计版本）
- SED 能谱分布图
- 银河消光计算
- 综合分析报告

用法：
    python astro_analyzer.py --ra 274.0554 --dec 49.8679 --name "AM_Her"
    python astro_analyzer.py --demo  # 运行AM Her演示
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
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# 配置
OUTPUT_DIR = "./output"
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def ensure_dirs():
    """确保输出目录存在"""
    for d in [OUTPUT_DIR, DATA_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)


# ==================== 数据查询模块 ====================

class SIMBADQuerier:
    """SIMBAD数据库查询"""
    
    @staticmethod
    def query(ra: float, dec: float, radius: float = 5) -> Dict:
        """查询SIMBAD数据库"""
        try:
            from astroquery.simbad import Simbad
            
            custom_simbad = Simbad()
            # 使用简单字段避免解析错误
            custom_simbad.add_votable_fields('otype', 'sp', 
                                             'fluxdata(U)', 'fluxdata(B)', 'fluxdata(V)',
                                             'fluxdata(R)', 'fluxdata(I)')
            
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
            
            # 测光数据
            mags = {}
            for band in ['U', 'B', 'V', 'R', 'I']:
                try:
                    key = f'FLUX_{band}'
                    if key in row.colnames and row[key] is not None:
                        val = float(row[key])
                        if np.isfinite(val):
                            mags[band] = val
                except:
                    pass
            
            if mags:
                data['magnitudes'] = mags
                if 'B' in mags and 'V' in mags:
                    data['bv'] = mags['B'] - mags['V']
            
            return data
            
        except Exception as e:
            return {'success': False, 'matched': False, 'error': str(e)}


class ExtinctionQuerier:
    """银河消光查询"""
    
    @staticmethod
    def query(ra: float, dec: float) -> Dict:
        """查询银河消光"""
        try:
            sys.path.insert(0, './src')
            from astro_tools import AstroTools
            
            tools = AstroTools()
            return tools.query_extinction(ra, dec)
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ==================== 周期分析模块 ====================

class PeriodAnalyzer:
    """光变曲线周期分析"""
    
    @staticmethod
    def find_period(times: np.ndarray, mags: np.ndarray, errors: np.ndarray,
                   fmin: float = 0.1, fmax: float = 50) -> Tuple:
        """Lomb-Scargle周期分析"""
        mask = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errors)
        t, y, dy = times[mask], mags[mask], errors[mask]
        
        if len(t) < 20:
            return None, None, None, None
        
        ls = LombScargle(t, y, dy)
        frequency, power = ls.autopower(minimum_frequency=fmin, 
                                        maximum_frequency=fmax, 
                                        samples_per_peak=15)
        
        best_idx = np.argmax(power)
        best_period = 1.0 / frequency[best_idx]
        
        try:
            false_alarm = ls.false_alarm_probability(power[best_idx])
            significance = 1.0 - false_alarm
        except:
            significance = power[best_idx] / np.mean(power)
        
        return best_period, frequency, power, significance
    
    @staticmethod
    def analyze_csv(csv_file: str, period_range: Tuple[float, float] = (0.08, 0.3)) -> Optional[Dict]:
        """分析CSV文件的周期"""
        try:
            df = pd.read_csv(csv_file)
            
            # 查找列
            time_col = next((c for c in ['time', 'TIME', 'bjd', 'btjd'] if c in df.columns), None)
            flux_col = next((c for c in ['pdcsap_flux', 'SAP_FLUX', 'flux', 'kspsap_flux'] 
                           if c in df.columns), None)
            
            if time_col is None or flux_col is None:
                return None
            
            # 清理数据
            df = df[(df[flux_col] > 0) & np.isfinite(df[flux_col])]
            
            times = df[time_col].values
            fluxes = df[flux_col].values
            
            # 转换为星等
            mags = -2.5 * np.log10(fluxes)
            mags = mags - np.median(mags)
            errors = np.ones_like(mags) * 0.001
            
            # 周期搜索
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
            print(f"  周期分析错误: {e}")
            return None


# ==================== 可视化模块 ====================

class Visualizer:
    """天文数据可视化"""
    
    @staticmethod
    def plot_periodogram(freq: np.ndarray, power: np.ndarray, period: float,
                        output_path: str, name: str = "Target") -> str:
        """绘制周期图"""
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
    def plot_folded_lc(times: np.ndarray, mags: np.ndarray, period: float,
                      output_path: str, name: str = "Target", subtitle: str = "") -> str:
        """绘制折叠光变曲线"""
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
        ax.scatter(phase_double, mags_double, c='red', s=3, alpha=0.5, 
                  label='Data', rasterized=True)
        
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
    def plot_sed(simbad_data: Dict, output_path: str, name: str = "Target") -> Optional[str]:
        """绘制SED图"""
        if not simbad_data.get('matched') or 'magnitudes' not in simbad_data:
            return None
        
        wavelengths = {
            'U': 3600, 'B': 4450, 'V': 5510, 'R': 6580, 'I': 8060,
            'J': 12350, 'H': 16620, 'K': 21590
        }
        
        mags = simbad_data['magnitudes']
        waves, mags_list, labels = [], [], []
        
        for band, mag in mags.items():
            if band in wavelengths and np.isfinite(mag):
                waves.append(wavelengths[band])
                mags_list.append(mag)
                labels.append(band)
        
        if len(waves) < 2:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(waves)))
        ax1.scatter(waves, mags_list, c=colors, s=150, edgecolors='black', linewidth=2, zorder=10)
        for w, m, l in zip(waves, mags_list, labels):
            ax1.annotate(l, (w, m), textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('Wavelength (Å)', fontsize=12)
        ax1.set_ylabel('Magnitude', fontsize=12)
        ax1.set_title(f'{name} - Magnitudes', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 流量密度
        fluxes = np.array([10**(-0.4 * m) for m in mags_list])
        ax2.scatter(waves, fluxes * 1e23, c=colors, s=150, edgecolors='black', linewidth=2, zorder=10)
        for w, f, l in zip(waves, fluxes * 1e23, labels):
            ax2.annotate(l, (w, f), textcoords="offset points", xytext=(0, 12),
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
    
    @staticmethod
    def create_summary(result: Dict, output_path: str) -> str:
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
        """
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # 消光
        ax_ext = fig.add_subplot(gs[0, 1])
        ext = result.get('extinction', {})
        if ext.get('success'):
            ax_ext.bar(['E(B-V)', 'A_V'], [ext.get('E_B_V', 0), ext.get('A_V', 0)],
                      color=['blue', 'red'], alpha=0.7)
            ax_ext.set_ylabel('Mag')
            ax_ext.set_title('Extinction')
        else:
            ax_ext.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax_ext.set_title('Extinction')
        
        # 测光
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
        
        # 周期图
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


# ==================== 主分析类 ====================

class AstroAnalyzer:
    """天体分析主类"""
    
    def __init__(self):
        ensure_dirs()
        self.visualizer = Visualizer()
        self.period_analyzer = PeriodAnalyzer()
    
    def analyze(self, ra: float, dec: float, name: str = "Target") -> Dict:
        """
        执行完整的天体分析
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
        
        Returns:
            分析结果字典
        """
        print("="*70)
        print(f"🔭 AstroAnalyzer: {name}")
        print(f"   Coordinates: RA={ra}°, DEC={dec}°")
        print("="*70)
        
        result = {
            'name': name, 'ra': ra, 'dec': dec,
            'timestamp': datetime.now().isoformat(),
            'figures': {}
        }
        
        # 1. SIMBAD查询
        print("\n【1/5】SIMBAD Query...")
        simbad = SIMBADQuerier.query(ra, dec)
        result['simbad'] = simbad
        
        if simbad.get('matched'):
            print(f"   ✅ ID: {simbad['main_id']}")
            print(f"   ✅ Type: {simbad.get('otype', 'N/A')}")
            print(f"   ✅ SpType: {simbad.get('sp_type', 'N/A')}")
            
            # 绘制SED
            sed_path = os.path.join(FIG_DIR, f"{name}_sed.png")
            if self.visualizer.plot_sed(simbad, sed_path, name):
                result['figures']['sed'] = sed_path
                print(f"   ✅ SED plot saved")
        else:
            print(f"   ⚠️ {simbad.get('message', simbad.get('error', 'Not matched'))}")
        
        # 2. 周期分析
        print("\n【2/5】Period Analysis...")
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
                print(f"   ✅ Significance: {period_result['significance']:.4f}")
                
                # 绘制周期图
                per_path = os.path.join(FIG_DIR, f"{name}_periodogram.png")
                self.visualizer.plot_periodogram(
                    period_result['frequency'], period_result['power'],
                    period, per_path, name
                )
                result['figures']['periodogram'] = per_path
                
                # 绘制折叠光变曲线
                fold_path = os.path.join(FIG_DIR, f"{name}_folded_lc.png")
                self.visualizer.plot_folded_lc(
                    period_result['times'], period_result['mags'],
                    period, fold_path, name, "(TESS)"
                )
                result['figures']['folded_lc'] = fold_path
            else:
                print("   ⚠️ Period analysis failed")
        else:
            print("   ⚠️ No TESS files found")
        
        # 3. 消光查询
        print("\n【3/5】Extinction Query...")
        ext = ExtinctionQuerier.query(ra, dec)
        result['extinction'] = ext
        if ext.get('success'):
            print(f"   ✅ E(B-V) = {ext.get('E_B_V', 'N/A')}, A_V = {ext.get('A_V', 'N/A')}")
        else:
            print(f"   ⚠️ Extinction query failed")
        
        # 4. 汇总图
        print("\n【4/5】Generating Summary...")
        summary_path = os.path.join(FIG_DIR, f"{name}_summary.png")
        self.visualizer.create_summary(result, summary_path)
        result['figures']['summary'] = summary_path
        print(f"   ✅ Summary plot saved")
        
        # 5. 保存结果
        print("\n【5/5】Saving Results...")
        result_file = os.path.join(OUTPUT_DIR, f"{name}_analysis.json")
        
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
        """打印分析汇总"""
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
        
        if 'period' in result:
            print(f"\nPeriod Analysis:")
            print(f"  P = {result['period']*24:.4f} hours")
            print(f"  Significance = {result['period_significance']:.4f}")
        
        print(f"\nGenerated Figures ({len(result['figures'])}):")
        for fig_type, fig_path in sorted(result['figures'].items()):
            print(f"  ✅ {fig_type}: {os.path.basename(fig_path)}")
        
        print("\n" + "="*70)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Astronomical Analysis System')
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
