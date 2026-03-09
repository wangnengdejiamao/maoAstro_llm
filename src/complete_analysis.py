#!/usr/bin/env python3
"""
完整天体分析流程
================
整合所有数据源和可视化
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from astropy.coordinates import SkyCoord

# 导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extended_tools import (
    SIMBADQuery, TESSQuery, GaiaHRDiagram, 
    SEDBuilder, ZTFAnalyzer
)
from astro_tools import AstroTools
from rag_system import AstronomyRAG


class CompleteAstroAnalysis:
    """完整天体分析类"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.fig_dir = os.path.join(output_dir, "figures")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        
        self.basic_tools = AstroTools()
        self.rag = AstronomyRAG()
    
    def analyze(self, ra: float, dec: float, name: str = None):
        """
        执行完整分析
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
        """
        name = name or f"Target_{ra:.4f}_{dec:.4f}"
        coord_str = f"RA={ra:.4f}, DEC={dec:.4f}"
        
        print(f"\n{'='*70}")
        print(f"🔭 完整天体分析: {name}")
        print(f"   坐标: {coord_str}")
        print(f"{'='*70}\n")
        
        result = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'timestamp': datetime.now().isoformat(),
            'data_dir': self.data_dir,
            'figures': {}
        }
        
        # ==================== 1. 基础数据查询 ====================
        print("【1/7】基础数据查询...")
        
        # 消光
        print("  - 查询消光...")
        ext = self.basic_tools.query_extinction(ra, dec)
        result['extinction'] = ext
        if ext.get('success'):
            print(f"    ✓ A_V = {ext['A_V']}, E(B-V) = {ext['E_B_V']}")
        
        # ZTF
        print("  - 查询ZTF...")
        ztf_summary = self.basic_tools.query_ztf(ra, dec)
        result['ztf_summary'] = ztf_summary
        if ztf_summary.get('success'):
            print(f"    ✓ {ztf_summary['n_points']} 个数据点")
        
        # ==================== 2. SIMBAD查询 ====================
        print("\n【2/7】SIMBAD数据库查询...")
        simbad = SIMBADQuery.query(ra, dec)
        result['simbad'] = simbad
        
        if simbad.get('matched'):
            print(f"    ✓ 匹配: {simbad['main_id']}")
            print(f"    ✓ 类型: {simbad.get('otype', 'N/A')}")
            print(f"    ✓ 光谱型: {simbad.get('sp_type', 'N/A')}")
            
            # 保存SIMBAD数据
            simbad_file = os.path.join(self.data_dir, f"{name}_simbad.json")
            with open(simbad_file, 'w') as f:
                json.dump(simbad, f, indent=2)
            print(f"    ✓ 数据保存: {simbad_file}")
            
            # 构建SED
            if 'magnitudes' in simbad:
                print("  - 构建SED...")
                sed_df = SEDBuilder.build_sed_from_simbad(simbad)
                if sed_df is not None:
                    sed_file = os.path.join(self.data_dir, f"{name}_sed.csv")
                    sed_df.to_csv(sed_file, index=False)
                    print(f"    ✓ SED数据保存: {sed_file}")
                    
                    # 绘制SED
                    sed_plot = os.path.join(self.fig_dir, f"{name}_sed.png")
                    SEDBuilder.plot_sed(sed_df, sed_plot, name)
                    result['figures']['sed'] = sed_plot
                    print(f"    ✓ SED图: {sed_plot}")
        else:
            print("    ✗ SIMBAD未匹配")
        
        # ==================== 3. TESS数据查询 ====================
        print("\n【3/7】TESS数据查询...")
        tess = TESSQuery.query(ra, dec)
        result['tess'] = tess
        
        if tess.get('found'):
            print(f"    ✓ 发现 {tess['n_sectors']} 个扇区")
            for m in tess.get('missions', []):
                print(f"      - {m['mission']}: {m['author']} ({m['year']})")
            
            # 下载TESS光变曲线
            print("  - 下载TESS光变曲线...")
            tess_plots = TESSQuery.download_and_plot(ra, dec, self.data_dir, name)
            if tess_plots:
                result['figures']['tess'] = tess_plots
                print(f"    ✓ 下载完成: {len(tess_plots)} 个文件")
        else:
            print(f"    ✗ {tess.get('message', '无TESS数据')}")
        
        # ==================== 4. Gaia数据与赫罗图 ====================
        print("\n【4/7】Gaia数据与赫罗图...")
        
        # 查询Gaia数据
        gaia_data = GaiaHRDiagram.query_gaia_data(ra, dec)
        if gaia_data is not None:
            gaia_file = os.path.join(self.data_dir, f"{name}_gaia.csv")
            gaia_data.write(gaia_file, overwrite=True, format='csv')
            print(f"    ✓ Gaia数据保存: {gaia_file}")
        
        # 绘制赫罗图
        hr_plot = os.path.join(self.fig_dir, f"{name}_hr_diagram.png")
        hr_result = GaiaHRDiagram.plot_hr_diagram(ra, dec, hr_plot)
        if hr_result:
            result['figures']['hr_diagram'] = hr_plot
            print(f"    ✓ 赫罗图: {hr_plot}")
        else:
            print("    ✗ 赫罗图绘制失败")
        
        # ==================== 5. ZTF折叠光变曲线 ====================
        print("\n【5/7】ZTF光变曲线分析...")
        
        # 尝试读取ZTF数据并绘制折叠光变曲线
        ztf_cache = f"./output/ztf_cache/ztf_{ra:.4f}_{dec:.4f}.csv"
        if os.path.exists(ztf_cache):
            try:
                ztf_df = pd.read_csv(ztf_cache)
                ztf_df.columns = [c.lower() for c in ztf_df.columns]
                
                # 重命名列以匹配标准格式
                col_map = {
                    'hjd': 'hjd',
                    'mjd': 'mjd',
                    'mag': 'mag',
                    'magerr': 'magerr',
                    'filtercode': 'filter'
                }
                ztf_df = ztf_df.rename(columns={k: v for k, v in col_map.items() if k in ztf_df.columns})
                
                if 'filter' not in ztf_df.columns and 'filtercode' in ztf_df.columns:
                    ztf_df['filter'] = ztf_df['filtercode']
                
                # 查找周期（如果有足够数据）
                if len(ztf_df) > 50:
                    print("  - 寻找周期...")
                    # 使用r波段
                    r_band = ztf_df[ztf_df['filter'] == 'zr'] if 'zr' in ztf_df['filter'].values else ztf_df
                    
                    if len(r_band) > 20:
                        try:
                            period, freq, freqs, power = ZTFAnalyzer.find_period(
                                r_band['mjd'].values,
                                r_band['mag'].values,
                                r_band['magerr'].values
                            )
                            print(f"    ✓ 最佳周期: {period:.6f} days ({period*24:.4f} hours)")
                            result['period'] = period
                            
                            # 绘制折叠光变曲线
                            ztf_plot = os.path.join(self.fig_dir, f"{name}_ztf_folded.png")
                            ZTFAnalyzer.plot_folded_lc(ztf_df, period, ztf_plot, name)
                            result['figures']['ztf_folded'] = ztf_plot
                            print(f"    ✓ ZTF折叠图: {ztf_plot}")
                        except Exception as e:
                            print(f"    ✗ 周期分析失败: {e}")
            except Exception as e:
                print(f"    ✗ ZTF分析失败: {e}")
        else:
            print("    ✗ 无ZTF缓存数据")
        
        # ==================== 6. RAG知识检索 ====================
        print("\n【6/7】检索天文知识...")
        
        # 根据SIMBAD类型选择查询关键词
        query_keyword = "variable star"
        if simbad.get('otype'):
            otype = simbad['otype'].lower()
            if 'cataclysmic' in otype or 'cv' in otype:
                query_keyword = "cataclysmic variable"
            elif 'puls' in otype:
                query_keyword = "pulsating variable"
            elif 'binary' in otype:
                query_keyword = "eclipsing binary"
        
        knowledge = self.rag.search(query_keyword)
        result['knowledge'] = knowledge[:1000]  # 限制大小
        print(f"    ✓ 检索知识: {query_keyword}")
        
        # ==================== 7. 生成汇总报告 ====================
        print("\n【7/7】生成汇总报告...")
        
        # 创建汇总图
        summary_plot = self._create_summary_plot(result, name)
        if summary_plot:
            result['figures']['summary'] = summary_plot
        
        # 保存完整结果
        result_file = os.path.join(self.output_dir, f"{name}_complete.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"    ✓ 完整结果: {result_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ 分析完成!")
        print(f"   数据目录: {self.data_dir}")
        print(f"   图像目录: {self.fig_dir}")
        print(f"{'='*70}\n")
        
        return result
    
    def _create_summary_plot(self, result: dict, name: str):
        """创建汇总图"""
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            ra = result['ra']
            dec = result['dec']
            
            fig.suptitle(f'{name} - Complete Analysis (RA={ra:.4f}, DEC={dec:.4f})',
                        fontsize=16, fontweight='bold')
            
            # 面板1: 基本信息
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.axis('off')
            
            info = f"Target: {name}\n\n"
            info += f"Coordinates:\n"
            info += f"  RA: {ra:.6f}°\n"
            info += f"  DEC: {dec:.6f}°\n\n"
            
            ext = result.get('extinction', {})
            if ext.get('success'):
                info += f"Extinction:\n"
                info += f"  A_V = {ext['A_V']}\n"
                info += f"  E(B-V) = {ext['E_B_V']}\n"
            
            simbad = result.get('simbad', {})
            if simbad.get('matched'):
                info += f"\nSIMBAD:\n"
                info += f"  ID: {simbad.get('main_id', 'N/A')}\n"
                info += f"  Type: {simbad.get('otype', 'N/A')}\n"
            
            ax1.text(0.1, 0.9, info, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # 面板2: 数据可用性
            ax2 = fig.add_subplot(gs[0, 1])
            
            data_status = []
            labels = []
            colors = []
            
            data_status.append(1 if ext.get('success') else 0)
            labels.append('Extinction')
            colors.append('green' if ext.get('success') else 'red')
            
            data_status.append(1 if result.get('simbad', {}).get('matched') else 0)
            labels.append('SIMBAD')
            colors.append('green' if result.get('simbad', {}).get('matched') else 'red')
            
            data_status.append(1 if result.get('tess', {}).get('found') else 0)
            labels.append('TESS')
            colors.append('green' if result.get('tess', {}).get('found') else 'red')
            
            data_status.append(1 if 'hr_diagram' in result.get('figures', {}) else 0)
            labels.append('Gaia HR')
            colors.append('green' if 'hr_diagram' in result.get('figures', {}) else 'red')
            
            y_pos = np.arange(len(labels))
            ax2.barh(y_pos, data_status, color=colors, alpha=0.6, edgecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlim(0, 1.5)
            ax2.set_title('Data Availability')
            
            # 面板3: SED预览（如果有）
            ax3 = fig.add_subplot(gs[0, 2])
            if 'sed' in result.get('figures', {}):
                sed_file = os.path.join(self.data_dir, f"{name}_sed.csv")
                if os.path.exists(sed_file):
                    sed_df = pd.read_csv(sed_file)
                    ax3.scatter(sed_df['wavelength'], sed_df['magnitude'],
                               s=100, c='red', edgecolors='black')
                    ax3.set_xlabel('Wavelength (A)')
                    ax3.set_ylabel('Magnitude')
                    ax3.set_title('SED Preview')
                    ax3.invert_yaxis()
            else:
                ax3.text(0.5, 0.5, 'No SED', ha='center', va='center',
                        transform=ax3.transAxes)
                ax3.set_title('SED')
            
            # 面板4-6: 图像缩略图位置
            for idx, (fig_type, title) in enumerate([('ztf_folded', 'ZTF Folded LC'),
                                                      ('tess', 'TESS LC'),
                                                      ('hr_diagram', 'HR Diagram')]):
                ax = fig.add_subplot(gs[1, idx])
                if fig_type in result.get('figures', {}):
                    ax.text(0.5, 0.5, f'{title}\n(Generated)', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='green')
                else:
                    ax.text(0.5, 0.5, f'{title}\n(Not Available)', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='red')
                ax.set_title(title)
                ax.axis('off')
            
            summary_path = os.path.join(self.fig_dir, f"{name}_summary.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return summary_path
            
        except Exception as e:
            print(f"  汇总图生成失败: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='完整天体分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python complete_analysis.py --ra 274.0554 --dec 49.8679 --name "AM_Her"
  python complete_analysis.py --ra 123.456 --dec 67.890 --name "MyStar"
        """
    )
    
    parser.add_argument('--ra', type=float, required=True, help='赤经（度）')
    parser.add_argument('--dec', type=float, required=True, help='赤纬（度）')
    parser.add_argument('--name', type=str, default=None, help='目标名称')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = CompleteAstroAnalysis(output_dir=args.output_dir)
    result = analyzer.analyze(args.ra, args.dec, args.name)
    
    # 打印结果摘要
    print("\n" + "="*70)
    print("📊 分析结果摘要")
    print("="*70)
    print(f"\n目标: {result['name']}")
    print(f"坐标: RA={result['ra']}, DEC={result['dec']}")
    
    print(f"\n生成的图像:")
    for fig_type, fig_path in result.get('figures', {}).items():
        if isinstance(fig_path, list):
            for p in fig_path:
                print(f"  - {os.path.basename(p)}")
        else:
            print(f"  - {os.path.basename(fig_path)}")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
