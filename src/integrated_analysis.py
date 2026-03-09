#!/usr/bin/env python3
"""
完整整合的天体分析系统
========================
整合功能：
1. Ollama Qwen 智能分析
2. VizieR SED 数据获取和绘图
3. LAMOST 背景的赫罗图
4. TESS TPF 绘图
5. 所有天文数据查询（消光、测光、SIMBAD等）
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
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from astro_tools import AstroTools
from extended_tools import SIMBADQuery, TESSQuery, ZTFAnalyzer
from sed_plotter import SEDAnalyzer
from hr_diagram_plotter import HRDiagramPlotter, TPFPlotter
from ollama_qwen_interface import OllamaQwenInterface
from spectrum_analyzer import SpectrumAnalyzer


class IntegratedAstroAnalysis:
    """
    整合天体分析类
    
    整合所有分析功能，提供一站式天体分析
    """
    
    def __init__(self, output_dir: str = "./output", use_ollama: bool = True, 
                 ollama_model: str = "qwen3:8b"):
        """
        初始化分析器
        
        Args:
            output_dir: 输出目录
            use_ollama: 是否使用 Ollama Qwen
            ollama_model: Ollama 模型名称
        """
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.fig_dir = os.path.join(output_dir, "figures")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        
        # 初始化工具
        self.basic_tools = AstroTools()
        self.sed_analyzer = SEDAnalyzer(output_dir=output_dir)
        
        # 初始化 Ollama 接口
        self.ollama = None
        if use_ollama:
            try:
                self.ollama = OllamaQwenInterface(model_name=ollama_model)
                print(f"✓ Ollama 接口初始化成功 ({ollama_model})")
            except Exception as e:
                print(f"⚠ Ollama 初始化失败: {e}")
        
        self.result = {}
    
    def analyze(self, ra: float, dec: float, name: str = None) -> dict:
        """
        执行完整分析流程
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
            
        Returns:
            分析结果字典
        """
        name = name or f"Target_{ra:.4f}_{dec:.4f}"
        
        print(f"\n{'='*70}")
        print(f"🔭 整合天体分析: {name}")
        print(f"   坐标: RA={ra:.6f}°, DEC={dec:.6f}°")
        print(f"{'='*70}\n")
        
        self.result = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'timestamp': datetime.now().isoformat(),
            'data_dir': self.data_dir,
            'figures': {},
            'analysis_text': ''
        }
        
        # ==================== 1. 基础数据查询 ====================
        print("【1/9】基础数据查询...")
        
        # 消光
        print("  - 查询消光...")
        ext = self.basic_tools.query_extinction(ra, dec)
        self.result['extinction'] = ext
        if ext.get('success'):
            print(f"    ✓ A_V = {ext['A_V']:.3f}, E(B-V) = {ext['E_B_V']:.3f}")
        
        # 测光
        print("  - 查询测光...")
        phot = self.basic_tools.query_photometry(ra, dec)
        self.result['photometry'] = phot
        if phot.get('success'):
            print(f"    ✓ 匹配 {phot.get('total_matches', 0)} 个目录")
        
        # ==================== 2. SIMBAD 查询 ====================
        print("\n【2/9】SIMBAD 数据库查询...")
        simbad = SIMBADQuery.query(ra, dec)
        self.result['simbad'] = simbad
        
        if simbad.get('matched'):
            print(f"    ✓ 匹配: {simbad['main_id']}")
            print(f"    ✓ 类型: {simbad.get('otype', 'N/A')}")
            print(f"    ✓ 光谱型: {simbad.get('sp_type', 'N/A')}")
        else:
            print("    ✗ SIMBAD 未匹配")
        
        # ==================== 3. 光谱分析 ====================
        print("\n【3/9】光谱分析...")
        spectrum_result = SpectrumAnalyzer.analyze_and_plot(
            ra, dec, name, self.output_dir
        )
        self.result['spectrum'] = spectrum_result
        
        if spectrum_result.get('success'):
            print(f"    ✓ 发现 {len(spectrum_result['spectra'])} 个光谱")
            for i, spec in enumerate(spectrum_result['spectra']):
                print(f"      - 光谱 {i+1}: {spec.get('type', 'Unknown')}")
        else:
            print("    ✗ 无本地光谱数据")
            print("    提示: 可将SDSS光谱放入 SDSS_Spectra_Downloads/ 目录")
        
        # ==================== 4. SED 分析 ====================
        print("\n【4/9】SED 分析...")
        sed_result = self.sed_analyzer.analyze(ra, dec, name)
        self.result['sed'] = sed_result
        
        if sed_result.get('success'):
            print(f"    ✓ SED 数据点: {sed_result.get('n_points', 0)}")
            print(f"    ✓ SED 图: {sed_result.get('sed_plot', 'N/A')}")
        else:
            print("    ✗ SED 分析失败")
        
        # ==================== 5. 赫罗图 ====================
        print("\n【5/9】赫罗图绘制...")
        hr_path = os.path.join(self.fig_dir, f"{name}_hr_diagram.png")
        hr_result = HRDiagramPlotter.plot_hr_diagram(
            ra, dec, name, 
            output_path=hr_path,
            extinction=ext
        )
        if hr_result:
            self.result['figures']['hr_diagram'] = hr_result
            print(f"    ✓ 赫罗图: {hr_result}")
        else:
            print("    ✗ 赫罗图绘制失败")
        
        # ==================== 6. TESS 数据 ====================
        print("\n【6/9】TESS 数据查询...")
        tess = TESSQuery.query(ra, dec)
        self.result['tess'] = tess
        
        if tess.get('found'):
            print(f"    ✓ 发现 {tess['n_sectors']} 个扇区")
            
            # 下载光变曲线
            print("  - 下载 TESS 光变曲线...")
            tess_plots = TESSQuery.download_and_plot(ra, dec, self.data_dir, name)
            if tess_plots:
                self.result['figures']['tess'] = tess_plots
                print(f"    ✓ 下载完成: {len(tess_plots)} 个文件")
            
            # TPF 绘图
            print("  - 绘制 TPF...")
            tpf_path = os.path.join(self.fig_dir, f"{name}_TPF.png")
            tpf_result = TPFPlotter.plot_tpf_simple(ra, dec, name, tpf_path)
            if tpf_result:
                self.result['figures']['tpf'] = tpf_result
                print(f"    ✓ TPF 图: {tpf_result}")
        else:
            print(f"    ✗ {tess.get('message', '无 TESS 数据')}")
        
        # ==================== 7. ZTF 分析 ====================
        print("\n【7/9】ZTF 光变曲线分析...")
        ztf_summary = self.basic_tools.query_ztf(ra, dec)
        self.result['ztf'] = ztf_summary
        
        if ztf_summary.get('success'):
            print(f"    ✓ {ztf_summary['n_points']} 个数据点")
        
        # ==================== 8. 创建汇总图 ====================
        print("\n【8/9】生成汇总图表...")
        summary_path = self._create_summary_plot(name)
        if summary_path:
            self.result['figures']['summary'] = summary_path
            print(f"    ✓ 汇总图: {summary_path}")
        
        # ==================== 9. AI 分析 ====================
        print("\n【9/9】AI 智能分析...")
        
        if self.ollama and self.ollama.is_available():
            print("  - 使用 Ollama Qwen 分析...")
            try:
                ai_analysis = self.ollama.analyze_target_summary(self.result)
                self.result['analysis_text'] = ai_analysis
                if ai_analysis.startswith('['):
                    print(f"    ⚠ AI 服务暂时不可用，已使用本地分析")
                else:
                    print(f"    ✓ AI 分析完成")
            except Exception as e:
                print(f"    ⚠ AI 分析失败: {e}")
                ai_analysis = self._local_analysis()
                self.result['analysis_text'] = ai_analysis
        else:
            print("  - Ollama 不可用，使用本地规则分析...")
            print("    (提示: 如需AI分析，请运行: ollama serve)")
            ai_analysis = self._local_analysis()
            self.result['analysis_text'] = ai_analysis
        
        # 保存完整结果
        self._save_results(name)
        
        print(f"\n{'='*70}")
        print(f"✅ 分析完成!")
        print(f"   数据目录: {self.data_dir}")
        print(f"   图像目录: {self.fig_dir}")
        print(f"{'='*70}\n")
        
        return self.result
    
    def _create_summary_plot(self, name: str) -> str:
        """创建汇总图"""
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
            
            ra = self.result['ra']
            dec = self.result['dec']
            
            fig.suptitle(f'{name} - Complete Analysis Report\n(RA={ra:.6f}°, DEC={dec:.6f}°)',
                        fontsize=16, fontweight='bold')
            
            # 面板1: 基本信息
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.axis('off')
            
            info_lines = [
                f"Target: {name}",
                f"",
                f"Coordinates:",
                f"  RA = {ra:.6f}°",
                f"  DEC = {dec:.6f}°",
                f"",
            ]
            
            ext = self.result.get('extinction', {})
            if ext.get('success'):
                info_lines.extend([
                    f"Extinction:",
                    f"  A_V = {ext['A_V']:.3f} mag",
                    f"  E(B-V) = {ext['E_B_V']:.3f} mag",
                ])
            
            simbad = self.result.get('simbad', {})
            if simbad.get('matched'):
                info_lines.extend([
                    f"",
                    f"SIMBAD:",
                    f"  ID: {simbad.get('main_id', 'N/A')}",
                    f"  Type: {simbad.get('otype', 'N/A')}",
                    f"  SpType: {simbad.get('sp_type', 'N/A')}",
                ])
            
            info_text = '\n'.join(info_lines)
            ax1.text(0.1, 0.95, info_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax1.set_title('Basic Information', fontsize=11, fontweight='bold')
            
            # 面板2: 数据可用性
            ax2 = fig.add_subplot(gs[0, 1])
            
            status_data = []
            labels = []
            colors = []
            
            # 消光
            status_data.append(1 if ext.get('success') else 0)
            labels.append('Extinction')
            colors.append('green' if ext.get('success') else 'red')
            
            # SIMBAD
            status_data.append(1 if simbad.get('matched') else 0)
            labels.append('SIMBAD')
            colors.append('green' if simbad.get('matched') else 'red')
            
            # SED
            sed = self.result.get('sed', {})
            status_data.append(1 if sed.get('success') else 0)
            labels.append('SED')
            colors.append('green' if sed.get('success') else 'red')
            
            # TESS
            tess = self.result.get('tess', {})
            status_data.append(1 if tess.get('found') else 0)
            labels.append('TESS')
            colors.append('green' if tess.get('found') else 'red')
            
            # HR Diagram
            status_data.append(1 if 'hr_diagram' in self.result.get('figures', {}) else 0)
            labels.append('HR Diagram')
            colors.append('green' if 'hr_diagram' in self.result.get('figures', {}) else 'red')
            
            y_pos = np.arange(len(labels))
            bars = ax2.barh(y_pos, status_data, color=colors, alpha=0.6, edgecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=9)
            ax2.set_xlim(0, 1.5)
            ax2.set_title('Data Availability', fontsize=11, fontweight='bold')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, status_data)):
                label = '✓' if val == 1 else '✗'
                ax2.text(val + 0.05, i, label, va='center', fontsize=12)
            
            # 面板3: 测光目录
            ax3 = fig.add_subplot(gs[0, 2])
            phot = self.result.get('photometry', {})
            if phot.get('success'):
                cats = phot.get('catalogs', {})
                cat_names = list(cats.keys())
                cat_values = [1 if str(v) not in ['0', 'Not found', '-1'] else 0 for v in cats.values()]
                colors_bar = ['green' if v == 1 else 'red' for v in cat_values]
                
                y_pos = np.arange(len(cat_names))
                ax3.barh(y_pos, cat_values, color=colors_bar, alpha=0.6, edgecolor='black')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(cat_names, fontsize=8)
                ax3.set_xlim(0, 1.5)
                ax3.set_title('Photometry Catalogs', fontsize=11, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Photometry', fontsize=11, fontweight='bold')
                ax3.axis('off')
            
            # 面板4: AI 分析结果
            ax4 = fig.add_subplot(gs[1, :2])
            ax4.axis('off')
            
            analysis = self.result.get('analysis_text', 'No analysis available.')
            # 截断文本
            if len(analysis) > 800:
                analysis = analysis[:797] + '...'
            
            ax4.text(0.05, 0.95, 'AI Analysis:', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top')
            ax4.text(0.05, 0.88, analysis, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top', wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            ax4.set_title('Intelligent Analysis', fontsize=11, fontweight='bold')
            
            # 面板5: 图像缩略图位置
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.axis('off')
            
            fig_types = [
                ('sed', 'SED'),
                ('hr_diagram', 'HR Diagram'),
                ('tess', 'TESS LC'),
                ('tpf', 'TPF'),
                ('ztf', 'ZTF LC')
            ]
            
            fig_status = []
            for fig_type, label in fig_types:
                if fig_type in self.result.get('figures', {}):
                    fig_status.append(f"✓ {label}")
                else:
                    fig_status.append(f"✗ {label}")
            
            status_text = '\n'.join(fig_status)
            ax5.text(0.1, 0.95, 'Generated Figures:', transform=ax5.transAxes,
                    fontsize=11, fontweight='bold', verticalalignment='top')
            ax5.text(0.1, 0.85, status_text, transform=ax5.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            ax5.set_title('Figure Status', fontsize=11, fontweight='bold')
            
            # 底部: 时间戳和说明
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')
            
            timestamp = self.result.get('timestamp', datetime.now().isoformat())
            footer_text = f"Generated: {timestamp} | Astro-AI Integrated Analysis System"
            ax6.text(0.5, 0.5, footer_text, transform=ax6.transAxes,
                    fontsize=9, ha='center', va='center', style='italic',
                    color='gray')
            
            # 保存
            summary_path = os.path.join(self.fig_dir, f"{name}_integrated_summary.png")
            plt.savefig(summary_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return summary_path
            
        except Exception as e:
            print(f"  ✗ 汇总图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _local_analysis(self) -> str:
        """本地规则分析（当 Ollama 不可用时）"""
        lines = []
        
        name = self.result.get('name', 'Unknown')
        lines.append(f"## {name} 分析报告")
        lines.append("")
        
        # SIMBAD 信息
        simbad = self.result.get('simbad', {})
        if simbad.get('matched'):
            lines.append(f"**天体类型**: {simbad.get('otype', 'Unknown')}")
            lines.append(f"**光谱型**: {simbad.get('sp_type', 'N/A')}")
            lines.append("")
        
        # 消光分析
        ext = self.result.get('extinction', {})
        if ext.get('success'):
            av = ext.get('A_V', 0)
            lines.append(f"**消光分析**: A_V = {av:.3f} mag")
            if av > 1.0:
                lines.append("该天体位于银道面方向，有显著星际消光。")
            lines.append("")
        
        # SED 分析
        sed = self.result.get('sed', {})
        if sed.get('success'):
            n_points = sed.get('n_points', 0)
            lines.append(f"**SED 分析**: 从 VizieR 获取了 {n_points} 个波段的数据点。")
            lines.append("SED 图显示了天体从紫外观测到红外的能量分布。")
            lines.append("")
        
        # TESS 分析
        tess = self.result.get('tess', {})
        if tess.get('found'):
            n_sectors = tess.get('n_sectors', 0)
            lines.append(f"**时域分析**: TESS 数据覆盖 {n_sectors} 个扇区。")
            lines.append("建议分析光变曲线的周期性特征。")
            lines.append("")
        
        lines.append("**建议**:")
        lines.append("1. 获取高分辨率光谱确定精确分类")
        lines.append("2. 分析光变曲线的周期特性")
        lines.append("3. 交叉匹配 X 射线源表")
        lines.append("4. 查阅相关文献了解该天体的研究历史")
        
        return '\n'.join(lines)
    
    def _save_results(self, name: str):
        """保存分析结果"""
        # 保存 JSON
        result_file = os.path.join(self.output_dir, f"{name}_integrated_analysis.json")
        
        # 移除不可序列化的数据
        result_clean = {}
        for k, v in self.result.items():
            if k != 'analysis_text':  # 单独保存文本
                try:
                    json.dumps({k: v})
                    result_clean[k] = v
                except:
                    result_clean[k] = str(v)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_clean, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"    ✓ 结果保存: {result_file}")
        
        # 保存 AI 分析文本
        if self.result.get('analysis_text'):
            text_file = os.path.join(self.output_dir, f"{name}_ai_analysis.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(self.result['analysis_text'])
            print(f"    ✓ AI分析保存: {text_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='整合天体分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python integrated_analysis.py --ra 274.0554 --dec 49.8679 --name "AM_Her"
  python integrated_analysis.py --ra 123.456 --dec 67.890 --name "MyStar" --no-ollama
        """
    )
    
    parser.add_argument('--ra', type=float, required=True, help='赤经（度）')
    parser.add_argument('--dec', type=float, required=True, help='赤纬（度）')
    parser.add_argument('--name', type=str, default=None, help='目标名称')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--no-ollama', action='store_true',
                       help='不使用 Ollama')
    parser.add_argument('--ollama-model', type=str, default='qwen3:8b',
                       help='Ollama 模型名称')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = IntegratedAstroAnalysis(
        output_dir=args.output_dir,
        use_ollama=not args.no_ollama,
        ollama_model=args.ollama_model
    )
    
    # 执行分析
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
    
    if result.get('analysis_text'):
        print(f"\n🤖 AI 分析摘要:")
        print(result['analysis_text'][:300] + "...")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
