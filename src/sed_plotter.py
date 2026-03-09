#!/usr/bin/env python3
"""
SED 画图工具（修复版）
=====================
参考 VSP_Photometry.py 实现，使用 VizieR SED 服务获取完整数据
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.ticker import MultipleLocator

# 添加 lib/vsp 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lib/vsp'))


class VizierSEDTool:
    """VizieR SED 数据获取工具"""
    
    @staticmethod
    def download_sed_vizier(ra: float, dec: float, radius: float = 1.5, 
                            output_dir: str = './output/data') -> str:
        """
        从 VizieR SED 服务下载 SED 数据
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            radius: 搜索半径（角秒）
            output_dir: 输出目录
            
        Returns:
            保存的 CSV 文件路径
        """
        import requests
        
        os.makedirs(output_dir, exist_ok=True)
        
        # VizieR SED URL
        url = f'http://vizier.u-strasbg.fr/viz-bin/sed?-c={ra}%20{dec}&-c.rs={radius}'
        
        temp_vot = os.path.join(output_dir, 'sed_vizier_download.vot')
        output_csv = os.path.join(output_dir, 'sed_vizier.csv')
        
        try:
            # 下载数据
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(temp_vot, 'wb') as f:
                    f.write(response.content)
                
                # 读取并处理
                try:
                    sed = Table.read(temp_vot)
                    
                    # 处理列名
                    if '_tab1_29' in sed.colnames:
                        sed.rename_column('_tab1_29', '_time(TCB)')
                    if '_tab1_30' in sed.colnames:
                        sed.remove_column('_tab1_30')
                    
                    # 保存为 CSV
                    sed.write(output_csv, format='ascii.csv', overwrite=True)
                    print(f'✓ SED 数据已保存: {output_csv}')
                    print(f'  数据点数: {len(sed)}')
                    
                    # 清理临时文件
                    if os.path.exists(temp_vot):
                        os.remove(temp_vot)
                    
                    return output_csv
                    
                except Exception as e:
                    print(f'✗ 处理 SED 数据失败: {e}')
                    return None
            else:
                print(f'✗ 下载失败: HTTP {response.status_code}')
                return None
                
        except Exception as e:
            print(f'✗ 下载 SED 数据失败: {e}')
            return None
    
    @staticmethod
    def screen_sed_data(tbl: Table) -> Table:
        """
        筛选 VizieR SED 数据，只保留有用的波段
        
        Args:
            tbl: SED 数据表
            
        Returns:
            筛选后的数据表
        """
        if len(tbl) == 0:
            return tbl
            
        # 筛选有用的波段（参考 VSP_Photometry.py）
        cut = (
            (tbl['_tabname'] == 'II/335/galex_ais') |  # GALEX
            (tbl['_tabname'] == 'I/355/gaiadr3') |      # Gaia DR3
            (tbl['_tabname'] == 'II/349/ps1') |         # Pan-STARRS
            ((tbl['_tabname'] == 'II/336/apass9') & (tbl['sed_filter'] == 'Johnson:B')) |
            ((tbl['_tabname'] == 'II/336/apass9') & (tbl['sed_filter'] == 'Johnson:V')) |
            (tbl['_tabname'] == 'II/328/allwise') |     # WISE
            (tbl['_tabname'] == 'II/363/unwise') |      # unWISE
            (tbl['_tabname'] == 'II/365/catwise') |     # CatWISE
            (tbl['_tabname'] == 'J/ApJS/249/18/table2') |  # 2MASS
            (tbl['_tabname'] == 'II/378/xmmom6s') |     # XMM
            (tbl['_tabname'] == 'II/368/sstsl2')        # Spitzer
        )
        
        return tbl[cut]
    
    @staticmethod
    def process_flux(data: Table) -> Table:
        """
        将 VizieR SED 数据转换为标准流量单位
        
        Args:
            data: SED 数据表
            
        Returns:
            处理后的数据表
        """
        if len(data) == 0:
            return None
        
        # 读取数据
        sed_freq = np.array(data['sed_freq'].tolist(), dtype=np.float64)
        sed_flux = np.array(data['sed_flux'].tolist(), dtype=np.float64)
        sed_eflux = np.array(data['sed_eflux'].tolist(), dtype=np.float64)
        filterdata = np.array(data['sed_filter'])
        tabnames = np.array(data['_tabname'])
        
        # 计算波长 (Å) 和流量
        # sed_freq 单位是 GHz
        # c = 2.998e9 Å/s per GHz = 2.998e18 Å/s per Hz
        # λ(Å) = 2.998e9 / ν(GHz)
        vizr_wave = 2.998e9 / sed_freq  # Å
        
        # 流量转换: sed_flux 单位是 Jy (1 Jy = 1e-23 erg/s/cm^2/Hz)
        # F_ν (erg/s/cm^2) = F_ν (Jy) * 1e-23
        vizr_flux = sed_flux * 1e-23  # erg/s/cm^2/Hz
        vizr_eflux = sed_eflux * 1e-23  # erg/s/cm^2/Hz
        
        # 创建输出表
        tbl = Table(
            names=['wave', 'flux', 'eflux', 'filter', 'source'],
            dtype=['f8', 'f8', 'f8', 'S30', 'S30'],
            units=[u.angstrom, u.erg / (u.cm ** 2 * u.s), u.erg / (u.cm ** 2 * u.s), None, None]
        )
        
        for i in range(len(data)):
            tbl.add_row([vizr_wave[i], vizr_flux[i], vizr_eflux[i], 
                        filterdata[i].decode() if isinstance(filterdata[i], bytes) else filterdata[i],
                        tabnames[i].decode() if isinstance(tabnames[i], bytes) else tabnames[i]])
        
        return tbl


class SEDPlotter:
    """SED 画图工具"""
    
    # 波段颜色和标记定义
    BAND_STYLES = {
        'XMM': {'color': 'tab:green', 'marker': 'o', 'size': 20, 'label': 'XMM'},
        'GALEX': {'color': 'tab:purple', 'marker': 's', 'size': 20, 'label': 'GALEX'},
        'GAIA': {'color': 'orange', 'marker': '^', 'size': 25, 'label': 'Gaia DR3'},
        '2MASS': {'color': 'tab:red', 'marker': 'D', 'size': 20, 'label': '2MASS'},
        'Spitzer': {'color': 'tab:red', 'marker': 'D', 'size': 20, 'label': 'IR'},
        'WISE': {'color': 'tab:brown', 'marker': 'v', 'size': 20, 'label': 'WISE'},
        'Pan-STARRS': {'color': 'tab:blue', 'marker': 'p', 'size': 20, 'label': 'Optical'},
        'APASS': {'color': 'tab:cyan', 'marker': 'h', 'size': 20, 'label': 'Optical'},
        'default': {'color': 'tab:blue', 'marker': 'o', 'size': 20, 'label': 'Other'}
    }
    
    @staticmethod
    def identify_band(filter_name: str, source: str) -> str:
        """
        识别波段类型
        
        Args:
            filter_name: 滤光片名称
            source: 数据源
            
        Returns:
            波段类型
        """
        filter_str = str(filter_name).upper()
        source_str = str(source).upper()
        
        if 'XMM' in filter_str or 'XMM' in source_str:
            return 'XMM'
        elif 'GALEX' in filter_str or 'GALEX' in source_str:
            return 'GALEX'
        elif 'GAIA' in filter_str or 'GAIA' in source_str or 'DR3' in source_str:
            return 'GAIA'
        elif '2MASS' in source_str:
            return '2MASS'
        elif 'SPITZER' in source_str:
            return 'Spitzer'
        elif 'WISE' in source_str or 'UNWISE' in source_str or 'CATWISE' in source_str:
            return 'WISE'
        elif 'PAN-STARRS' in source_str or 'PS1' in source_str:
            return 'Pan-STARRS'
        elif 'APASS' in source_str or 'JOHNSON' in filter_str:
            return 'APASS'
        else:
            return 'default'
    
    @staticmethod
    def plot_sed(data_file: str, output_path: str = None, title: str = "SED",
                 ifshow: bool = False) -> str:
        """
        绘制 SED 图（参考 VSP_Photometry.py 的 plot_sed 函数）
        
        Args:
            data_file: SED 数据文件路径（CSV格式）
            output_path: 输出图像路径
            title: 图标题
            ifshow: 是否显示图像
            
        Returns:
            输出图像路径
        """
        try:
            # 读取数据
            sed = Table.read(data_file)
            
            # 提取数据
            wave = sed['wave'] if 'wave' in sed.colnames else sed['vizr_wave']
            flux = sed['flux'] if 'flux' in sed.colnames else sed['vizr_flux']
            eflux = sed['eflux'] if 'eflux' in sed.colnames else sed['vizr_eflux']
            filters = sed['filter'] if 'filter' in sed.colnames else sed['vizr_filter']
            sources = sed['source'] if 'source' in sed.colnames else sed.get('vizier_tabname', ['']*len(wave))
            
            # 将 F_nu (erg/s/cm^2/Hz) 转换为 F_lambda (erg/s/cm^2/Å)
            # F_lambda = F_nu * c / lambda^2
            c = 2.998e18  # Å/s
            flux_lambda = flux * c / (wave**2)  # erg/s/cm^2/Å
            eflux_lambda = eflux * c / (wave**2)
            
            # 用于显示的流量 (erg/s/cm^2/Å)
            flux_display = flux_lambda
            eflux_display = eflux_lambda
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            ax.grid(True, linewidth=0.2, alpha=0.16, c='grey', zorder=0)
            
            # 按波段类型分组绘制
            plotted_bands = {}
            valid_flux = []  # 用于存储有效流量值
            
            for i in range(len(wave)):
                band_type = SEDPlotter.identify_band(str(filters[i]), str(sources[i]))
                style = SEDPlotter.BAND_STYLES.get(band_type, SEDPlotter.BAND_STYLES['default'])
                
                # 计算显示的流量值
                flux_val = flux_display[i] / wave[i]
                eflux_val = eflux_display[i] / wave[i]
                
                # 检查有效性
                if np.isfinite(flux_val) and flux_val > 0:
                    valid_flux.append(flux_val)
                    
                    # 绘制数据点
                    scatter = ax.scatter(
                        wave[i], 
                        flux_val,
                        marker=style['marker'],
                        edgecolors='k',
                        linewidths=0.3,
                        s=style['size'],
                        alpha=0.8,
                        c=style['color'],
                        zorder=4,
                        label=style['label'] if band_type not in plotted_bands else ""
                    )
                    
                    # 绘制误差条
                    if np.isfinite(eflux_val) and eflux_val > 0:
                        ax.errorbar(
                            wave[i],
                            flux_val,
                            yerr=eflux_val,
                            elinewidth=1,
                            capsize=0,
                            capthick=1,
                            alpha=0.8,
                            c=style['color'],
                            zorder=4
                        )
                
                plotted_bands[band_type] = True
            
            # 添加波长参考线
            wave_marks = [2000, 3000, 5000, 9000, 12000, 16000, 22000, 34000, 46000, 120000, 220000]
            wave_labels = ['200nm', '300nm', '500nm', '900nm', '1.2μm', '1.6μm', '2.2μm', 
                          '3.4μm', '4.6μm', '12μm', '22μm']
            
            # 获取有效的 y 轴范围
            if valid_flux:
                y_min = min(valid_flux) * 0.5
                y_max = max(valid_flux) * 2.0
            else:
                y_min, y_max = 1e-20, 1e-10
            
            for i, (wm, wl) in enumerate(zip(wave_marks, wave_labels)):
                ax.axvline(x=wm, linewidth=1, alpha=0.2, c='gray', zorder=1)
                ax.text(wm, y_max, wl, zorder=1, alpha=0.5, c='gray', fontsize=6, 
                       verticalalignment='bottom', horizontalalignment='center')
            
            # 设置坐标轴
            ax.set_ylabel(r'$F_\lambda$ ($\rm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)', 
                         fontsize=10)
            ax.set_xlabel(r'Wavelength ($\rm{\AA}$)', fontsize=10)
            ax.set_xlim(1000, 1000000)
            ax.set_ylim(y_min, y_max)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(labelsize=8)
            
            # 图例
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
            ax.set_title(f'VizieR SED - {title}', fontsize=11)
            
            plt.tight_layout()
            
            # 保存图像
            if output_path is None:
                output_path = data_file.replace('.csv', '.png')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if ifshow:
                plt.show()
            
            plt.close(fig)
            plt.close('all')
            
            print(f'✓ SED 图已保存: {output_path}')
            return output_path
            
        except Exception as e:
            print(f'✗ 绘制 SED 图失败: {e}')
            import traceback
            traceback.print_exc()
            return None


class SEDAnalyzer:
    """SED 分析器"""
    
    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        self.fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
    
    def analyze(self, ra: float, dec: float, name: str = "Target") -> dict:
        """
        完整的 SED 分析流程
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
            
        Returns:
            分析结果字典
        """
        result = {
            'success': False,
            'name': name,
            'ra': ra,
            'dec': dec
        }
        
        print(f"\n【SED 分析】{name}")
        print(f"  坐标: RA={ra:.6f}, DEC={dec:.6f}")
        
        # 1. 下载 SED 数据
        print("  正在从 VizieR 下载 SED 数据...")
        sed_file = VizierSEDTool.download_sed_vizier(
            ra, dec, 
            radius=1.5,
            output_dir=self.data_dir
        )
        
        if sed_file is None or not os.path.exists(sed_file):
            print("  ✗ SED 数据下载失败")
            return result
        
        # 2. 读取并筛选数据
        try:
            sed_raw = Table.read(sed_file)
            print(f"  ✓ 原始数据点: {len(sed_raw)}")
            
            sed_filtered = VizierSEDTool.screen_sed_data(sed_raw)
            print(f"  ✓ 筛选后数据点: {len(sed_filtered)}")
            
            if len(sed_filtered) == 0:
                print("  ✗ 筛选后无数据")
                return result
            
            # 3. 处理流量
            sed_processed = VizierSEDTool.process_flux(sed_filtered)
            
            if sed_processed is None or len(sed_processed) == 0:
                print("  ✗ 流量处理失败")
                return result
            
            # 保存处理后的数据
            processed_file = os.path.join(self.data_dir, f'{name}_sed_processed.csv')
            sed_processed.write(processed_file, format='ascii.csv', overwrite=True)
            print(f"  ✓ 处理后的数据: {processed_file}")
            
            # 4. 绘制 SED 图
            sed_plot = os.path.join(self.fig_dir, f'{name}_sed.png')
            SEDPlotter.plot_sed(processed_file, sed_plot, title=name)
            
            result['success'] = True
            result['sed_data'] = processed_file
            result['sed_plot'] = sed_plot
            result['n_points'] = len(sed_processed)
            
        except Exception as e:
            print(f"  ✗ SED 分析失败: {e}")
            import traceback
            traceback.print_exc()
        
        return result


# ==================== 便捷函数 ====================

def plot_sed_fixed(ra: float, dec: float, name: str = "Target", 
                   output_dir: str = './output') -> dict:
    """
    修复的 SED 画图函数
    
    Args:
        ra: 赤经（度）
        dec: 赤纬（度）
        name: 目标名称
        output_dir: 输出目录
        
    Returns:
        分析结果字典
    """
    analyzer = SEDAnalyzer(output_dir=output_dir)
    return analyzer.analyze(ra, dec, name)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SED 画图工具测试")
    print("=" * 60)
    
    # 测试 AM Her
    ra, dec = 274.0554, 49.8679
    name = "AM_Her"
    
    result = plot_sed_fixed(ra, dec, name)
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  成功: {result['success']}")
    print(f"  数据点: {result.get('n_points', 0)}")
    print(f"  SED图: {result.get('sed_plot', 'N/A')}")
    print("=" * 60)
