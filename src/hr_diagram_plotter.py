#!/usr/bin/env python3
"""
赫罗图画图工具（修复版）
========================
使用 LAMOST DR10 数据作为背景，参考 VSP_GAIA.py 实现
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.ticker import MultipleLocator

# 添加 lib/vsp 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lib/vsp'))


class HRDiagramPlotter:
    """赫罗图画图工具"""
    
    # LAMOST CMD 背景数据路径（参考 VSP_GAIA.py）
    LAMOST_CMD_FILE = './lib/LM_DR10_FOR_CMD_every30.fits'
    
    # 消光系数（参考 VSP_GAIA.py）
    R_G = 2.364   # G 波段消光系数
    R_BP = 2.998  # BP 波段消光系数
    R_RP = 1.737  # RP 波段消光系数
    
    _background_data = None  # 缓存背景数据
    
    @classmethod
    def load_background(cls):
        """加载 LAMOST 背景数据"""
        if cls._background_data is not None:
            return cls._background_data
        
        try:
            from astropy.io import fits
            
            if not os.path.exists(cls.LAMOST_CMD_FILE):
                print(f"✗ 背景数据文件不存在: {cls.LAMOST_CMD_FILE}")
                return None
            
            print(f"  加载 LAMOST 背景数据...")
            with fits.open(cls.LAMOST_CMD_FILE) as hdul:
                data = hdul[1].data
                bprp0 = data['BPRP0_green19']  # 改正消光后的 BP-RP
                mg0 = data['MG_rgeo_green19']   # 改正消光后的绝对星等
                
                # 过滤有效数据
                mask = np.isfinite(bprp0) & np.isfinite(mg0) & (mg0 > -5) & (mg0 < 16) & (bprp0 > -1) & (bprp0 < 5)
                
                cls._background_data = {
                    'bprp0': bprp0[mask],
                    'mg0': mg0[mask]
                }
                print(f"  ✓ 背景星数: {len(cls._background_data['bprp0'])}")
                return cls._background_data
                
        except Exception as e:
            print(f"✗ 加载背景数据失败: {e}")
            return None
    
    @staticmethod
    def query_gaia_dr3(ra: float, dec: float, radius: float = 2.0):
        """
        查询 Gaia DR3 数据
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            radius: 搜索半径（角秒）
            
        Returns:
            Gaia 数据表或 None
        """
        try:
            from astroquery.gaia import Gaia
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
            
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            
            # 查询中心星
            job = Gaia.cone_search_async(coord, radius=radius * u.arcsec)
            result = job.get_results()
            
            if len(result) == 0:
                print("  Gaia DR3 无匹配")
                return None
            
            return result
            
        except Exception as e:
            print(f"  Gaia 查询失败: {e}")
            return None
    
    @staticmethod
    def calculate_absolute_magnitude(g_mag, bp_mag, rp_mag, ebv, distance):
        """
        计算绝对星等和色指数
        
        Args:
            g_mag: G 波段星等
            bp_mag: BP 波段星等
            rp_mag: RP 波段星等
            ebv: E(B-V) 消光值
            distance: 距离（秒差距）
            
        Returns:
            (MG, BPRP0) 绝对星等和改正色指数
        """
        # 消光改正
        ag = HRDiagramPlotter.R_G * ebv
        abp = HRDiagramPlotter.R_BP * ebv
        arp = HRDiagramPlotter.R_RP * ebv
        
        g0 = g_mag - ag
        bp0 = bp_mag - abp
        rp0 = rp_mag - arp
        
        # 计算绝对星等
        mg = g0 + 5 - 5 * np.log10(distance)
        bprp0 = bp0 - rp0
        
        return mg, bprp0
    
    @staticmethod
    def plot_hr_diagram(ra: float, dec: float, name: str = "Target",
                        output_path: str = None, extinction: dict = None) -> str:
        """
        绘制赫罗图
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
            output_path: 输出路径
            extinction: 消光数据字典 {'A_V': x, 'E_B_V': y}
            
        Returns:
            输出图像路径
        """
        print(f"\n【赫罗图绘制】{name}")
        
        try:
            # 加载背景数据
            bg_data = HRDiagramPlotter.load_background()
            
            # 查询 Gaia 数据
            gaia_data = HRDiagramPlotter.query_gaia_dr3(ra, dec, radius=2.0)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            ax.minorticks_on()
            
            # 绘制背景星
            if bg_data is not None:
                ax.scatter(
                    bg_data['bprp0'], bg_data['mg0'],
                    marker='o', s=1, alpha=0.4, c='gray',
                    edgecolors='None', rasterized=True, zorder=0,
                    label='LAMOST DR10'
                )
            
            # 绘制目标星
            target_plotted = False
            if gaia_data is not None and len(gaia_data) > 0:
                star = gaia_data[0]
                
                # 提取数据
                try:
                    g_mag = float(star['phot_g_mean_mag'])
                    bp_mag = float(star['phot_bp_mean_mag']) if 'phot_bp_mean_mag' in star.colnames else None
                    rp_mag = float(star['phot_rp_mean_mag']) if 'phot_rp_mean_mag' in star.colnames else None
                    parallax = float(star['parallax']) if 'parallax' in star.colnames else None
                    
                    if parallax and parallax > 0:
                        distance = 1000.0 / parallax
                        
                        # 获取消光值
                        ebv = 0.0
                        if extinction and extinction.get('success'):
                            ebv = extinction.get('E_B_V', 0)
                        
                        # 计算绝对星等
                        if bp_mag is not None and rp_mag is not None:
                            mg, bprp0 = HRDiagramPlotter.calculate_absolute_magnitude(
                                g_mag, bp_mag, rp_mag, ebv, distance
                            )
                            
                            ax.scatter(
                                [bprp0], [mg],
                                marker='*', s=300, alpha=0.99,
                                edgecolors='k', facecolor='orange',
                                linewidths=0.5, rasterized=True, zorder=3,
                                label=f'{name}'
                            )
                            
                            # 添加标注
                            ax.annotate(
                                name,
                                (bprp0, mg),
                                xytext=(10, 10),
                                textcoords='offset points',
                                fontsize=9,
                                color='red',
                                fontweight='bold'
                            )
                            
                            target_plotted = True
                            print(f"  ✓ 目标星位置: BPRP0={bprp0:.2f}, MG={mg:.2f}")
                            
                            # 添加信息文本
                            info_text = f"G={g_mag:.2f}, dist={distance:.0f}pc, EBV={ebv:.2f}"
                            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                                   fontsize=8, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    print(f"  处理 Gaia 数据失败: {e}")
            
            if not target_plotted:
                print("  ⚠ 无法绘制目标星（无有效 Gaia 数据）")
            
            # 设置坐标轴
            ax.set_xlim(-1, 4.4)
            ax.set_ylim(16, -5)
            ax.set_xlabel(r'$(BP-RP)_0$', fontsize=12)
            ax.set_ylabel(r'$M_{\rm{G}}$', fontsize=12)
            ax.set_title(f'Hertzsprung-Russell Diagram - {name}', fontsize=13)
            
            # 刻度和网格
            ax.tick_params(axis='both', which='major', direction='in', width=1, length=5)
            ax.tick_params(axis='both', which='minor', direction='in', width=1, length=3)
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            
            ax.legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            
            # 保存图像
            if output_path is None:
                output_path = f'./output/figures/{name}_hr_diagram.png'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.close('all')
            
            print(f"  ✓ 赫罗图已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ 绘制赫罗图失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class TPFPlotter:
    """TESS TPF (Target Pixel File) 画图工具"""
    
    @staticmethod
    def plot_tpf(ra: float, dec: float, name: str = "Target", 
                 output_path: str = None, sector: int = None) -> str:
        """
        使用 tpfplotter 绘制 TESS TPF 图
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
            output_path: 输出路径
            sector: TESS 扇区号（可选）
            
        Returns:
            输出图像路径
        """
        print(f"\n【TPF 绘图】{name}")
        
        try:
            import tpfplotter
            from tpfplotter import tpfplotter as tp
            
            # 创建临时目录
            temp_dir = './output/temp'
            os.makedirs(temp_dir, exist_ok=True)
            
            # 构建 tpfplotter 参数
            if output_path is None:
                output_path = f'./output/figures/{name}_TPF.png'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 调用 tpfplotter
            # 注意：tpfplotter 通常通过命令行调用，这里尝试使用其 API
            try:
                # 尝试导入并调用 tpfplotter 的函数
                import lightkurve as lk
                
                # 搜索 TPF
                search_result = lk.search_targetpixelfile(
                    f"{ra} {dec}", 
                    radius=60 * u.arcsec
                )
                
                if len(search_result) == 0:
                    print("  ✗ 未找到 TPF 数据")
                    return None
                
                # 选择扇区
                if sector is not None:
                    mask = [f"tess-s{sector:04d}" in str(mission).lower() 
                           for mission in search_result.mission]
                    if any(mask):
                        idx = mask.index(True)
                    else:
                        idx = 0
                else:
                    idx = 0
                
                # 下载 TPF
                tpf = search_result[idx].download()
                
                if tpf is None:
                    print("  ✗ TPF 下载失败")
                    return None
                
                # 绘制 TPF
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 绘制原始图像
                tpf.plot(ax=axes[0], aperture_mask='pipeline', show_colorbar=True)
                axes[0].set_title(f'{name} - TPF Pipeline Mask')
                
                # 绘制中值图像
                tpf.plot(ax=axes[1], show_colorbar=True, aperture_mask='threshold')
                axes[1].set_title('Threshold Mask')
                
                # 绘制光变曲线
                lc = tpf.to_lightcurve()
                lc.plot(ax=axes[2])
                axes[2].set_title('SAP Light Curve')
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ TPF 图已保存: {output_path}")
                return output_path
                
            except ImportError:
                print("  ✗ tpfplotter 或 lightkurve 未安装")
                return None
                
        except Exception as e:
            print(f"✗ TPF 绘图失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def plot_tpf_simple(ra: float, dec: float, name: str = "Target",
                        output_path: str = None) -> str:
        """
        简化版 TPF 绘图（不依赖 tpfplotter）
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
            output_path: 输出路径
            
        Returns:
            输出图像路径
        """
        print(f"\n【TPF 绘图（简化版）】{name}")
        
        try:
            import lightkurve as lk
            import numpy as np
            
            # 搜索 TPF
            search_result = lk.search_targetpixelfile(f"{ra} {dec}", radius=60 * u.arcsec)
            
            if len(search_result) == 0:
                print("  ✗ 未找到 TPF 数据")
                return None
            
            print(f"  找到 {len(search_result)} 个 TPF 数据")
            
            # 下载第一个
            tpf = search_result[0].download()
            
            if tpf is None:
                print("  ✗ TPF 下载失败")
                return None
            
            # 获取TPF数据
            flux = tpf.flux.value
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 计算中值图像
            if len(flux.shape) == 3:
                median_flux = np.nanmedian(flux, axis=0)
            else:
                median_flux = flux
            
            # 绘制中值图像
            im1 = axes[0, 0].imshow(median_flux, origin='lower', cmap='viridis')
            axes[0, 0].set_title(f'{name} - TPF Median Flux')
            axes[0, 0].set_xlabel('Pixel Column')
            axes[0, 0].set_ylabel('Pixel Row')
            plt.colorbar(im1, ax=axes[0, 0], label='Flux')
            
            # 绘制第一帧
            if len(flux.shape) == 3:
                im2 = axes[0, 1].imshow(flux[0], origin='lower', cmap='viridis')
                axes[0, 1].set_title('First Frame')
            else:
                im2 = axes[0, 1].imshow(flux, origin='lower', cmap='viridis')
                axes[0, 1].set_title('Flux')
            axes[0, 1].set_xlabel('Pixel Column')
            axes[0, 1].set_ylabel('Pixel Row')
            plt.colorbar(im2, ax=axes[0, 1], label='Flux')
            
            # 绘制标准差图像
            if len(flux.shape) == 3:
                std_flux = np.nanstd(flux, axis=0)
                im3 = axes[1, 0].imshow(std_flux, origin='lower', cmap='hot')
                axes[1, 0].set_title('Flux Standard Deviation')
                plt.colorbar(im3, ax=axes[1, 0], label='Std Dev')
            else:
                axes[1, 0].text(0.5, 0.5, 'No time series data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_xlabel('Pixel Column')
            axes[1, 0].set_ylabel('Pixel Row')
            
            # 光变曲线
            try:
                lc = tpf.to_lightcurve()
                time = lc.time.value
                flux_lc = lc.flux.value
                axes[1, 1].plot(time, flux_lc, 'k.', markersize=2, alpha=0.5)
                axes[1, 1].set_xlabel('Time (BTJD)')
                axes[1, 1].set_ylabel('Flux (e-/s)')
                axes[1, 1].set_title('SAP Light Curve')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Light curve error:\n{str(e)}', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.suptitle(f'TESS TPF Analysis - {name}\n(RA={ra:.4f}, DEC={dec:.4f}, Sector={tpf.sector})', 
                        fontsize=14)
            plt.tight_layout()
            
            # 保存
            if output_path is None:
                output_path = f'./output/figures/{name}_TPF.png'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ TPF 图已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ TPF 绘图失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== 便捷函数 ====================

def plot_hr_diagram_fixed(ra: float, dec: float, name: str = "Target",
                          output_path: str = None, extinction: dict = None) -> str:
    """
    修复的赫罗图绘制函数
    
    Args:
        ra: 赤经（度）
        dec: 赤纬（度）
        name: 目标名称
        output_path: 输出路径
        extinction: 消光数据
        
    Returns:
        输出图像路径
    """
    return HRDiagramPlotter.plot_hr_diagram(ra, dec, name, output_path, extinction)


def plot_tpf(ra: float, dec: float, name: str = "Target",
             output_path: str = None) -> str:
    """
    TPF 绘图函数
    
    Args:
        ra: 赤经（度）
        dec: 赤纬（度）
        name: 目标名称
        output_path: 输出路径
        
    Returns:
        输出图像路径
    """
    return TPFPlotter.plot_tpf_simple(ra, dec, name, output_path)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("赫罗图和 TPF 绘图工具测试")
    print("=" * 60)
    
    # 测试 AM Her
    ra, dec = 274.0554, 49.8679
    name = "AM_Her"
    
    # 测试赫罗图
    print("\n1. 测试赫罗图...")
    extinction = {'success': True, 'A_V': 1.12, 'E_B_V': 0.36}
    hr_path = plot_hr_diagram_fixed(ra, dec, name, extinction=extinction)
    
    if hr_path:
        print(f"   ✓ 赫罗图: {hr_path}")
    
    # 测试 TPF
    print("\n2. 测试 TPF 绘图...")
    tpf_path = plot_tpf(ra, dec, name)
    
    if tpf_path:
        print(f"   ✓ TPF图: {tpf_path}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
