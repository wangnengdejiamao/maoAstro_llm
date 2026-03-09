#!/usr/bin/env python3
"""
光变曲线下载与分析模块 (Lightcurve Download & Analysis Module)
=========================================================

整合所有光变数据源:
- TESS/Kepler/K2 (lightkurve + MAST)
- ZTF (VSP_ZTF)
- OGLE (astroquery)
- WISE 时序 (IRSA)

包含 pj4 (PDM-based) 周期计算算法

作者: AI Assistant
日期: 2026-03-04
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """周期计算结果"""
    period: float
    period_error: float
    significance: float
    method: str
    power: Optional[np.ndarray] = None
    periods: Optional[np.ndarray] = None
    best_phase: Optional[np.ndarray] = None
    best_mag: Optional[np.ndarray] = None


@dataclass
class LightcurveData:
    """光变曲线数据"""
    source: str  # 'TESS', 'ZTF', 'OGLE', 'WISE'
    band: str    # 'g', 'r', 'i', 'W1', 'W2', etc.
    time: np.ndarray
    mag: np.ndarray
    error: np.ndarray
    filename: Optional[str] = None
    metadata: Optional[Dict] = None


class PJ4PeriodFinder:
    """
    PJ4 周期计算算法 (Phase Dispersion Minimization)
    
    基于PDM的改进算法，使用4个相位bin进行优化
    """
    
    def __init__(self):
        self.n_bins = 4  # pj4 uses 4 bins
    
    def calculate(self, time: np.ndarray, mag: np.ndarray, 
                  error: Optional[np.ndarray] = None,
                  min_period: float = 0.01, max_period: float = 100,
                  n_periods: int = 10000) -> PeriodResult:
        """
        使用pj4算法计算周期
        
        Parameters:
        -----------
        time : np.ndarray
            时间数组 (JD/MJD)
        mag : np.ndarray
            星等数组
        error : np.ndarray, optional
            误差数组
        min_period : float
            最小周期（天）
        max_period : float
            最大周期（天）
        n_periods : int
            周期搜索步数
        
        Returns:
        --------
        PeriodResult : 周期计算结果
        """
        # 生成周期网格
        periods = np.linspace(min_period, max_period, n_periods)
        theta = np.zeros(n_periods)
        
        for i, period in enumerate(periods):
            theta[i] = self._pdm_theta(time, mag, period)
        
        # 找到最小theta（最小离散度）
        best_idx = np.argmin(theta)
        best_period = periods[best_idx]
        
        # 计算显著性
        significance = self._calculate_significance(theta, best_idx)
        
        # 估计周期误差
        period_error = self._estimate_error(periods, theta, best_idx)
        
        # 计算最佳相位
        phase, phase_mag = self._fold_lightcurve(time, mag, best_period)
        
        return PeriodResult(
            period=best_period,
            period_error=period_error,
            significance=significance,
            method='pj4',
            power=1 - theta,  # 转换为power（越大越好）
            periods=periods,
            best_phase=phase,
            best_mag=phase_mag
        )
    
    def _pdm_theta(self, time: np.ndarray, mag: np.ndarray, 
                   period: float) -> float:
        """
        计算PDM统计量 theta
        theta = sum(sigma_bin^2) / sigma_total^2
        """
        # 折叠相位
        phase = (time / period) % 1.0
        
        # 创建4个相位bin
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        
        total_var = np.var(mag)
        if total_var == 0:
            return 1.0
        
        bin_var_sum = 0
        n_bins_used = 0
        
        for i in range(self.n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if i == self.n_bins - 1:  # 最后一个bin包含1.0
                mask = (phase >= bin_edges[i]) & (phase <= bin_edges[i + 1])
            
            bin_mag = mag[mask]
            if len(bin_mag) > 1:
                bin_var_sum += np.var(bin_mag)
                n_bins_used += 1
        
        if n_bins_used == 0:
            return 1.0
        
        theta = bin_var_sum / (n_bins_used * total_var)
        return theta
    
    def _fold_lightcurve(self, time: np.ndarray, mag: np.ndarray, 
                         period: float) -> Tuple[np.ndarray, np.ndarray]:
        """折叠光变曲线"""
        phase = (time / period) % 1.0
        # 排序
        sort_idx = np.argsort(phase)
        return phase[sort_idx], mag[sort_idx]
    
    def _calculate_significance(self, theta: np.ndarray, 
                                best_idx: int) -> float:
        """计算周期显著性"""
        best_theta = theta[best_idx]
        mean_theta = np.mean(theta)
        std_theta = np.std(theta)
        
        if std_theta == 0:
            return 0.0
        
        # 显著性 = (mean - best) / std
        significance = (mean_theta - best_theta) / std_theta
        return significance
    
    def _estimate_error(self, periods: np.ndarray, theta: np.ndarray,
                        best_idx: int) -> float:
        """估计周期误差（基于FWHM）"""
        best_theta = theta[best_idx]
        best_period = periods[best_idx]
        
        # 找到半高宽点
        threshold = best_theta + 0.5 * (np.max(theta) - best_theta)
        
        # 找到峰值附近的区域
        above_threshold = theta < threshold
        if not np.any(above_threshold):
            return 0.01 * best_period
        
        # 找到连续区域
        indices = np.where(above_threshold)[0]
        if len(indices) < 2:
            return 0.01 * best_period
        
        # FWHM对应的周期范围
        fwhm_periods = periods[indices]
        error = (np.max(fwhm_periods) - np.min(fwhm_periods)) / 2
        
        return max(error, 0.001 * best_period)


class LombScarglePeriodFinder:
    """Lomb-Scargle周期计算"""
    
    def calculate(self, time: np.ndarray, mag: np.ndarray,
                  error: Optional[np.ndarray] = None,
                  min_period: float = 0.01, max_period: float = 100,
                  n_periods: int = 10000) -> PeriodResult:
        """使用Lomb-Scargle计算周期"""
        try:
            from astropy.timeseries import LombScargle
            
            # 转换为频率
            frequency = np.linspace(1/max_period, 1/min_period, n_periods)
            
            if error is not None:
                ls = LombScargle(time, mag, error)
            else:
                ls = LombScargle(time, mag)
            
            power = ls.power(frequency)
            
            best_idx = np.argmax(power)
            best_period = 1 / frequency[best_idx]
            
            # 计算显著性
            false_alarm_prob = ls.false_alarm_probability(power[best_idx])
            significance = -np.log10(false_alarm_prob + 1e-10)
            
            # 相位折叠
            phase, phase_mag = self._fold(time, mag, best_period)
            
            return PeriodResult(
                period=best_period,
                period_error=0.01 * best_period,
                significance=significance,
                method='lomb_scargle',
                power=power,
                periods=1/frequency,
                best_phase=phase,
                best_mag=phase_mag
            )
        except Exception as e:
            logger.error(f"Lomb-Scargle计算失败: {e}")
            return PeriodResult(
                period=0, period_error=0, significance=0,
                method='lomb_scargle'
            )
    
    def _fold(self, time, mag, period):
        phase = (time / period) % 1.0
        sort_idx = np.argsort(phase)
        return phase[sort_idx], mag[sort_idx]


class LightcurveDownloader:
    """光变曲线下载器"""
    
    def __init__(self, output_dir: str = "./lightcurves"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_tess_kepler(self, ra: float, dec: float, 
                              target_name: Optional[str] = None) -> List[LightcurveData]:
        """
        下载 TESS/Kepler/K2 光变曲线 (使用lightkurve)
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称
        
        Returns:
        --------
        List[LightcurveData] : 光变数据列表
        """
        results = []
        
        try:
            import lightkurve as lk
            
            # 搜索光变曲线
            search_result = lk.search_lightcurve(
                f"{ra} {dec}", 
                radius=2  # arcsec
            )
            
            if len(search_result) == 0:
                logger.warning("未找到TESS/Kepler/K2数据")
                return results
            
            logger.info(f"找到 {len(search_result)} 个光变曲线")
            
            for i, result in enumerate(search_result):
                try:
                    # 跳过某些pipeline
                    if result.author[0] in ['EVEREST', 'K2VARCAT']:
                        continue
                    
                    # 下载
                    lc = result.download()
                    if lc is None:
                        continue
                    
                    # 转换为标准格式
                    time = np.array(lc.time.value)
                    mag = np.array(lc.flux.value)
                    error = np.array(lc.flux_err.value) if hasattr(lc, 'flux_err') else np.zeros_like(mag)
                    
                    # 处理flux到mag的转换（如果数据是flux）
                    if np.mean(mag) > 100:  # 可能是flux
                        mag = -2.5 * np.log10(mag) + 20  # 简单转换
                        error = 1.0857 * error / np.power(10, (mag - 20) / -2.5)
                    
                    mission = result.mission[0]
                    author = result.author[0]
                    
                    # 保存文件
                    filename = f"{target_name or 'target'}_TESS_{i+1:02d}_{mission.replace(' ', '_')}_{author}.csv"
                    filepath = self.output_dir / filename
                    
                    df = pd.DataFrame({
                        'time': time,
                        'mag': mag,
                        'error': error
                    })
                    df.to_csv(filepath, index=False)
                    
                    lc_data = LightcurveData(
                        source='TESS' if 'TESS' in mission else mission,
                        band='TESS',
                        time=time,
                        mag=mag,
                        error=error,
                        filename=str(filepath),
                        metadata={
                            'mission': mission,
                            'author': author,
                            'exptime': result.exptime[0].value if hasattr(result.exptime[0], 'value') else result.exptime[0]
                        }
                    )
                    results.append(lc_data)
                    logger.info(f"✓ 下载成功: {filename}")
                    
                except Exception as e:
                    logger.warning(f"下载失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"lightkurve查询失败: {e}")
        
        return results
    
    def download_ztf(self, ra: float, dec: float,
                     target_name: Optional[str] = None) -> List[LightcurveData]:
        """
        下载 ZTF 光变曲线
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称
        
        Returns:
        --------
        List[LightcurveData] : 光变数据列表
        """
        results = []
        
        try:
            # 使用VSP_ZTF的方法
            sys.path.insert(0, '/mnt/c/Users/Administrator/Desktop/astro-ai-demo/lib/vsp')
            from VSP_ZTF import generate_ztf_url, download as download_ztf_file, read_lc
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # 下载ZTF数据
            filename = f"{target_name or 'target'}_ZTF.csv"
            filepath = self.output_dir / filename
            
            result = download_ztf_file(coord, str(filepath))
            
            if result and os.path.exists(filepath):
                # 读取数据
                tb_g, tb_r, tb_i = read_lc(str(filepath))
                
                bands = [
                    ('g', tb_g['hjd_g'].value, tb_g['mag_g'].value, tb_g['magerr_g'].value),
                    ('r', tb_r['hjd_r'].value, tb_r['mag_r'].value, tb_r['magerr_r'].value),
                    ('i', tb_i['hjd_i'].value, tb_i['mag_i'].value, tb_i['magerr_i'].value),
                ]
                
                for band, time, mag, error in bands:
                    if len(time) > 0:
                        # 保存单个波段
                        band_filename = f"{target_name or 'target'}_ZTF_{band}.csv"
                        band_filepath = self.output_dir / band_filename
                        
                        df = pd.DataFrame({
                            'time': time,
                            'mag': mag,
                            'error': error
                        })
                        df.to_csv(band_filepath, index=False)
                        
                        lc_data = LightcurveData(
                            source='ZTF',
                            band=band,
                            time=np.array(time),
                            mag=np.array(mag),
                            error=np.array(error),
                            filename=str(band_filepath),
                            metadata={'n_points': len(time)}
                        )
                        results.append(lc_data)
                        logger.info(f"✓ ZTF {band}波段: {len(time)} 个点")
            else:
                logger.warning("未找到ZTF数据")
            
        except Exception as e:
            logger.error(f"ZTF下载失败: {e}")
        
        return results
    
    def download_wise(self, ra: float, dec: float,
                      target_name: Optional[str] = None) -> List[LightcurveData]:
        """
        下载 WISE 时序数据 (NEOWISE)
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称
        
        Returns:
        --------
        List[LightcurveData] : 光变数据列表
        """
        results = []
        
        try:
            from astroquery.irsa import Irsa
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # 查询NEOWISE数据
            wise_table = Irsa.query_region(
                coord,
                catalog='neowiser_p1bs_psd',
                spatial='Cone',
                radius=2 * u.arcsec
            )
            
            if wise_table is None or len(wise_table) == 0:
                logger.warning("未找到WISE数据")
                return results
            
            logger.info(f"找到 {len(wise_table)} 条WISE记录")
            
            # 处理W1和W2波段
            for band, band_name in [(1, 'W1'), (2, 'W2')]:
                mask = wise_table['w1mpro'] if band == 1 else wise_table['w2mpro']
                valid = ~np.isnan(mask)
                
                if np.sum(valid) > 0:
                    time = np.array(wise_table['mjd'][valid])
                    mag = np.array(wise_table[f'w{band}mpro'][valid])
                    error = np.array(wise_table[f'w{band}sigmpro'][valid])
                    
                    # 保存
                    filename = f"{target_name or 'target'}_WISE_{band_name}.csv"
                    filepath = self.output_dir / filename
                    
                    df = pd.DataFrame({
                        'time': time,
                        'mag': mag,
                        'error': error
                    })
                    df.to_csv(filepath, index=False)
                    
                    lc_data = LightcurveData(
                        source='WISE',
                        band=band_name,
                        time=time,
                        mag=mag,
                        error=error,
                        filename=str(filepath),
                        metadata={'n_points': len(time)}
                    )
                    results.append(lc_data)
                    logger.info(f"✓ WISE {band_name}: {len(time)} 个点")
            
        except Exception as e:
            logger.error(f"WISE下载失败: {e}")
        
        return results
    
    def download_ogle(self, ra: float, dec: float,
                      target_name: Optional[str] = None) -> List[LightcurveData]:
        """
        下载 OGLE 光变数据
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称
        
        Returns:
        --------
        List[LightcurveData] : 光变数据列表
        """
        results = []
        
        try:
            from astroquery.ogle import Ogle
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            
            # 查询OGLE（主要是消光数据，但也可以用于光变）
            ogle_table = Ogle.query_region(
                coord=coord,
                width=0.1 * u.deg
            )
            
            if ogle_table is None or len(ogle_table) == 0:
                logger.warning("未找到OGLE数据")
                return results
            
            logger.info(f"找到 {len(ogle_table)} 条OGLE记录")
            
            # OGLE数据格式特殊，这里主要是恒星参数
            # 保存查询结果
            filename = f"{target_name or 'target'}_OGLE_catalog.csv"
            filepath = self.output_dir / filename
            ogle_table.write(filepath, format='csv', overwrite=True)
            
            logger.info(f"✓ OGLE星表已保存")
            
        except Exception as e:
            logger.error(f"OGLE下载失败: {e}")
        
        return results
    
    def download_all(self, ra: float, dec: float, 
                     target_name: Optional[str] = None) -> Dict[str, List[LightcurveData]]:
        """
        下载所有可用光变数据
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        target_name : str, optional
            目标名称
        
        Returns:
        --------
        Dict[str, List[LightcurveData]] : 各类光变数据
        """
        logger.info(f"开始下载光变数据: RA={ra}, DEC={dec}")
        
        results = {
            'TESS': [],
            'ZTF': [],
            'WISE': [],
            'OGLE': [],
        }
        
        # TESS/Kepler/K2
        logger.info("="*50)
        logger.info("下载 TESS/Kepler/K2 数据...")
        results['TESS'] = self.download_tess_kepler(ra, dec, target_name)
        
        # ZTF
        logger.info("="*50)
        logger.info("下载 ZTF 数据...")
        results['ZTF'] = self.download_ztf(ra, dec, target_name)
        
        # WISE
        logger.info("="*50)
        logger.info("下载 WISE 数据...")
        results['WISE'] = self.download_wise(ra, dec, target_name)
        
        # OGLE
        logger.info("="*50)
        logger.info("查询 OGLE 数据...")
        results['OGLE'] = self.download_ogle(ra, dec, target_name)
        
        logger.info("="*50)
        total = sum(len(v) for v in results.values())
        logger.info(f"总共下载 {total} 条光变曲线")
        
        return results


class LightcurveAnalyzer:
    """光变曲线分析器"""
    
    def __init__(self, output_dir: str = "./lightcurves"):
        self.output_dir = Path(output_dir)
        self.pj4 = PJ4PeriodFinder()
        self.ls = LombScarglePeriodFinder()
    
    def analyze(self, lc_data: LightcurveData, 
                min_period: float = 0.01,
                max_period: float = 100) -> Dict:
        """
        分析单条光变曲线
        
        Parameters:
        -----------
        lc_data : LightcurveData
            光变数据
        min_period, max_period : float
            周期搜索范围（天）
        
        Returns:
        --------
        Dict : 分析结果
        """
        results = {
            'source': lc_data.source,
            'band': lc_data.band,
            'n_points': len(lc_data.time),
            'time_range': (float(np.min(lc_data.time)), float(np.max(lc_data.time))),
            'mag_range': (float(np.min(lc_data.mag)), float(np.max(lc_data.mag))),
            'periods': {},
            'plots': {}
        }
        
        if len(lc_data.time) < 10:
            logger.warning(f"数据点太少 ({len(lc_data.time)}), 跳过周期分析")
            return results
        
        # PJ4 周期
        logger.info(f"使用pj4算法计算周期 ({lc_data.source} {lc_data.band})...")
        pj4_result = self.pj4.calculate(
            lc_data.time, lc_data.mag, lc_data.error,
            min_period=min_period, max_period=max_period
        )
        results['periods']['pj4'] = {
            'period': float(pj4_result.period),
            'period_error': float(pj4_result.period_error),
            'significance': float(pj4_result.significance),
        }
        
        # Lomb-Scargle 周期
        logger.info(f"使用Lomb-Scargle算法计算周期...")
        ls_result = self.ls.calculate(
            lc_data.time, lc_data.mag, lc_data.error,
            min_period=min_period, max_period=max_period
        )
        results['periods']['lomb_scargle'] = {
            'period': float(ls_result.period),
            'period_error': float(ls_result.period_error),
            'significance': float(ls_result.significance),
        }
        
        # 绘制周期图
        self._plot_periodogram(lc_data, pj4_result, ls_result)
        
        # 绘制相位图
        self._plot_phase(lc_data, pj4_result, ls_result)
        
        return results
    
    def _plot_periodogram(self, lc_data: LightcurveData,
                          pj4_result: PeriodResult,
                          ls_result: PeriodResult):
        """绘制周期图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            
            # PJ4 periodogram
            if pj4_result.periods is not None and pj4_result.power is not None:
                axes[0].plot(pj4_result.periods, pj4_result.power, 'b-', linewidth=0.8)
                axes[0].axvline(pj4_result.period, color='r', linestyle='--', 
                               label=f"P={pj4_result.period:.4f} d")
                axes[0].set_xlabel('Period (days)')
                axes[0].set_ylabel('Power (1-theta)')
                axes[0].set_title(f'PJ4 Periodogram ({lc_data.source} {lc_data.band})')
                axes[0].legend()
                axes[0].set_xlim(0, min(10, np.max(pj4_result.periods)))
            
            # Lomb-Scargle periodogram
            if ls_result.periods is not None and ls_result.power is not None:
                axes[1].plot(ls_result.periods, ls_result.power, 'g-', linewidth=0.8)
                axes[1].axvline(ls_result.period, color='r', linestyle='--',
                               label=f"P={ls_result.period:.4f} d")
                axes[1].set_xlabel('Period (days)')
                axes[1].set_ylabel('Power')
                axes[1].set_title(f'Lomb-Scargle Periodogram ({lc_data.source} {lc_data.band})')
                axes[1].legend()
                axes[1].set_xlim(0, min(10, np.max(ls_result.periods)))
            
            plt.tight_layout()
            
            filename = f"{lc_data.source}_{lc_data.band}_periodogram.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 周期图已保存: {filename}")
            
        except Exception as e:
            logger.warning(f"绘制周期图失败: {e}")
    
    def _plot_phase(self, lc_data: LightcurveData,
                    pj4_result: PeriodResult,
                    ls_result: PeriodResult):
        """绘制相位图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # PJ4 相位图
            if pj4_result.best_phase is not None and pj4_result.best_mag is not None:
                axes[0].scatter(pj4_result.best_phase, pj4_result.best_mag, 
                              c='b', s=10, alpha=0.6)
                axes[0].set_xlabel('Phase')
                axes[0].set_ylabel('Magnitude')
                axes[0].set_title(f'PJ4 Phase (P={pj4_result.period:.4f} d)')
                axes[0].invert_yaxis()
            
            # LS 相位图
            if ls_result.best_phase is not None and ls_result.best_mag is not None:
                axes[1].scatter(ls_result.best_phase, ls_result.best_mag,
                              c='g', s=10, alpha=0.6)
                axes[1].set_xlabel('Phase')
                axes[1].set_ylabel('Magnitude')
                axes[1].set_title(f'Lomb-Scargle Phase (P={ls_result.period:.4f} d)')
                axes[1].invert_yaxis()
            
            plt.tight_layout()
            
            filename = f"{lc_data.source}_{lc_data.band}_phase.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ 相位图已保存: {filename}")
            
        except Exception as e:
            logger.warning(f"绘制相位图失败: {e}")
    
    def analyze_all(self, lightcurves: Dict[str, List[LightcurveData]]) -> Dict:
        """分析所有光变数据"""
        all_results = {}
        
        for source, lc_list in lightcurves.items():
            for lc in lc_list:
                key = f"{lc.source}_{lc.band}"
                logger.info(f"\n分析 {key}...")
                result = self.analyze(lc)
                all_results[key] = result
        
        return all_results


def main():
    """主函数 - 测试 WD_OPC"""
    
    # WD_OPC 参数
    ra = 281.0805527
    dec = -17.89272442
    target_name = "WD_OPC"
    
    # 创建输出目录
    output_dir = f"/mnt/c/Users/Administrator/Desktop/astro-ai-demo/output/{target_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*70)
    logger.info(f"光变曲线分析 - {target_name}")
    logger.info(f"坐标: RA={ra}, DEC={dec}")
    logger.info("="*70)
    
    # 下载光变数据
    downloader = LightcurveDownloader(output_dir)
    lc_data = downloader.download_all(ra, dec, target_name)
    
    # 分析光变数据
    analyzer = LightcurveAnalyzer(output_dir)
    results = analyzer.analyze_all(lc_data)
    
    # 保存结果
    result_file = os.path.join(output_dir, "period_analysis_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("="*70)
    logger.info(f"分析完成！结果保存在: {output_dir}")
    logger.info("="*70)
    
    # 打印汇总
    print("\n周期分析汇总:")
    print("-"*70)
    for key, result in results.items():
        if 'periods' in result and result['periods']:
            print(f"\n{key}:")
            for method, period_info in result['periods'].items():
                print(f"  {method}: P = {period_info['period']:.6f} ± {period_info['period_error']:.6f} d "
                      f"(S/N = {period_info['significance']:.2f})")


if __name__ == "__main__":
    main()
