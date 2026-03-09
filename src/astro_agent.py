#!/usr/bin/env python3
"""
天文AI智能体 - 整合Qwen + RAG + Tools
====================================
Astronomical AI Agent with Qwen-VL, RAG Knowledge Base, and Tool Use
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

warnings.filterwarnings('ignore')

# ==================== 配置 ====================

@dataclass
class Config:
    """系统配置"""
    # 模型路径（优先使用Astro_qwen目录下的模型）
    QWEN_MODEL_PATH: str = "../Astro_qwen/models/qwen/Qwen-VL-Chat-Int4"
    
    # 备用路径
    QWEN_MODEL_PATH_ALT: str = "./models/Qwen-VL-Chat-Int4"
    
    # 数据文件
    EXTINCTION_FILE: str = "./data/sfd_ebv.fits"
    EXTINCTION_FILE_ALT: str = "../Astro_qwen/csfd_ebv.fits"
    
    KNOWLEDGE_BASE: str = "./data/astronomy_kb.json"
    
    # 输出目录
    OUTPUT_DIR: str = "./output"
    
    # 搜索半径
    ZTF_RADIUS: float = 3.0  # arcsec
    SDSS_RADIUS: float = 3.0
    VIZIER_RADIUS: float = 5.0

CONFIG = Config()

# 自动检测模型路径
if not os.path.exists(CONFIG.QWEN_MODEL_PATH):
    if os.path.exists(CONFIG.QWEN_MODEL_PATH_ALT):
        CONFIG.QWEN_MODEL_PATH = CONFIG.QWEN_MODEL_PATH_ALT
        print(f"使用备用模型路径: {CONFIG.QWEN_MODEL_PATH}")
    else:
        print(f"⚠ 警告: Qwen模型未找到")
        print(f"  主路径: {CONFIG.QWEN_MODEL_PATH}")
        print(f"  备用路径: {CONFIG.QWEN_MODEL_PATH_ALT}")
        print(f"  如需使用AI功能，请下载模型到上述路径之一")

# 自动检测消光文件路径
if not os.path.exists(CONFIG.EXTINCTION_FILE):
    if os.path.exists(CONFIG.EXTINCTION_FILE_ALT):
        CONFIG.EXTINCTION_FILE = CONFIG.EXTINCTION_FILE_ALT
        print(f"使用备用消光文件: {CONFIG.EXTINCTION_FILE}")
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)


# ==================== 1. RAG知识库 ====================

class AstronomyRAG:
    """
    天文知识库 (RAG - Retrieval Augmented Generation)
    包含变星、白矮星、双星等专业知识
    """
    
    def __init__(self):
        self.knowledge = self._init_knowledge()
    
    def _init_knowledge(self) -> Dict[str, str]:
        """初始化天文知识库"""
        return {
            "polar": """
Polar（激变变星/高偏振星）是强磁场的激变变星（CV）:
- 磁场强度: 10-100 MG（兆高斯）
- 周期: 1-2小时（轨道周期）
- 光变特征: 
  * 强偏振（线性+圆形）
  * X射线辐射
  * 准周期振荡（QPO）
  * 吸积柱辐射
- 光谱特征:
  * 强发射线（Hα, He II）
  *  cyclotron辐射（红外/光学）
- 著名例子: AM Her, EF Eri, VV Pup
""",
            "cataclysmic_variable": """
激变变星（Cataclysmic Variables, CVs）是白矮星吸积伴星物质的系统:
- 组成: 白矮星 + 红矮星/亚巨星
- 特征:
  * 轨道周期 < 1天
  * 不规则爆发（矮新星爆发）
  * 吸积盘（非磁系统）或吸积柱（磁系统）
- 子类型:
  * 矮新星（Dwarf Novae）
  * 新星（Novae）
  * 极星/Polar（强磁场）
  * 中介极星（Intermediate Polars）
""",
            "rr_lyrae": """
RR Lyrae变星是水平分支上的脉动变星:
- 周期: 0.2-1天
- 振幅: 0.3-2星等
- 周期-光度关系: M_v ≈ 0.6
- 分类:
  * RRab: 不对称光变，快速上升
  * RRc: 更对称，周期较短
- 用途: 标准烛光，测量距离
""",
            "eclipsing_binary": """
食双星（Eclipsing Binary）是轨道面与视线接近的双星:
- 大陵五型（Algol）: 主星不充满洛希瓣
- 渐台二型（Beta Lyrae）: 物质交换
- 大熊W型（W UMa）: 公共包层
- 光变特征:
  * 主极小: 较暗星遮挡较亮星
  * 次极小: 较亮星遮挡较暗星
""",
            "white_dwarf": """
白矮星（White Dwarf）是恒星演化的最终产物:
- 质量: 0.17-1.33 M_sun（钱德拉塞卡极限）
- 半径: ~地球大小
- 分类:
  * DA: H线主导（最常见）
  * DB: He I线
  * DC: 连续谱
  * DO: He II线
- 演化: 冷却轨迹，从热到冷
""",
        }
    
    def search(self, query: str) -> str:
        """搜索相关知识"""
        query_lower = query.lower()
        results = []
        
        for key, content in self.knowledge.items():
            if any(word in query_lower for word in key.split('_')):
                results.append(f"【{key}】\n{content}")
        
        return "\n\n".join(results) if results else "未找到相关知识。"


# ==================== 2. 天文工具集 ====================

class AstroTools:
    """
    天文数据查询工具集
    包含消光、测光、光变曲线等工具
    """
    
    def __init__(self):
        self._extinction_data = None
    
    def query_extinction(self, ra: float, dec: float) -> Dict:
        """查询银河消光"""
        try:
            from astropy.io import fits
            
            if self._extinction_data is None:
                with fits.open(CONFIG.EXTINCTION_FILE) as hdul:
                    data = hdul[1].data
                    self._extinction_data = np.vstack([row['T'] for row in data])
            
            # 简化坐标转换
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            gal = coord.galactic
            l, b = gal.l.deg, gal.b.deg
            
            ny, nx = self._extinction_data.shape
            x = int((l / 360.0) * nx)
            y = int(((90 - b) / 180.0) * ny)
            x = max(0, min(x, nx - 1))
            y = max(0, min(y, ny - 1))
            
            ebv = float(self._extinction_data[y, x])
            return {
                'success': True,
                'A_V': round(3.1 * ebv, 3),
                'E_B_V': round(ebv, 3)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def query_ztf(self, ra: float, dec: float, timeout: int = 30) -> Dict:
        """查询ZTF光变曲线"""
        try:
            import requests
            
            cache_file = os.path.join(CONFIG.OUTPUT_DIR, f"ztf_{ra:.4f}_{dec:.4f}.csv")
            
            # 检查缓存
            if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
                df = pd.read_csv(cache_file)
            else:
                radius_deg = CONFIG.ZTF_RADIUS / 3600.0
                url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?" \
                      f"POS=CIRCLE+{ra}+{dec}+{radius_deg}&COLLECTION=&FORMAT=csv"
                
                response = requests.get(url, timeout=timeout)
                if response.status_code != 200 or len(response.content) < 1000:
                    return {'success': False, 'error': 'No data'}
                
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                df = pd.read_csv(cache_file)
            
            # 简单分析
            df.columns = [c.lower() for c in df.columns]
            if 'catflags' in df.columns:
                df = df[df['catflags'] < 32768]
            
            return {
                'success': True,
                'n_points': len(df),
                'mean_mag': float(df['mag'].mean()) if len(df) > 0 else None,
                'time_span': float(df['mjd'].max() - df['mjd'].min()) if len(df) > 0 else 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def query_photometry(self, ra: float, dec: float) -> Dict:
        """查询多波段测光"""
        try:
            from astroquery.vizier import Vizier
            
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            v = Vizier(row_limit=3, columns=['**'])
            
            catalogs = {
                'Gaia': 'I/355/gaiadr3',
                '2MASS': 'II/246/out',
                'AllWISE': 'II/328/allwise',
            }
            
            results = {}
            for name, cat_id in catalogs.items():
                try:
                    result = v.query_region(coord, radius=CONFIG.VIZIER_RADIUS * u.arcsec, 
                                           catalog=cat_id)
                    if result and len(result) > 0:
                        results[name] = 'Found'
                    else:
                        results[name] = 'Not found'
                except:
                    results[name] = 'Error'
            
            return {'success': True, 'catalogs': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# ==================== 3. Qwen模型接口 ====================

class QwenInterface:
    """Qwen-VL模型接口"""
    
    def __init__(self):
        self.agent = None
        self._load_model()
    
    def _load_model(self):
        """加载Qwen模型"""
        try:
            # 尝试导入并加载
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from qwen_vl_loader import QwenVLAgent
            
            if os.path.exists(CONFIG.QWEN_MODEL_PATH):
                self.agent = QwenVLAgent(
                    model_name=CONFIG.QWEN_MODEL_PATH,
                    load_in_4bit=True
                )
                print("✓ Qwen模型加载成功")
            else:
                print(f"⚠ Qwen模型未找到: {CONFIG.QWEN_MODEL_PATH}")
        except Exception as e:
            print(f"⚠ Qwen加载失败: {e}")
    
    def analyze(self, prompt: str) -> str:
        """使用Qwen分析"""
        if self.agent:
            try:
                response, _ = self.agent.chat(prompt)
                return response
            except Exception as e:
                return f"Qwen分析错误: {e}"
        else:
            return "Qwen模型未加载，使用规则分析。"


# ==================== 4. 主智能体 ====================

class AstroAIAgent:
    """
    天文AI智能体
    整合: RAG知识库 + 工具调用 + Qwen大模型
    """
    
    def __init__(self):
        print("🚀 初始化天文AI智能体...")
        self.rag = AstronomyRAG()
        self.tools = AstroTools()
        self.qwen = QwenInterface()
        print("✓ 初始化完成\n")
    
    def analyze_target(self, ra: float, dec: float, name: str = None) -> Dict[str, Any]:
        """
        分析天体
        
        Args:
            ra: 赤经（度）
            dec: 赤纬（度）
            name: 目标名称
        
        Returns:
            分析结果字典
        """
        name = name or f"Target_{ra:.2f}_{dec:.2f}"
        
        print(f"🔭 分析天体: {name}")
        print(f"   坐标: RA={ra}°, DEC={dec}°")
        print("="*60)
        
        # 1. 查询消光
        print("\n[Step 1] 查询消光...")
        ext = self.tools.query_extinction(ra, dec)
        if ext['success']:
            print(f"   ✓ A_V = {ext['A_V']}, E(B-V) = {ext['E_B_V']}")
        
        # 2. 查询ZTF
        print("\n[Step 2] 查询ZTF...")
        ztf = self.tools.query_ztf(ra, dec)
        if ztf['success']:
            print(f"   ✓ {ztf['n_points']} 个数据点")
        
        # 3. 查询测光
        print("\n[Step 3] 查询测光...")
        phot = self.tools.query_photometry(ra, dec)
        if phot['success']:
            print(f"   ✓ 测光目录: {list(phot['catalogs'].keys())}")
        
        # 4. RAG知识检索
        print("\n[Step 4] 检索相关知识...")
        knowledge = self.rag.search("polar cataclysmic variable")
        print(f"   ✓ 检索到 {len(knowledge)} 条相关知识")
        
        # 5. Qwen综合分析
        print("\n[Step 5] AI综合分析...")
        prompt = self._build_analysis_prompt(name, ra, dec, ext, ztf, phot, knowledge)
        analysis = self.qwen.analyze(prompt)
        
        result = {
            'name': name,
            'ra': ra,
            'dec': dec,
            'timestamp': datetime.now().isoformat(),
            'extinction': ext,
            'ztf': ztf,
            'photometry': phot,
            'analysis': analysis
        }
        
        # 保存结果
        output_file = os.path.join(CONFIG.OUTPUT_DIR, f"{name}_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 分析完成!")
        print(f"   结果保存: {output_file}")
        
        return result
    
    def _build_analysis_prompt(self, name: str, ra: float, dec: float, 
                               ext: Dict, ztf: Dict, phot: Dict, knowledge: str) -> str:
        """构建分析提示词"""
        
        ext_str = f"A_V={ext.get('A_V', 'N/A')}, E(B-V)={ext.get('E_B_V', 'N/A')}" if ext.get('success') else "消光数据不可用"
        ztf_str = f"{ztf.get('n_points', 'N/A')} 个数据点" if ztf.get('success') else "ZTF数据不可用"
        
        return f"""你是一位专业的天体物理学家。请根据以下数据对天体进行详细分析。

## 目标信息
- 名称: {name}
- 坐标: RA={ra}°, DEC={dec}°

## 观测数据
- 消光: {ext_str}
- ZTF: {ztf_str}

## 相关知识
{knowledge}

## 请回答
1. 基于消光值，这个天体的距离大约是多少？
2. 可能是什么类型的天体？
3. 建议进行哪些后续观测？

请用中文回答，保持专业但易懂。"""


# ==================== 演示函数 ====================

def demo_am_her():
    """
    AM Her (Polar) 天体分析演示
    AM Her是著名的激变变星（Polar类型）
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           AM Her (Polar) 天体分析演示                            ║
╠══════════════════════════════════════════════════════════════════╣
║  AM Her是历史上第一个被发现的Polar（激变变星/高偏振星）          ║
║  坐标: RA = 274.0554°, DEC = +49.8679°                          ║
║  特征:                                                            ║
║    - 强磁场白矮星 (~10 MG)                                        ║
║    - 强X射线辐射                                                  ║
║    - 高偏振（线性和圆形）                                          ║
║    - 轨道周期约3小时                                              ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # 初始化智能体
    agent = AstroAIAgent()
    
    # 分析AM Her
    result = agent.analyze_target(
        ra=274.0554,
        dec=49.8679,
        name="AM_Her_Polar"
    )
    
    # 打印分析结果
    print("\n" + "="*60)
    print("📊 AI分析结果")
    print("="*60)
    print(result['analysis'])
    print("="*60)
    
    return result


def main():
    """主函数 - 支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='天文AI智能分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python astro_agent.py --ra 274.0554 --dec 49.8679 --name "AM_Her"
  python astro_agent.py --ra 123.456 --dec 67.890 --name "MyStar"
  python astro_agent.py --demo  # 运行AM Her演示
        """
    )
    
    parser.add_argument('--ra', type=float, help='赤经（度）')
    parser.add_argument('--dec', type=float, help='赤纬（度）')
    parser.add_argument('--name', type=str, default=None, help='目标名称')
    parser.add_argument('--demo', action='store_true', help='运行AM Her演示')
    parser.add_argument('--output-dir', type=str, default='./output', 
                       help='输出目录（默认: ./output）')
    
    args = parser.parse_args()
    
    # 更新输出目录
    global CONFIG
    CONFIG.OUTPUT_DIR = args.output_dir
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    if args.demo or (args.ra is None and args.dec is None):
        # 运行演示
        demo_am_her()
    else:
        if args.ra is None or args.dec is None:
            print("错误: 请同时提供 --ra 和 --dec 参数")
            return 1
        
        # 使用命令行参数分析
        agent = AstroAIAgent()
        result = agent.analyze_target(
            ra=args.ra,
            dec=args.dec,
            name=args.name
        )
        
        # 打印详细结果
        print("\n" + "="*70)
        print("📊 详细分析结果")
        print("="*70)
        print(f"\n目标: {result['name']}")
        print(f"坐标: RA={result['ra']}°, DEC={result['dec']}°")
        print(f"时间: {result['timestamp']}")
        
        # 消光
        ext = result.get('extinction', {})
        print(f"\n【消光】")
        if ext.get('success'):
            print(f"  ✓ A_V = {ext['A_V']} mag")
            print(f"  ✓ E(B-V) = {ext['E_B_V']} mag")
        else:
            print(f"  ✗ {ext.get('error', '查询失败')}")
        
        # ZTF
        ztf = result.get('ztf', {})
        print(f"\n【ZTF光变曲线】")
        if ztf.get('success'):
            print(f"  ✓ 数据点数: {ztf['n_points']}")
            if ztf.get('mean_mag'):
                print(f"  ✓ 平均星等: {ztf['mean_mag']:.2f}")
        else:
            print(f"  ✗ {ztf.get('error', '查询失败')}")
        
        # 测光
        phot = result.get('photometry', {})
        print(f"\n【多波段测光】")
        if phot.get('success'):
            print(f"  ✓ 匹配目录数: {phot.get('total_matches', 0)}")
            for cat, count in phot.get('catalogs', {}).items():
                try:
                    count_val = int(count) if count != -1 else 0
                    status = "✓" if count_val > 0 else "✗"
                except:
                    status = "?"
                    count_val = count
                print(f"    {status} {cat}: {count_val} 条记录")
        else:
            print(f"  ✗ {phot.get('error', '查询失败')}")
        
        # AI分析
        print(f"\n【AI分析】")
        print(result['analysis'])
        
        print("\n" + "="*70)
        print(f"✅ 完整结果已保存: {os.path.join(CONFIG.OUTPUT_DIR, result['name'] + '_analysis.json')}")
        print("="*70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
