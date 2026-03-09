# 统一天文数据查询接口 - 使用指南

## 项目概述

基于 `astroquery` 的多源数据整合查询工具，**只需要提供 RA 和 DEC**，即可自动查询多个天文数据库。

### 支持的数据源

| 数据源 | 功能 | 数据内容 |
|--------|------|----------|
| **SIMBAD** | 天体识别与分类 | 名称、类型、光谱型、测光、视差 |
| **Gaia DR3** | 天体测量 | 视差、自行、距离、温度、多波段测光 |
| **VizieR** | 多星表查询 | 2MASS、WISE、SDSS、变星表等 |
| **SDSS** | 光谱与测光 | 光谱数据、红移、分类 |
| **SkyView** | 图像数据 | DSS、2MASS、WISE 图像 |
| **NED** | 河外天体 | 星系、类星体数据 |
| **HEASARC** | 高能数据 | X射线、伽马射线源 |

---

## 快速开始

### 安装依赖

```bash
pip install astroquery astropy numpy pandas
```

### 最简单用法

```python
from unified_astro_query import AstroSourceQuerier

# 1. 初始化
querier = AstroSourceQuerier()

# 2. 提供坐标查询
ra, dec = 101.2875, -16.7161  # Sirius (天狼星)
results = querier.query_all(ra, dec)

# 3. 查看结果
simbad = results['simbad']
if simbad.success and simbad.data.get('matched'):
    print(f"名称: {simbad.data['main_id']}")
    print(f"类型: {simbad.data['otype']}")
    print(f"光谱型: {simbad.data['sp_type']}")
```

---

## 详细功能

### 1. 初始化查询器

```python
from unified_astro_query import AstroSourceQuerier

# 创建查询器实例
querier = AstroSourceQuerier(cache_dir="./cache")

# 查看可用的服务
print(querier.services)
# {'simbad': True, 'gaia': True, 'vizier': True, ...}
```

### 2. 全源查询

```python
# 查询所有可用数据源
results = querier.query_all(
    ra=101.2875,      # 赤经 (度)
    dec=-16.7161,     # 赤纬 (度)
    radius=10.0       # 搜索半径 (角秒)
)

# 返回包含各数据源结果的字典
# results = {
#     'simbad': QueryResult(...),
#     'gaia': QueryResult(...),
#     'vizier': QueryResult(...),
#     'sdss': QueryResult(...),
# }
```

### 3. 单独查询某个数据源

```python
# 只查询 SIMBAD
simbad_result = querier.query_simbad(ra, dec, radius=5.0)

# 只查询 Gaia
gaia_result = querier.query_gaia(ra, dec, radius=5.0)

# 只查询 VizieR
vizier_result = querier.query_vizier(ra, dec, radius=5.0)

# 只查询 SDSS
sdss_result = querier.query_sdss(ra, dec, radius=5.0)
```

### 4. 获取详细数据

#### SIMBAD 数据

```python
result = querier.query_simbad(ra, dec)
if result.success and result.data.get('matched'):
    data = result.data
    
    # 基本信息
    print(f"名称: {data['main_id']}")
    print(f"类型: {data['otype']}")  # 如: Star, Galaxy, QSO
    print(f"光谱型: {data['sp_type']}")  # 如: A1V, G2V
    
    # 天体测量
    print(f"视差: {data['plx']} mas")
    print(f"径向速度: {data['rv']} km/s")
    print(f"红移: {data['redshift']}")
    
    # 测光数据
    mags = data['magnitudes']  # 字典: {'V': -1.46, 'B': -1.52, ...}
    print(f"星等: V={mags.get('V')}, B={mags.get('B')}")
    
    # 文献统计
    print(f"参考文献数: {data['n_references']}")
```

#### Gaia 数据

```python
result = querier.query_gaia(ra, dec)
if result.success and result.data.get('found'):
    data = result.data
    
    # 基本识别
    print(f"Source ID: {data['source_id']}")
    
    # 测光
    print(f"G = {data['g_mag']:.2f}")
    print(f"BP = {data['bp_mag']:.2f}")
    print(f"RP = {data['rp_mag']:.2f}")
    print(f"BP-RP = {data['bp_rp']:.2f}")
    
    # 天体测量
    print(f"视差: {data['parallax']:.2f} ± {data['parallax_error']:.2f} mas")
    print(f"距离: {data['distance']:.1f} pc")
    print(f"自行: ({data['pmra']:.2f}, {data['pmdec']:.2f}) mas/yr")
    
    # 物理参数
    print(f"有效温度: {data['teff']:.0f} K")
    print(f"表面重力: {data['logg']:.2f}")
    print(f"径向速度: {data['radial_velocity']:.1f} km/s")
    
    # 质量指标
    print(f"RUWE: {data['ruwe']:.2f}")  # < 1.4 表示好的天体测量
```

### 5. 生成报告

```python
# 生成文本报告
report = querier.generate_summary_report(results, ra, dec)
print(report)
```

输出示例:
```
======================================================================
天体数据查询报告
坐标: RA=101.287500°, DEC=-16.716100°
时间: 2026-03-04 15:30:00
======================================================================

【SIMBAD 数据库】
  ID: NAME SIRIUS
  类型: Star
  光谱型: A1V
  视差: 379.21 mas
  星等: U=-1.51, B=-1.52, V=-1.46, ...

【Gaia DR3】
  Source ID: 2947050466531873024
  G = -1.77
  视差: 379.21 ± 0.15 mas
  距离: 2.64 pc
  温度: 9940 K
  
【SDSS】
  无 SDSS 数据
```

### 6. 保存结果

```python
# 保存为 JSON 文件
filename = querier.save_results(results, ra, dec, output_dir="./output")
# 输出: ./output/astro_query_RA101.2875_DEC-16.7161.json

# 同时生成文本报告
# ./output/astro_query_RA101.2875_DEC-16.7161_report.txt
```

---

## 数据结构

### QueryResult 数据类

```python
@dataclass
class QueryResult:
    source: str           # 数据源名称
    success: bool         # 查询是否成功
    data: Optional[Dict]  # 查询结果数据
    error_message: Optional[str]  # 错误信息
    query_time: Optional[str]     # 查询时间
```

### SIMBAD 数据结构

```python
{
    'matched': True,
    'main_id': 'NAME SIRIUS',
    'otype': 'Star',
    'sp_type': 'A1V',
    'ra': 101.2875,
    'dec': -16.7161,
    'plx': 379.21,
    'rv': -5.50,
    'redshift': None,
    'magnitudes': {
        'U': -1.51, 'B': -1.52, 'V': -1.46,
        'R': -1.44, 'I': -1.43, ...
    },
    'n_references': 2345,
}
```

### Gaia 数据结构

```python
{
    'found': True,
    'source_id': '2947050466531873024',
    'ra': 101.2875,
    'dec': -16.7161,
    'parallax': 379.21,
    'parallax_error': 0.15,
    'pmra': -546.01,
    'pmdec': -1223.07,
    'g_mag': -1.77,
    'bp_mag': -1.36,
    'rp_mag': -2.28,
    'bp_rp': 0.92,
    'distance': 2.64,
    'g_abs': 1.61,
    'teff': 9940,
    'logg': 4.33,
    'radial_velocity': -5.50,
    'ruwe': 1.03,
    'all_sources': [...],  # 附近其他源
}
```

---

## 运行示例

### 基本示例

```bash
python src/astro_query_example.py
```

### 完整演示

```bash
python src/unified_astro_query.py
```

---

## astroquery 功能概览

本项目基于 `astroquery`，它提供了对多个天文数据服务的统一访问接口：

### 主要模块

| 模块 | 功能 | 本项目使用 |
|------|------|-----------|
| `astroquery.simbad` | SIMBAD数据库查询 | ✓ |
| `astroquery.gaia` | Gaia天体测量数据 | ✓ |
| `astroquery.vizier` | VizieR星表服务 | ✓ |
| `astroquery.sdss` | SDSS光谱和测光 | ✓ |
| `astroquery.skyview` | 天空图像服务 | ✓ |
| `astroquery.ned` | NASA河外数据库 | ✓ |
| `astroquery.heasarc` | 高能天体物理 | ✓ |
| `astroquery.mast` | MAST数据档案 | - |
| `astroquery.irsa` | IRSA红外数据 | - |
| `astroquery.eso` | ESO数据档案 | - |

### VizieR 常用星表

```python
# 通过 VizieR 查询的常用星表
catalogs = [
    'I/355/gaiadr3',    # Gaia DR3
    'II/246/out',       # 2MASS
    'II/328/allwise',   # AllWISE
    'II/349/ps1',       # Pan-STARRS1
    'V/147/sdss12',     # SDSS DR12
    'B/vsx/vsx',        # 变星总表
    'B/gcvs/gcvs_cat',  # GCVS变星表
    'J/A+A/...',        # A&A 论文附表
]
```

---

## 优化点总结

与原代码相比，优化后的接口提供了：

1. **统一入口**: 一行代码查询多个数据库
   ```python
   results = querier.query_all(ra, dec)
   ```

2. **自动错误处理**: 单个服务失败不影响其他服务

3. **数据标准化**: 统一的数据结构，易于处理

4. **智能报告**: 自动生成可读性强的文本报告

5. **结果缓存**: 自动保存结果到 JSON 文件

6. **详细日志**: 记录查询过程和错误信息

---

## 扩展建议

### 添加新的数据源

```python
def query_new_service(self, ra, dec):
    try:
        from astroquery.new_service import NewService
        # 查询逻辑
        return QueryResult('new_service', True, data=data)
    except Exception as e:
        return QueryResult('new_service', False, error_message=str(e))
```

### 批量查询

```python
targets = [(name1, ra1, dec1), (name2, ra2, dec2), ...]
for name, ra, dec in targets:
    results = querier.query_all(ra, dec)
    querier.save_results(results, ra, dec)
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `unified_astro_query.py` | 主要接口实现 |
| `astro_query_example.py` | 使用示例 |
| `ASTROQUERY_GUIDE.md` | 本使用文档 |

---

**作者**: AI Assistant  
**日期**: 2026-03-04  
**版本**: v1.0
