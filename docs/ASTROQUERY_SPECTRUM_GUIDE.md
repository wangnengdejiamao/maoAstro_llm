# Astroquery 光谱与数据下载完整指南

## 📋 概述

本文档详细介绍如何使用 `astroquery` 下载各类天文数据，包括**光谱、图像、时序数据和星表**。

---

## 🎯 支持的下载服务

### 一、光谱服务

| 服务名称 | 类型 | 覆盖波段 | 主要用途 |
|---------|------|---------|---------|
| **SDSS** | 光学光谱 | 3800-9200 Å | 恒星、星系、类星体 |
| **HEASARC** | X射线/高能 | 0.1-100 keV | X射线双星、AGN |
| **IRSA (IRS)** | 红外光谱 | 5-40 μm | 原恒星、尘埃 |
| **ESO Archive** | 光学/红外 | 多波段 | VLT、NTT光谱 |
| **MAST** | 紫外/光学 | 多波段 | HST、JWST |
| **Splatalogue** | 分子谱线 | 射电/亚毫米 | 分子云、恒星形成 |
| **NIST** | 原子谱线 | 全波段 | 谱线识别 |
| **JPLSpec** | 分子谱线 | 射电 | 天体化学 |
| **HITRAN** | 分子吸收 | 红外 | 行星大气 |

### 二、图像服务

| 服务名称 | 巡天/仪器 | 波段 |
|---------|----------|------|
| **SkyView** | DSS、2MASS、WISE、GALEX | 多波段 |
| **ESASky** | XMM、Herschel、Gaia | X射线/红外 |
| **HiPS2FITS** | DSS、CDS巡天 | 多波段 |
| **CADC** | CFHT、Gemini | 光学/红外 |
| **IRSA** | WISE、Spitzer | 红外 |

### 三、时序数据服务

| 服务名称 | 任务 | 数据类型 |
|---------|------|---------|
| **MAST** | Kepler、TESS、K2 | 空间光变 |
| **OGLE** | OGLE-III/IV | 地面光变 |
| **IRSA** | WISE (NEOWISE) | 红外时序 |
| **VizieR** | VSX、ASAS、NSVS | 变星数据 |

---

## 🔧 详细使用示例

### 1. SDSS 光谱下载

```python
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u

# 目标坐标
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 1. 查询光谱数据
spectra = SDSS.query_region(coord, spectro=True, radius=5*u.arcsec)
print(f"找到 {len(spectra)} 条光谱")

# 2. 下载光谱文件（需要 plate, mjd, fiberID）
if len(spectra) > 0:
    spec = spectra[0]
    sp_files = SDSS.get_spectra(
        plate=int(spec['plate']),
        mjd=int(spec['mjd']),
        fiberID=int(spec['fiberid'])
    )
    print(f"光谱文件已下载: {sp_files}")
```

### 2. SkyView 图像下载

```python
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u

# 目标坐标
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 下载多个巡天的图像
surveys = ['DSS2 Red', 'DSS2 Blue', '2MASS-J', '2MASS-K', 'WISE 3.4', 'GALEX Near UV']

for survey in surveys:
    try:
        images = SkyView.get_images(
            position=coord,
            survey=survey,
            radius=0.1 * u.deg,  # 图像半径
            pixels=500           # 图像大小（像素）
        )
        
        # 保存 FITS 文件
        if images:
            filename = f"{survey.replace(' ', '_')}.fits"
            images[0][0].writeto(filename, overwrite=True)
            print(f"✓ {survey} 下载完成")
    except Exception as e:
        print(f"✗ {survey}: {e}")
```

### 3. VizieR 星表查询与下载

```python
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u

# 设置 VizieR
vizier = Vizier(columns=['**'])
vizier.ROW_LIMIT = 1000

# 目标坐标
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 常用星表列表
useful_catalogs = [
    ('I/355/gaiadr3', 'Gaia DR3'),          # Gaia 天体测量
    ('II/246/out', '2MASS'),                # 2MASS 近红外测光
    ('II/328/allwise', 'AllWISE'),          # WISE 红外测光
    ('B/vsx/vsx', 'VSX'),                   # 变星总表
    ('V/147/sdss12', 'SDSS DR12'),          # SDSS 测光
    ('J/AJ/161/176/table1', 'LAMOST DR8'),  # LAMOST 光谱参数
]

# 查询每个星表
for catalog, name in useful_catalogs:
    try:
        result = vizier.query_region(
            coord,
            radius=1 * u.arcmin,
            catalog=catalog
        )
        
        if result and len(result) > 0:
            table = result[0]
            filename = f"{catalog.replace('/', '_')}.csv"
            table.write(filename, format='csv', overwrite=True)
            print(f"✓ {name}: {len(table)} 条记录")
        else:
            print(f"○ {name}: 无数据")
    except Exception as e:
        print(f"✗ {name}: {e}")
```

### 4. HEASARC X射线数据查询

```python
from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord

heasarc = Heasarc()

# 查询不同的 X射线任务
target = "Cen X-3"  # X射线双星

missions = {
    'xmm': 'xmmmaster',      # XMM-Newton
    'chandra': 'chanmaster', # Chandra
    'suzaku': 'suzamaster',  # Suzaku
    'nustar': 'numaster',    # NuSTAR
    'swift': 'swiftmastr',   # Swift
}

for mission, table in missions.items():
    try:
        result = heasarc.query_object(target, mission=table)
        if result and len(result) > 0:
            print(f"✓ {mission.upper()}: {len(result)} 次观测")
            # 输出观测信息
            for obs in result[:3]:  # 只显示前3个
                print(f"   - OBSID: {obs.get('OBSID', 'N/A')}, "
                      f"Date: {obs.get('START_DATE', 'N/A')}")
    except Exception as e:
        print(f"✗ {mission}: {e}")
```

### 5. 分子谱线查询（Splatalogue）

```python
from astroquery.splatalogue import Splatalogue
from astropy import units as u

# 查询 CO 分子在 100-300 GHz 的谱线
result = Splatalogue.query_lines(
    min_frequency=100 * u.GHz,
    max_frequency=300 * u.GHz,
    chemical_name='CO',
    energy_range=(0, 1000),
    energy_type='eu_k',
    line_lists=['JPL']
)

print(f"找到 {len(result)} 条 CO 谱线")
if len(result) > 0:
    print("\n前5条谱线:")
    for line in result[:5]:
        print(f"  {line['Species']}: {line['Freq-GHz']:.4f} GHz, "
              f"E_up={line['E_U (K)']:.1f} K")
    
    # 保存结果
    result.write('co_spectral_lines.csv', format='csv', overwrite=True)
```

### 6. 原子谱线查询（NIST）

```python
from astroquery.nist import Nist

# 查询氢原子在 H-alpha 附近的谱线
result = Nist.query(
    minwav=6500,
    maxwav=6600,
    wavelength_unit='Angstrom',
    element='H',
    ion_state=0  # 中性氢
)

print(f"找到 {len(result)} 条氢谱线")
for line in result[:10]:
    print(f"  λ={line['Ritz']} Å, Aki={line['Aki']}, ")
```

### 7. MAST 光变曲线查询

```python
from astroquery.mast import Mast

# 查询 TESS 光变曲线
target = "Kepler-186"  # 已知有行星的恒星

# 查询 TESS 观测
observations = Mast.query_object(
    target,
    service="Mast.Caom.Cone",
    params={"obs_collection": "TESS"}
)

if observations and len(observations) > 0:
    print(f"找到 {len(observations)} 次 TESS 观测")
    
    # 获取数据产品列表
    data_products = Mast.get_product_list(observations[0])
    
    # 筛选光变曲线文件
    lc_files = [p for p in data_products 
                if 'lc' in str(p.get('productFilename', '')).lower()]
    
    print(f"其中 {len(lc_files)} 个是光变曲线文件")
```

### 8. ESASky 多波段查询

```python
from astroquery.esasky import ESASky
from astropy.coordinates import SkyCoord
from astropy import units as u

# 目标坐标
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 查询 XMM-Newton 观测
xmm_maps = ESASky.query_region_maps(
    position=coord,
    radius=1 * u.deg,
    missions='XMM'
)

print(f"XMM 观测: {len(xmm_maps)} 个")

# 查询 Herschel 观测
herschel_maps = ESASky.query_region_maps(
    position=coord,
    radius=1 * u.deg,
    missions='HERSCHEL'
)

print(f"Herschel 观测: {len(herschel_maps)} 个")
```

### 9. HiPS 图像下载

```python
from astroquery.hips2fits import hips2fits

# 常用 HiPS 巡天
hips_surveys = [
    'CDS/P/DSS2/color',          # DSS2 彩色
    'CDS/P/2MASS/color',         # 2MASS 彩色
    'CDS/P/allWISE/color',       # WISE 彩色
    'CDS/P/SDSS9/color',         # SDSS DR9 彩色
    'CDS/P/Spitzer/IRAC134',     # Spitzer IRAC
]

ra, dec = 10.6847, 41.2687

for survey in hips_surveys:
    try:
        result = hips2fits.query(
            hips=survey,
            width=500,
            height=500,
            ra=ra,
            dec=dec,
            fov=0.5  # 视场（度）
        )
        
        if result:
            filename = f"hips_{survey.replace('/', '_')}.fits"
            result.writeto(filename, overwrite=True)
            print(f"✓ {survey}")
    except Exception as e:
        print(f"✗ {survey}: {e}")
```

---

## 📊 实际应用：恒星分类完整流程

```python
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u

class StarAnalyzer:
    """恒星分析器"""
    
    def __init__(self):
        self.vizier = Vizier(columns=['**'])
        self.vizier.ROW_LIMIT = 100
    
    def analyze(self, ra, dec, radius=5):
        """分析指定位置的恒星"""
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        
        results = {}
        
        # 1. SIMBAD 基础信息
        try:
            simbad = Simbad.query_region(coord, radius=radius*u.arcsec)
            if simbad:
                results['simbad'] = {
                    'name': simbad[0]['MAIN_ID'],
                    'type': simbad[0]['OTYPE_S'],
                    'sp_type': simbad[0]['SP_TYPE'],
                }
        except:
            pass
        
        # 2. Gaia DR3 天体测量
        try:
            gaia = self.vizier.query_region(
                coord, radius=radius*u.arcsec, catalog='I/355/gaiadr3'
            )
            if gaia:
                g = gaia[0]
                results['gaia'] = {
                    'g_mag': float(g['Gmag']),
                    'bp_rp': float(g['BP-RP']) if 'BP-RP' in g.colnames else None,
                    'parallax': float(g['Plx']) if 'Plx' in g.colnames else None,
                    'teff': float(g['Teff']) if 'Teff' in g.colnames else None,
                }
        except:
            pass
        
        # 3. 下载多波段图像
        try:
            images = SkyView.get_images(
                coord, 
                survey=['DSS2 Red', '2MASS-J', 'WISE 3.4'],
                radius=0.05*u.deg
            )
            results['images'] = len(images)
        except:
            pass
        
        return results

# 使用示例
analyzer = StarAnalyzer()
result = analyzer.analyze(10.6847, 41.2687)
print(json.dumps(result, indent=2))
```

---

## 🗂️ 输出目录结构建议

```
downloads/
├── spectra/           # 光谱文件
│   ├── sdss_spec_*.fits
│   ├── heasarc_*.fits
│   └── irs_*.fits
├── images/            # 图像文件
│   ├── skyview_*.fits
│   ├── hips_*.fits
│   └── esasky_*.fits
├── lightcurves/       # 光变曲线
│   ├── tess_*.fits
│   ├── kepler_*.fits
│   └── ogle_*.csv
└── catalogs/          # 星表数据
    ├── gaia_dr3_*.csv
    ├── vsx_*.csv
    └── sdss_*.csv
```

---

## ⚠️ 注意事项

1. **网络限制**：部分服务（如 Gaia）可能有访问频率限制
2. **数据量**：批量下载时请控制请求频率，避免被封禁
3. **数据版权**：下载的数据请遵守各数据中心的版权规定
4. **查询限制**：设置合理的 `ROW_LIMIT` 避免内存溢出

---

## 📚 参考链接

- [Astroquery 官方文档](https://astroquery.readthedocs.io/)
- [SDSS SkyServer](http://skyserver.sdss.org/)
- [HEASARC 数据档案](https://heasarc.gsfc.nasa.gov/)
- [ESASky](https://sky.esa.int/)
- [VizieR 星表服务](http://vizier.u-strasbg.fr/)
