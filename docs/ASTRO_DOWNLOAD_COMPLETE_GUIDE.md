# Astroquery 完整数据下载指南

## 概述

本文档全面介绍如何使用 `astroquery` 下载天文数据，包括**光谱、图像、光变曲线和星表**。

---

## 快速对比：VSP vs Astroquery

| 功能 | VSP | Astroquery | 适用场景 |
|-----|-----|-----------|---------|
| 本地光谱 | ✅ LAMOST DR10 | ❌ 不支持 | 批量分析本地数据 |
| 在线光谱 | ❌ 不支持 | ✅ SDSS/HEASARC | 下载公开光谱 |
| 图像下载 | ❌ 不支持 | ✅ SkyView/HiPS | 获取多波段图像 |
| 光变曲线 | ✅ ZTF/TESS | ✅ MAST/OGLE | 时序分析 |
| 天体测量 | ✅ Gaia DR3 | ✅ Gaia Archive | 视差、自行数据 |
| 天体识别 | ✅ SIMBAD/VSX | ✅ SIMBAD/VizieR | 分类查询 |
| 速度 | ✅ 快（本地） | ⚠️ 依赖网络 | 实时查询 |
| 离线能力 | ✅ 支持 | ❌ 需网络 | 无网络环境 |

**结论**: VSP 和 Astroquery **互补使用**，VSP 适合本地批量处理，Astroquery 适合在线数据下载。

---

## 一、光谱下载服务

### 1. SDSS 光学光谱

```python
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u

# 坐标
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 查询光谱数据
spectra = SDSS.query_region(coord, spectro=True, radius=5*u.arcsec)

if spectra is not None and len(spectra) > 0:
    # 获取第一个光谱
    spec = spectra[0]
    print(f"找到光谱: Plate={spec['plate']}, MJD={spec['mjd']}, Fiber={spec['fiberid']}")
    print(f"红移: {spec['z']:.4f}, 类型: {spec['class']}")
    
    # 下载光谱文件
    sp_files = SDSS.get_spectra(
        plate=int(spec['plate']),
        mjd=int(spec['mjd']),
        fiberID=int(spec['fiberid'])
    )
    print(f"光谱文件: {sp_files}")
```

**输出示例**:
```
找到光谱: Plate=751, MJD=52251, Fiber=3
红移: 0.0321, 类型: GALAXY
光谱文件: ['/tmp/tmp.../spPlate-0751-52251-003.fits']
```

### 2. HEASARC X射线光谱

```python
from astroquery.heasarc import Heasarc

heasarc = Heasarc()

# 查询 Cen X-3 (X射线双星)
result = heasarc.query_object('Cen X-3', mission='xmmmaster')

if result is not None:
    print(f"XMM-Newton 观测次数: {len(result)}")
    for obs in result[:3]:
        print(f"  OBSID: {obs['OBSID']}, 日期: {obs['START_DATE']}, 曝光: {obs['EXPOSURE']}s")
```

### 3. 分子谱线查询 (Splatalogue)

```python
from astroquery.splatalogue import Splatalogue
from astropy import units as u

# 查询 CO 在 100-200 GHz 的谱线
result = Splatalogue.query_lines(
    min_frequency=100 * u.GHz,
    max_frequency=200 * u.GHz,
    chemical_name='CO',
    energy_range=(0, 500),
    energy_type='eu_k',
    line_lists=['JPL']
)

print(f"找到 {len(result)} 条 CO 谱线")
for line in result[:5]:
    print(f"  {line['Species']}: {line['Freq-GHz']:.4f} GHz, E_up={line['E_U (K)']:.1f} K")
```

**输出示例**:
```
找到 14 条 CO 谱线
  COv=0: 115.2712 GHz, E_up=5.5 K
  13COv=0: 110.2014 GHz, E_up=5.3 K
  C18Ov=0: 109.7822 GHz, E_up=5.3 K
  ...
```

### 4. 原子谱线查询 (NIST)

```python
from astroquery.nist import Nist

# 查询 H-alpha (6563 Å) 附近的氢谱线
result = Nist.query(
    minwav=6500,
    maxwav=6600,
    wavelength_unit='Angstrom',
    element='H',
    ion_state=0  # 中性氢
)

print(f"找到 {len(result)} 条氢谱线")
for line in result:
    if 'Ritz' in line.colnames and line['Ritz']:
        print(f"  λ={line['Ritz']} Å")
```

---

## 二、图像下载服务

### 1. SkyView 多波段图像

```python
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u

# 目标: M31 中心
ra, dec = 10.6847, 41.2687
coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

# 可用的巡天项目
surveys = [
    'DSS2 Red',      # DSS2 光学红波段
    'DSS2 Blue',     # DSS2 光学蓝波段
    '2MASS-J',       # 2MASS J波段 (1.25 μm)
    '2MASS-H',       # 2MASS H波段 (1.65 μm)
    '2MASS-K',       # 2MASS K波段 (2.17 μm)
    'WISE 3.4',      # WISE W1 (3.4 μm)
    'WISE 4.6',      # WISE W2 (4.6 μm)
    'WISE 12',       # WISE W3 (12 μm)
    'WISE 22',       # WISE W4 (22 μm)
    'GALEX Near UV', # GALEX NUV (227 nm)
    'GALEX Far UV',  # GALEX FUV (152 nm)
]

# 下载图像
for survey in surveys:
    try:
        images = SkyView.get_images(
            position=coord,
            survey=survey,
            radius=0.2 * u.deg,  # 0.2度视场
            pixels=500
        )
        
        if images:
            filename = f"m31_{survey.replace(' ', '_')}.fits"
            images[0][0].writeto(filename, overwrite=True)
            print(f"✓ {survey} -> {filename}")
    except Exception as e:
        print(f"✗ {survey}: {e}")
```

### 2. HiPS2FITS 图像服务

```python
from astroquery.hips2fits import hips2fits

# 常用 HiPS 巡天
hips_surveys = {
    'DSS2': 'CDS/P/DSS2/color',
    '2MASS': 'CDS/P/2MASS/color',
    'AllWISE': 'CDS/P/allWISE/color',
    'SDSS9': 'CDS/P/SDSS9/color',
    'Spitzer': 'CDS/P/Spitzer/IRAC134',
    'Herschel': 'CDS/P/Herschel/PlanckP125-353-857',
}

# 下载不同巡天的图像
for name, hips_id in hips_surveys.items():
    try:
        result = hips2fits.query(
            hips=hips_id,
            width=500,
            height=500,
            ra=10.6847,
            dec=41.2687,
            fov=0.5  # 0.5度视场
        )
        
        if result:
            filename = f"hips_{name}.fits"
            result.writeto(filename, overwrite=True)
            print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
```

### 3. ESASky ESA多波段

```python
from astroquery.esasky import ESASky
from astropy.coordinates import SkyCoord
from astropy import units as u

coord = SkyCoord(ra=10.6847, dec=41.2687, unit=(u.deg, u.deg))

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

---

## 三、光变曲线服务

### 1. MAST TESS/Kepler

```python
from astroquery.mast import Mast

# 查询目标
target = "Kepler-186"

# 查询 TESS 观测
observations = Mast.query_object(
    target,
    service="Mast.Caom.Cone",
    params={"obs_collection": "TESS"}
)

if observations is not None:
    print(f"TESS 观测次数: {len(observations)}")
    
    # 获取数据产品
    if len(observations) > 0:
        products = Mast.get_product_list(observations[0])
        print(f"数据产品: {len(products)} 个")
        
        # 筛选光变曲线
        lc_products = [p for p in products 
                       if 'lc' in str(p.get('productFilename', '')).lower()]
        print(f"光变曲线文件: {len(lc_products)} 个")
```

### 2. OGLE 变星

```python
from astroquery.ogle import Ogle
from astropy.coordinates import SkyCoord
from astropy import units as u

coord = SkyCoord(ra=270.9042, dec=-29.0078, unit=(u.deg, u.deg))

# 查询 OGLE 数据（主要是消光）
result = Ogle.query_region(coord=coord, width=0.1 * u.deg)

if result is not None:
    print(f"找到 {len(result)} 颗恒星")
    # 保存结果
    result.write('ogle_result.csv', format='csv', overwrite=True)
```

---

## 四、星表查询服务

### 1. VizieR 综合星表

```python
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u

vizier = Vizier(columns=['**'])
vizier.ROW_LIMIT = 1000

coord = SkyCoord(ra=10.6847, dec=41.2687, unit=(u.deg, u.deg))

# 常用星表
catalogs = [
    ('I/355/gaiadr3', 'Gaia DR3'),
    ('II/246/out', '2MASS'),
    ('II/328/allwise', 'AllWISE'),
    ('II/311/wise', 'WISE'),
    ('V/147/sdss12', 'SDSS DR12'),
    ('B/vsx/vsx', 'VSX (变星总表)'),
    ('B/gcvs/gcvs_cat', 'GCVS (变星总表)'),
]

for cat_id, cat_name in catalogs:
    try:
        result = vizier.query_region(
            coord,
            radius=1 * u.arcmin,
            catalog=cat_id
        )
        
        if result and len(result) > 0:
            table = result[0]
            print(f"✓ {cat_name}: {len(table)} 条记录")
            
            # 保存
            filename = f"vizier_{cat_id.replace('/', '_')}.csv"
            table.write(filename, format='csv', overwrite=True)
        else:
            print(f"○ {cat_name}: 无数据")
    except Exception as e:
        print(f"✗ {cat_name}: {e}")
```

### 2. SIMBAD 天体识别

```python
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u

# 配置 SIMBAD 返回更多列
Simbad.add_votable_fields('otype', 'sp', 'plx', 'rvz_redshift')

coord = SkyCoord(ra=10.6847, dec=41.2687, unit=(u.deg, u.deg))

# 查询
result = Simbad.query_region(coord, radius=1 * u.arcmin)

if result is not None:
    print(f"找到 {len(result)} 个天体")
    for obj in result[:5]:
        print(f"  {obj['MAIN_ID']}: 类型={obj['OTYPE']}, 光谱型={obj['SP_TYPE']}")
```

### 3. NED 河外天体

```python
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
from astropy import units as u

coord = SkyCoord(ra=10.6847, dec=41.2687, unit=(u.deg, u.deg))

# 查询河外天体
result = Ned.query_region(coord, radius=0.1 * u.deg)

if result is not None:
    print(f"找到 {len(result)} 个河外天体")
    for obj in result[:5]:
        print(f"  {obj['Object Name']}: z={obj.get('Redshift', 'N/A')}")
```

---

## 五、综合应用示例

### 完整恒星分析流程

```python
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.skyview import SkyView
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
import json

class CompleteStarAnalysis:
    """完整的恒星分析工具"""
    
    def __init__(self):
        self.vizier = Vizier(columns=['**'])
        self.vizier.ROW_LIMIT = 100
        Simbad.add_votable_fields('otype', 'sp', 'plx', 'distance')
    
    def analyze(self, ra, dec, radius_arcsec=10):
        """
        分析指定位置的恒星
        
        Parameters:
        -----------
        ra, dec : float
            坐标（度）
        radius_arcsec : float
            搜索半径（角秒）
        
        Returns:
        --------
        dict : 分析结果
        """
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        result = {
            'coordinates': {'ra': ra, 'dec': dec},
            'identification': {},
            'photometry': {},
            'astrometry': {},
            'spectroscopy': {},
            'images': {},
        }
        
        # 1. SIMBAD 识别
        try:
            simbad = Simbad.query_region(coord, radius=radius_arcsec*u.arcsec)
            if simbad:
                result['identification'] = {
                    'name': str(simbad[0]['MAIN_ID']),
                    'type': str(simbad[0].get('OTYPE', 'Unknown')),
                    'spectral_type': str(simbad[0].get('SP_TYPE', 'Unknown')),
                }
        except Exception as e:
            result['identification']['error'] = str(e)
        
        # 2. Gaia DR3 天体测量
        try:
            gaia = self.vizier.query_region(
                coord, radius=radius_arcsec*u.arcsec, catalog='I/355/gaiadr3'
            )
            if gaia:
                g = gaia[0]
                result['astrometry'] = {
                    'parallax_mas': float(g.get('Plx', 0)) if 'Plx' in g.colnames else None,
                    'pm_ra_masyr': float(g.get('pmRA', 0)) if 'pmRA' in g.colnames else None,
                    'pm_dec_masyr': float(g.get('pmDE', 0)) if 'pmDE' in g.colnames else None,
                    'ruwe': float(g.get('RUWE', 0)) if 'RUWE' in g.colnames else None,
                }
                result['photometry'] = {
                    'g_mag': float(g.get('Gmag', 0)) if 'Gmag' in g.colnames else None,
                    'bp_rp': float(g.get('BP-RP', 0)) if 'BP-RP' in g.colnames else None,
                    'teff_k': float(g.get('Teff', 0)) if 'Teff' in g.colnames else None,
                    'logg': float(g.get('logg', 0)) if 'logg' in g.colnames else None,
                }
        except Exception as e:
            result['astrometry']['error'] = str(e)
        
        # 3. SDSS 光谱
        try:
            sdss = SDSS.query_region(coord, spectro=True, radius=radius_arcsec*u.arcsec)
            if sdss:
                result['spectroscopy'] = {
                    'has_spectrum': True,
                    'redshift': float(sdss[0]['z']) if 'z' in sdss.colnames else None,
                    'classification': str(sdss[0]['class']) if 'class' in sdss.colnames else None,
                    'sn_median': float(sdss[0]['snMedian']) if 'snMedian' in sdss.colnames else None,
                }
        except Exception as e:
            result['spectroscopy']['has_spectrum'] = False
        
        # 4. 多波段图像
        try:
            surveys = ['DSS2 Red', '2MASS-J', 'WISE 3.4']
            images = SkyView.get_images(
                coord, survey=surveys, 
                radius=0.05*u.deg, pixels=300
            )
            result['images'] = {
                'downloaded': len(images),
                'surveys': surveys
            }
        except Exception as e:
            result['images']['error'] = str(e)
        
        return result
    
    def print_report(self, result):
        """打印分析报告"""
        print("\n" + "="*60)
        print("恒星分析报告")
        print("="*60)
        print(f"坐标: RA={result['coordinates']['ra']:.4f}, "
              f"DEC={result['coordinates']['dec']:.4f}")
        
        # 识别
        print("\n【天体识别】")
        if 'name' in result['identification']:
            iden = result['identification']
            print(f"  名称: {iden['name']}")
            print(f"  类型: {iden['type']}")
            print(f"  光谱型: {iden['spectral_type']}")
        
        # 测光
        if result['photometry']:
            print("\n【测光数据 (Gaia DR3)】")
            phot = result['photometry']
            if phot.get('g_mag'):
                print(f"  G = {phot['g_mag']:.2f}")
            if phot.get('bp_rp'):
                print(f"  BP-RP = {phot['bp_rp']:.2f}")
            if phot.get('teff_k'):
                print(f"  Teff = {phot['teff_k']:.0f} K")
        
        # 天体测量
        if result['astrometry']:
            print("\n【天体测量】")
            astro = result['astrometry']
            if astro.get('parallax_mas'):
                print(f"  视差: {astro['parallax_mas']:.2f} mas")
                if astro['parallax_mas'] > 0:
                    dist_pc = 1000 / astro['parallax_mas']
                    print(f"  距离: ~{dist_pc:.0f} pc")
        
        # 光谱
        if result['spectroscopy'].get('has_spectrum'):
            print("\n【SDSS 光谱】")
            spec = result['spectroscopy']
            print(f"  红移: {spec.get('redshift', 'N/A')}")
            print(f"  分类: {spec.get('classification', 'N/A')}")
        
        # 图像
        print("\n【多波段图像】")
        print(f"  下载了 {result['images'].get('downloaded', 0)} 张图像")
        
        print("\n" + "="*60)


# 使用示例
if __name__ == "__main__":
    analyzer = CompleteStarAnalysis()
    
    # 分析 M31 中心附近的一颗星
    result = analyzer.analyze(10.6847, 41.2687, radius_arcsec=30)
    analyzer.print_report(result)
    
    # 保存为JSON
    with open('star_analysis_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
```

---

## 六、输出目录结构建议

```
projects/
├── downloads/
│   ├── spectra/               # 光谱文件
│   │   ├── sdss/
│   │   │   ├── spec-0751-52251-003.fits
│   │   │   └── spec-0752-52252-004.fits
│   │   └── heasarc/
│   │       └── xmm_observations.csv
│   ├── images/                # 图像文件
│   │   ├── skyview/
│   │   │   ├── m31_dss2_red.fits
│   │   │   ├── m31_2mass_j.fits
│   │   │   └── m31_wise_w1.fits
│   │   └── hips/
│   │       ├── m31_dss2_color.fits
│   │       └── m31_2mass_color.fits
│   ├── lightcurves/           # 光变曲线
│   │   ├── tess/
│   │   │   ├── kepler186_lc.fits
│   │   │   └── tic12345_lc.fits
│   │   └── ogle/
│   │       └── ogle_field.csv
│   └── catalogs/              # 星表数据
│       ├── gaia_dr3/
│       │   └── gaia_dr3_field.csv
│       ├── vsx/
│       │   └── vsx_field.csv
│       └── vizier/
│           └── vizier_allwise.csv
└── analysis/                  # 分析结果
    └── star_analysis_result.json
```

---

## 七、常见问题

### Q1: 下载速度慢怎么办？

- 使用较小的 `radius` 减少查询范围
- 使用 `ROW_LIMIT` 限制返回行数
- 批量下载时添加延时避免被封禁

### Q2: Gaia 查询失败？

- Gaia Archive 有访问频率限制
- 尝试使用 VizieR 的 Gaia DR3 星表 (`I/355/gaiadr3`)
- 考虑使用本地缓存

### Q3: 如何下载多个目标？

```python
targets = [
    (10.6847, 41.2687, "M31 Center"),
    (201.3650, -43.0192, "Centaurus A"),
    (266.4168, -29.0078, "Galactic Center"),
]

for ra, dec, name in targets:
    print(f"处理 {name}...")
    result = analyzer.analyze(ra, dec)
    # 保存结果...
    time.sleep(1)  # 避免请求过快
```

---

## 八、参考资料

- [Astroquery 官方文档](https://astroquery.readthedocs.io/)
- [SDSS SkyServer](http://skyserver.sdss.org/)
- [HEASARC](https://heasarc.gsfc.nasa.gov/)
- [ESASky](https://sky.esa.int/)
- [VizieR](http://vizier.u-strasbg.fr/)
- [SIMBAD](http://simbad.u-strasbg.fr/)
- [Splatalogue](https://splatalogue.online/)
- [NIST Atomic Spectra](https://physics.nist.gov/cgi-bin/ASD/energy1.pl)
