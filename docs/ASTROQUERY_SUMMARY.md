# Astroquery 数据下载功能总结

## 已完成工作

### 1. 代码文件

| 文件 | 描述 | 行数 |
|-----|------|------|
| `src/complete_astro_download.py` | 完整数据下载接口 | ~650 |
| `src/astro_download_demo.py` | 服务演示（离线模式） | ~320 |
| `src/unified_astro_query.py` | 基础查询接口（7服务） | ~400 |

### 2. 文档文件

| 文件 | 描述 | 内容 |
|-----|------|------|
| `docs/ASTROQUERY_SPECTRUM_GUIDE.md` | 光谱下载指南 | SDSS/HEASARC/Splatalogue/NIST |
| `docs/ASTRO_DOWNLOAD_COMPLETE_GUIDE.md` | 完整使用指南 | 光谱/图像/光变/星表 |
| `docs/ASTROQUERY_SUMMARY.md` | 本汇总文档 | 全部功能总结 |

### 3. 生成的数据文件

| 文件 | 描述 |
|-----|------|
| `astroquery_services.json` | 服务列表JSON |

---

## 支持的数据服务 (18个)

### 🔬 光谱服务 (6个)
- **SDSS** - 斯隆数字巡天光学光谱
- **HEASARC** - X射线/高能光谱
- **Splatalogue** - 分子谱线数据库
- **NIST** - 原子谱线数据库
- **IRSA (IRS)** - Spitzer 红外光谱
- **ESO Archive** - 欧洲南方台光谱

### 🖼️ 图像服务 (4个)
- **SkyView** - DSS/2MASS/WISE/GALEX 图像
- **HiPS2FITS** - HiPS 多波段图像
- **ESASky** - XMM/Herschel 图像
- **ImageCutouts** - 巡天图像切片

### 📈 光变曲线 (2个)
- **MAST** - TESS/Kepler 光变
- **OGLE** - OGLE 光变数据

### 📚 星表服务 (4个)
- **VizieR** - CDS 星表服务
- **SIMBAD** - 天体识别
- **NED** - 河外数据库
- **Gaia** - Gaia DR3 天体测量

### 🔧 其他服务 (2个)
- **ALMA** - 射电干涉数据
- **Exoplanet Archives** - 系外行星数据

---

## 关键代码示例

### 下载 SDSS 光谱
```python
from astroquery.sdss import SDSS
from astropy import units as u

spectra = SDSS.query_region(coord, spectro=True, radius=5*u.arcsec)
if spectra:
    files = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)
```

### 下载多波段图像
```python
from astroquery.skyview import SkyView

images = SkyView.get_images(
    position=coord,
    survey=['DSS2 Red', '2MASS-J', 'WISE 3.4'],
    radius=0.1*u.deg
)
```

### 查询分子谱线
```python
from astroquery.splatalogue import Splatalogue
from astropy import units as u

result = Splatalogue.query_lines(
    min_frequency=100*u.GHz,
    max_frequency=200*u.GHz,
    chemical_name='CO'
)
```

### 查询原子谱线
```python
from astroquery.nist import Nist

result = Nist.query(
    minwav=6500, maxwav=6600,
    element='H', ion_state=0
)
```

---

## VSP vs Astroquery 对比

| 功能 | VSP | Astroquery | 备注 |
|-----|-----|-----------|------|
| 本地LAMOST | ✅ | ❌ | VSP优势 |
| SDSS光谱 | ❌ | ✅ | Astroquery优势 |
| X射线光谱 | ❌ | ✅ | HEASARC独有 |
| 分子谱线 | ❌ | ✅ | Splatalogue独有 |
| 多波段图像 | ❌ | ✅ | SkyView独有 |
| 光变曲线 | ✅ ZTF/TESS | ✅ MAST/OGLE | 互补 |
| 天体测量 | ✅ Gaia | ✅ Gaia | 两者皆可 |
| 速度 | ⚡ 快 | 🌐 依赖网络 | VSP更快 |
| 离线使用 | ✅ 支持 | ❌ 需网络 | VSP优势 |

**结论**: VSP 和 Astroquery **互补使用**，根据场景选择合适工具。

---

## 下一步建议

1. **集成到 VSP**: 在 `lib/vsp/VSP_Download.py` 中添加在线下载功能
2. **缓存机制**: 为 Astroquery 添加本地缓存避免重复下载
3. **批处理**: 实现多目标批量下载功能
4. **可视化**: 结合下载的数据创建分析图表

---

*生成时间: 2026-03-04*
*作者: AI Assistant*
