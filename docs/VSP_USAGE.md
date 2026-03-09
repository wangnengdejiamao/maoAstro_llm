# VSP (Variable Star Pipeline) 工具包使用指南

## 目录结构

```
lib/
├── vsp/                          # VSP 工具包目录
│   ├── __init__.py              # 统一入口，导出所有功能
│   ├── VSP_Auxiliary.py         # 辅助函数
│   ├── VSP_Lib.py               # 本地星表查询 (VSX, GCVS, CV等)
│   ├── VSP_GAIA.py              # Gaia 数据查询
│   ├── VSP_LAMOST.py            # LAMOST 光谱查询
│   ├── VSP_ZTF.py               # ZTF 光变数据
│   ├── VSP_TESS.py              # TESS 数据查询
│   ├── VSP_ASASSN.py            # ASAS-SN 数据查询
│   ├── VSP_EBV.py               # 消光查询
│   ├── VSP_Photometry.py        # 多波段测光
│   ├── VSP_Integrate.py         # 数据整合
│   ├── VSP_Kinematics.py        # 运动学计算
│   ├── VSP_Period.py            # 周期分析
│   └── README.md                # VSP 内部文档
│
├── LAMOST_DR10_LRS_*.fits       # LAMOST DR10 数据文件
├── VSX20230729_*.fits           # VSX 变星表
├── GCVS2022_reduced.fits        # GCVS 变星表
├── CV_catalog_output_*.xlsx     # CV 样本
├── MWDD-export_*.xlsx           # 白矮星表
├── NASA_*.fits                  # NASA 系外行星数据
├── pne.csv                      # 行星状星云表
├── catnorth_qso_cand_reduced.csv # QSO 候选体
├── TheGoldSample_447.fits       # CV 金样本
└── hlsp_atlas-var_reduced.fits  # ATLAS 变星
```

## 快速开始

### 方法1: 使用统一的 VSP 类 (推荐)

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib/vsp'))

from VSP import VSP, query_target

# 创建 VSP 实例
vsp = VSP()

# 执行完整查询
result = vsp.query_all(
    ra=13.1316273124,      # 赤经（度）
    dec=53.8584719271,     # 赤纬（度）
    radius=2.0,            # 搜索半径（角秒）
    name="EV_UMa"          # 目标名称（可选）
)

# 查询结果包含:
# - vsx: VSX 变星数据
# - gcvs: GCVS 变星数据
# - ebv_sfd: 消光值
# - gaia: Gaia DR3 数据
# - lamost: LAMOST 光谱数据
# - ztf: ZTF 光变数据
# - tess: TESS 数据
# - asassn: ASAS-SN 数据
# - photometry: 多波段测光数据
```

### 方法2: 使用便捷函数

```python
from VSP import query_target

# 一键查询
result = query_target(
    ra=274.0554,
    dec=49.8679,
    radius=2.0,
    name="AM_Her"
)
```

### 方法3: 单独查询特定数据库

```python
from VSP import (
    query_vsx_local,      # VSX 变星
    query_gcvs_local,     # GCVS 变星
    query_cv_local,       # CV 金样本
    query_atlas_local,    # ATLAS 变星
    find_LM_DR10LRS,      # LAMOST 光谱
    query_ztf,            # ZTF 光变
    query_tess,           # TESS
    query_asassn,         # ASAS-SN
    query_ebv_sfd,        # 消光
    get_gaia_dr3,         # Gaia
)

from astropy.coordinates import SkyCoord
from astropy import units as u

coord = SkyCoord(ra=13.13, dec=53.86, unit=(u.deg, u.deg))

# 查询 VSX
vsx_data = query_vsx_local(coord, radius=2.0)

# 查询 LAMOST
lamost_data = find_LM_DR10LRS(coord, radius=2.0)

# 查询消光
ebv = query_ebv_sfd(13.13, 53.86)
```

## 专项查询

### 变星数据库查询

```python
# 查询所有变星数据库
var_results = vsp.query_variable_star(ra, dec, radius=2.0)
# 返回: vsx, gcvs, atlas, cv_gold
```

### 光谱数据查询

```python
# 查询光谱数据库
spec_results = vsp.query_spectroscopic(ra, dec, radius=2.0)
# 返回: lamost 等
```

### 时域数据查询

```python
# 查询时域数据
ts_results = vsp.query_time_series(ra, dec, radius=2.0)
# 返回: ztf, tess, asassn
```

## 高级用法

### LAMOST 光谱下载和绘图

```python
from VSP import find_LM_DR10LRS, download_lamost_DR10, draw_spectrum

coord = SkyCoord(ra=13.13, dec=53.86, unit=(u.deg, u.deg))

# 查找 LAMOST 数据
result = find_LM_DR10LRS(coord, radius=2.0)
if result:
    idx_tbl, data_tbl, obsid_list = result
    obsid = obsid_list[0]
    
    # 下载光谱
    url, filepath = download_lamost_DR10(obsid, './spectra/')
    
    # 绘制光谱
    draw_spectrum(filepath, save_file='./spectrum.png')
```

### Gaia 数据和赫罗图

```python
from VSP import get_gaia_dr3, plot_cmd_target

coord = SkyCoord(ra=13.13, dec=53.86, unit=(u.deg, u.deg))

# 查询 Gaia DR3
gaia_data = get_gaia_dr3(coord, radius=2.0)

# 绘制赫罗图
plot_cmd_target(gaia_data, 'target_name', save_path='./cmd.png')
```

### ZTF 光变曲线

```python
from VSP import query_ztf, plot_ztf

# 查询 ZTF
ztf_data = query_ztf(ra=13.13, dec=53.86, radius=2.0)

# 绘制光变曲线
if ztf_data:
    plot_ztf(ztf_data, target_name='EV_UMa', save_file='./ztf.png')
```

## 数据文件说明

所有本地数据文件都位于 `lib/` 目录下：

| 文件 | 说明 | 用途 |
|------|------|------|
| `VSX20230729_*.fits` | VSX 变星表 | 变星识别 |
| `GCVS2022_reduced.fits` | GCVS 变星表 | 变星识别 |
| `LAMOST_DR10_LRS_*.fits` | LAMOST DR10 数据 | 光谱查询 |
| `CV_catalog_output_*.xlsx` | CV 样本 | CV 识别 |
| `MWDD-*.xlsx` | 白矮星表 | WD 识别 |
| `NASA_*.fits` | 系外行星数据 | 行星系统 |
| `pne.csv` | 行星状星云 | PNe 识别 |
| `catnorth_qso_cand_reduced.csv` | QSO 候选体 | 类星体识别 |
| `TheGoldSample_447.fits` | CV 金样本 | CV 识别 |
| `hlsp_atlas-var_reduced.fits` | ATLAS 变星 | 变星识别 |

## 注意事项

1. **路径设置**: 使用 VSP 前确保添加路径:
   ```python
   sys.path.insert(0, 'lib/vsp')
   ```

2. **依赖包**: VSP 需要以下 Python 包:
   - astropy
   - astroquery
   - numpy
   - pandas
   - matplotlib
   - scipy

3. **网络连接**: 部分功能需要网络连接 (Gaia, Vizier, TESS等)

4. **数据文件**: 确保所有 `.fits`, `.csv`, `.xlsx` 文件在 `lib/` 目录下

## 故障排除

### 问题: "File not found" 错误
**解决**: 检查数据文件是否在 `lib/` 目录下

### 问题: "Module not found" 错误
**解决**: 确保已添加 `lib/vsp` 到 Python 路径

### 问题: 查询返回空结果
**解决**: 
- 检查坐标是否正确
- 增大搜索半径
- 检查网络连接（在线查询）

## 示例脚本

参见 `examples/vsp_example.py` 获取完整示例。
