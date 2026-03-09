# 数据目录

此目录用于存放项目运行所需的消光地图数据文件。

## 📥 下载数据

### 快速下载（推荐）

```bash
# 下载所有消光文件（约 900MB）
python scripts/download_extinction.py
```

### 手动下载

如果自动下载失败，请访问以下链接手动下载：

1. **CSFD 消光地图**（推荐）
   - GitHub: https://github.com/CPPMariner/CSFD
   - 文件: `csfd_ebv.fits`, `lss_intensity.fits`, `lss_error.fits`, `mask.fits`

2. **SFD 原始地图**（备选）
   - GitHub: https://github.com/kbarbary/sfddata
   - 文件: `sfd_ebv.fits`

## 📋 文件清单

| 文件名 | 大小 | 说明 | 必需 |
|--------|------|------|------|
| `csfd_ebv.fits` | ~200 MB | CSFD E(B-V) 修正消光地图 | ⭐ 推荐 |
| `sfd_ebv.fits` | ~200 MB | SFD E(B-V) 原始消光地图 | 备选 |
| `lss_intensity.fits` | ~200 MB | LSS 强度图 | 可选 |
| `lss_error.fits` | ~200 MB | LSS 误差图 | 可选 |
| `mask.fits` | ~100 MB | 数据质量掩码 | 可选 |

## 🔍 验证下载

```bash
python scripts/check_setup.py
```

## 📖 使用说明

下载完成后，可以在代码中使用：

```python
from src.astro_tools import query_extinction

# 查询消光
result = query_extinction(ra=13.1316, dec=53.8585)
print(f"A_V = {result['av']:.3f} mag")
```

## 📚 数据引用

使用这些数据时，请引用：

```bibtex
@article{csfd2019,
  title={CSFD: Corrected SFD Dust Map},
  author={...},
  journal={...},
  year={2019}
}

@article{sfd1998,
  title={Maps of Dust Infrared Emission for Use in Estimation of Reddening and CMBR Foregrounds},
  author={Schlegel, D. J. and Finkbeiner, D. P. and Davis, M.},
  journal={ApJ},
  year={1998}
}
```

---

详细配置指南: [SETUP_GUIDE.md](../SETUP_GUIDE.md)
