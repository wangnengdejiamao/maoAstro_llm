# 数据目录

此目录存放项目运行所需的数据文件。

## ⚠️ 重要说明

由于数据文件体积较大，**这些文件没有包含在 Git 仓库中**。你需要通过以下方式获取：

## 快速下载

使用提供的下载脚本：

```bash
# 下载消光地图数据 (~800MB)
python download_data.py --extinction

# 下载模型文件 (~900MB)
python download_data.py --models

# 下载所有数据
python download_data.py --all
```

## 手动下载

### 1. 消光地图数据 (CSFD/SFD)

**必需文件：**

| 文件名 | 大小 | 说明 | 下载地址 |
|--------|------|------|----------|
| `csfd_ebv.fits` | ~200MB | CSFD E(B-V) 修正消光地图 | [CSFD GitHub](https://github.com/CPPMariner/CSFD) |
| `sfd_ebv.fits` | ~200MB | SFD E(B-V) 原始消光地图 | [Legacy Survey](https://www.legacysurvey.org/) |
| `lss_intensity.fits` | ~200MB | LSS 强度图 | 随消光地图提供 |
| `lss_error.fits` | ~200MB | LSS 误差图 | 随消光地图提供 |
| `mask.fits` | ~100MB | 数据质量掩码 | 随消光地图提供 |

**下载步骤：**

1. 访问 [CSFD GitHub 仓库](https://github.com/CPPMariner/CSFD)
2. 下载发布的消光地图文件
3. 解压后将 `.fits` 文件放入此目录

### 2. LAMOST DR10 星表数据

**文件位置：** `lib/` 目录（与代码仓库分开）

**数据内容：**
- LAMOST DR10 低分辨率光谱星表
- Gaia EDR3 白矮星星表
- GCVS 变星星表

**下载步骤：**

1. 访问 [LAMOST DR10 官网](http://www.lamost.org/dr10/v2.0/)
2. 注册账号并登录
3. 下载所需星表文件
4. 创建 `lib/` 目录并放入文件

> **注意：** LAMOST 数据总大小约 30GB，请确保有足够的磁盘空间。

### 3. 模型文件

**文件位置：** 项目根目录

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `contamination_umap_reducer_*.joblib` | ~900MB | UMAP 降维器 |
| `contamination_random_forest_*.joblib` | ~27MB | 随机森林分类器 |
| `contamination_threshold_*.npy` | ~1KB | 分类阈值 |

**获取方式：**
- 方式 1: 联系项目维护者获取
- 方式 2: 使用 `download_data.py --models` 下载
- 方式 3: 训练自己的模型（见 `docs/model_training.md`）

### 4. LLM 模型文件

**文件位置：** `models/` 目录

项目使用 Ollama 本地部署的 Qwen 模型，需要：

1. 安装 [Ollama](https://ollama.com/)
2. 拉取模型：
   ```bash
   ollama pull qwen3:8b
   ```

或配置其他支持的模型：
```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

## 数据组织结构

下载完成后，项目结构应如下：

```
astro-ai-demo/
├── data/                       # 消光地图数据
│   ├── csfd_ebv.fits
│   ├── sfd_ebv.fits
│   ├── lss_intensity.fits
│   ├── lss_error.fits
│   └── mask.fits
│
├── lib/                        # 星表数据（手动创建）
│   ├── LAMOST_DR10_LRS_CMD_Galactic_part01-12.fits
│   ├── LAMOST_DR10_LRS_CMD_Galactic_part02-12.fits
│   └── ...
│
├── models/                     # LLM 模型（Ollama 自动管理）
│   └── .cache/
│
├── contamination_*.joblib      # 污染检测模型（项目根目录）
├── contamination_*.npy
│
└── src/                        # 源代码
```

## 数据许可

使用这些数据时，请遵守相应的数据使用政策：

- **LAMOST 数据**: 遵守 [LAMOST 数据政策](http://www.lamost.org/dr10/v2.0/doc/data-policy)
- **Gaia 数据**: 遵守 [Gaia 使用条款](https://www.cosmos.esa.int/web/gaia-users/rights-and-obligations)
- **CSFD 数据**: 遵循相应许可证（通常为学术使用）

## 数据引用

如果在研究中使用了这些数据，请引用原始论文：

```bibtex
% LAMOST DR10
@article{lamost_dr10,
  title={LAMOST DR10: ...},
  author={...},
  journal={...},
  year={2024}
}

% CSFD
@article{csfd,
  title={CSFD: Corrected SFD Dust Map},
  author={...},
  journal={...},
  year={...}
}

% Gaia
@article{gaia_dr3,
  title={Gaia Data Release 3},
  author={Gaia Collaboration},
  journal={A&A},
  year={2022}
}
```

## 常见问题

### Q: 可以不下载这些数据吗？

A: 部分功能可以在没有本地数据的情况下运行：
- ✅ SIMBAD/VizieR 查询（在线）
- ✅ ZTF/TESS 光变曲线（在线）
- ✅ SED 分析（在线）
- ❌ 消光查询（需要本地消光地图）
- ❌ 赫罗图绘制（需要 LAMOST 星表）
- ❌ 污染检测（需要预训练模型）

### Q: 数据下载太慢怎么办？

A: 建议：
1. 使用迅雷等下载工具
2. 联系项目维护者获取百度网盘/阿里云盘链接
3. 在服务器上下载后传输到本地

### Q: 磁盘空间不足怎么办？

A: 最小运行配置：
- 仅消光查询：~1GB
- 完整功能：~35GB
- 可以只下载需要的部分

---

如有问题，请在 GitHub Issues 中提问。
