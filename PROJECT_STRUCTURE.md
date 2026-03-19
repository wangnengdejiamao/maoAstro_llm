# Astro-AI 项目结构说明

## 🚀 快速开始

```bash
# 使用终极分析脚本
python ultimate_analysis_fixed.py --ra 196.9744 --dec 53.8585 --name "EV_UMa"

# 或使用启动脚本
./run_analysis.sh 196.9744 53.8585 EV_UMa
```

---

## 📂 项目结构

```
astro-ai-demo/
│
├── 🎯 核心文件
│   ├── ultimate_analysis_fixed.py    # ⭐ 主分析脚本（使用这个）
│   ├── ultimate_analysis.py          # 原始版本
│   ├── run_analysis.sh               # 便捷启动脚本
│   ├── requirements.txt              # Python依赖
│   └── setup.py                      # 安装配置
│
├── 📖 文档
│   ├── README.md                     # 项目说明
│   ├── ASTROSAGE_VSP_README.md       # VSP集成文档
│   ├── ZTF_USAGE.md                  # ZTF数据下载说明
│   ├── QUICK_START.md                # 快速开始
│   ├── OLLAMA_SETUP.md               # Ollama设置
│   └── INTEGRATED_SYSTEM_README.md   # 系统说明
│
├── 🔬 核心源代码 src/
│   ├── astro_tools.py                # 基础工具（消光、测光）
│   ├── extended_tools.py             # SIMBAD、TESS查询
│   ├── sed_plotter.py                # SED分析
│   ├── hr_diagram_plotter.py         # 赫罗图
│   ├── spectrum_analyzer.py          # 光谱分析
│   ├── ollama_qwen_interface.py      # AI接口
│   └── ...（其他参考模块）
│
├── 🛠️ VSP工具包 lib/vsp/
│   ├── VSP_Lib.py                    # 核心库
│   ├── VSP_GAIA.py                   # Gaia查询
│   ├── VSP_LAMOST.py                 # LAMOST数据
│   ├── VSP_TESS.py                   # TESS数据
│   ├── VSP_ZTF.py                    # ZTF数据
│   └── ...（其他VSP模块）
│
├── 💾 数据目录 data/
│   ├── csfd_ebv.fits                 # CSFD消光地图
│   ├── sfd_ebv.fits                  # SFD消光地图
│   └── ...（其他FITS文件）
│
├── 📤 输出目录 output/               # 分析结果（自动生成）
│
├── 📚 归档目录 archive/              # 非核心文件
│   ├── literature/                   # 文献数据(.ecsv)
│   └── old_scripts/                  # 旧版本脚本
│
├── 🧪 测试目录 tests/                # 测试脚本
│
└── 📊 文档目录 docs/
    ├── figures/                      # 图片文件
    ├── internal/                     # 开发文档
    └── ...（其他文档）
```

---

## ⚡ 核心功能

`ultimate_analysis_fixed.py` 提供12步完整分析：

1. ✅ 消光查询 (CSFD/SFD)
2. ✅ 测光查询 (多波段)
3. ✅ SIMBAD 数据库查询
4. ✅ Gaia DR3 查询
5. ✅ SDSS/LAMOST 光谱检查
6. ✅ SED 分析 + 黑体拟合
7. ✅ ZTF 光变曲线 (wget下载)
8. ✅ TESS 光变曲线
9. ✅ pj4 周期分析
10. ✅ 赫罗图 (LAMOST背景)
11. ✅ TPF 绘图
12. ✅ AI 智能分析 (Ollama)

---

## 📦 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- astropy, astroquery
- numpy, pandas, scipy
- matplotlib
- lightkurve
- requests

---

## 🔧 数据要求

必须数据文件（已包含在 data/）：
- `csfd_ebv.fits` - CSFD消光地图 (~200MB)
- `sfd_ebv.fits` - SFD消光地图 (~200MB)

---

## 💡 使用示例

### 分析 EV UMa
```bash
python ultimate_analysis_fixed.py --ra 196.9744 --dec 53.8585 --name "EV_UMa"
```

### 分析 Sirius
```bash
python ultimate_analysis_fixed.py --ra 101.2875 --dec -16.7161 --name "Sirius"
```

### 查看结果
```bash
cat output/EV_UMa_analysis.json
```

---

## 📝 文件统计

| 目录 | 文件数 | 说明 |
|------|--------|------|
| 根目录 | ~10 | 核心启动文件 |
| src/ | ~25 | 源代码模块 |
| lib/vsp/ | ~13 | VSP工具包 |
| data/ | 5 | FITS数据文件 |
| archive/ | 100+ | 归档文件 |
| docs/ | 10+ | 文档和图片 |

---

## 🗑️ 已归档文件

以下文件已移动到 `archive/` 或 `docs/`：

- 文献数据 (.ecsv) → `archive/literature/`
- 旧版脚本 → `archive/old_scripts/`
- 测试文件 → `tests/`
- 图片文件 → `docs/figures/`
- 开发文档 → `docs/internal/`

如需恢复，请从相应目录复制回来。
