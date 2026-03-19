# Astro-AI 整合系统使用说明

## 系统概述

本整合系统将 Ollama Qwen、SED 画图、赫罗图、TPFPlotter 等功能整合在一起，提供一站式天体分析能力。

## 新增文件

```
src/
├── ollama_qwen_interface.py    # Ollama Qwen 接口
├── sed_plotter.py              # 修复的 SED 画图（参考 VSP_Photometry）
├── hr_diagram_plotter.py       # 修复的赫罗图（LAMOST 背景）
├── integrated_analysis.py      # 完整整合分析（含 TESS）

run_analysis.py                 # 主运行脚本（完整版）
run_quick_analysis.py           # 快速运行脚本（跳过 TESS）
```

## 快速开始

### 1. 快速分析（推荐）

跳过耗时的 TESS 数据下载，专注于核心分析：

```bash
python run_quick_analysis.py --target "AM_Her" --ra 274.0554 --dec 49.8679
```

### 2. 完整分析（包含 TESS）

```bash
python run_analysis.py --target "AM_Her" --ra 274.0554 --dec 49.8679 --quick
```

### 3. 不使用 Ollama（仅本地规则分析）

```bash
python run_quick_analysis.py --target "AM_Her" --ra 274.0554 --dec 49.8679 --no-ollama
```

## 功能说明

### 1. Ollama Qwen 接口 (`ollama_qwen_interface.py`)

- 与本地 Ollama 服务通信
- 支持文本分析和图像分析
- 专门针对天文图像的 AI 分析

**使用方法**:
```python
from src.ollama_qwen_interface import OllamaQwenInterface

interface = OllamaQwenInterface(model_name="qwen3:8b")

# 文本分析
result = interface.analyze_text("分析激变变星的特征")

# 图像分析
result = interface.analyze_sed_plot("path/to/sed.png")
```

### 2. SED 画图 (`sed_plotter.py`)

修复了原版的 SED 参数计算错误：
- 正确计算波长：频率(GHz) → 波长(Å)
- 正确计算流量：F_ν → F_λ
- 参考 VSP_Photometry.py 的实现

**使用方法**:
```python
from src.sed_plotter import plot_sed_fixed

result = plot_sed_fixed(
    ra=274.0554, 
    dec=49.8679, 
    name="AM_Her",
    output_dir='./output'
)
```

### 3. 赫罗图 (`hr_diagram_plotter.py`)

使用 LAMOST DR10 数据作为背景：
- 加载 LM_DR10_FOR_CMD_every30.fits
- 正确的消光系数（参考 VSP_GAIA.py）
- 自动计算绝对星等

**使用方法**:
```python
from src.hr_diagram_plotter import plot_hr_diagram_fixed

plot_hr_diagram_fixed(
    ra=274.0554,
    dec=49.8679,
    name="AM_Her",
    extinction={'success': True, 'A_V': 1.12, 'E_B_V': 0.36}
)
```

### 4. TPF 绘图 (`hr_diagram_plotter.py`)

使用 lightkurve 绘制 TESS TPF：

```python
from src.hr_diagram_plotter import plot_tpf

plot_tpf(ra=274.0554, dec=49.8679, name="AM_Her")
```

## 输出文件

分析完成后，在 `output/` 目录下生成：

```
output/
├── figures/
│   ├── {name}_sed.png              # SED 图
│   ├── {name}_hr_diagram.png       # 赫罗图
│   ├── {name}_TPF.png              # TPF 图（完整版）
│   └── {name}_quick_summary.png    # 汇总图
├── data/
│   ├── sed_vizier.csv              # VizieR SED 原始数据
│   └── {name}_sed_processed.csv    # 处理后的 SED 数据
├── {name}_quick_analysis.json      # 分析结果 JSON
└── {name}_ai_analysis.txt          # AI 分析文本
```

## 修复的问题

### 1. SED 参数不准确

**原问题**: 
- 波长计算错误（频率单位混淆）
- 流量单位转换错误

**修复**:
- `sed_freq` 单位是 GHz，不是 Hz
- 正确转换：λ(Å) = 2.998e9 / ν(GHz)
- F_λ = F_ν × c / λ²

### 2. 赫罗图背景

**修复**:
- 使用 lib/LM_DR10_FOR_CMD_every30.fits 作为背景
- 正确的消光系数：R_G=2.364, R_BP=2.998, R_RP=1.737

## 依赖要求

```
- ollama (本地服务)
- qwen3:8b 模型（或其他 Ollama 模型）
- lightkurve
- astropy
- matplotlib
- numpy
- pandas
- requests
```

## 注意事项

1. **Ollama 服务**: 确保 Ollama 服务正在运行 (`ollama serve`)
2. **网络连接**: VizieR SED 下载需要网络
3. **Gaia 服务**: 赫罗图目标星位置依赖 Gaia 在线服务
4. **TESS 数据**: 完整分析模式下，TESS 下载可能耗时较长

## 示例输出

分析 AM Her (激变变星):

```
🔭 快速天体分析: AM_Her
   坐标: RA=274.055400°, DEC=49.867900°

【1/5】基础数据查询...
    ✓ 消光: A_V=1.117, E(B-V)=0.360

【2/5】SIMBAD 查询...
    ✓ 匹配: V* AM Her
    ✓ 类型: CataclyV*

【3/5】SED 分析...
    ✓ SED 数据点: 37

【4/5】赫罗图...
    ✓ 赫罗图已生成

【5/5】AI 分析...
    ✓ AI 分析完成

✅ 分析完成!
```

## 下一步开发建议

1. 添加 ZTF 光变曲线折叠分析
2. 添加 VSX 变星表查询
3. 优化 AI 分析的提示词
4. 添加更多背景星表选择
5. 支持批量目标分析
