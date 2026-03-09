# 项目整理方案

## 🎯 核心保留文件（用于 ultimate_analysis_fixed.py）

### 1. 主脚本（保留）
- `ultimate_analysis_fixed.py` - 终极分析脚本 ⭐核心
- `ultimate_analysis.py` - 原始版本
- `requirements.txt` - Python依赖
- `setup.py` - 安装配置

### 2. src/ 目录 - 核心模块（保留）
必需模块：
- `astro_tools.py` - 基础天文工具（消光、测光查询）
- `extended_tools.py` - SIMBAD、TESS查询
- `sed_plotter.py` - SED分析
- `hr_diagram_plotter.py` - 赫罗图绘制
- `spectrum_analyzer.py` - 光谱分析
- `ollama_qwen_interface.py` - Ollama AI接口

可选模块（保留但非必需）：
- `complete_analysis_system.py` - 完整分析系统参考
- `integrated_analysis.py` - 整合分析参考

### 3. lib/vsp/ 目录 - VSP工具包（完整保留）
- 所有 VSP_*.py 文件
- `__init__.py`

### 4. data/ 目录 - 消光数据（保留）
- `csfd_ebv.fits` - CSFD消光地图
- `sfd_ebv.fits` - SFD消光地图
- `lss_intensity.fits` - LSS强度图
- `lss_error.fits` - LSS误差图
- `mask.fits` - 数据掩码

### 5. 模型文件（保留）
- `contamination_random_forest_*.joblib`
- `contamination_umap_reducer_*.joblib`
- `contamination_threshold_*.npy`

### 6. 文档（选择性保留）
- `README.md` - 项目说明
- `ASTROSAGE_VSP_README.md` - VSP集成文档
- `ZTF_USAGE.md` - ZTF使用说明

---

## 🗑️ 建议删除/移动的文件

### 1. 文献数据文件（大量 .ecsv）
```bash
# 2023A&A...*.ecsv - 天文文献数据（约90个文件）
# 2023AJ...*.ecsv
# 2023ApJ...*.ecsv
# ...等等
```
**建议**：移动到 `literature/` 子目录或删除

### 2. 测试文件
```bash
test_*.py           # 各种测试脚本
test_*.sh
test_*.log
```
**建议**：移动到 `tests/` 子目录

### 3. 部署日志
```bash
deploy_*.log
deploy.pid
```
**建议**：删除

### 4. 旧版本脚本
```bash
main.py
main_integrated.py
main_opcwd.py
ev_uma_analysis.py
ev_uma_final_demo.py
run_*.py            # 除 run_astrosage_now.py 外
```
**建议**：移动到 `archive/` 子目录

### 5. 图片文件
```bash
extinction_*.png    # 消光对比图（已在output中生成）
langgraph_workflow.png
langgraph_workflow.pdf
distillation_architecture.png
```
**建议**：移动到 `docs/figures/` 或删除

### 6. 开发文档（可选删除）
```bash
CLAUDE_ISSUE_REPORT.md
MODEL_EVALUATION_EXPLANATION.md
MODEL_QUALITY_ASSESSMENT.md
AI_RAG_ENHANCEMENT_GUIDE.md
功能对比.md
```
**建议**：移动到 `docs/internal/`

### 7. 缓存和临时文件
```bash
__pycache__/
*.pyc
*.log
cache/
```
**建议**：删除

### 8. 大数据文件
```bash
TAP_1632913590914.csv    # 33MB CSV文件
联合模型预测v20251211.3b1.ipynb    # 大notebook
```
**建议**：移动到 `data/large_files/` 或删除

---

## 📂 建议的目录结构

```
astro-ai-demo/
├── ultimate_analysis_fixed.py    # 主脚本
├── requirements.txt
├── setup.py
├── README.md
│
├── src/                          # 核心源代码
│   ├── astro_tools.py
│   ├── extended_tools.py
│   ├── sed_plotter.py
│   ├── hr_diagram_plotter.py
│   ├── spectrum_analyzer.py
│   └── ollama_qwen_interface.py
│
├── lib/
│   └── vsp/                      # VSP工具包
│
├── data/                         # 数据文件
│   ├── csfd_ebv.fits
│   ├── sfd_ebv.fits
│   └── ...
│
├── output/                       # 输出目录（自动生成）
│
├── docs/                         # 文档（整理后）
│   ├── ASTROSAGE_VSP_README.md
│   ├── ZTF_USAGE.md
│   └── figures/
│
├── tests/                        # 测试文件（移动后）
│
├── archive/                      # 旧版本（移动后）
│
└── .gitignore
```

---

## 🚀 清理命令

```bash
# 1. 创建归档目录
mkdir -p archive/literature archive/old_scripts tests docs/figures

# 2. 移动文献数据
mv 202*.ecsv archive/literature/ 2>/dev/null

# 3. 移动旧脚本
mv main*.py ev_uma*.py run_analysis.py run_quick_analysis.py archive/old_scripts/ 2>/dev/null

# 4. 移动测试文件
mv test_*.py test_*.sh tests/ 2>/dev/null

# 5. 移动图片
mv *.png *.pdf docs/figures/ 2>/dev/null

# 6. 删除日志和临时文件
rm -f *.log deploy.pid
rm -rf __pycache__ */__pycache__

# 7. 删除开发文档（可选）
mkdir -p docs/internal
mv CLAUDE_ISSUE_REPORT.md MODEL_*.md AI_RAG_*.md 功能对比.md docs/internal/ 2>/dev/null
```

---

## 📊 整理后预期效果

| 项目 | 整理前 | 整理后 |
|------|--------|--------|
| 根目录文件数 | ~150个 | ~20个 |
| 项目大小 | ~1.5GB | ~1.2GB |
| 核心文件 | 分散 | 集中 |
| 可读性 | 差 | 好 |
