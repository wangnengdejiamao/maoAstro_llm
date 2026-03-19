# 🚀 Astro-AI 快速开始指南

## 1分钟快速上手

### 第一步：运行分析

```bash
# 分析指定坐标（示例：MyStar）
python src/astro_agent.py --ra 123.456 --dec 67.890 --name "MyStar"
```

**输出示例**：
```
🔭 分析天体: MyStar
   坐标: RA=123.456°, DEC=67.89°
============================================================

[Step 1] 查询消光...
   ✓ A_V = 0.106, E(B-V) = 0.034

[Step 2] 查询ZTF...

[Step 3] 查询测光...
   ✓ 测光目录: ['Gaia', '2MASS', 'AllWISE']

[Step 4] 检索相关知识...
   ✓ 检索到 466 条相关知识

[Step 5] AI综合分析...

✅ 分析完成!
   结果保存: ./output/MyStar_analysis.json
```

### 第二步：查看结果

```bash
# 查看详细结果
python src/view_results.py --name MyStar

# 或者查看所有结果
python src/view_results.py --all
```

**显示内容**：
- ✅ 银河消光: A_V = 0.106 mag
- ✅ 距离估计: ~100-500 pc
- ✅ 测光匹配: AllWISE成功
- ⚠️ ZTF数据: 无数据

### 第三步：生成图表

```bash
# 生成可视化图表
python src/visualize.py --name MyStar
```

**生成文件**：
- `output/MyStar_summary.png` - 汇总图表

### 第四步：查看图表

图表包含：
1. 基本信息面板 - 坐标、消光、距离估计
2. 数据状态面板 - 各数据源可用性
3. 测光分布面板 - 各巡天匹配情况
4. AI分析面板 - 智能分析摘要

---

## 📂 文件说明

### 输出文件

```
output/
├── MyStar_analysis.json       # 完整数据（JSON格式）
├── MyStar_summary.png         # 可视化图表
└── ...
```

### JSON数据内容

```json
{
  "name": "MyStar",              // 目标名称
  "ra": 123.456,                 // 赤经（度）
  "dec": 67.89,                  // 赤纬（度）
  "timestamp": "...",            // 分析时间
  "extinction": {                // 消光数据
    "A_V": 0.106,                // V波段消光
    "E_B_V": 0.034               // 色余
  },
  "ztf": {...},                  // ZTF光变曲线
  "photometry": {...},           // 多波段测光
  "analysis": "..."              // AI分析结果
}
```

---

## 🎯 常用命令

### 分析不同坐标

```bash
# AM Her (著名的Polar)
python src/astro_agent.py --ra 274.0554 --dec 49.8679 --name "AM_Her"

# RR Lyrae 示例（假设坐标）
python src/astro_agent.py --ra 180.0 --dec 30.0 --name "RR_Lyr"

# 白矮星示例
python src/astro_agent.py --ra 150.0 --dec 20.0 --name "White_Dwarf"
```

### 批量分析

```bash
# 创建坐标列表文件 targets.txt
# 格式: RA DEC NAME
cat > targets.txt << EOF
274.0554 49.8679 AM_Her
123.456 67.890 MyStar1
150.0 20.0 WD_Test
EOF

# 批量分析
while read ra dec name; do
    python src/astro_agent.py --ra $ra --dec $dec --name $name --no-qwen
done < targets.txt
```

### 结果管理

```bash
# 列出所有分析结果
python src/view_results.py --list

# 导出CSV摘要
python src/view_results.py --export

# 生成所有图表
python src/visualize.py --all
```

---

## 🔍 查看结果的方法

### 方法1: 命令行查看

```bash
python src/view_results.py --name MyStar
```

### 方法2: 直接查看JSON

```bash
cat output/MyStar_analysis.json | python -m json.tool
```

### 方法3: 查看图表

```bash
# 在Linux/WSL中
xdg-open output/MyStar_summary.png

# 或在Windows中
start output/MyStar_summary.png
```

### 方法4: Python读取

```python
import json

with open('output/MyStar_analysis.json', 'r') as f:
    data = json.load(f)

print(f"消光 A_V: {data['extinction']['A_V']}")
print(f"ZTF点数: {data['ztf'].get('n_points', 'N/A')}")
```

---

## ⚙️ 高级用法

### 使用不同输出目录

```bash
python src/astro_agent.py --ra 123.456 --dec 67.890 \
    --name "MyStar" \
    --output-dir "./my_results"
```

### 启用完整AI分析（需要Qwen模型）

```bash
# 安装依赖
pip install tiktoken

# 运行分析（自动使用Qwen）
python src/astro_agent.py --ra 123.456 --dec 67.890 --name "MyStar"
```

### 仅运行演示

```bash
# AM Her演示
python src/astro_agent.py --demo
```

---

## 🛠️ 故障排除

### 问题："消光文件未找到"

**解决**：确保消光文件存在
```bash
ls -la data/sfd_ebv.fits
# 或
ls -la ../Astro_qwen/csfd_ebv.fits
```

### 问题："ZTF无数据"

**原因**：该坐标可能没有ZTF观测
**解决**：尝试其他坐标，如AM Her
```bash
python src/astro_agent.py --ra 274.0554 --dec 49.8679 --name "AM_Her"
```

### 问题："Qwen模型未找到"

**原因**：模型路径不正确
**解决**：下载模型
```bash
python download_model.py
```

### 问题：图表中文显示为方块

**原因**：matplotlib字体问题
**解决**：修改visualize.py中的字体设置，或使用英文标签

---

## 📊 数据解读

### 消光值 A_V

| A_V 值 | 距离估计 | 说明 |
|--------|----------|------|
| < 0.1 | < 100 pc | 本地泡内 |
| 0.1-0.5 | 100-500 pc | 近距天体 |
| 0.5-1.0 | 500 pc - 1 kpc | 中距离 |
| 1.0-2.0 | 1-2 kpc | 远距离 |
| > 2.0 | > 2 kpc | 银道面方向 |

### 测光匹配

- ✓ **Found**: 成功匹配到数据
- ✗ **Not found**: 未匹配（可能太暗或超出巡天范围）
- **Error**: 查询出错

### ZTF数据

- **n_points**: 数据点数量（>50为良好）
- **period_hours**: 光变周期（可用于变星分类）
- **amplitude**: 振幅（星等变化幅度）

---

## 🎓 学习路径

### 初学者

1. 运行 `--demo` 查看AM Her分析
2. 尝试几个不同的坐标
3. 查看生成的JSON和PNG文件
4. 阅读 `TEST_REPORT.md` 了解功能

### 进阶用户

1. 阅读源代码 `src/astro_agent.py`
2. 了解RAG知识库 `src/rag_system.py`
3. 自定义天文工具 `src/astro_tools.py`
4. 扩展新的数据源

### 开发者

1. 阅读LaTeX论文 `docs/article.tex`
2. 了解系统架构
3. 修改可视化样式
4. 贡献代码

---

## 📞 获取帮助

### 查看帮助信息

```bash
# 主程序帮助
python src/astro_agent.py --help

# 结果查看帮助
python src/view_results.py --help

# 可视化帮助
python src/visualize.py --help
```

### 文档索引

- `README.md` - 项目说明
- `QUICK_START.md` - 本文件
- `TEST_REPORT.md` - 测试报告
- `PROJECT_SUMMARY.md` - 项目总结
- `docs/article.tex` - LaTeX论文

---

## ✅ 快速检查清单

- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 运行分析: `python src/astro_agent.py --ra 123.456 --dec 67.890 --name "Test"`
- [ ] 查看结果: `python src/view_results.py --name Test`
- [ ] 生成图表: `python src/visualize.py --name Test`
- [ ] 检查输出: `ls -la output/`

---

**祝使用愉快！** 🔭✨
