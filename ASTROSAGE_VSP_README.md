# AstroSage + VSP 集成系统

将本地部署的 `astrosage-local` 模型与 VSP (Variable Star Pipeline) 集成，实现输入 RA/DEC 坐标自动查询天文数据并进行 AI 分析。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `astrosage_with_vsp.py` | 主程序，提供交互式查询界面 |
| `query_ra_dec.py` | 命令行快速查询工具 |
| `test_astrosage_vsp.py` | 系统测试脚本 |
| `Modelfile.astrosage-vsp` | Ollama 模型配置文件 |

## 🚀 快速开始

### 1. 确保环境就绪

```bash
# 检查 Ollama 服务
ollama list

# 确保 astrosage-local 模型存在
# 应该看到类似输出：
# NAME                   ID              SIZE      MODIFIED
# astrosage-local:latest 6d33b4e237a3    4.9 GB    7 seconds ago
```

### 2. 运行测试

```bash
# 运行所有测试
python test_astrosage_vsp.py

# 或测试特定模块
python test_astrosage_vsp.py vsp      # 测试 VSP 模块
python test_astrosage_vsp.py ollama   # 测试 Ollama 连接
python test_astrosage_vsp.py coord    # 测试坐标解析
python test_astrosage_vsp.py query    # 测试完整查询
```

### 3. 使用交互式界面

```bash
python astrosage_with_vsp.py
```

**使用示例：**

```
======================================================================
🌟 AstroSage + VSP 智能天文助手
======================================================================
✓ VSP 模块已加载
✓ Ollama 服务正常，模型 astrosage-local 可用

======================================================================
✓ 系统初始化完成！
======================================================================

======================================================================
💡 使用说明:
   • 直接输入问题: 询问天文知识
   • 输入坐标: 如 '分析 13.1316, 53.8585' 或 '150.5 23.8'
   • 命令:
     /help  - 显示帮助
     /clear - 清屏
     /exit  - 退出
======================================================================

🌟 你: 分析 13.1316, 53.8585

📍 识别到坐标: RA=13.1316, DEC=53.8585

🔍 正在查询 VSP 数据库: RA=13.131600, DEC=53.858500
------------------------------------------------------------
============================================================
VSP 天体数据查询结果
============================================================
📍 坐标: RA=13.1316, DEC=53.8585

【VSX 变星数据库】
  ✓ 找到变星: EV UMa
  变星类型: EW
  周期: 0.4 天

【消光】
  E(B-V) = 0.0234 (SFD)
  A_V ≈ 0.0725

【Gaia DR3】
  ✓ 找到 1 个 Gaia 源
  Source ID: 123456789012345
  G 星等: 12.34
  视差: 5.67 mas
  距离: 176.4 pc
  ...

🤖 正在使用 AI 分析数据...
------------------------------------------------------------

🤖 AstroSage:
[AI 分析结果...]
```

### 4. 使用命令行工具

```bash
# 基本查询
python query_ra_dec.py 13.1316 53.8585

# 指定搜索半径
python query_ra_dec.py 150.5 23.8 --radius 10

# 只查询数据，不调用 AI
python query_ra_dec.py 200.0 -30.0 --no-ai

# 保存结果到文件
python query_ra_dec.py 13.1316 53.8585 --save ./ev_uma_analysis.txt
```

## 🎯 支持的坐标格式

系统可以自动识别多种坐标输入格式：

```
# 标准格式
分析 13.1316, 53.8585
查询 150.5 23.8

# 带关键词
RA=150.5 DEC=23.8
ra=150.5 dec=23.8

# 直接输入数字（以分析/查询开头）
分析 150.5 23.8
查询 200.5 -30.2

# 逗号分隔
150.5, 23.8
200.0,-30.0
```

## 📊 查询的数据源

输入 RA/DEC 后，系统会自动查询以下数据库：

| 数据源 | 数据内容 |
|--------|----------|
| **VSX** | 变星名称、类型、周期、星等 |
| **Gaia DR3** | 视差、自行、测光、距离、温度 |
| **LAMOST** | 光谱数据 |
| **SIMBAD** | 天体分类、光谱型 |
| **ZTF** | 光变曲线数据 |
| **TESS** | 空间光变数据 |
| **消光** | E(B-V), A_V |

## 🧠 AI 分析内容

获取数据后，AI 会分析并给出：

1. **天体类型判断** - 基于数据的分类识别
2. **物理参数分析** - 距离、光度、温度、消光
3. **变星特征分析** - 周期、振幅、光变机制
4. **科学价值评估** - 研究意义
5. **观测建议** - 后续观测方案

## 🔧 高级配置

### 创建优化版 Ollama 模型

```bash
# 使用提供的 Modelfile 创建增强版模型
ollama create astrosage-vsp -f Modelfile.astrosage-vsp

# 使用新模型运行
ollama run astrosage-vsp
```

### 修改默认参数

编辑 `astrosage_with_vsp.py` 中的 `OllamaInterface` 类：

```python
ollama = OllamaInterface(
    model_name="astrosage-local",  # 改为你的模型名
    base_url="http://localhost:11434"  # Ollama 地址
)
```

## 🐛 故障排除

### VSP 模块加载失败

```bash
# 检查 VSP 路径
python -c "import sys; sys.path.insert(0, './lib/vsp'); from lib.vsp import VSP; print('OK')"

# 安装依赖
pip install astropy astroquery numpy pandas matplotlib
```

### Ollama 连接失败

```bash
# 检查 Ollama 服务
ollama serve

# 测试 API
curl http://localhost:11434/api/tags
```

### 查询超时

- 检查网络连接
- 部分服务（Gaia）可能需要代理
- 使用 `--no-ai` 先测试数据查询

## 📝 示例天体坐标

| 名称 | RA | DEC | 类型 |
|------|-----|-----|------|
| EV UMa | 13.1316 | 53.8585 | 食双星 |
| Sirius | 101.2875 | -16.7161 | 主序星 |
| M31 | 10.6847 | 41.2687 | 星系 |
| M33 | 23.4621 | 30.6602 | 星系 |
| AM CVn | 230.236 | 36.133 | 激变变星 |

## 📚 相关文件

- `lib/vsp/` - VSP 核心模块
- `src/unified_astro_query.py` - 统一天文查询接口
- `src/ollama_qwen_interface.py` - Ollama 接口

## 🤝 集成到现有系统

如果你想将 VSP 查询集成到其他应用：

```python
from astrosage_with_vsp import VSPAnalyzer, OllamaInterface

# 查询数据
vsp = VSPAnalyzer()
results = vsp.query_source(ra=13.1316, dec=53.8585)

# 生成分析提示词
prompt = vsp.generate_analysis_prompt(results)

# 调用 AI
ollama = OllamaInterface(model_name="astrosage-local")
analysis = ollama.generate(prompt)

print(analysis)
```

## 📄 License

与主项目保持一致。
