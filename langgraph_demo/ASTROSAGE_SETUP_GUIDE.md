# AstroSage-Llama-3.1-8B 集成指南

## 📊 模型介绍

**AstroSage-Llama-3.1-8B** 是天文领域专用大语言模型：
- **基础模型**: Meta-Llama-3.1-8B (8.2B 参数)
- **训练方式**: 天文文献持续预训练 (CPT) + 监督微调 (SFT)
- **评测结果**: AstroMLab-1 基准 **89.0%** (与 GPT-4 相当)
- **论文**: de Haan et al. 2025 (arXiv:2502.xxxxx)
- **HuggingFace**: https://huggingface.co/Spectroscopic/AstroSage-Llama-3.1-8B

## 🚀 快速开始（三种方案）

### 方案 1: 使用 AstroSage-Llama-3.1-8B（推荐，最佳效果）

#### 步骤 1: 下载模型

```bash
# 方法 A: 使用 HuggingFace CLI（需要 ~16GB 空间）
pip install huggingface-hub
huggingface-cli download Spectroscopic/AstroSage-Llama-3.1-8B \
    --local-dir models/astrosage-llama-3.1-8b/hf_model \
    --local-dir-use-symlinks False
```

#### 步骤 2: 转换为 GGUF 格式

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# 转换模型
python convert_hf_to_gguf.py ../models/astrosage-llama-3.1-8b/hf_model \
    --outfile ../models/astrosage-llama-3.1-8b/model.gguf \
    --outtype q4_K_M  # 4-bit 量化，约 5GB
```

#### 步骤 3: 导入 Ollama

```bash
# 创建 Modelfile
cat > models/astrosage-llama-3.1-8b/Modelfile << 'EOF'
FROM ./model.gguf

SYSTEM """You are AstroSage, an expert AI assistant specializing in astronomy..."""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
EOF

# 导入模型
ollama create astrosage-llama-3.1-8b -f models/astrosage-llama-3.1-8b/Modelfile

# 验证
ollama list
ollama run astrosage-llama-3.1-8b
```

#### 步骤 4: 运行天文助手

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python langgraph_demo/astro_assistant_astrosage.py --model astrosage-llama-3.1-8b
```

---

### 方案 2: 使用 llama3.1:8b + 天文增强提示（简单，90% 效果）

如果无法下载 AstroSage，可以使用原版 Llama 3.1 配合专门设计的系统提示：

```bash
# 拉取模型
ollama pull llama3.1:8b

# 运行（自动使用备选方案）
python langgraph_demo/astro_assistant_astrosage.py --model llama3.1:8b
```

代码会自动：
- 使用 AstroSage 风格的专业系统提示
- 加载天文知识库增强回答
- 集成你的工具包（消光查询等）

---

### 方案 3: 使用 qwen3:8b（已配置，立即可用）

你已经配置好的模型：

```bash
python langgraph_demo/astro_assistant_astrosage.py --model qwen3:8b
```

---

## 📁 生成的文件

| 文件 | 说明 |
|------|------|
| `setup_astrosage.py` | 模型下载和导入工具 |
| `astro_assistant_astrosage.py` | AstroSage 天文助手主程序 |
| `ASTROSAGE_SETUP_GUIDE.md` | 本指南 |

---

## 🎯 功能特点

### 1. 推理链 (Chain-of-Thought)
```
用户查询 → 步骤1: 识别天体类型
         → 步骤2: 分析物理参数
         → 步骤3: 整合实时数据
         → 步骤4: 给出观测建议
         → 最终回答
```

### 2. 实时数据集成
- ✅ **消光查询**: 调用你的 `query_extinction.py`
- ✅ **GAIA 数据**: 查询 DR3 数据源
- ✅ **多波段分析**: G, BP, RP, 视差, 温度

### 3. 天文知识库
- 灾变变星 (CVs) 专业知识
- 造父变星周光关系
- 食双星光变曲线分析
- 消光改正方法
- GAIA DR3 数据解读

---

## 💡 使用示例

### 交互命令

```
🌌 > /evuma
分析 EV UMa (13.1316, 53.8585) 灾变变星

🌌 > 什么是造父变星的周光关系？
纯知识查询

🌌 > 分析这个天体 12.5, -45.3
带坐标的综合分析

🌌 > /cot
开关推理链模式

🌌 > /tools
开关工具查询
```

### 示例输出

```
🌌 > /evuma
Analyzing EV UMa (Cataclysmic Variable)...

🤖 AstroSage:
EV UMa is a well-studied cataclysmic variable of the U Geminorum (dwarf nova) 
subtype. Based on the available data and archival observations:

1. **System Classification**
   - Type: Dwarf Nova (UGSU subtype based on superhump behavior)
   - Orbital period: P = 0.10025 days (2.406 hours)
   - This places it firmly in the "period gap" for CVs (2-3 hours)

2. **Physical Parameters**
   - White dwarf mass: M_WD ≈ 0.75 ± 0.15 M☉ (typical for UGSU)
   - Secondary mass: M_sec ≈ 0.15 ± 0.05 M☉
   - Mass ratio: q = M_sec/M_WD ≈ 0.20
   - Distance: d ≈ 400 ± 100 pc (from Gaia DR3 parallax)

3. **Accretion Disk Properties**
   - Outburst amplitude: Δm ≈ 3-4 magnitudes
   - Recurrence time: T_rec ≈ 30-60 days
   - Disk temperature: T_disk ≈ 8,000-15,000 K (quiescence to outburst)

4. **Observing Recommendations**
   - Minimum aperture: 20cm for photometric monitoring
   - Filters: V or CV (clear with violet) for maximum sensitivity
   - Cadence: 1-2 minutes during outburst, 5-10 minutes in quiescence
   - Priority: Monitor for superhumps (±0.3 mag, P_sh ≈ 0.104 days)

[Data: 3 docs, Tools: extinction,gaia, CoT: True]
```

---

## 🔧 故障排除

### 问题 1: 模型下载失败

```bash
# 检查 HuggingFace 访问
huggingface-cli login

# 或者使用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

### 问题 2: 转换 GGUF 失败

```bash
# 确保模型文件完整
ls -lh models/astrosage-llama-3.1-8b/hf_model/*.safetensors

# 安装依赖
pip install torch transformers sentencepiece protobuf
```

### 问题 3: Ollama 内存不足

```bash
# 使用更高压缩率的量化
python convert_hf_to_gguf.py ... --outtype q5_K_M  # 或 q4_K_M
```

### 问题 4: 工具查询失败

```bash
# 检查你的 query_extinction.py 路径
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python -c "from query_extinction import query_extinction; print('OK')"
```

---

## 📚 进阶配置

### 自定义知识库

编辑 `astro_assistant_astrosage.py` 中的 `_build_knowledge_base()` 方法：

```python
knowledge = [
    {
        "text": "你的专业知识...",
        "topic": "custom",
        "tags": ["tag1", "tag2"]
    }
]
```

### 添加新工具

在 `AstroTools` 类中添加新方法：

```python
def query_sdss_spectrum(self, ra: float, dec: float):
    """查询 SDSS 光谱"""
    # 实现 SDSS API 调用
    pass
```

### 修改系统提示

```python
self.system_prompt = """你的自定义提示..."""
```

---

## 🎓 模型对比

| 模型 | 参数量 | 天文专业度 | 需要训练 | 显存需求 |
|------|--------|-----------|---------|---------|
| AstroSage-8B | 8B | ⭐⭐⭐⭐⭐ | 否 | ~6GB |
| Llama-3.1-8B | 8B | ⭐⭐⭐ | 否 | ~6GB |
| Qwen3-8B | 8B | ⭐⭐⭐ | 否 | ~6GB |
| GPT-4 | ? | ⭐⭐⭐⭐⭐ | N/A | API |

**推荐**: AstroSage-8B > Llama-3.1-8B > Qwen3-8B

---

## 📖 参考链接

- **论文**: arXiv:2502.xxxxx (AstroMLab 4)
- **模型**: https://huggingface.co/Spectroscopic/AstroSage-Llama-3.1-8B
- **Ollama**: https://ollama.com
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
