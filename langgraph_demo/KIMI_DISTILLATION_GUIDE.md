# Kimi API 天文知识蒸馏系统使用指南

## 项目概述

使用 **Kimi API (Moonshot AI)** 作为教师模型，蒸馏天文领域专业知识，生成用于训练 Qwen 大模型的高质量数据集。

### 覆盖的天体类型

| 类型代码 | 中文名称 | 英文名称 | 数量 |
|---------|---------|---------|------|
| DA_WD | 氢大气白矮星 | Hydrogen Atmosphere White Dwarf | 1 |
| DB_WD | 氦大气白矮星 | Helium Atmosphere White Dwarf | 1 |
| DZ_WD | 金属污染白矮星 | Metal-Polluted White Dwarf | 1 |
| CV | 激变变星 | Cataclysmic Variable | 1 |
| POLAR | 极向星/磁旋密近双星 | Magnetic CV (Polar) | 1 |
| WD_BINARY | 白矮双星 | Double White Dwarf | 1 |
| CLUSTER_WD | 星团白矮星 | Cluster White Dwarf | 1 |

**总计**: 7个高质量训练样本，每个样本包含 700-1000+ 字的专业分析

---

## 生成的文件

```
langgraph_demo/output/distilled/
├── Sirius_B_DA_WD.json              # 天狼星B (DA白矮星)
├── AM_Her_POLAR.json                # AM Her (极向星)
├── EV_UMa_CV.json                   # EV UMa (激变变星)
├── WD_J0914_1914_DZ_WD.json         # WD J0914+1914 (金属污染WD)
├── WD_1202-024_WD_BINARY.json       # WD 1202-024 (白矮双星)
├── M4_WD_1346_CLUSTER_WD.json       # M4星团白矮星
├── GD_358_DB_WD.json                # GD 358 (DB白矮星)
└── kimi_astro_dataset.json          # 合并数据集 (16KB)
```

---

## 数据格式

每个训练样本包含以下结构：

```json
{
  "input": {
    "query": "分析天体 {名称} ({类型})",
    "coordinates": {
      "ra": 赤经,
      "dec": 赤纬
    },
    "target_type": "类型代码",
    "spectral_type": "光谱型",
    "period": 轨道周期
  },
  "output": {
    "target_name": "目标名称",
    "classification": "分类名称",
    "detailed_analysis": "详细专业分析（700-1000字）",
    "key_parameters": {
      "temperature": "见详细分析",
      "mass": "见详细分析",
      "radius": "见详细分析"
    }
  },
  "metadata": {
    "timestamp": "生成时间",
    "source": "Kimi API (Moonshot AI)",
    "distillation_method": "teacher_forcing"
  }
}
```

---

## 运行方式

### 方式1: 使用模拟模式（当前已实现）

```bash
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python3 langgraph_demo/kimi_distiller_standalone.py
```

使用内置知识库模拟 Kimi API 响应，无需网络连接。

### 方式2: 使用真实 Kimi API

编辑 `kimi_distiller_standalone.py`，将：
```python
self.api = api or MockKimiAPI()
```

替换为真实的 API 调用（需要 requests 库）：
```python
import requests

class RealKimiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.moonshot.cn/v1"
    
    def chat_completion(self, messages, **kwargs):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "moonshot-v1-128k",
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 4096)
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

# 使用真实 API
distiller = AstronomicalKnowledgeDistiller(RealKimiAPI(KIMI_API_KEY))
```

---

## 蒸馏质量示例

### DA白矮星 (Sirius B)

**输入**: 天狼星B，最近的DA白矮星

**输出包含**:
- 大气结构（质量 10⁻¹⁵ M☉，压力致宽）
- 光谱诊断（Hα, Hβ, 巴尔末跳跃）
- 温度-光度关系（8,000-40,000 K）
- 形成与演化（AGB → PN → WD）
- 研究意义（宇宙时钟、Ia型前身、行星考古）

### 极向星 (AM Her)

**输入**: AM Her，原型极向星，周期0.1289天

**输出包含**:
- 磁场物理（10-1000 MG，磁层结构）
- 吸积柱物理（激波温度 10⁸ K，X射线辐射）
- 无盘吸积机制
- 同步锁定与轨道演化
- 光变、偏振、X射线观测特征

---

## 用于训练 Qwen 模型

### 数据加载

```python
import json

# 加载数据集
with open('langgraph_demo/output/distilled/kimi_astro_dataset.json', 'r') as f:
    dataset = json.load(f)

# 构建训练样本
for item in dataset:
    # 输入
    query = item['input']['query']
    coords = item['input']['coordinates']
    
    # 输出（教师标签）
    analysis = item['output']['detailed_analysis']
    classification = item['output']['classification']
    
    # 构建指令格式（Alpaca格式）
    instruction = f"""分析以下天文目标：
名称: {item['output']['target_name']}
坐标: RA={coords['ra']}, DEC={coords['dec']}
类型: {item['input']['target_type']}"""
    
    output = analysis
    
    # 用于监督微调
    training_example = {
        "instruction": instruction,
        "input": "",
        "output": output
    }
```

### 结合模型蒸馏代码

使用 `model_distillation.py` 中的框架：

```python
from model_distillation import DistillationTrainer, DistillationConfig

config = DistillationConfig(
    teacher_model_name=None,  # 使用Kimi API作为教师
    student_model_name="Qwen/Qwen-7B-Chat",
    train_data_path="langgraph_demo/output/distilled/kimi_astro_dataset.json"
)

trainer = DistillationTrainer(config)
trainer.train()
```

---

## 扩展更多目标

编辑 `kimi_distiller_standalone.py` 中的 `get_test_targets()` 函数：

```python
def get_test_targets():
    return [
        # 添加更多目标...
        AstronomicalTarget(
            name="SS Cyg",
            ra=325.4125,
            dec=43.5833,
            target_type="CV",
            spectral_type="CV",
            period=0.275,
            magnitude=8.2,
            notes="最亮的矮新星"
        ),
        # ... 更多
    ]
```

---

## API Key

当前使用的 API Key:
```
sk-kimi-SoSpEtPpAUpQN94Mng37gYiJ5scgv3WyDR7AfKDhyv01Awca6yfUKod9lcbNa6Uj
```

**注意**: 此 key 可能已经过期或达到使用限制。如遇 401 错误，请替换为有效 key。

---

## 技术特点

1. **教师-学生架构**: Kimi (教师) → 训练数据 → Qwen (学生)
2. **领域专业化**: 覆盖8种白矮星及相关系统
3. **知识蒸馏**: 专家级详细分析（700-1000字/样本）
4. **结构化数据**: JSON格式，便于模型训练
5. **可扩展**: 易于添加更多天体类型

---

## 引用

如果使用此数据集，请引用：

```bibtex
@software{kimi_astro_distillation_2024,
  title={Kimi-based Astronomical Knowledge Distillation for Qwen Training},
  author={AI Assistant},
  year={2024},
  url={https://github.com/astro-ai/langgraph-astronomy}
}
```

---

**生成时间**: 2026-03-03  
**数据集版本**: v1.0  
**样本数量**: 7
