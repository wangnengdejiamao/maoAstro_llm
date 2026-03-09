# Astro AI 助手 - 具体代码问题报告

> 本报告整理当前系统中的具体代码问题和测试问题，供 Claude 修复参考。

---

## 🔴 严重问题（需要立即修复）

### 问题 1: `AstroTools.query_extinction()` 接口不匹配

**位置**: `langgraph_demo/astro_assistant_astrosage.py` 第 76-83 行

**问题描述**: 
`AstroTools.query_extinction()` 方法调用 `query_extinction(ra, dec)`，但 `query_extinction.py` 中的函数签名需要 3 个参数：
```python
# astro_assistant_astrosage.py 中的调用
return self._extinction(ra, dec)  # 缺少 maps 参数

# query_extinction.py 中的实际函数签名
def query_extinction(ra_deg, dec_deg, maps):
```

**代码问题**:
```python
# 当前代码 (astro_assistant_astrosage.py)
def query_extinction(self, ra: float, dec: float) -> Optional[Dict]:
    if self._extinction:
        try:
            return self._extinction(ra, dec)  # ❌ 缺少 maps 参数
        except Exception as e:
            return {"error": str(e)}
    return None
```

**修复建议**:
1. 在 `AstroTools.__init__` 中预加载 `maps` 数据
2. 或修改 `query_extinction()` 调用方式

---

### 问题 2: `ollama_rag_light.py` 相同问题

**位置**: `langgraph_demo/ollama_rag_light.py` 第 95-104 行

**问题描述**:
`AstronomyTools.get_extinction()` 同样没有传入 `maps` 参数。

```python
def get_extinction(self, ra: float, dec: float) -> Optional[Dict]:
    if self.extinction_query is None:
        return None
    try:
        result = self.extinction_query(ra, dec)  # ❌ 缺少 maps 参数
        return result
    except Exception as e:
        print(f"  消光查询错误: {e}")
        return None
```

---

### 问题 3: `rag_tool_system.py` 相同问题

**位置**: `langgraph_demo/rag_tool_system.py` 第 235-256 行

**问题描述**:
`AstronomyTools.query_extinction_data()` 同样调用方式错误。

```python
def query_extinction_data(self, ra: float, dec: float) -> Dict:
    if self.query_extinction is None:
        return {"tool": "extinction_query", "success": False, "error": "..."}
    try:
        result = self.query_extinction(ra, dec)  # ❌ 缺少 maps 参数
        return {"tool": "extinction_query", "success": True, "data": result}
    except Exception as e:
        return {"tool": "extinction_query", "success": False, "error": str(e)}
```

---

## 🟡 中等问题（影响功能完整性）

### 问题 4: RAG 关键词检索过于简单

**位置**: `langgraph_demo/astro_assistant_astrosage.py` 第 49-57 行

**问题描述**:
当前 RAG 只有简单的关键词匹配，没有 TF-IDF 权重，也没有语义检索能力。

```python
def search(self, query: str, k: int = 3) -> List[Dict]:
    words = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', query.lower()))
    scores = {}
    for w in words:
        if w in self.index:
            for idx in self.index[w]:
                scores[idx] = scores.get(idx, 0) + 1  # 简单计数，无权重
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{**self.docs[i], 'score': s} for i, s in top]
```

**期望行为**:
1. 添加 TF-IDF 权重计算
2. 支持同义词扩展
3. 考虑添加轻量级向量检索（可选）

---

### 问题 5: GAIA 查询超时处理不完善

**位置**: `langgraph_demo/astro_assistant_astrosage.py` 第 85-119 行

**问题描述**:
1. 超时时间固定为 30 秒，没有重试机制
2. 错误信息不够详细
3. 没有备用查询端点

```python
def query_gaia(self, ra: float, dec: float, radius: float = 0.1) -> List[Dict]:
    try:
        # ...
        r = requests.get(url, params={...}, timeout=30)  # 固定超时，无重试
        # ...
    except Exception as e:
        print(f"  GAIA 查询错误: {e}")  # 错误信息过于简单
        return []
```

**期望行为**:
1. 添加指数退避重试（最多 3 次）
2. 支持多个 TAP 端点（Heidelberg, ESA, ARI）
3. 更详细的错误分类

---

### 问题 6: `query_extinction.py` 全局变量问题

**位置**: `query_extinction.py` 第 21-24 行

**问题描述**:
使用全局变量存储坐标，不利于作为模块导入使用。

```python
# ========== 用户坐标 ==========
RA_DEG = 13.1316273124      # 赤经 (度)
DEC_DEG = 53.8584719271     # 赤纬 (度)
# ==============================
```

当作为模块导入时，这些全局变量没有意义。

**期望行为**:
1. 移除全局坐标变量
2. 提供默认参数或强制要求传入参数
3. 添加 `__all__` 导出列表

---

## 🟢 小问题（代码质量改进）

### 问题 7: 重复代码过多

**三个文件中都有几乎相同的 RAG 实现**:
- `astro_assistant_astrosage.py`: `AstroRAG` 类
- `ollama_rag_light.py`: `SimpleRAG` 类
- `rag_tool_system.py`: 使用 FAISS 的向量检索

**期望**: 提取公共模块，避免重复代码。

---

### 问题 8: 硬编码路径过多

**位置**: 多个文件

```python
# astro_assistant_astrosage.py 第 244 行
train_path = "langgraph_demo/output/training_dataset.json"

# ollama_rag_light.py 第 237 行
data_path = "langgraph_demo/output/training_dataset.json"

# rag_tool_system.py 第 425 行
docs_path = "langgraph_demo/output/kimi_generated/rag_documents.json"
```

**问题**:
1. 路径硬编码，不易移植
2. 没有检查文件存在性
3. 没有配置化

---

### 问题 9: 模型名称检查不严格

**位置**: `astro_assistant_astrosage.py` 第 458-470 行

```python
def check_model_available(model_name: str) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            for m in models:
                if model_name in m or m in model_name:  # ❌ 过于宽松的匹配
                    return True
    except:
        pass
    return False
```

**问题**: 使用子串匹配可能导致误判，如 "llama" 会匹配 "llama3.1" 和 "llama2"。

---

## 🔬 测试问题

### 测试 1: `test_astrosage_vsp.py` 缺少对 `astro_assistant_astrosage` 的测试

**当前测试覆盖**:
- ✓ VSP 模块加载
- ✓ Ollama 连接
- ✓ 坐标解析
- ✓ 查询流程

**缺少测试**:
- ✗ `AstroRAG.search()` 功能测试
- ✗ `AstroTools.query_extinction()` 功能测试
- ✗ `AstroSageAssistant.query()` 集成测试

---

### 测试 2: `test_fixes.py` 引用了不存在的模块

**位置**: `test_fixes.py` 第 21, 45, 69, 81 行

```python
from spectrum_analyzer import SpectrumAnalyzer        # 不存在
from ollama_qwen_interface import OllamaQwenInterface # 不存在
from hr_diagram_plotter import TPFPlotter             # 不存在
from integrated_analysis import IntegratedAstroAnalysis # 不存在
```

**问题**: 这些模块在 `src/` 目录下不存在，测试会直接失败。

---

### 测试 3: 缺少端到端测试

**缺少场景**:
1. 完整查询流程测试（输入 → RAG → Tool → LLM → 输出）
2. 错误恢复测试（网络中断、模型不可用等）
3. 并发测试（多个查询同时处理）

---

## 📋 给 Claude 的修复任务清单

### 任务 1: 修复消光查询接口（优先级：🔴 高）

**需要修改的文件**:
1. `langgraph_demo/astro_assistant_astrosage.py`
2. `langgraph_demo/ollama_rag_light.py`
3. `langgraph_demo/rag_tool_system.py`

**修复方案**:
```python
# 方案 A: 在 __init__ 中预加载 maps
class AstroTools:
    def __init__(self):
        self._extinction = None
        self._maps = None  # 添加 maps 缓存
        self._load_tools()
    
    def _load_tools(self):
        try:
            from query_extinction import query_extinction, load_extinction_maps
            self._extinction = query_extinction
            self._maps = load_extinction_maps(data_dir='data')  # 预加载
            print("✓ 消光查询工具已加载")
        except Exception as e:
            print(f"⚠ 消光工具加载失败: {e}")
    
    def query_extinction(self, ra: float, dec: float) -> Optional[Dict]:
        if self._extinction and self._maps:
            try:
                return self._extinction(ra, dec, self._maps)  # 传入 maps
            except Exception as e:
                return {"error": str(e)}
        return None
```

---

### 任务 2: 改进 RAG 检索算法（优先级：🟡 中）

**目标**: 添加 TF-IDF 权重和同义词支持

**参考实现**:
```python
class ImprovedRAG:
    def __init__(self):
        self.docs = []
        self.index = {}
        self.doc_freq = {}  # 文档频率
        self.total_docs = 0
    
    def add(self, documents: List[Dict]):
        # 现有代码...
        
        # 计算 IDF
        for word in set(words):
            self.doc_freq[word] = self.doc_freq.get(word, 0) + 1
        self.total_docs = len(self.docs)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        words = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', query.lower()))
        scores = {}
        for w in words:
            if w in self.index:
                # TF-IDF 权重
                idf = math.log(self.total_docs / (self.doc_freq.get(w, 1) + 1))
                for idx in self.index[w]:
                    scores[idx] = scores.get(idx, 0) + idf
        # ...
```

---

### 任务 3: 添加 GAIA 查询重试机制（优先级：🟡 中）

**参考实现**:
```python
def query_gaia(self, ra: float, dec: float, radius: float = 0.1, max_retries: int = 3) -> List[Dict]:
    endpoints = [
        "http://dc.zah.uni-heidelberg.de/tap/sync",
        "https://gea.esac.esa.int/tap-server/tap/sync",
    ]
    
    for attempt in range(max_retries):
        for url in endpoints:
            try:
                # ... 查询代码
                return data
            except requests.Timeout:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            except Exception as e:
                continue
    
    return []
```

---

### 任务 4: 重构重复代码（优先级：🟢 低）

**建议**:
1. 创建 `langgraph_demo/core/` 目录
2. 提取公共类:
   - `core/rag.py`: 统一 RAG 实现
   - `core/tools.py`: 统一工具类
   - `core/utils.py`: 通用函数

---

### 任务 5: 修复测试文件（优先级：🔴 高）

**需要做的**:
1. 更新 `test_fixes.py` 中的导入路径
2. 添加对 `astro_assistant_astrosage.py` 的专门测试
3. 添加消光查询接口的单元测试

---

## 📁 相关文件位置

```
/mnt/c/Users/Administrator/Desktop/astro-ai-demo/
├── langgraph_demo/
│   ├── astro_assistant_astrosage.py  # 主程序（问题 1, 4, 5, 8, 9）
│   ├── ollama_rag_light.py           # 轻量版（问题 2）
│   ├── rag_tool_system.py            # 完整系统（问题 3, FAISS 依赖）
│   └── output/                       # 数据文件目录
├── query_extinction.py               # 消光查询（问题 6）
├── test_astrosage_vsp.py             # 测试文件（测试问题 1）
└── test_fixes.py                     # 测试文件（测试问题 2）
```

---

## 🎯 验证修复的方法

修复后，运行以下命令验证:

```bash
# 1. 测试消光查询修复
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
python -c "
from langgraph_demo.astro_assistant_astrosage import AstroTools
tools = AstroTools()
result = tools.query_extinction(13.1316, 53.8585)
print('消光查询结果:', result)
"

# 2. 运行测试
python test_astrosage_vsp.py all

# 3. 启动交互模式测试
python langgraph_demo/astro_assistant_astrosage.py
```

---

*报告生成时间: 2026-03-05*
