# 天文 AI 助手代码文档

## 项目概览

基于 Ollama/AstroSage + RAG + Tool 的天文领域 AI 助手系统。

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户查询 (RA, DEC, 问题)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     系统架构                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Ollama LLM   │  │ RAG 知识库    │  │ Tool (你的代码)       │  │
│  │ - qwen3:8b   │  │ - 天文知识    │  │ - query_extinction   │  │
│  │ - AstroSage  │  │ - 文档检索    │  │ - GAIA 查询          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心文件说明

### 1. 主程序

| 文件 | 功能 | 状态 |
|------|------|------|
| `astro_assistant_astrosage.py` | AstroSage + RAG + Tool 集成 | ✅ 完整 |
| `start_astro_assistant.py` | 简化版天文助手 (qwen3:8b) | ✅ 可用 |
| `ollama_rag_light.py` | 轻量级 RAG (无 FAISS) | ✅ 可用 |

### 2. 模型部署

| 文件 | 功能 | 状态 |
|------|------|------|
| `deploy_astrosage.py` | 交互式部署 AstroSage | ✅ 完整 |
| `deploy_astrosage_auto.py` | 自动部署脚本 | ✅ 完整 |
| `setup_astrosage.py` | 模型安装工具 | ✅ 完整 |
| `check_astrosage_status.py` | 部署状态检查 | ✅ 完整 |
| `monitor_download.py` | 下载进度监控 | ✅ 完整 |

### 3. 数据生成 (Kimi API)

| 文件 | 功能 | 状态 |
|------|------|------|
| `kimi_data_generator.py` | Kimi API 数据生成器 | ✅ 完整 |
| `kimi_distillation.py` | Kimi 蒸馏训练 | ⚠️ 需优化 |

### 4. 其他

| 文件 | 功能 | 状态 |
|------|------|------|
| `rag_tool_system.py` | RAG + Tool 完整系统 | ⚠️ 依赖重 |
| `run_kimi_ollama.py` | 启动脚本 | ✅ 可用 |

---

## 核心功能模块

### 模块 1: RAG 知识库 (`SimpleRAG`)

```python
class SimpleRAG:
    """简化版 RAG - 关键词匹配"""
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """基于关键词的文档检索"""
        # 实现: 分词 → 倒排索引 → TF 排序
```

**功能**: 
- 关键词倒排索引
- TF (词频) 相关性排序
- 无需 FAISS/向量模型

**问题**: 
- 语义检索能力弱（只能关键词匹配）
- 无法处理同义词

---

### 模块 2: 天文工具 (`AstroTools`)

```python
class AstroTools:
    def query_extinction(self, ra: float, dec: float) -> Dict:
        """调用你的 query_extinction.py"""
        
    def query_gaia(self, ra: float, dec: float) -> List[Dict]:
        """查询 GAIA DR3"""
```

**功能**:
- ✅ 集成你的 `query_extinction.py`
- ✅ GAIA TAP 查询
- ✅ 天区综合分析

**问题**:
- GAIA 查询偶尔超时
- 缺少 SDSS、ZTF 等数据源

---

### 模块 3: LLM 接口 (`AstroSageAssistant`)

```python
class AstroSageAssistant:
    def query(self, user_query: str, ra: float = None, dec: float = None):
        """
        流程:
        1. RAG 检索知识
        2. Tool 查询实时数据
        3. 构建增强提示
        4. 调用 Ollama/AstroSage
        """
```

**功能**:
- 推理链 (Chain-of-Thought)
- 上下文增强
- 支持多模型切换

---

## 已知问题

### 🔴 严重问题

1. **向量检索依赖**
   - `rag_tool_system.py` 依赖 FAISS，在某些环境安装失败
   - **解决**: 使用 `ollama_rag_light.py` 或 `start_astro_assistant.py`（关键词检索）

2. **网络下载问题**
   - HuggingFace 访问不稳定
   - **解决**: 使用 `deploy_astrosage_mirror.py` 或 Git 克隆

### 🟡 中等问题

3. **内存占用**
   - AstroSage 8B 模型需要 ~6GB 显存/内存
   - **解决**: 使用 4-bit 量化版

4. **工具集成不完善**
   - 缺少 ZTF、ASAS-SN、TESS 查询
   - SDSS 光谱查询未实现

### 🟢 小问题

5. **错误处理**
   - 部分异常未捕获
   - API 超时处理不完善

---

## 给 Claude 的开发任务

你可以让 Claude 协助完成以下任务：

### 任务 1: 增强 RAG 系统

```
当前: 关键词匹配 (SimpleRAG)
目标: 语义检索 + 关键词混合

请 Claude:
1. 添加 Sentence-BERT 嵌入 (轻量级)
2. 实现混合检索: 关键词 + 余弦相似度
3. 添加重排序 (reranking)
```

**参考实现**:
```python
class HybridRAG:
    def __init__(self):
        self.keyword_rag = SimpleRAG()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = {}
    
    def search(self, query: str, k: int = 5):
        # 1. 关键词检索
        keyword_results = self.keyword_rag.search(query, k=k)
        
        # 2. 向量检索
        query_vec = self.embedding_model.encode([query])
        semantic_results = self.vector_search(query_vec, k=k)
        
        # 3. 融合排序
        return self.reciprocal_rank_fusion(keyword_results, semantic_results)
```

---

### 任务 2: 添加更多天文工具

```
当前工具:
- query_extinction (你的代码)
- GAIA 查询

需要添加:
- ZTF 光变曲线查询
- ASAS-SN 变星数据
- TESS/Kepler 光变曲线
- SDSS 光谱查询
- Simbad 天体表查询
```

**参考接口**:
```python
class AstronomyDataHub:
    def query_ztf_lightcurve(self, ra: float, dec: float, radius: float = 0.01):
        """查询 ZTF 光变曲线"""
        url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
        # 实现查询逻辑
        
    def query_tess_tce(self, tic_id: int):
        """查询 TESS 掩星候选体"""
        # 使用 Lightkurve 或 TESScut
        
    def query_sdss_spectrum(self, ra: float, dec: float):
        """查询 SDSS 光谱"""
        # 使用 astroquery SDSS
```

---

### 任务 3: 改进提示工程

```
当前: 固定系统提示
目标: 动态提示 + 少样本示例
```

**参考实现**:
```python
class DynamicPromptBuilder:
    def build_prompt(self, query: str, context: Dict) -> str:
        # 1. 查询分类
        query_type = self.classify_query(query)
        
        # 2. 选择模板
        if query_type == "variable_star":
            template = self.VARIABLE_STAR_TEMPLATE
            examples = self.get_few_shot_examples("variable")
        elif query_type == "observation_planning":
            template = self.OBSERVATION_TEMPLATE
            
        # 3. 组装提示
        return template.format(
            context=context,
            examples=examples,
            query=query
        )
```

---

### 任务 4: 添加评估框架

```
需要: 评估 RAG 和 Tool 效果
```

**参考实现**:
```python
class AstroQAEvaluator:
    def __init__(self):
        self.test_set = self.load_test_questions()
    
    def evaluate_rag(self):
        """评估 RAG 检索质量"""
        metrics = {
            'recall@k': [],
            'precision@k': [],
            'mrr': []
        }
        for q in self.test_set:
            results = self.rag.search(q['question'])
            # 计算指标
        return metrics
    
    def evaluate_end_to_end(self):
        """端到端评估"""
        for q in self.test_set:
            response = self.assistant.query(q['question'])
            # 对比参考答案
```

---

### 任务 5: Web UI 界面

```
目标: Gradio/Streamlit Web 界面
```

**参考实现**:
```python
import gradio as gr

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 天文 AI 助手")
        
        with gr.Row():
            ra = gr.Number(label="RA (deg)")
            dec = gr.Number(label="DEC (deg)")
        
        query = gr.Textbox(label="查询内容")
        submit = gr.Button("分析")
        output = gr.Markdown(label="结果")
        
        submit.click(
            fn=lambda r,d,q: assistant.query(q, r, d)['response'],
            inputs=[ra, dec, query],
            outputs=output
        )
    
    demo.launch()
```

---

## 当前下载状态

```bash
# 检查 AstroSage 下载进度
cd /mnt/c/Users/Administrator/Desktop/astro-ai-demo
tail -n 20 models/astrosage-llama-3.1-8b/git.log 2>/dev/null || echo "下载进行中..."
du -sh models/astrosage-llama-3.1-8b/hf_model/ 2>/dev/null || echo "目录尚未创建"
```

---

## 快速启动命令

```bash
# 1. 检查下载状态
python langgraph_demo/check_astrosage_status.py

# 2. 启动体验版 (qwen3:8b)
python run_astrosage_now.py

# 3. 监控下载
python langgraph_demo/monitor_download.py

# 4. 部署完成后的启动
python run_astrosage.py
```

---

## 与 Claude 协作的最佳实践

1. **提供完整上下文**
   ```
   给 Claude 的文件:
   - astro_assistant_astrosage.py (主程序)
   - query_extinction.py (你的工具)
   - 具体需求描述
   ```

2. **明确任务边界**
   ```
   好: "在 AstroTools 类中添加 ZTF 查询方法，接口类似 query_gaia"
   差: "添加更多数据源"
   ```

3. **迭代开发**
   ```
   第1轮: 基础功能
   第2轮: 错误处理
   第3轮: 性能优化
   ```

4. **测试验证**
   ```
   每次修改后运行:
   - python check_astrosage_status.py
   - 实际查询测试
   ```
