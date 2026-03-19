# Claude 协助开发提示模板

## 当前项目状态

### ✅ 已完成
- 基于 qwen3:8b 的天文 AI 助手系统
- RAG 知识库（关键词检索）
- 工具集成（消光查询、GAIA）
- Kimi API 数据生成器

### ⚠️ 进行中
- AstroSage-Llama-3.1-8B 模型下载（遇到网络问题）

### 📋 需要协助

---

## 提示模板 1: 增强 RAG 检索

```markdown
请帮我增强天文 AI 助手的 RAG 系统。

当前代码文件: `langgraph_demo/astro_assistant_astrosage.py`
当前 SimpleRAG 类使用关键词匹配，我需要添加语义检索能力。

需求:
1. 添加轻量级 Sentence-BERT 嵌入 (如 'all-MiniLM-L6-v2' 或 'BAAI/bge-small-en')
2. 实现混合检索: 关键词匹配 + 向量相似度
3. 保持轻量，不依赖 FAISS（使用 numpy/scipy 计算余弦相似度）
4. 添加缓存机制避免重复编码

参考代码结构:
```python
class HybridRAG:
    def __init__(self, use_embedding=True):
        self.keyword_rag = SimpleRAG()
        if use_embedding:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.vectors = []
        
    def add_documents(self, docs):
        # 同时更新关键词索引和向量库
        
    def search(self, query, k=3, alpha=0.5):
        # alpha: 混合权重，0=纯关键词, 1=纯向量
        # 返回融合排序的结果
```

约束:
- 依赖尽量轻量 (sentence-transformers, numpy, scipy)
- 支持中文和英文天文文档
- 检索时间 < 500ms (1000 篇文档)
```

---

## 提示模板 2: 添加 ZTF 数据查询

```markdown
请帮我在天文工具集中添加 ZTF (Zwicky Transient Facility) 数据查询功能。

当前代码文件: `langgraph_demo/astro_assistant_astrosage.py`
当前 AstroTools 类已有 query_gaia 和 query_extinction 方法。

需求:
1. 添加 query_ztf_lightcurve(ra, dec, radius=0.01) 方法
2. 使用 IPAC/IRSA API (https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves)
3. 返回标准化格式: [{mjd, mag, magerr, filter, objid}, ...]
4. 添加数据缓存避免重复查询

接口参考:
```python
def query_ztf_lightcurve(self, ra: float, dec: float, radius: float = 0.01) -> Dict:
    """
    查询 ZTF 光变曲线
    
    Returns:
        {
            'success': bool,
            'count': int,
            'lightcurve': [
                {
                    'mjd': float,
                    'mag': float,
                    'magerr': float,
                    'filter': str,  # 'zg', 'zr', 'zi'
                    'object_id': str
                }
            ],
            'metadata': {
                'ra': float,
                'dec': float,
                'search_radius': float
            }
        }
    """
```

约束:
- 处理 API 限流和超时
- 支持多波段 (g, r, i) 数据筛选
- 错误处理友好
```

---

## 提示模板 3: 改进提示工程

```markdown
请帮我改进天文 AI 助手的提示工程系统。

当前代码文件: `langgraph_demo/astro_assistant_astrosage.py`
当前使用固定系统提示，需要改为动态提示。

需求:
1. 实现查询分类器 (variable_star, binary_system, exoplanet, general)
2. 根据分类选择不同提示模板
3. 添加少样本示例 (few-shot examples)
4. 实现上下文压缩（处理长上下文）

参考代码:
```python
class DynamicPromptBuilder:
    TEMPLATES = {
        'variable_star': """你是变星专家...""",
        'binary_system': """你是双星系统专家...""",
        'observation': """你是观测策略专家...""",
        'general': """你是天文学家..."""
    }
    
    FEW_SHOT_EXAMPLES = {
        'variable_star': [
            {'input': '...', 'output': '...'},
        ]
    }
    
    def classify_query(self, query: str) -> str:
        # 使用关键词或轻量分类器
        
    def build_prompt(self, query: str, context: Dict, use_cot: bool = True) -> str:
        # 1. 分类
        # 2. 选择模板
        # 3. 添加上下文
        # 4. 添加示例（如果相关）
        # 5. 组装
```

约束:
- 分类器轻量（不要用大模型）
- 响应时间 < 100ms
- 模板可配置（从文件加载）
```

---

## 提示模板 4: 创建 Web UI

```markdown
请为天文 AI 助手创建 Gradio Web 界面。

当前后端代码: `langgraph_demo/astro_assistant_astrosage.py`
类: AstroSageAssistant

需求:
1. 创建 `langgraph_demo/web_ui.py`
2. 输入: RA, DEC, 查询文本
3. 输出: 格式化回答（Markdown）
4. 可选: 光变曲线可视化 (Plotly/Matplotlib)
5. 显示使用的数据源（RAG 文档、Tool 结果）

参考界面:
```python
import gradio as gr

def query_assistant(ra, dec, query, use_rag, use_tools):
    assistant = AstroSageAssistant()
    result = assistant.query(query, ra, dec, use_rag, use_tools)
    return result['response']

with gr.Blocks() as demo:
    gr.Markdown("# 🔭 天文 AI 助手")
    
    with gr.Row():
        with gr.Column():
            ra = gr.Number(label="RA (度)", value=13.1316)
            dec = gr.Number(label="DEC (度)", value=53.8585)
            query = gr.Textbox(label="查询", lines=3)
            use_rag = gr.Checkbox(label="使用 RAG", value=True)
            use_tools = gr.Checkbox(label="使用工具", value=True)
            submit = gr.Button("分析")
        
        with gr.Column():
            output = gr.Markdown(label="分析结果")
    
    submit.click(query_assistant, 
                 inputs=[ra, dec, query, use_rag, use_tools],
                 outputs=output)

demo.launch()
```

约束:
- 美观专业（天文学主题）
- 响应式设计
- 支持中文
```

---

## 提示模板 5: 添加模型评估

```markdown
请为天文 AI 助手添加评估框架。

当前代码: `langgraph_demo/astro_assistant_astrosage.py`

需求:
1. 创建 `langgraph_demo/evaluation.py`
2. 定义测试集（天文问答对）
3. 评估指标:
   - RAG: Recall@K, MRR, NDCG
   - 端到端: 回答准确性（与参考答案对比）
   - 工具: 查询成功率、响应时间

参考实现:
```python
class AstroEvaluator:
    def __init__(self, test_file='test_questions.json'):
        self.test_set = json.load(open(test_file))
        self.assistant = AstroSageAssistant()
    
    def evaluate_rag(self, k=5):
        """评估 RAG 检索质量"""
        metrics = {'recall@k': [], 'precision@k': [], 'mrr': []}
        
        for item in self.test_set:
            query = item['question']
            relevant_docs = item['relevant_docs']
            
            results = self.assistant.rag.search(query, k=k)
            retrieved = [r['topic'] for r in results]
            
            # 计算指标
            recall = len(set(retrieved) & set(relevant_docs)) / len(relevant_docs)
            metrics['recall@k'].append(recall)
            
        return {k: sum(v)/len(v) for k, v in metrics.items()}
    
    def evaluate_tools(self):
        """评估工具调用"""
        test_coords = [(13.1316, 53.8585), (180.0, 0.0)]
        
        for ra, dec in test_coords:
            # 测试消光查询
            start = time.time()
            result = self.assistant.tools.query_extinction(ra, dec)
            latency = time.time() - start
            
            # 记录成功率和延迟
    
    def evaluate_llm(self):
        """评估 LLM 回答质量"""
        # 使用简单匹配或 BLEU/ROUGE 分数
```

约束:
- 可重复运行
- 生成详细报告（JSON/Markdown）
```

---

## 提示模板 6: 解决下载问题

```markdown
我遇到了 AstroSage 模型下载问题，请帮我解决。

当前状态:
- Git 克隆失败: gnutls_handshake() failed
- HuggingFace 镜像访问失败 (401)
- 模型: Spectroscopic/AstroSage-Llama-3.1-8B

需求:
1. 提供备选下载方案
2. 或使用替代模型 (llama3.1:8b 或 qwen3:8b)
3. 创建 Modelfile 优化天文能力

当前已有:
- qwen3:8b 在 Ollama 中可用
- 天文知识库 (RAG)
- 工具集成

目标:
在无法下载 AstroSage 的情况下，通过提示工程和 RAG 达到 90% 效果。

参考方案:
```python
# 创建优化的 Modelfile
MODEFILE = '''
FROM qwen3:8b

SYSTEM """你是一位顶尖天文学家...
[详细的天文专业角色设定]
..."""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
'''

# 导入 Ollama
# ollama create astro-qwen -f Modelfile
```
```

---

## 使用 Claude 的工作流程

1. **准备上下文**
   ```bash
   # 导出当前代码
   cat langgraph_demo/astro_assistant_astrosage.py | xclip -selection clipboard
   ```

2. **选择提示模板**
   - 根据需求选择上面的模板
   - 或组合多个模板

3. **与 Claude 对话**
   - 粘贴模板
   - 说明具体约束
   - 迭代修改

4. **验证代码**
   ```bash
   python -m py_compile new_file.py
   python new_file.py
   ```

5. **集成测试**
   ```bash
   python langgraph_demo/check_astrosage_status.py
   python run_astrosage_now.py
   ```
