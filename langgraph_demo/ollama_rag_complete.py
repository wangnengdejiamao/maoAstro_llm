#!/usr/bin/env python3
"""
Ollama + RAG + Tool + 实时数据 完整方案
========================================
重要说明：Ollama 的 GGUF 模型不支持直接微调！
本方案提供替代方法增强模型能力：
1. RAG 检索 - 注入专业知识
2. Tool 调用 - 查询实时数据
3. Modelfile - 优化系统提示
4. 上下文工程 - 动态增强输入

作者: AI Assistant
"""

import os
import sys
import json
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Config:
    """配置"""
    ollama_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    kimi_api_key: str = "19cb2d77-5ef2-8672-8000-0000a0d97edd"
    
    # 向量数据库
    vector_db_path: str = "langgraph_demo/output/astro_knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"  # 小型中文模型
    
    # 数据路径
    training_data_path: str = "langgraph_demo/output/training_dataset.json"
    kimi_data_path: str = "langgraph_demo/output/kimi_generated"


class EmbeddingModel:
    """轻量级 Embedding 模型"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load()
    
    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"加载 Embedding 模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Embedding 模型加载成功")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            raise
    
    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


class SimpleVectorStore:
    """简单向量存储 (FAISS)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding = None
        self.index = None
        self.documents = []
        self._init()
    
    def _init(self):
        try:
            import faiss
            self.faiss = faiss
            self.embedding = EmbeddingModel(self.config.embedding_model)
            self._load()
            print(f"✓ 向量库就绪 (当前 {len(self.documents)} 条)")
        except ImportError:
            print("请安装: pip install faiss-cpu sentence-transformers")
            raise
    
    def add(self, texts: List[str], metadatas: List[Dict] = None):
        """添加文档"""
        if not texts:
            return
        
        vectors = self.embedding.encode(texts)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        if self.index is None:
            dim = vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(dim)
        
        self.index.add(vectors.astype('float32'))
        
        for i, text in enumerate(texts):
            doc = {"text": text, "metadata": metadatas[i] if metadatas else {}}
            self.documents.append(doc)
        
        self._save()
        print(f"  添加 {len(texts)} 条文档，总计 {len(self.documents)} 条")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关文档"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        query_vec = self.embedding.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        scores, indices = self.index.search(query_vec.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result["score"] = float(score)
                results.append(result)
        return results
    
    def _save(self):
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        if self.index:
            self.faiss.write_index(self.index, 
                os.path.join(self.config.vector_db_path, "index.faiss"))
        with open(os.path.join(self.config.vector_db_path, "docs.json"), 'w') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        index_path = os.path.join(self.config.vector_db_path, "index.faiss")
        docs_path = os.path.join(self.config.vector_db_path, "docs.json")
        if os.path.exists(index_path):
            self.index = self.faiss.read_index(index_path)
        if os.path.exists(docs_path):
            with open(docs_path, 'r') as f:
                self.documents = json.load(f)


class AstronomyDataTools:
    """天文数据工具 - 集成你的现有代码"""
    
    def __init__(self):
        self.extinction_query = None
        self._init_tools()
    
    def _init_tools(self):
        """初始化工具"""
        try:
            from query_extinction import query_extinction
            self.extinction_query = query_extinction
            print("✓ 消光查询工具加载成功")
        except Exception as e:
            print(f"⚠ 消光查询工具加载失败: {e}")
    
    def query_extinction(self, ra: float, dec: float) -> Dict:
        """查询消光"""
        if self.extinction_query is None:
            return {"error": "工具未加载"}
        try:
            result = self.extinction_query(ra, dec)
            return {"success": True, "data": result}
        except Exception as e:
            return {"error": str(e)}
    
    def query_gaia_simple(self, ra: float, dec: float, radius: float = 0.1) -> Dict:
        """简单 Gaia 查询（无需完整 astroquery）"""
        try:
            import requests
            # TAP 查询
            query = f"""
            SELECT TOP 5 source_id, ra, dec, phot_g_mean_mag, bp_rp
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(POINT('ICRS', ra, dec), 
                          CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
            AND phot_g_mean_mag < 18
            ORDER BY phot_g_mean_mag ASC
            """
            
            url = "http://dc.zah.uni-heidelberg.de/tap/sync"
            params = {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "json",
                "QUERY": query
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "data": data.get('data', [])}
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_sky_conditions(self, ra: float, dec: float) -> Dict:
        """获取天区观测条件"""
        # 简化的观测条件评估
        extinction = self.query_extinction(ra, dec)
        
        conditions = {
            "coordinates": {"ra": ra, "dec": dec},
            "extinction_available": "error" not in extinction,
            "recommended_filters": ["V", "R", "I"],
            "notes": "基于消光数据推荐"
        }
        
        if extinction.get("success"):
            data = extinction["data"]
            conditions["extinction_data"] = data
        
        return conditions


class OllamaRAGSystem:
    """
    Ollama + RAG + Tool 集成系统
    核心思想：不微调模型，而是优化输入上下文
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        print("=" * 70)
        print("初始化 Ollama + RAG + Tool 系统")
        print("=" * 70)
        
        # 初始化组件
        self.vector_store = SimpleVectorStore(self.config)
        self.tools = AstronomyDataTools()
        
        # 系统提示模板 - 这是增强模型能力的关键！
        self.system_prompt = """你是一位专业天文学家，精通变星、双星和系外行星研究。

**角色设定：**
- 你熟悉 GAIA、SDSS、ZTF 等巡天数据
- 你了解光度测量、光谱分析、时序分析
- 你能根据坐标查询天文数据库

**回答规范：**
1. 天体分类要精确到子类型
2. 物理参数必须包含数值和单位
3. 提供具体的观测建议（望远镜参数、滤镜、曝光时间）
4. 引用数据来源

**当前可用工具：**
- query_extinction: 查询银河消光
- query_gaia: 查询 GAIA 数据源
- get_sky_conditions: 获取观测条件

**知识库：**
如果提供了[相关知识]，请优先基于这些知识回答。"""

        print("\n✓ 系统初始化完成")
    
    def build_knowledge_base(self, use_kimi: bool = True):
        """
        构建知识库
        可以用 Kimi API 生成高质量内容，或使用本地数据
        """
        print("\n" + "=" * 70)
        print("构建知识库")
        print("=" * 70)
        
        if use_kimi and self._check_kimi():
            self._build_with_kimi()
        else:
            self._build_with_local_data()
    
    def _check_kimi(self) -> bool:
        """检查 Kimi API 是否可用"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.kimi_api_key, 
                          base_url="https://api.moonshot.cn/v1")
            # 简单测试
            client.models.list()
            print("✓ Kimi API 可用")
            return True
        except Exception as e:
            print(f"⚠ Kimi API 不可用: {e}")
            return False
    
    def _build_with_kimi(self):
        """使用 Kimi 生成知识库"""
        from openai import OpenAI
        client = OpenAI(api_key=self.config.kimi_api_key,
                       base_url="https://api.moonshot.cn/v1")
        
        topics = [
            "灾变变星 (Cataclysmic Variables) - 分类和物理特征",
            "造父变星 (Cepheid Variables) - 周光关系和应用",
            "食双星 (Eclipsing Binaries) - 测光分析和参数测定",
            "米拉变星 (Mira Variables) - 脉动机制和光变特征",
            "系外行星探测 - 凌星法和径向速度法",
            "GAIA 数据源 - 变星应用",
            "天文测光 - 技术要点和校准",
        ]
        
        print(f"使用 Kimi API 生成 {len(topics)} 篇专业知识...")
        
        documents = []
        metadatas = []
        
        for topic in topics:
            print(f"  生成: {topic}...")
            try:
                response = client.chat.completions.create(
                    model="moonshot-v1-32k",
                    messages=[{
                        "role": "system",
                        "content": "你是天文学专家。请提供详细的专业知识，适合作为知识库。"
                    }, {
                        "role": "user",
                        "content": f"详细介绍: {topic}\n\n请包括：定义、分类、特征、观测方法、重要实例。"
                    }],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                documents.append(content)
                metadatas.append({"topic": topic, "source": "kimi", "created": datetime.now().isoformat()})
                
                import time
                time.sleep(0.5)  # 避免限流
                
            except Exception as e:
                print(f"    失败: {e}")
        
        self.vector_store.add(documents, metadatas)
        print(f"✓ 知识库构建完成: {len(documents)} 篇")
    
    def _build_with_local_data(self):
        """使用本地数据构建知识库"""
        # 从你的训练数据加载
        if os.path.exists(self.config.training_data_path):
            with open(self.config.training_data_path, 'r') as f:
                data = json.load(f)
            
            documents = []
            metadatas = []
            
            for item in data:
                # 构建文本
                input_info = item.get('input', {})
                output_info = item.get('output', {})
                
                text = f"""天体分析
坐标: RA={input_info.get('coordinates', {}).get('ra')}, 
      DEC={input_info.get('coordinates', {}).get('dec')}
分类: {output_info.get('classification', 'Unknown')}
分析: {output_info.get('reasoning', [])}
"""
                documents.append(text)
                metadatas.append({"source": "local_training_data"})
            
            self.vector_store.add(documents, metadatas)
            print(f"✓ 从本地加载: {len(documents)} 条")
        else:
            print(f"⚠ 本地数据不存在: {self.config.training_data_path}")
            print("  使用内置知识...")
            self._add_builtin_knowledge()
    
    def _add_builtin_knowledge(self):
        """添加内置知识"""
        knowledge = [
            """灾变变星 (Cataclysmic Variables, CV) 是包含白矮星主星和正常伴星的双星系统。
白矮星通过吸积盘吸积伴星物质，产生剧烈的能量释放。
分类: 新星 (Novae), 矮新星 (Dwarf Novae), 类新星 (Nova-like), 磁激变变星 (Polars/Intermediate Polars)
观测: 需要时序测光，时间分辨率 1-10 分钟，V/R 滤镜。""",
            
            """造父变星 (Cepheid Variables) 是脉动变星，光度和周期存在线性关系。
周光关系: M_v = -2.81 log(P) - 1.43
应用: 宇宙距离尺度测定，哈勃常数测量
周期: 1-50 天，光变幅度 0.5-2 等
观测: V 滤镜，覆盖完整周期。""",
            
            """食双星 (Eclipsing Binaries) 是轨道平面与视线对齐的双星系统，发生周期性掩食。
类型: 大陵型 (Algol), 渐台型 (Beta Lyrae), 大熊座 W 型 (W UMa)
分析: 测光曲线拟合，推导轨道参数、恒星半径、表面温度
观测: 高时间分辨率测光，多滤镜。""",
        ]
        
        metadatas = [{"topic": "builtin", "source": "builtin"} for _ in knowledge]
        self.vector_store.add(knowledge, metadatas)
        print(f"✓ 添加内置知识: {len(knowledge)} 条")
    
    def query(self, user_query: str, ra: float = None, dec: float = None, 
              use_tools: bool = True, use_rag: bool = True) -> Dict:
        """
        处理用户查询
        
        核心逻辑：
        1. RAG 检索相关知识
        2. Tool 查询实时数据
        3. 构建增强提示
        4. 调用 Ollama
        """
        context_parts = []
        
        # 1. RAG 检索
        if use_rag:
            print("  [RAG] 检索相关知识...")
            docs = self.vector_store.search(user_query, top_k=3)
            if docs:
                context_parts.append("[相关知识]")
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"{i}. {doc['text'][:300]}...")
                    context_parts.append(f"   (来源: {doc['metadata'].get('topic', 'Unknown')})")
        
        # 2. Tool 查询
        tool_results = {}
        if use_tools and ra is not None and dec is not None:
            print(f"  [Tool] 查询实时数据: RA={ra}, DEC={dec}...")
            
            # 消光查询
            ext_result = self.tools.query_extinction(ra, dec)
            if ext_result.get("success"):
                tool_results["extinction"] = ext_result["data"]
                context_parts.append(f"\n[实时数据 - 消光]")
                context_parts.append(json.dumps(ext_result["data"], indent=2, ensure_ascii=False))
            
            # Gaia 查询
            gaia_result = self.tools.query_gaia_simple(ra, dec)
            if gaia_result.get("success"):
                tool_results["gaia"] = gaia_result["data"]
                if gaia_result["data"]:
                    context_parts.append(f"\n[实时数据 - GAIA 近邻源]")
                    for star in gaia_result["data"][:3]:
                        context_parts.append(f"  G={star[3]:.2f}, BP-RP={star[4]:.3f}" if len(star) > 4 else str(star))
        
        # 3. 构建增强提示
        enhanced_prompt = self._build_prompt(user_query, context_parts)
        
        # 4. 调用 Ollama
        print("  [Ollama] 生成回答...")
        response = self._call_ollama(enhanced_prompt)
        
        return {
            "query": user_query,
            "coordinates": {"ra": ra, "dec": dec},
            "context_used": {
                "rag": use_rag,
                "tools": use_tools,
                "retrieved_docs": len(docs) if use_rag else 0,
                "tool_results": list(tool_results.keys())
            },
            "response": response
        }
    
    def _build_prompt(self, user_query: str, context_parts: List[str]) -> str:
        """构建完整提示"""
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""{self.system_prompt}

{'='*50}
{context_str}
{'='*50}

[用户问题]
{user_query}

请基于以上信息，提供专业、详细的回答。如果提供了实时数据，请在分析中引用。"""
        
        return prompt
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """调用 Ollama"""
        url = f"{self.config.ollama_host}/api/generate"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {
                "num_ctx": 8192,  # 增大上下文窗口
                "num_predict": 2048  # 限制生成长度
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            return response.json().get('response', '')
        except Exception as e:
            return f"Ollama 调用失败: {e}"
    
    def create_ollama_modelfile(self):
        """
        创建 Ollama Modelfile
        这可以永久改变模型的系统提示！
        """
        modelfile_content = f'''FROM {self.config.ollama_model}

SYSTEM """你是一位专业天文学家，精通变星、双星和系外行星研究。

**角色设定：**
- 你熟悉 GAIA、SDSS、ZTF 等巡天数据
- 你了解光度测量、光谱分析、时序分析
- 你能根据坐标查询天文数据库

**回答规范：**
1. 天体分类要精确到子类型
2. 物理参数必须包含数值和单位
3. 提供具体的观测建议
4. 引用数据来源

当用户提供坐标时，主动询问是否需要查询相关数据。"""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
'''
        
        output_path = "langgraph_demo/output/Modelfile.astro"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"\n✓ Modelfile 已创建: {output_path}")
        print("\n使用以下命令创建自定义模型:")
        print(f"  ollama create astro-assistant -f {output_path}")
        print("  ollama run astro-assistant")
    
    def interactive(self):
        """交互模式"""
        print("\n" + "=" * 70)
        print("Ollama + RAG + Tool 交互模式")
        print("=" * 70)
        print("命令:")
        print("  /exit      - 退出")
        print("  /rag on/off- 开关 RAG")
        print("  /tools on/off - 开关 Tools")
        print("  /modelfile - 生成 Modelfile")
        print("  /search <query> - 仅搜索知识库")
        print("\n输入坐标格式: RA, DEC 或 直接提问")
        print("-" * 70)
        
        use_rag = True
        use_tools = True
        
        while True:
            try:
                user_input = input("\n🌟 你: ").strip()
                
                if not user_input:
                    continue
                
                # 命令处理
                if user_input == "/exit":
                    print("再见!")
                    break
                
                if user_input == "/rag on":
                    use_rag = True
                    print("✓ RAG 已开启")
                    continue
                
                if user_input == "/rag off":
                    use_rag = False
                    print("✓ RAG 已关闭")
                    continue
                
                if user_input == "/tools on":
                    use_tools = True
                    print("✓ Tools 已开启")
                    continue
                
                if user_input == "/tools off":
                    use_tools = False
                    print("✓ Tools 已关闭")
                    continue
                
                if user_input == "/modelfile":
                    self.create_ollama_modelfile()
                    continue
                
                if user_input.startswith("/search "):
                    query = user_input[8:]
                    results = self.vector_store.search(query)
                    print(f"\n找到 {len(results)} 条结果:")
                    for r in results:
                        print(f"  [{r['score']:.3f}] {r['text'][:100]}...")
                    continue
                
                # 解析坐标
                import re
                coords = re.findall(r'([\d.]+)\s*,\s*([+-]?[\d.]+)', user_input)
                ra, dec = None, None
                if coords:
                    ra, dec = float(coords[0][0]), float(coords[0][1])
                    print(f"  [检测到坐标: RA={ra}, DEC={dec}]")
                
                # 执行查询
                result = self.query(user_input, ra, dec, use_tools, use_rag)
                
                print(f"\n🤖 AI:\n{result['response']}")
                
                # 显示使用的上下文
                ctx = result['context_used']
                print(f"\n[上下文: RAG={ctx['rag']}({ctx['retrieved_docs']}条), "
                      f"Tools={ctx['tools']}({','.join(ctx['tool_results']) or '无'})]")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama + RAG + Tool 系统")
    parser.add_argument("--build-only", action="store_true", help="仅构建知识库")
    parser.add_argument("--no-kimi", action="store_true", help="不使用 Kimi API")
    args = parser.parse_args()
    
    # 初始化系统
    config = Config()
    system = OllamaRAGSystem(config)
    
    # 构建知识库
    system.build_knowledge_base(use_kimi=not args.no_kimi)
    
    if args.build_only:
        print("\n知识库构建完成，退出")
        return
    
    # 启动交互模式
    system.interactive()


if __name__ == "__main__":
    main()
