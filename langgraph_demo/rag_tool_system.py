#!/usr/bin/env python3
"""
RAG + Tool 集成系统
===================
结合 Kimi API 和本地 Ollama qwen3:8b 的完整解决方案

架构：
1. RAG: 使用向量数据库存储天文知识，增强问答质量
2. Tool: 集成天文数据查询工具（Gaia、SDSS、消光等）
3. 路由: Kimi API 用于复杂推理，Ollama 用于快速响应

组件：
- 向量数据库: FAISS / Chroma
- 嵌入模型: 本地模型或 API
- LLM: Ollama qwen3:8b (本地) + Kimi API (备用)
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RAGConfig:
    """RAG 配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"  # 中文 Embedding 模型
    # 备选: "sentence-transformers/all-MiniLM-L6-v2" (英文)
    vector_db_path: str = "langgraph_demo/output/vector_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 3


@dataclass
class ToolConfig:
    """工具配置"""
    gaia_table: str = "gaiadr3.gaia_source"
    sdss_url: str = "https://skyserver.sdss.org/dr18/en/tools/search/x_sql.aspx"
    extinction_service: str = "local"  # 或 "irsa"


class SimpleEmbedding:
    """简化版 Embedding 模型（无需额外依赖）"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"加载 Embedding 模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Embedding 模型加载成功")
        except ImportError:
            print("✗ 请安装 sentence-transformers: pip install sentence-transformers")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        return self.model.encode(texts, convert_to_numpy=True)


class SimpleVectorDB:
    """简化版向量数据库（使用 FAISS）"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding = None
        self.index = None
        self.documents = []
        self._init_db()
    
    def _init_db(self):
        """初始化向量数据库"""
        try:
            import faiss
            self.faiss = faiss
            
            # 初始化 Embedding
            self.embedding = SimpleEmbedding(self.config.embedding_model)
            
            # 创建 FAISS 索引（在第一次添加数据时初始化）
            self.index = None
            
            # 加载已有数据
            self._load()
            
            print("✓ 向量数据库初始化完成")
        except ImportError:
            print("✗ 请安装 faiss: pip install faiss-cpu")
            raise
    
    def add_documents(self, documents: List[Dict]):
        """添加文档到向量库"""
        if not documents:
            return
        
        print(f"添加 {len(documents)} 篇文档到向量库...")
        
        # 提取文本内容
        texts = [doc.get('content', '') for doc in documents]
        
        # 生成向量
        vectors = self.embedding.encode(texts)
        
        # 初始化索引（如果是第一次）
        if self.index is None:
            dim = vectors.shape[1]
            self.index = self.faiss.IndexFlatIP(dim)  # 内积相似度
        
        # 归一化向量（用于余弦相似度）
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # 添加到索引
        self.index.add(vectors.astype('float32'))
        
        # 保存文档
        self.documents.extend(documents)
        
        self._save()
        print(f"✓ 向量库现在包含 {len(self.documents)} 篇文档")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """搜索相关文档"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        top_k = top_k or self.config.top_k
        
        # 编码查询
        query_vector = self.embedding.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 搜索
        scores, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        # 返回结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        
        return results
    
    def _save(self):
        """保存向量库"""
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # 保存 FAISS 索引
        if self.index is not None:
            self.faiss.write_index(
                self.index, 
                os.path.join(self.config.vector_db_path, "index.faiss")
            )
        
        # 保存文档
        with open(os.path.join(self.config.vector_db_path, "documents.json"), 'w') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """加载向量库"""
        index_path = os.path.join(self.config.vector_db_path, "index.faiss")
        docs_path = os.path.join(self.config.vector_db_path, "documents.json")
        
        if os.path.exists(index_path):
            self.index = self.faiss.read_index(index_path)
            print(f"  加载已有索引: {self.index.ntotal} 条向量")
        
        if os.path.exists(docs_path):
            with open(docs_path, 'r') as f:
                self.documents = json.load(f)
            print(f"  加载已有文档: {len(self.documents)} 篇")


class AstronomyTools:
    """天文数据查询工具集"""
    
    def __init__(self, config: ToolConfig = None):
        self.config = config or ToolConfig()
        self.query_extinction_func = None  # 重命名避免冲突
        self._maps = None  # 预加载的消光地图
        self._init_tools()
    
    def _init_tools(self):
        """初始化工具"""
        # 导入本地消光查询模块
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from query_extinction import query_extinction as qe, load_extinction_maps
            self.query_extinction_func = qe
            # 预加载消光地图
            try:
                self._maps = load_extinction_maps(data_dir='data')
                print("✓ 消光查询工具加载成功（maps 预加载成功）")
            except Exception as e:
                print(f"⚠ 消光地图加载失败: {e}")
                self._maps = None
        except Exception as e:
            print(f"⚠ 消光查询工具加载失败: {e}")
    
    def query_gaia(self, ra: float, dec: float, radius: float = 0.1) -> Dict:
        """查询 Gaia 数据"""
        try:
            from astroquery.gaia import Gaia
            
            query = f"""
            SELECT TOP 10 *
            FROM {self.config.gaia_table}
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius})
            ) = 1
            AND phot_g_mean_mag IS NOT NULL
            ORDER BY phot_g_mean_mag ASC
            """
            
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            
            return {
                "tool": "gaia_query",
                "success": True,
                "count": len(results),
                "data": results.to_pandas().to_dict('records')
            }
        except Exception as e:
            return {
                "tool": "gaia_query",
                "success": False,
                "error": str(e)
            }
    
    def query_extinction_data(self, ra: float, dec: float) -> Dict:
        """查询消光数据"""
        if self.query_extinction_func is None:
            return {
                "tool": "extinction_query",
                "success": False,
                "error": "消光查询模块未加载",
                "ra": ra, "dec": dec
            }
        if self._maps is None:
            return {
                "tool": "extinction_query",
                "success": False,
                "error": "消光地图未加载",
                "ra": ra, "dec": dec
            }
        
        try:
            result = self.query_extinction_func(ra, dec, self._maps)
            return {
                "tool": "extinction_query",
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "tool": "extinction_query",
                "success": False,
                "error": str(e),
                "ra": ra, "dec": dec
            }
    
    def query_light_curve(self, ra: float, dec: float, radius: float = 0.01) -> Dict:
        """查询光变曲线（模拟）"""
        # 实际实现需要连接到 ZTF、ASAS-SN 等数据库
        return {
            "tool": "light_curve_query",
            "success": True,
            "message": "光变曲线查询功能需要连接到具体数据库",
            "suggested_services": ["ZTF", "ASAS-SN", "TESS", "Kepler"]
        }
    
    def get_available_tools(self) -> List[Dict]:
        """获取可用工具列表"""
        return [
            {
                "name": "query_gaia",
                "description": "查询 Gaia DR3 数据源，获取恒星的天体测量和测光数据",
                "parameters": {
                    "ra": {"type": "float", "description": "赤经 (度)"},
                    "dec": {"type": "float", "description": "赤纬 (度)"},
                    "radius": {"type": "float", "default": 0.1, "description": "搜索半径 (度)"}
                }
            },
            {
                "name": "query_extinction_data",
                "description": "查询指定位置的消光和reddening数据",
                "parameters": {
                    "ra": {"type": "float", "description": "赤经 (度)"},
                    "dec": {"type": "float", "description": "赤纬 (度)"}
                }
            },
            {
                "name": "query_light_curve",
                "description": "查询变星的光变曲线数据",
                "parameters": {
                    "ra": {"type": "float", "description": "赤经 (度)"},
                    "dec": {"type": "float", "description": "赤纬 (度)"},
                    "radius": {"type": "float", "default": 0.01, "description": "搜索半径 (度)"}
                }
            }
        ]


class LLMRouter:
    """LLM 路由管理器
    
    策略：
    - 简单查询 → Ollama (本地，快速)
    - 复杂推理 → Kimi API (云端，强大)
    """
    
    def __init__(self, ollama_model: str = "qwen3:8b", kimi_api_key: str = None):
        self.ollama_model = ollama_model
        self.kimi_api_key = kimi_api_key
        self.ollama_host = "http://localhost:11434"
        self.kimi_client = None
        
        if kimi_api_key:
            self._init_kimi()
    
    def _init_kimi(self):
        """初始化 Kimi 客户端"""
        try:
            from openai import OpenAI
            self.kimi_client = OpenAI(
                api_key=self.kimi_api_key,
                base_url="https://api.moonshot.cn/v1"
            )
        except Exception as e:
            print(f"⚠ Kimi 客户端初始化失败: {e}")
    
    def query_ollama(self, prompt: str, system: str = None) -> str:
        """查询 Ollama"""
        import requests
        
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            return response.json().get('response', '')
        except Exception as e:
            return f"Ollama 查询失败: {e}"
    
    def query_kimi(self, prompt: str, system: str = None) -> str:
        """查询 Kimi API"""
        if not self.kimi_client:
            return "Kimi API 未配置"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.kimi_client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Kimi 查询失败: {e}"
    
    def route(self, query: str, complexity: str = "auto") -> str:
        """
        路由查询到合适的 LLM
        
        Args:
            query: 用户查询
            complexity: "simple" | "complex" | "auto"
        """
        if complexity == "auto":
            # 自动判断复杂度
            complexity = self._assess_complexity(query)
        
        system_prompt = "你是一位专业天文学家，熟悉变星、双星和系外行星研究。"
        
        if complexity == "complex" and self.kimi_client:
            print("  [路由到 Kimi API - 复杂查询]")
            return self.query_kimi(query, system_prompt)
        else:
            print("  [路由到 Ollama - 本地处理]")
            return self.query_ollama(query, system_prompt)
    
    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        complex_keywords = [
            "分析", "比较", "为什么", "原因", "机制", "详细", 
            "推导", "计算", "模型", "理论"
        ]
        
        if any(kw in query for kw in complex_keywords) or len(query) > 100:
            return "complex"
        return "simple"


class AstronomyAssistant:
    """天文助手 - RAG + Tool + LLM 集成"""
    
    def __init__(self, use_kimi: bool = True):
        print("=" * 70)
        print("初始化天文 AI 助手")
        print("=" * 70)
        
        # 初始化 RAG
        self.rag_config = RAGConfig()
        self.vector_db = SimpleVectorDB(self.rag_config)
        
        # 初始化 Tools
        self.tool_config = ToolConfig()
        self.tools = AstronomyTools(self.tool_config)
        
        # 初始化 LLM 路由
        kimi_key = "19cb2d77-5ef2-8672-8000-0000a0d97edd" if use_kimi else None
        self.llm_router = LLMRouter(ollama_model="qwen3:8b", kimi_api_key=kimi_key)
        
        print("\n✓ 助手初始化完成")
    
    def load_kimi_documents(self, docs_path: str = None):
        """加载 Kimi 生成的文档"""
        if docs_path is None:
            docs_path = "langgraph_demo/output/kimi_generated/rag_documents.json"
        
        if not os.path.exists(docs_path):
            print(f"⚠ 文档文件不存在: {docs_path}")
            print("  请先运行 kimi_data_generator.py 生成文档")
            return
        
        with open(docs_path, 'r') as f:
            documents = json.load(f)
        
        self.vector_db.add_documents(documents)
    
    def query(self, user_query: str, ra: float = None, dec: float = None) -> Dict:
        """
        处理用户查询
        
        流程：
        1. 判断是否需要 Tool
        2. 从向量库检索相关知识
        3. 构建增强提示
        4. 路由到合适的 LLM
        """
        print("\n" + "-" * 70)
        print(f"用户查询: {user_query}")
        print("-" * 70)
        
        # 1. 检查是否需要 Tool
        tool_results = []
        if ra is not None and dec is not None:
            print("\n[步骤 1] 执行工具查询...")
            
            # 查询消光
            extinction_result = self.tools.query_extinction_data(ra, dec)
            if extinction_result['success']:
                tool_results.append(extinction_result)
                print(f"  ✓ 消光数据: {extinction_result['data']}")
            
            # 查询 Gaia
            gaia_result = self.tools.query_gaia(ra, dec, radius=0.1)
            if gaia_result['success']:
                tool_results.append(gaia_result)
                print(f"  ✓ Gaia 数据: {gaia_result['count']} 个源")
        
        # 2. RAG 检索
        print("\n[步骤 2] 检索相关知识...")
        relevant_docs = self.vector_db.search(user_query, top_k=3)
        
        if relevant_docs:
            print(f"  ✓ 找到 {len(relevant_docs)} 篇相关文档:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"    {i}. {doc.get('metadata', {}).get('topic', 'Unknown')} "
                      f"(相似度: {doc.get('similarity_score', 0):.3f})")
        
        # 3. 构建增强提示
        context_parts = []
        
        # 添加工具结果
        if tool_results:
            context_parts.append("[观测数据]")
            for result in tool_results:
                context_parts.append(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 添加 RAG 结果
        if relevant_docs:
            context_parts.append("\n[相关知识]")
            for doc in relevant_docs:
                content = doc.get('content', '')[:500]  # 限制长度
                context_parts.append(f"主题: {doc.get('metadata', {}).get('topic', '')}\n{content}\n")
        
        # 构建最终提示
        enhanced_prompt = f"""基于以下信息回答用户问题：

{chr(10).join(context_parts)}

[用户问题]
{user_query}

请提供详细的分析，结合提供的数据和知识。"""
        
        # 4. 路由到 LLM
        print("\n[步骤 3] 生成回答...")
        response = self.llm_router.route(enhanced_prompt, complexity="auto")
        
        return {
            "query": user_query,
            "context_used": {
                "tools": len(tool_results),
                "documents": len(relevant_docs)
            },
            "tool_results": tool_results,
            "relevant_documents": relevant_docs,
            "response": response
        }
    
    def chat(self):
        """交互式聊天模式"""
        print("\n" + "=" * 70)
        print("天文 AI 助手 - 交互模式")
        print("=" * 70)
        print("命令:")
        print("  /exit - 退出")
        print("  /tools - 查看可用工具")
        print("  /docs - 查看知识库状态")
        print("输入查询直接提问，可包含坐标 (RA, DEC)")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n🌟 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/exit":
                    print("再见!")
                    break
                
                if user_input == "/tools":
                    tools = self.tools.get_available_tools()
                    print("\n可用工具:")
                    for tool in tools:
                        print(f"  - {tool['name']}: {tool['description']}")
                    continue
                
                if user_input == "/docs":
                    print(f"\n知识库状态:")
                    print(f"  文档数量: {len(self.vector_db.documents)}")
                    continue
                
                # 解析可能的坐标
                ra, dec = None, None
                # 简单解析：查找数字对
                import re
                coords = re.findall(r'(\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)', user_input)
                if coords:
                    ra, dec = float(coords[0][0]), float(coords[0][1])
                    print(f"  [检测到坐标: RA={ra}, DEC={dec}]")
                
                # 执行查询
                result = self.query(user_input, ra, dec)
                
                print(f"\n🤖 AI: {result['response']}")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """主函数 - 演示 RAG + Tool 系统"""
    
    # 初始化助手
    assistant = AstronomyAssistant(use_kimi=True)
    
    # 加载 Kimi 生成的文档
    assistant.load_kimi_documents()
    
    # 示例查询
    print("\n" + "=" * 70)
    print("示例查询")
    print("=" * 70)
    
    # 示例 1: 纯知识查询
    print("\n【示例 1】知识查询")
    result = assistant.query("什么是灾变变星？它们有哪些类型？")
    print(f"回答:\n{result['response'][:500]}...")
    
    # 示例 2: 带坐标的数据查询
    print("\n【示例 2】数据查询 (EV UMa)")
    result = assistant.query("分析这个天体", ra=13.1316, dec=53.8585)
    print(f"使用的上下文: {result['context_used']}")
    print(f"回答:\n{result['response'][:500]}...")
    
    # 进入交互模式
    assistant.chat()


if __name__ == "__main__":
    main()
