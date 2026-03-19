#!/usr/bin/env python3
"""
Ollama + RAG + Tool 轻量版
=========================
无需 FAISS 和 sentence-transformers，使用简单关键词匹配
适合快速启动和使用

功能：
1. RAG - 关键词检索知识库
2. Tool - 集成你的消光查询代码
3. 实时数据 - 查询 Gaia、消光等
4. 上下文增强 - 优化 Ollama 输入

重要：Ollama 模型不能直接微调，本方案通过上下文工程增强能力
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math

class SimpleRAG:
    """简化版 RAG - 基于 TF-IDF 权重"""
    
    def __init__(self):
        self.documents = []
        self.index = {}       # 关键词 -> 文档索引
        self.tf = {}          # (doc_idx, word) -> tf值
        self.df = {}          # word -> 包含该词的文档数
        self.total_docs = 0
    
    def add_documents(self, documents: List[Dict]):
        """添加文档，计算 TF"""
        start_idx = len(self.documents)
        
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            
            # 提取关键词
            text = doc.get('text', '')
            words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())
            word_count = len(words) if words else 1
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            # 计算 TF 并更新索引
            for w, freq in word_freq.items():
                self.tf[(idx, w)] = freq / word_count
                if w not in self.index:
                    self.index[w] = []
                    self.df[w] = 0
                if idx not in self.index[w]:
                    self.index[w].append(idx)
                    self.df[w] += 1
        
        self.total_docs = len(self.documents)
        print(f"  添加 {len(documents)} 条文档，总计 {self.total_docs} 条")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """TF-IDF 搜索"""
        query_words = set(re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}', query.lower()))
        
        if not query_words or self.total_docs == 0:
            return []
        
        # TF-IDF 评分
        scores = {}
        for word in query_words:
            if word in self.index:
                # IDF = log(总文档数 / (含该词文档数 + 1)) + 1
                idf = math.log((self.total_docs + 1) / (self.df.get(word, 1) + 1)) + 1
                for idx in self.index[word]:
                    tf = self.tf.get((idx, word), 0)
                    scores[idx] = scores.get(idx, 0) + tf * idf
        
        # 排序并返回
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            doc = self.documents[idx].copy()
            doc['match_score'] = round(score, 4)
            results.append(doc)
        
        return results


class AstronomyTools:
    """天文数据工具"""
    
    def __init__(self):
        self.extinction_query = None
        self._maps = None  # 预加载的消光地图
        self._init()
    
    def _init(self):
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from query_extinction import query_extinction, load_extinction_maps
            self.extinction_query = query_extinction
            # 预加载消光地图
            try:
                self._maps = load_extinction_maps(data_dir='data')
                print("✓ 消光查询工具就绪（maps 预加载成功）")
            except Exception as e:
                print(f"⚠ 消光地图加载失败: {e}")
                self._maps = None
        except Exception as e:
            print(f"⚠ 消光查询工具: {e}")
    
    def get_extinction(self, ra: float, dec: float) -> Optional[Dict]:
        """获取消光数据"""
        if self.extinction_query is None:
            return {"error": "消光查询模块未加载", "ra": ra, "dec": dec}
        if self._maps is None:
            return {"error": "消光地图未加载", "ra": ra, "dec": dec}
        try:
            return self.extinction_query(ra, dec, self._maps)
        except Exception as e:
            print(f"  消光查询错误: {e}")
            return {"error": str(e), "ra": ra, "dec": dec}
    
    def get_gaia_sources(self, ra: float, dec: float, radius: float = 0.1) -> List[Dict]:
        """简单 Gaia 查询"""
        try:
            import requests
            
            query = f"""
            SELECT TOP 5 source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(POINT('ICRS', ra, dec), 
                          CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
            AND phot_g_mean_mag IS NOT NULL
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
                if 'data' in data:
                    sources = []
                    for row in data['data'][:5]:
                        sources.append({
                            'source_id': row[0],
                            'ra': row[1],
                            'dec': row[2],
                            'g_mag': row[3],
                            'bp_rp': row[4],
                            'parallax': row[5]
                        })
                    return sources
            
            return []
        except Exception as e:
            print(f"  Gaia 查询错误: {e}")
            return []


class OllamaRAGSystem:
    """Ollama + RAG + Tool 系统"""
    
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
        self.host = "http://localhost:11434"
        self.rag = SimpleRAG()
        self.tools = AstronomyTools()
        
        # 专业系统提示
        self.system_prompt = """你是一位资深天文学家，专门研究变星和双星。

核心能力：
1. 精确的天体分类（具体到子类型）
2. 物理参数分析（周期、光度、质量、距离）
3. 观测建议（望远镜参数、滤镜、曝光时间）
4. 数据解读（GAIA、消光、测光数据）

回答要求：
- 使用专业术语但解释清楚
- 数值必须带单位
- 基于提供的数据进行分析
- 不确定时说明限制"""
        
        print("=" * 70)
        print("Ollama + RAG + Tool 系统")
        print("=" * 70)
        print(f"模型: {model}")
        print("✓ 系统初始化完成\n")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("构建知识库...")
        
        # 内置专业知识
        knowledge = [
            {
                "text": """灾变变星 (Cataclysmic Variables, CV) 是包含白矮星和正常伴星的密近双星系统。
白矮星通过吸积盘吸积伴星物质。分类包括：新星 (Novae)、矮新星 (Dwarf Novae, DN)、
类新星 (Nova-like, NL)、磁激变变星 (Polars 和 Intermediate Polars)。
周期范围 78 分钟到数小时。光变特征包括爆发和准周期振荡。
观测建议：时序测光，时间分辨率 1-5 分钟，V/R/I 滤镜。""",
                "topic": "CV",
                "keywords": ["灾变变星", "cataclysmic", "CV", "白矮星", "吸积盘"]
            },
            {
                "text": """造父变星 (Cepheid Variables) 是经典的标准烛光，用于测量宇宙距离。
周光关系：M_v = -2.81 log(P) - 1.43，其中 P 是周期（天）。
周期范围 1-50 天，典型光变幅度 0.5-2 等。
分为经典造父变星 (Type I) 和室女座W型变星 (Type II)。
应用于哈勃常数测量、星系距离测定。""",
                "topic": "Cepheid",
                "keywords": ["造父变星", "cepheid", "周光关系", "标准烛光", "距离"]
            },
            {
                "text": """食双星 (Eclipsing Binaries) 是双星系统，轨道平面与视线对齐导致周期性掩食。
大陵型 (Algol-type)：主星被次星掩食，深度不等。
渐台型 (Beta Lyrae-type)：两星变形，连续光变。
大熊座W型 (W UMa-type)：相接双星，周期短（<1天），等深掩食。
通过测光曲线可推导：轨道周期、恒星半径比、倾角、表面温度比。""",
                "topic": "EB",
                "keywords": ["食双星", "eclipsing", "binary", "掩食", "测光"]
            },
            {
                "text": """消光 (Extinction) 是星际尘埃对星光的吸收和散射。
常用参数：E(B-V) 色余，A_V = 3.1 × E(B-V) 是 V 波段消光。
来源：SFD (Schlegel, Finkbeiner & Davis) 1998 全天消光图，
或基于 GAIA 银盘消光模型。
影响：变星测光需改正消光，影响周期-光度关系。""",
                "topic": "extinction",
                "keywords": ["消光", "extinction", "reddening", "E(B-V)", "A_V"]
            },
            {
                "text": """GAIA DR3 提供高精度天体测量和测光数据。
关键参数：G 星等（宽带，330-1050nm），BP（蓝端，330-680nm），
RP（红端，640-1050nm），视差（距离），自行。
变星数据：Epoch Photometry，分类标签，周期。
应用：变星识别、双星轨道测定、距离校准。""",
                "topic": "GAIA",
                "keywords": ["GAIA", "DR3", "星等", "视差", "测光"]
            },
        ]
        
        self.rag.add_documents(knowledge)
        
        # 尝试加载本地训练数据
        data_path = "langgraph_demo/output/training_dataset.json"
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                local_docs = []
                for item in data:
                    coords = item.get('input', {}).get('coordinates', {})
                    output = item.get('output', {})
                    
                    text = f"天体 {coords.get('ra')}, {coords.get('dec')}: "
                    text += f"分类 {output.get('classification')}. "
                    text += f"分析: {'; '.join(output.get('reasoning', []))}"
                    
                    local_docs.append({
                        "text": text,
                        "topic": "local",
                        "source": data_path
                    })
                
                self.rag.add_documents(local_docs)
            except Exception as e:
                print(f"  加载本地数据失败: {e}")
        
        print(f"✓ 知识库就绪: {len(self.rag.documents)} 条\n")
    
    def query(self, user_query: str, ra: float = None, dec: float = None,
              use_rag: bool = True, use_tools: bool = True) -> Dict:
        """处理查询"""
        
        context_parts = []
        
        # RAG 检索
        if use_rag:
            docs = self.rag.search(user_query, top_k=3)
            if docs:
                context_parts.append("[天文知识库]")
                for doc in docs:
                    context_parts.append(f"• {doc['text'][:300]}...")
        
        # 工具查询
        if use_tools and ra is not None and dec is not None:
            context_parts.append(f"\n[实时数据 - 坐标 {ra:.4f}, {dec:.4f}]")
            
            # 消光
            ext = self.tools.get_extinction(ra, dec)
            if ext:
                context_parts.append(f"消光数据: {json.dumps(ext, ensure_ascii=False)}")
            
            # Gaia
            gaia = self.tools.get_gaia_sources(ra, dec)
            if gaia:
                context_parts.append(f"GAIA 近邻源 ({len(gaia)} 个):")
                for s in gaia[:3]:
                    context_parts.append(f"  G={s['g_mag']:.2f}, BP-RP={s['bp_rp']:.3f}, "
                                       f"π={s['parallax']:.2f}mas")
        
        # 构建提示
        context = "\n".join(context_parts)
        
        prompt = f"""{self.system_prompt}

{'='*60}
{context}
{'='*60}

[用户问题]
{user_query}

请提供专业、详细的分析。如有实时数据，请结合分析。"""
        
        # 调用 Ollama
        response = self._call_ollama(prompt)
        
        return {
            "query": user_query,
            "coordinates": {"ra": ra, "dec": dec},
            "response": response,
            "context": {
                "rag_used": use_rag and len(docs) > 0,
                "tools_used": use_tools and ra is not None,
                "docs_count": len(docs) if use_rag else 0
            }
        }
    
    def _call_ollama(self, prompt: str) -> str:
        """调用 Ollama API"""
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "options": {
                "num_ctx": 8192,
                "num_predict": 2048
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            return response.json().get('response', '')
        except Exception as e:
            return f"错误: {e}"
    
    def interactive(self):
        """交互模式"""
        print("=" * 70)
        print("交互模式 - 命令:")
        print("  /exit     - 退出")
        print("  /rag      - 开关 RAG")
        print("  /tools    - 开关 Tools")
        print("  /evuma    - 分析 EV UMa 示例")
        print("输入格式: 查询 或 查询 RA,DEC")
        print("=" * 70 + "\n")
        
        use_rag = True
        use_tools = True
        
        while True:
            try:
                user_input = input("🌟 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/exit":
                    print("再见!")
                    break
                
                if user_input == "/rag":
                    use_rag = not use_rag
                    print(f"RAG: {'开' if use_rag else '关'}")
                    continue
                
                if user_input == "/tools":
                    use_tools = not use_tools
                    print(f"Tools: {'开' if use_tools else '关'}")
                    continue
                
                if user_input == "/evuma":
                    print("分析 EV UMa (13.1316, 53.8585)...")
                    result = self.query("分析 EV UMa 这个天体", ra=13.1316, dec=53.8585)
                    print(f"\n🤖 AI:\n{result['response']}")
                    continue
                
                # 解析坐标
                coords = re.findall(r'([\d.]+)\s*,\s*([+-]?[\d.]+)', user_input)
                ra, dec = None, None
                if coords:
                    ra, dec = float(coords[0][0]), float(coords[0][1])
                    print(f"  [坐标: RA={ra}, DEC={dec}]")
                
                # 查询
                result = self.query(user_input, ra, dec, use_rag, use_tools)
                
                print(f"\n🤖 AI:\n{result['response']}")
                
                ctx = result['context']
                print(f"\n[使用: RAG={ctx['rag_used']}, Tools={ctx['tools_used']}, 文档={ctx['docs_count']}]")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")


import requests


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama + RAG + Tool")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama 模型名")
    args = parser.parse_args()
    
    # 检查 Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama 未运行")
            return
    except:
        print("无法连接到 Ollama，请运行: ollama serve")
        return
    
    # 启动系统
    system = OllamaRAGSystem(model=args.model)
    system.build_knowledge_base()
    system.interactive()


if __name__ == "__main__":
    main()
