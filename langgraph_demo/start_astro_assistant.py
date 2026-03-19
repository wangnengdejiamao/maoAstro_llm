#!/usr/bin/env python3
"""
天文 AI 助手启动脚本
====================
Ollama qwen3:8b + RAG + Tool 完整方案

用法:
    python start_astro_assistant.py

功能:
1. 使用内置知识库增强问答
2. 集成你的消光查询代码
3. 查询 GAIA 实时数据
4. 无需微调模型，通过上下文工程增强能力
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleRAG:
    """简化版 RAG - 关键词匹配"""
    
    def __init__(self):
        self.documents = []
        self.index = {}
    
    def add(self, docs: List[Dict]):
        start = len(self.documents)
        for i, doc in enumerate(docs):
            idx = start + i
            self.documents.append(doc)
            text = doc.get('text', '').lower()
            words = set(re.findall(r'[a-z\u4e00-\u9fff]{2,}', text))
            for w in words:
                if w not in self.index:
                    self.index[w] = []
                self.index[w].append(idx)
        print(f"  添加 {len(docs)} 条，共 {len(self.documents)} 条")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        words = set(re.findall(r'[a-z\u4e00-\u9fff]{2,}', query.lower()))
        scores = {}
        for w in words:
            if w in self.index:
                for idx in self.index[w]:
                    scores[idx] = scores.get(idx, 0) + 1
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.documents[i], 'score': s} for i, s in top]


class AstroTools:
    """天文工具"""
    
    def __init__(self):
        self._ext = None
        try:
            from query_extinction import query_extinction
            self._ext = query_extinction
            print("✓ 消光查询工具就绪")
        except:
            print("⚠ 消光工具不可用")
    
    def extinction(self, ra: float, dec: float):
        if self._ext:
            try:
                return self._ext(ra, dec)
            except Exception as e:
                return {"error": str(e)}
        return None
    
    def gaia(self, ra: float, dec: float):
        try:
            import requests
            q = f"SELECT TOP 5 source_id,ra,dec,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},0.1))=1 ORDER BY phot_g_mean_mag"
            r = requests.get("http://dc.zah.uni-heidelberg.de/tap/sync", 
                           params={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":q},
                           timeout=30)
            if r.status_code == 200:
                data = r.json().get('data', [])
                return [{'g': row[3], 'bp_rp': row[4]} for row in data[:5]]
        except:
            pass
        return []


class AstroAssistant:
    """天文助手"""
    
    def __init__(self):
        self.model = "qwen3:8b"
        self.host = "http://localhost:11434"
        self.rag = SimpleRAG()
        self.tools = AstroTools()
        
        self.system = """你是资深天文学家，专门研究变星和双星。
能力：精确分类、参数分析、观测建议、数据解读。
回答要求：专业术语解释清楚、数值带单位、基于数据、不确定时说明。"""
        
        print("=" * 70)
        print("天文 AI 助手 (Ollama qwen3:8b + RAG + Tool)")
        print("=" * 70)
        self._build_kb()
    
    def _build_kb(self):
        """构建知识库"""
        print("\n构建知识库...")
        
        docs = [
            {"text": "灾变变星(CV)是白矮星+伴星双星系统，通过吸积盘吸积物质。分类：新星、矮新星、类新星、磁激变变星。周期78分钟到数小时。观测：时序测光，1-5分钟分辨率，V/R/I滤镜。", "topic": "CV"},
            {"text": "造父变星(Cepheid)是标准烛光，周光关系Mv=-2.81log(P)-1.43。周期1-50天，光变幅度0.5-2等。分Type I和Type II。用于哈勃常数测量、星系距离。", "topic": "Cepheid"},
            {"text": "食双星(EB)轨道平面与视线对齐致周期性掩食。大陵型(Algol)：主星被掩食；渐台型(Beta Lyrae)：连续光变；大熊W型(W UMa)：相接双星。测光曲线可推轨道参数。", "topic": "EB"},
            {"text": "消光(Extinction)是星际尘埃吸收散射。E(B-V)色余，Av=3.1×E(B-V)。来源SFD1998全天图或GAIA模型。变星测光需改正，影响周光关系。", "topic": "extinction"},
            {"text": "GAIA DR3提供高精度天体测量和测光。G星等(330-1050nm)，BP/RP分光测光，视差测距。变星数据包括Epoch测光、分类标签、周期。用于变星识别、双星测定。", "topic": "GAIA"},
        ]
        
        self.rag.add(docs)
        
        # 加载本地数据
        path = "langgraph_demo/output/training_dataset.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            local = []
            for item in data:
                c = item.get('input', {}).get('coordinates', {})
                o = item.get('output', {})
                text = f"天体({c.get('ra')},{c.get('dec')}): {o.get('classification')}. 分析: {'; '.join(o.get('reasoning', []))}"
                local.append({"text": text, "topic": "local"})
            self.rag.add(local)
        
        print(f"✓ 知识库: {len(self.rag.documents)} 条\n")
    
    def query(self, q: str, ra=None, dec=None, use_rag=True, use_tools=True):
        ctx = []
        
        if use_rag:
            docs = self.rag.search(q)
            if docs:
                ctx.append("[知识库]\n" + "\n".join(f"• {d['text'][:250]}" for d in docs))
        
        if use_tools and ra is not None:
            ctx.append(f"\n[实时数据 {ra:.4f},{dec:.4f}]")
            ext = self.tools.extinction(ra, dec)
            if ext:
                ctx.append(f"消光: {json.dumps(ext, ensure_ascii=False)}")
            gaia = self.tools.gaia(ra, dec)
            if gaia:
                ctx.append(f"GAIA源: " + "; ".join(f"G={s['g']:.2f}" for s in gaia[:3]))
        
        context_str = '\n'.join(ctx)
        prompt = f"{self.system}\n\n{'='*50}\n{context_str}\n{'='*50}\n\n问题: {q}\n\n请详细分析。"
        
        # 调用 Ollama
        try:
            import requests
            r = requests.post(f"{self.host}/api/generate",
                            json={"model": self.model, "prompt": prompt, "stream": False,
                                  "temperature": 0.7, "options": {"num_ctx": 8192, "num_predict": 2048}},
                            timeout=120)
            return r.json().get('response', '错误')
        except Exception as e:
            return f"错误: {e}"
    
    def run(self):
        """交互模式"""
        print("命令: /exit, /rag, /tools, /evuma")
        print("输入: 查询 或 查询 RA,DEC")
        print("-" * 70 + "\n")
        
        use_rag, use_tools = True, True
        
        while True:
            try:
                inp = input("🌟 你: ").strip()
                if not inp:
                    continue
                
                if inp == "/exit":
                    print("再见!")
                    break
                if inp == "/rag":
                    use_rag = not use_rag
                    print(f"RAG: {'开' if use_rag else '关'}")
                    continue
                if inp == "/tools":
                    use_tools = not use_tools
                    print(f"Tools: {'开' if use_tools else '关'}")
                    continue
                if inp == "/evuma":
                    print("🤖 AI:\n" + self.query("分析EV UMa", 13.1316, 53.8585))
                    continue
                
                # 解析坐标
                m = re.search(r'([\d.]+)\s*,\s*([+-]?[\d.]+)', inp)
                ra, dec = (float(m.group(1)), float(m.group(2))) if m else (None, None)
                
                print(f"🤖 AI:\n{self.query(inp, ra, dec, use_rag, use_tools)}")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    # 检查 Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200 or "qwen3:8b" not in str(r.json()):
            print("错误: Ollama 未运行或没有 qwen3:8b")
            return
    except:
        print("错误: 无法连接到 Ollama")
        print("请运行: ollama serve")
        return
    
    # 启动
    assistant = AstroAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
