#!/usr/bin/env python3
"""
天文 AI 助手 - AstroSage-Llama-3.1-8B 版本
==========================================
使用天文领域专用大模型 AstroSage-Llama-3.1-8B

模型特点：
- 70B/8B 参数，基于 Llama 3.1
- 天文文献持续预训练
- AstroMLab-1 基准 89.0% (与 GPT-4 相当)
- 支持推理链（CoT）

使用方法：
1. 先运行 setup_astrosage.py 导入模型到 Ollama
2. 运行本脚本: python astro_assistant_astrosage.py

或者使用替代方案（无需下载）：
- 使用 llama3.1:8b + 天文专用系统提示
"""

import os
import sys
import json
import re
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math

class AstroRAG:
    """天文知识库 - 支持 TF-IDF 权重"""
    
    def __init__(self):
        self.docs = []
        self.index = {}       # word -> [doc_idx, ...]
        self.tf = {}          # (doc_idx, word) -> tf值
        self.df = {}          # word -> 包含该词的文档数
        self.total_docs = 0
    
    def add(self, documents: List[Dict]):
        """添加文档，计算 TF"""
        start = len(self.docs)
        for i, doc in enumerate(documents):
            idx = start + i
            self.docs.append(doc)
            text = doc.get('text', '').lower()
            words = re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', text)
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
        self.total_docs = len(self.docs)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """TF-IDF 检索"""
        words = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', query.lower()))
        if not words or self.total_docs == 0:
            return []
        
        scores = {}
        for w in words:
            if w in self.index:
                # IDF = log(总文档数 / (含该词文档数 + 1)) + 1
                idf = math.log((self.total_docs + 1) / (self.df.get(w, 1) + 1)) + 1
                for idx in self.index[w]:
                    tf = self.tf.get((idx, w), 0)
                    scores[idx] = scores.get(idx, 0) + tf * idf
        
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.docs[i], 'score': round(s, 4)} for i, s in top]


class AstroTools:
    """天文工具集 - 集成你的包"""
    
    def __init__(self):
        self._extinction = None
        self._maps = None  # 预加载的消光地图
        self._load_tools()
    
    def _load_tools(self):
        """加载你的工具"""
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from query_extinction import query_extinction, load_extinction_maps
            self._extinction = query_extinction
            # 预加载消光地图
            try:
                self._maps = load_extinction_maps(data_dir='data')
                print("✓ 消光查询工具已加载（maps 预加载成功）")
            except Exception as e:
                print(f"⚠ 消光地图加载失败: {e}")
                self._maps = None
        except Exception as e:
            print(f"⚠ 消光工具加载失败: {e}")
    
    def query_extinction(self, ra: float, dec: float) -> Optional[Dict]:
        """查询消光数据"""
        if self._extinction is None:
            return {"error": "消光查询模块未加载", "ra": ra, "dec": dec}
        if self._maps is None:
            return {"error": "消光地图未加载", "ra": ra, "dec": dec}
        try:
            return self._extinction(ra, dec, self._maps)
        except Exception as e:
            return {"error": str(e), "ra": ra, "dec": dec}
    
    def query_gaia(self, ra: float, dec: float, radius: float = 0.1, max_retries: int = 3) -> List[Dict]:
        """查询 GAIA 数据 - 支持多端点和重试"""
        import time
        
        # 多个 TAP 端点，增加可靠性
        endpoints = [
            "http://dc.zah.uni-heidelberg.de/tap/sync",
            "https://gea.esac.esa.int/tap-server/tap/sync",
            "https://gaia.ari.uni-heidelberg.de/tap/sync",
        ]
        
        query = f"""
        SELECT TOP 10 source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax,
               teff_gspphot, radius_gspphot, lum_flame
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                      CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
        AND phot_g_mean_mag IS NOT NULL
        ORDER BY phot_g_mean_mag ASC
        """
        
        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": query
        }
        
        last_error = None
        
        for attempt in range(max_retries):
            for url in endpoints:
                try:
                    r = requests.get(url, params=params, timeout=15)
                    r.raise_for_status()
                    
                    data = r.json().get('data', [])
                    return [{
                        'source_id': row[0],
                        'ra': row[1], 'dec': row[2],
                        'g_mag': row[3],
                        'bp_rp': row[4],
                        'parallax': row[5],
                        'teff': row[6],
                        'radius': row[7],
                        'luminosity': row[8]
                    } for row in data]
                    
                except requests.Timeout:
                    wait_time = 2 ** attempt
                    print(f"  GAIA 查询超时（{url}），{wait_time}s 后重试...")
                    time.sleep(wait_time)
                    last_error = "timeout"
                    break  # 切换到下一个端点
                    
                except requests.ConnectionError:
                    print(f"  GAIA 连接失败（{url}），尝试下一端点...")
                    last_error = "connection"
                    continue
                    
                except Exception as e:
                    print(f"  GAIA 查询错误（{url}）: {type(e).__name__}: {e}")
                    last_error = str(e)
                    continue
        
        print(f"  ⚠ 所有 GAIA 端点均不可用（最后错误: {last_error}）")
        return []
    
    def analyze_sky_region(self, ra: float, dec: float) -> Dict:
        """综合分析天区"""
        result = {
            'coordinates': {'ra': ra, 'dec': dec},
            'extinction': self.query_extinction(ra, dec),
            'gaia_sources': self.query_gaia(ra, dec),
            'analysis_time': str(datetime.now()) if 'datetime' in dir() else 'N/A'
        }
        return result


class AstroSageAssistant:
    """
    AstroSage-Llama-3.1-8B 天文助手
    
    特色：
    - 使用天文专用大模型
    - 深度集成你的工具包
    - RAG 增强知识检索
    - 支持推理链（Chain-of-Thought）
    """
    
    def __init__(self, model: str = "astrosage-llama-3.1-8b"):
        self.model = model
        self.host = "http://localhost:11434"
        self.rag = AstroRAG()
        self.tools = AstroTools()
        
        # AstroSage 专用系统提示
        self.system_prompt = """You are AstroSage, an expert AI assistant specializing in astronomy, astrophysics, and space science.

Your training includes extensive astronomical literature, enabling you to:
- Analyze variable stars, binary systems, and exoplanets with precision
- Interpret light curves, spectra, and time-series data
- Query and analyze data from GAIA, SDSS, ZTF, TESS, and other surveys
- Provide detailed observing strategies and recommendations
- Explain complex astrophysical concepts clearly

When responding:
1. Use domain-specific terminology accurately
2. Include quantitative estimates with uncertainties when possible
3. Cite specific surveys, catalogs, or literature where relevant
4. Consider observational constraints (extinction, sky brightness, etc.)
5. If coordinates are provided, analyze the specific sky region

You have access to real-time astronomical databases through function calls.
Always use available tools to enrich your analysis with current data."""
        
        print("=" * 70)
        print("AstroSage-Llama-3.1-8B 天文助手")
        print("=" * 70)
        print(f"模型: {model}")
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """构建天文知识库"""
        print("\n构建天文知识库...")
        
        # 天文专业知识（AstroSage 增强版）
        knowledge = [
            {
                "text": """Cataclysmic Variables (CVs) are interacting binary systems with a white dwarf primary accreting from a Roche-lobe filling secondary star. 
Key subtypes: Novae (thermonuclear runaway), Dwarf Novae (disk instability), Nova-likes (steady accretion), Polars/Intermediate Polars (magnetic accretion).
Observational characteristics: Recurrent outbursts (2-6 mag for DNe, >10 mag for novae), orbital periods 78 min to ~10 hours, strong UV/X-ray emission.
Key surveys: ZTF, ASAS-SN, Gaia Alerts for discovery; TESS for short-cadence light curves.
Physical parameters: Mass transfer rates 10^-11 to 10^-8 Msun/yr, white dwarf masses 0.6-1.4 Msun.""",
                "topic": "CV",
                "tags": ["cataclysmic", "variable", "white dwarf", "accretion"]
            },
            {
                "text": """Cepheid Variables are pulsating supergiants with Period-Luminosity (P-L) relation: M_V = -2.81 log(P) - 1.43 (classical) or M_V = -2.43 log(P) - 1.62 (Type II).
Periods: 1-50 days (Classical), 10-100 days (Type II/W Virginis). Amplitudes: 0.5-2 mag in V.
Applications: Primary distance indicators, H_0 determination, galactic structure.
Key catalogs: Gaia DR3 Cepheid sample, OGLE-IV (Magellanic Clouds), HST samples (extragalactic).
Systematics: Metallicity dependence, reddening, crowding in dense fields.""",
                "topic": "Cepheid",
                "tags": ["cepheid", "distance", "period-luminosity", "standard candle"]
            },
            {
                "text": """Eclipsing Binaries (EBs) provide fundamental stellar parameters through light curve analysis.
Types: Algol (semidetached, distinct eclipses), Beta Lyrae (contact/semidetached, continuous variation), W UMa (contact, nearly equal depths).
Analysis methods: Wilson-Devinney code, PHOEBE, JKTEBOP.
Derive: Masses (M1, M2), radii (R1, R2), temperatures (T1, T2), orbital inclination (i), eccentricity (e).
TESS revolution: 2-min cadence enables asteroseismology of binary components.
Gaia contribution: Astrometric orbits, distance-independent mass estimates.""",
                "topic": "EB",
                "tags": ["binary", "eclipsing", "light curve", "stellar parameters"]
            },
            {
                "text": """Interstellar Extinction follows: A_V = 3.1 × E(B-V) = R_V × E(B-V), with R_V ≈ 3.1 for diffuse ISM.
Total-to-selective extinction: R_V varies 2.1-5.5 depending on environment.
Correction methods: SFD98 (Schlegel et al.) all-sky map based on IRAS/DIRBE, Bayestar19 (3D dust map), GAIA XP spectra (star-by-star).
Impact on variables: Systematic errors in P-L relations, distance moduli, color-temperature calibrations.
Best practices: Use 3D dust maps when distance known, multiple bands for reddening-independent indices (Wesenheit magnitude).""",
                "topic": "extinction",
                "tags": ["extinction", "reddening", "dust", "A_V", "E(B-V)"]
            },
            {
                "text": """Gaia DR3 Variable Star Analysis: 
15,541,806 variable candidates classified into 25+ types.
Epoch photometry: Typically 30-100 epochs over 34 months for G<17 mag.
Variability metrics: RUWE (astrometric goodness-of-fit), excess astrometric noise.
Specific classes: Cepheids (8,951), RR Lyrae (270,905), eclipsing binaries (2,245,514), long-period variables (1,447,273).
Value-added: Periods, amplitudes, Fourier parameters, classifications from SOS (Specific Object Study).
Limitations: Saturation at G<6, crowding in dense regions, 10% completeness at G~20.""",
                "topic": "Gaia",
                "tags": ["Gaia", "DR3", "variability", "survey"]
            },
            {
                "text": """Time-Domain Surveys for Variable Stars:
ZTF (Zwicky Transient Facility): 3-day cadence, g/r filters, mag limit 20.5, 47% sky coverage.
ASAS-SN (All-Sky Automated Survey): 2-3 day cadence, V filter, ~17 mag limit, all-sky coverage.
TESS (Transiting Exoplanet Survey Satellite): 2-min or 10-min cadence, 27-day sectors, I_c filter, 6.5 < T < 16.
Legacy surveys: ASAS (10+ years), NSVS, Catalina, MACHO/OGLE/EROS (Magellanic Clouds).
Selection effects: Cadence-dependent period sensitivity, magnitude limits, galactic latitude bias.""",
                "topic": "surveys",
                "tags": ["ZTF", "ASAS-SN", "TESS", "time-domain", "survey"]
            }
        ]
        
        self.rag.add(knowledge)
        
        # 加载本地训练数据
        train_path = "langgraph_demo/output/training_dataset.json"
        if os.path.exists(train_path):
            with open(train_path) as f:
                data = json.load(f)
            
            local = []
            for item in data:
                c = item.get('input', {}).get('coordinates', {})
                o = item.get('output', {})
                text = f"Object at ({c.get('ra')}, {c.get('dec')}): "
                text += f"Classified as {o.get('classification')}. "
                text += f"Analysis indicates: {'. '.join(o.get('reasoning', []))}"
                local.append({"text": text, "topic": "local_data"})
            
            self.rag.add(local)
        
        print(f"✓ 知识库: {len(self.rag.docs)} 条\n")
    
    def query(self, user_query: str, ra: float = None, dec: float = None,
              use_cot: bool = True, use_tools: bool = True) -> Dict:
        """
        处理查询
        
        Args:
            use_cot: 使用推理链（Chain-of-Thought）
            use_tools: 使用天文工具查询实时数据
        """
        context_parts = []
        
        # RAG 检索
        docs = self.rag.search(user_query)
        if docs:
            context_parts.append("[Astronomical Knowledge Base]")
            for doc in docs:
                context_parts.append(f"• {doc['text'][:400]}...")
        
        # 工具查询
        tool_data = {}
        if use_tools and ra is not None:
            context_parts.append(f"\n[Real-time Data for RA={ra:.4f}, DEC={dec:.4f}]")
            
            # 消光
            ext = self.tools.query_extinction(ra, dec)
            if ext:
                tool_data['extinction'] = ext
                context_parts.append(f"Extinction: {json.dumps(ext, ensure_ascii=False)[:500]}")
            
            # GAIA
            gaia = self.tools.query_gaia(ra, dec)
            if gaia:
                tool_data['gaia'] = gaia
                context_parts.append(f"GAIA sources: {len(gaia)} found")
                for s in gaia[:3]:
                    context_parts.append(f"  G={s['g_mag']:.2f}, BP-RP={s['bp_rp']:.3f}, "
                                       f"Teff={s.get('teff', 'N/A')}")
        
        # 构建增强提示
        context = "\n".join(context_parts)
        
        if use_cot:
            # 推理链提示
            prompt = f"""{self.system_prompt}

{'='*60}
{context}
{'='*60}

User Query: {user_query}

Please analyze this step by step:
1. Identify the type of astronomical object or phenomenon
2. Consider relevant physical parameters and their typical ranges
3. Incorporate any real-time data provided above
4. Provide quantitative estimates where possible
5. Conclude with observational recommendations

Your analysis:"""
        else:
            prompt = f"""{self.system_prompt}

{context}

Query: {user_query}

Response:"""
        
        # 调用模型
        response = self._call_ollama(prompt)
        
        return {
            "query": user_query,
            "coordinates": {"ra": ra, "dec": dec},
            "response": response,
            "data_used": {
                "knowledge_docs": len(docs),
                "tool_results": list(tool_data.keys()),
                "cot_enabled": use_cot
            }
        }
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """调用 Ollama API"""
        try:
            import requests
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": 4096,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=180
            )
            return r.json().get('response', 'Error: No response')
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def interactive(self):
        """交互模式"""
        print("=" * 70)
        print("AstroSage 交互模式")
        print("=" * 70)
        print("Commands:")
        print("  /exit     - Exit")
        print("  /cot      - Toggle Chain-of-Thought")
        print("  /tools    - Toggle tool usage")
        print("  /evuma    - Analyze EV UMa")
        print("  /alhena   - Analyze Alhena (Gemini)")
        print("\nInput formats:")
        print("  Question only          - General knowledge query")
        print("  Question RA,DEC        - With coordinates (tools enabled)")
        print("-" * 70 + "\n")
        
        use_cot = True
        use_tools = True
        
        while True:
            try:
                user_input = input("🌌 > ").strip()
                
                if not user_input:
                    continue
                
                # 命令处理
                if user_input == "/exit":
                    print("Goodbye!")
                    break
                
                if user_input == "/cot":
                    use_cot = not use_cot
                    print(f"Chain-of-Thought: {'ON' if use_cot else 'OFF'}")
                    continue
                
                if user_input == "/tools":
                    use_tools = not use_tools
                    print(f"Tools: {'ON' if use_tools else 'OFF'}")
                    continue
                
                if user_input == "/evuma":
                    print("\nAnalyzing EV UMa (Cataclysmic Variable)...")
                    result = self.query(
                        "Analyze the cataclysmic variable EV UMa. Provide physical parameters, "
                        "accretion disk characteristics, and observing recommendations.",
                        ra=13.1316, dec=53.8585,
                        use_cot=use_cot, use_tools=use_tools
                    )
                    print(f"\n🤖 AstroSage:\n{result['response']}")
                    continue
                
                if user_input == "/alhena":
                    print("\nAnalyzing Alhena (Gamma Geminorum)...")
                    result = self.query(
                        "Analyze the bright star Alhena in Gemini.",
                        ra=99.427, dec=16.399,
                        use_cot=use_cot, use_tools=use_tools
                    )
                    print(f"\n🤖 AstroSage:\n{result['response']}")
                    continue
                
                # 解析坐标
                coords = re.search(r'([\d.]+)\s*,\s*([+-]?[\d.]+)', user_input)
                ra, dec = None, None
                if coords:
                    ra, dec = float(coords.group(1)), float(coords.group(2))
                    print(f"  [Coordinates: RA={ra}, DEC={dec}]")
                
                # 执行查询
                result = self.query(user_input, ra, dec, use_cot, use_tools)
                
                print(f"\n🤖 AstroSage:\n{result['response']}")
                
                # 显示使用的数据
                d = result['data_used']
                print(f"\n[Data: {d['knowledge_docs']} docs, "
                      f"Tools: {','.join(d['tool_results']) or 'none'}, "
                      f"CoT: {d['cot_enabled']}]")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


import requests
from datetime import datetime


def check_model_available(model_name: str) -> bool:
    """检查模型是否可用 - 精确匹配或基础名称匹配"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            # 获取输入模型的基础名称（去掉 tag）
            model_base = model_name.split(':')[0]
            for m in models:
                # 完全匹配
                if m == model_name:
                    return True
                # 基础名称匹配（去掉 tag 后）
                m_base = m.split(':')[0]
                if m_base == model_base:
                    return True
    except Exception as e:
        print(f"  检查模型可用性失败: {e}")
    return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AstroSage 天文助手")
    parser.add_argument("--model", default="astrosage-llama-3.1-8b",
                       help="Ollama 模型名称")
    parser.add_argument("--fallback", default="llama3.1:8b",
                       help="备选模型（如果 AstroSage 不可用）")
    args = parser.parse_args()
    
    # 检查 Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Error: Ollama is not running")
            print("Please run: ollama serve")
            return
    except:
        print("Error: Cannot connect to Ollama")
        return
    
    # 选择模型
    model = args.model
    if not check_model_available(model):
        print(f"⚠ Model '{model}' not found in Ollama")
        print(f"\nOptions:")
        print(f"1. Run setup_astrosage.py to import AstroSage-Llama-3.1-8B")
        print(f"2. Use fallback model: {args.fallback}")
        
        if check_model_available(args.fallback):
            choice = input(f"\nUse fallback model '{args.fallback}'? (y/n): ").strip().lower()
            if choice == 'y':
                model = args.fallback
                print(f"Using fallback: {model}\n")
            else:
                return
        else:
            print(f"Fallback model also not available.")
            print("Please pull a model: ollama pull llama3.1:8b")
            return
    else:
        print(f"✓ Using model: {model}")
    
    # 启动助手
    assistant = AstroSageAssistant(model=model)
    assistant.interactive()


if __name__ == "__main__":
    main()
