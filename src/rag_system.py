#!/usr/bin/env python3
"""
RAG知识库系统
=============
天文专业知识库，支持变星、白矮星、激变变星等查询
增强版：支持从文件加载知识库，包含 AM CVn 等关键知识
"""

import os
import json
import glob
from typing import Dict, List, Optional
from pathlib import Path


class AstronomyRAG:
    """
    天文知识库 (RAG - Retrieval Augmented Generation)
    
    包含天体物理专业知识，支持关键词检索
    支持从文本文件加载外部知识库
    """
    
    def __init__(self, kb_path: str = None, knowledge_dir: str = None):
        """
        初始化知识库
        
        Args:
            kb_path: 知识库JSON文件路径，None则使用默认知识
            knowledge_dir: 知识库文本文件目录，默认 astro_knowledge/
        """
        self.kb_path = kb_path
        self.knowledge_dir = knowledge_dir or self._find_knowledge_dir()
        self.knowledge: Dict[str, str] = {}
        self.documents: List[Dict] = []  # 用于更细粒度的检索
        self._load_knowledge()
    
    def _find_knowledge_dir(self) -> str:
        """自动查找知识库目录"""
        # 可能的目录位置
        possible_dirs = [
            "astro_knowledge",
            "../astro_knowledge",
            "./astro_knowledge",
            "/mnt/c/Users/Administrator/Desktop/astro-ai-demo/astro_knowledge"
        ]
        for d in possible_dirs:
            if os.path.isdir(d):
                return d
        # 默认返回项目根目录下的 astro_knowledge
        return "astro_knowledge"
    
    def _load_knowledge(self):
        """加载知识库"""
        # 1. 加载默认内置知识
        self.knowledge = self._default_knowledge()
        
        # 2. 如果指定了JSON文件，加载它
        if self.kb_path and os.path.exists(self.kb_path):
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                file_knowledge = json.load(f)
                self.knowledge.update(file_knowledge)
        
        # 3. 从文本文件加载知识库
        self._load_text_knowledge()
        
        print(f"✓ RAG知识库已加载: {len(self.knowledge)} 个主题, {len(self.documents)} 个文档块")
    
    def _load_text_knowledge(self):
        """从文本文件加载知识"""
        if not os.path.isdir(self.knowledge_dir):
            return
        
        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用文件名作为主题键
                topic_key = os.path.splitext(os.path.basename(filepath))[0]
                
                # 添加到知识库
                if topic_key not in self.knowledge:
                    self.knowledge[topic_key] = content
                
                # 按段落切分为文档块，用于更细粒度的检索
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if len(para) > 30:  # 只保留有意义的段落
                        self.documents.append({
                            'content': para,
                            'source': filepath,
                            'topic': topic_key
                        })
                        
            except Exception as e:
                print(f"  ⚠ 加载知识文件失败 {filepath}: {e}")
    
    def _default_knowledge(self) -> Dict[str, str]:
        """默认天文知识库 - 包含关键修正"""
        return {
            "am_cvn": """
AM CVn Systems - CRITICAL FACTS (Common errors to avoid!):

【系统组成 - 重要！】
- 白矮星（吸积星）+ 氦星（供体星）
- ⚠️ 绝对不含中子星！这是最常见的错误
- 超紧密双星系统，轨道周期仅 5-65 分钟

【周期-光度关系 - 注意区别！】
- 短周期 → 高质量转移率 → 更亮（统计趋势）
- 周期范围：5-65 分钟（极短！）
- ❌ 错误：不存在 ΔF/F = (1+q)*sin(...) 这样的公式
- ✅ 正确：Mdot ∝ P^(-10/3)（引力波驱动系统）

【光变原因 - 不是潮汐！】
- 吸积盘不稳定性（盘不稳定性模型 DIM）
- 超涨（superhump）：来自盘进动
- 直接撞击吸积（P < 10分钟时无吸积盘）
- ❌ 错误：不能用"潮汐互动"解释光变
- ✅ 正确：光变来自吸积过程

【关键例子】
- HM Cnc: P = 5.4 分钟（最短已知）
- AM CVn本身: P = 17.1 分钟

【引力波源】
- LISA空间任务的主要目标
- 已知参数的"验证双星"
""",
            "polar": """
Polar（激变变星/高偏振星）是强磁场的激变变星（Cataclysmic Variable, CV）:

【物理特征】
- 磁场强度: 10-100 MG（兆高斯）
- 白矮星质量: 0.5-1.0 太阳质量
- 轨道周期: 1-4小时（典型值约2小时）

【观测特征】
- 强偏振: 线性偏振可达10-30%，圆形偏振可达10-20%
- X射线辐射: 来自吸积柱的硬X射线
- 光学波段: cyclotron辐射（回旋辐射）
- 红外波段: 强烈的cyclotron发射

【光变特征】
- 准周期振荡（QPO）: 时标秒到分钟
- 轨道调制: 因白矮星自转和轨道运动
- 吸积状态变化: 高态/低态转变

【光谱特征】
- 强发射线: Hα, Hβ, He II λ4686
- cyclotron谐波: 红外到光学波段
- 高激发线: 来自吸积柱的高温气体

【子类型】
- 硬X射线Polar: 强X射线，弱光学
- 软X射线Polar: 弱X射线，强光学
- 异步Polar: 白矮星自转与轨道不同步

【著名例子】
- AM Her: 第一个被发现的Polar（1976年）
- EF Eri: 第一个被观测到cyclotron辐射的Polar
- VV Pup: 高偏振度Polar
- HU Aqr: 双吸积柱系统

【形成与演化】
- 从非磁激变变星演化而来
- 磁场抑制吸积盘形成
- 物质沿磁力线直接吸积到磁极
""",
            "cataclysmic_variable": """
激变变星（Cataclysmic Variables, CVs）是紧密双星系统:

【系统组成】
- 主星: 白矮星（质量0.6-1.2 M_sun）
- 伴星: 红矮星或亚巨星（质量<1 M_sun）
- 轨道周期: 通常<1天（1-12小时）

【物理过程】
- 洛希瓣溢出: 伴星充满洛希瓣，物质转移
- 吸积: 物质流向白矮星
- 角动量转移: 磁制动、引力辐射

【分类】
1. 新星（Novae）: 热核爆发，亮度增加6-19星等
2. 再发新星（Recurrent Novae）: 多次爆发
3. 矮新星（Dwarf Novae）: 吸积盘不稳定，亮度增加2-6星等
   - U Gem型: 标准矮新星
   - SU UMa型: 超爆发+正常爆发
   - Z Cam型: 静止态
4. 极星/Polar（AM Her型）: 强磁场，无吸积盘
5. 中介极星（Intermediate Polars）: 中等磁场，有吸积盘
6. 磁激变变星: 磁场>1 MG的系统

【演化】
- 轨道周期缺口: 2-3小时的周期分布间隙
- 周期变化: 长期演化导致周期增减
- 最终命运: 可能是Ia型超新星（如果质量接近钱德拉塞卡极限）
""",
            "rr_lyrae": """
RR Lyrae变星是水平分支上的脉动变星:

【基本特征】
- 周期: 0.2-1.0天（典型0.5天）
- 振幅: 0.3-2.0星等（V波段）
- 光谱型: A-F型
- 质量: 0.5-0.7 太阳质量
- 光度: ~50 L_sun

【脉动模式】
- 基频模式（F模式）: 径向脉动
- 第一泛音: 部分RRc型
- 双模脉动: RRd型，同时有基频和第一泛音

【子类型】
1. RRab型:
   - 不对称光变曲线
   - 快速上升，缓慢下降
   - 占RR Lyrae的约90%
   
2. RRc型:
   - 更对称的光变曲线
   - 周期较短（0.2-0.4天）
   - 振幅较小
   
3. RRd型:
   - 双模脉动
   - 两个周期比值约0.74

【周期-光度关系】
- M_v ≈ 0.6 ± 0.1
- 可作为标准烛光
- 用于测量银河系内距离

【金属丰度效应】
- 球状星团中的Oosterhoff分类
- OoI型: <[Fe/H]> ≈ -1.6
- OoII型: <[Fe/H]> ≈ -2.0
- 周期-金属丰度关系

【应用】
- 距离测量: 银河系、球状星团、矮星系
- 银河系结构: 研究银晕
- 恒星演化: 水平分支星的研究
""",
            "eclipsing_binary": """
食双星（Eclipsing Binary）是发生周期性掩食的双星:

【几何条件】
- 轨道面与视线方向夹角接近90°
- 能够观测到相互掩食
- 轨道周期: 数小时到数年

【分类】
1. 大陵五型（Algol型）:
   - 主星不充满洛希瓣
   - 典型的分离双星
   - 光变深度大

2. 渐台二型（Beta Lyrae型）:
   - 一颗星充满洛希瓣
   - 存在物质交换
   - 光变曲线连续变化

3. 大熊W型（W UMa型）:
   - 两颗星都充满洛希瓣
   - 接触或公共包层双星
   - 周期通常<1天

【光变曲线特征】
- 主极小: 较暗星遮挡较亮星，深度最大
- 次极小: 较亮星遮挡较暗星，深度较小
- 次极小深度: 反映两星温度比

【物理参数确定】
- 质量比: q = M_2/M_1
- 轨道倾角: i
- 半径比: R_2/R_1
- 表面温度比: T_2/T_1

【应用】
- 恒星质量测量: 直接测定恒星质量
- 恒星半径测量
- 表面温度测定
- 距离测量: 结合光谱分析
- 星震学: 脉动食双星
""",
            "white_dwarf": """
白矮星（White Dwarf, WD）是恒星演化的最终产物:

【基本特征】
- 质量: 0.17-1.33 M_sun（钱德拉塞卡极限）
- 半径: ~0.01 R_sun（地球大小）
- 密度: 10^6 g/cm³
- 光度: 10^-4 到 10^-1 L_sun
- 表面温度: 4000-150,000 K

【物理支持机制】
- 电子简并压力: 抵抗引力坍缩
- 非热核反应: 余热辐射

【光谱分类】
- DA型: 氢线主导（Balmer线），占80%
- DB型: 氦线主导（He I），无氢
- DC型: 连续谱，无明显谱线
- DO型: He II线主导
- DZ型: 金属线（Ca, Mg, Fe等）
- DQ型: 碳特征（C2 Swan带或CI线）
- 混合类型: DA+DB, DBA等

【冷却轨迹】
- 热WD: T_eff > 20,000 K
- 温WD: 10,000 < T_eff < 20,000 K
- 冷WD: T_eff < 10,000 K
- 冷却时标: 数十亿年

【质量-半径关系】
- R ∝ M^(-1/3)
- 质量越大，半径越小
- 相对论性修正在大质量时重要

【形成途径】
1. 单星演化:
   - 主序星 → 红巨星 → AGB → 行星状星云 → WD
   - 初始质量: 0.8-8 M_sun

2. 双星演化:
   - 洛希瓣溢出
   - 公共包层演化
   - 双白矮星并合

【应用】
- 恒星演化: 研究恒星最终命运
- 宇宙时标: WD冷却定年
- 太阳系外行星: WD行星系统
- 超新星: Ia型超新星前身星
- 星震学: WD脉动（DAV, DBV, GW Vir）
""",
            "delta_scuti": """
δ Scuti变星是A-F型主序或近主序脉动变星:

【基本特征】
- 周期: 0.02-0.3小时（18分钟到8小时）
- 振幅: 0.003-0.9星等
- 光谱型: A-F型（A2-F5）
- 脉动模式: p-mode（压力模）
- 质量: 1.5-2.5 M_sun

【脉动特性】
- 多周期脉动: 通常同时存在多个频率
- 径向和非径向脉动: l=0,1,2,...
- 频率范围: 5-80 d^-1

【子类型】
1. 高振幅δ Scuti (HADS):
   - 振幅 > 0.1 mag
   - 通常单周期或少数几个周期
   - 基频脉动为主

2. 低振幅δ Scuti (LADS):
   - 振幅 < 0.1 mag
   - 多周期脉动
   - 大量的脉动模式

【与RR Lyrae的区别】
- δ Scuti: 主序/近主序，质量较大，周期较短
- RR Lyrae: 水平分支，质量较小（0.6 M_sun），周期较长

【星震学应用】
- 研究恒星内部结构
- 测量恒星参数（质量、半径、年龄）
- 检验恒星演化模型
- 对流区研究

【银河系结构】
- 年轻星族成分
- 盘面和旋臂示踪
- 金属丰度梯度研究
"""
        }
    
    def search(self, query: str, top_k: int = 3) -> str:
        """
        搜索相关知识 - 增强版支持关键词匹配和语义检索
        
        Args:
            query: 查询关键词
            top_k: 返回最多几条知识
            
        Returns:
            相关知识文本
        """
        query_lower = query.lower()
        
        # 特殊关键词映射（常见错误纠正）- 优先匹配
        keyword_mappings = {
            "am cvn": ["am_cvn", "cataclysmic_variable", "period_luminosity_relations", "binary_systems"],
            "amcvn": ["am_cvn", "cataclysmic_variable"],
            "周期-光度": ["period_luminosity_relations", "rr_lyrae", "am_cvn"],
            "周期光度": ["period_luminosity_relations", "rr_lyrae", "am_cvn"],
            "周期范围": ["am_cvn", "period_luminosity_relations", "delta_scuti", "rr_lyrae"],
            "潮汐": ["binary_systems", "am_cvn"],
            "tidal": ["binary_systems", "am_cvn"],
            "中子星": ["am_cvn", "binary_systems"],
            "neutron star": ["am_cvn", "binary_systems"],
            "激变变星": ["cataclysmic_variable", "polar", "am_cvn"],
            "双星": ["binary_systems", "eclipsing_binary", "cataclysmic_variable"],
            "白矮星": ["white_dwarf", "cataclysmic_variable", "am_cvn"],
            "引力波": ["am_cvn", "cataclysmic_variable"],
            "gravitational wave": ["am_cvn", "cataclysmic_variable"],
            "lisa": ["am_cvn"],
        }
        
        matches = []
        priority_matches = []  # 高优先级匹配
        
        # 检查特殊关键词映射（最高优先级）
        for key, mapped_topics in keyword_mappings.items():
            if key in query_lower:
                for topic in mapped_topics:
                    if topic in self.knowledge and topic not in [m[1] for m in priority_matches]:
                        priority_matches.append((20, topic, self.knowledge[topic]))
        
        # 通用关键词匹配
        for key, content in self.knowledge.items():
            # 计算匹配分数
            score = 0
            query_words = query_lower.replace('-', ' ').replace('_', ' ').split()
            key_lower = key.lower().replace('_', ' ')
            
            # 完全匹配键名获得高分
            if query_lower in key_lower or key_lower in query_lower:
                score += 10
            
            for word in query_words:
                if len(word) < 2:  # 跳过单字
                    continue
                if word in key_lower:
                    score += 3
                if word in content.lower():
                    score += 1
            
            if score > 0:
                matches.append((score, key, content))
        
        # 从文档块中搜索
        for doc in self.documents:
            score = 0
            for word in query_lower.split():
                if len(word) < 2:
                    continue
                if word in doc['content'].lower():
                    score += 1
            if score > 0:
                matches.append((score - 0.5, doc['topic'], doc['content']))
        
        # 合并优先级匹配和普通匹配
        all_matches = priority_matches + matches
        
        # 排序并去重
        all_matches.sort(reverse=True, key=lambda x: x[0])
        
        # 去重并选择前k个
        seen_topics = set()
        results = []
        for score, key, content in all_matches:
            if key not in seen_topics:
                seen_topics.add(key)
                results.append((score, key, content))
                if len(results) >= top_k:
                    break
        
        if results:
            output = []
            for score, key, content in results:
                # 根据内容长度决定是否截断
                if len(content) > 800:
                    content = content[:800] + "...\n[内容已截断，完整内容请查看知识库文件]"
                output.append(f"【{key.upper()}】(相关度: {score:.1f})\n{content}")
            return "\n\n" + "="*60 + "\n".join(output)
        else:
            return "未找到相关知识。可用主题: " + ", ".join(self.list_topics()[:10]) + "..."
    
    def search_am_cvn(self) -> str:
        """
        专门搜索 AM CVn 相关知识（用于纠正常见错误）
        
        Returns:
            AM CVn 完整知识
        """
        am_cvn_keys = ['am_cvn', 'amcvn', 'period_luminosity_relations', 'binary_systems']
        results = []
        for key in am_cvn_keys:
            if key in self.knowledge:
                results.append(f"【{key.upper()}】\n{self.knowledge[key]}")
        
        if results:
            return "\n\n".join(results)
        return "AM CVn 知识未找到"
    
    def get(self, key: str) -> str:
        """
        获取特定主题的知识
        
        Args:
            key: 知识键名
            
        Returns:
            知识内容
        """
        # 尝试直接匹配
        if key in self.knowledge:
            return self.knowledge[key]
        
        # 尝试模糊匹配
        key_lower = key.lower()
        for k, v in self.knowledge.items():
            if key_lower in k.lower() or k.lower() in key_lower:
                return v
        
        return f"未找到 '{key}' 的知识。可用主题: {', '.join(self.list_topics())}"
    
    def list_topics(self) -> List[str]:
        """列出所有知识主题"""
        return list(self.knowledge.keys())
    
    def add_knowledge(self, key: str, content: str):
        """添加新知识"""
        self.knowledge[key] = content
        if self.kb_path:
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
    
    def get_common_mistakes_hint(self, topic: str = None) -> str:
        """
        获取常见错误提示
        
        Args:
            topic: 主题，如 'am_cvn'
            
        Returns:
            常见错误提示文本
        """
        mistakes = {
            "am_cvn": """
⚠️ AM CVn 常见错误提醒：
1. ❌ 绝不能说含有中子星 - 是白矮星+氦星
2. ❌ 绝不能用"潮汐互动"解释光变 - 是吸积盘过程
3. ❌ 绝不能编造公式 - AM CVn没有标准周期-光度公式
4. ✅ 周期范围：5-65分钟（这是关键特征）
5. ✅ 光变来自吸积盘不稳定性
""",
            "period_luminosity": """
⚠️ 周期-光度关系常见错误：
1. ❌ 不是所有变星都有严格的P-L关系
2. ✅ Cepheid: 有严格P-L关系，用于测距
3. ✅ RR Lyrae: 几乎恒定光度，M_V ≈ 0.6
4. ✅ AM CVn: 统计相关，无严格公式
5. ❌ 绝不能编造数学公式
"""
        }
        
        if topic and topic in mistakes:
            return mistakes[topic]
        return "\n".join(mistakes.values())


# ==================== 便捷函数 ====================

def get_rag(knowledge_dir: str = None) -> AstronomyRAG:
    """
    获取RAG实例
    
    Args:
        knowledge_dir: 知识库目录路径
        
    Returns:
        AstronomyRAG实例
    """
    return AstronomyRAG(knowledge_dir=knowledge_dir)


def quick_search(query: str, knowledge_dir: str = None) -> str:
    """
    快速搜索知识库
    
    Args:
        query: 查询词
        knowledge_dir: 知识库目录
        
    Returns:
        搜索结果
    """
    rag = AstronomyRAG(knowledge_dir=knowledge_dir)
    return rag.search(query)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("天文RAG知识库测试 (增强版)")
    print("=" * 70)
    
    rag = AstronomyRAG()
    
    print("\n可用主题:")
    for topic in rag.list_topics():
        print(f"  - {topic}")
    
    # 测试 AM CVn 查询
    print("\n" + "=" * 70)
    print("测试1: AM CVn 周期-光度关系")
    print("=" * 70)
    result = rag.search("AM CVn 周期-光度关系")
    print(result[:1500])
    
    # 测试常见错误提示
    print("\n" + "=" * 70)
    print("测试2: 常见错误提示")
    print("=" * 70)
    print(rag.get_common_mistakes_hint("am_cvn"))
    
    # 测试潮汐相关查询
    print("\n" + "=" * 70)
    print("测试3: 潮汐互动查询")
    print("=" * 70)
    result = rag.search("潮汐互动 光变")
    print(result[:1000])
    
    print("\n测试完成!")
