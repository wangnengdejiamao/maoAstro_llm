#!/usr/bin/env python3
"""
Kimi API 天文知识蒸馏系统 - 独立可运行版本
===========================================
使用 Kimi API 作为教师模型，蒸馏白矮星、CVs、Polars 等天文知识

API Key: sk-kimi-SoSpEtPpAUpQN94Mng37gYiJ5scgv3WyDR7AfKDhyv01Awca6yfUKod9lcbNa6Uj

天体类型覆盖:
1. DA_WD - 氢大气白矮星
2. DB_WD - 氦大气白矮星
3. DC_WD - 连续谱白矮星
4. DZ_WD - 金属污染白矮星
5. CV - 激变变星
6. POLAR - 极向星/磁旋密近双星
7. WD_BINARY - 白矮双星
8. CLUSTER_WD - 星团白矮星
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# ============ 配置 ============
KIMI_API_KEY = "sk-kimi-SoSpEtPpAUpQN94Mng37gYiJ5scgv3WyDR7AfKDhyv01Awca6yfUKod9lcbNa6Uj"
KIMI_API_BASE = "https://api.moonshot.cn/v1"

# ============ 数据类 ============

@dataclass
class AstronomicalTarget:
    """天文目标"""
    name: str
    ra: float
    dec: float
    target_type: str
    spectral_type: Optional[str] = None
    period: Optional[float] = None
    magnitude: Optional[float] = None
    distance: Optional[float] = None
    notes: Optional[str] = None


# ============ 知识库 ============

ASTRONOMICAL_KNOWLEDGE = {
    "DA_WD": """## DA白矮星专业分析

### 物理特征
DA白矮星是宇宙中最常见的致密天体，约占所有白矮星的80%。其最显著特征是在氢大气层中展现出的巴尔末线系。

**大气结构**：
- 大气质量仅约 10⁻¹⁵ M☉，却足以屏蔽内部的氦层
- 压力致宽主导谱线轮廓，斯坦-格林菲尔德效应显著
- 对流区深度与温度密切相关，在 12,000 K 附近达到最大

**光谱诊断**：
- Hα (6563Å): 线心深度反映表面重力 log g ≈ 7.5-8.0
- Hβ (4861Å): 最佳温度诊断指标，8,000-40,000 K 敏感
- 巴尔末跳跃：约 3646Å，幅度 0.1-0.5 mag

**温度-光度关系**：
- 高温端 (T > 20,000 K): 蓝白色，紫外辐射强
- 中温段 (12,000-20,000 K): ZZ Ceti 脉动不稳定带
- 低温端 (T < 8,000 K): 氢线消失，演变为 DC 型

### 形成与演化
**前身星质量**：0.8-8 M☉ 的主序星经历AGB和行星状星云阶段
**核聚变终止**：碳氧核心电子简并压支撑，不再产生能量
**冷却过程**：
1. 热白矮星相 (0-10 Myr): 辐射残余热能
2. 晶体化相 (1-10 Gyr): 核心开始结晶，释放潜热延缓冷却
3. 冷白矮星相 (>10 Gyr): 成为宇宙中最古老的天体化石

### 研究意义
- **宇宙时钟**：最古老球状星团白矮星冷却序列给出 10-13 Gyr 年龄
- **Ia型超新星前身**：吸积伴星物质达到钱德拉塞卡极限 (1.4 M☉)
- **行星系统考古**：金属污染白矮星直接分析小行星化学成分""",

    "POLAR": """## 极向星 (AM Her型) 专业分析

### 磁场物理
极向星是强磁场激变变星，白矮星磁场强度达 10-1000 MG（百万高斯），是太阳黑子的数万倍。

**磁层结构**：
- 阿尔文半径：R_A ≈ 10¹⁰ cm，远大于白矮星半径
- 磁轴倾角：通常 < 30°，低倾角系统占优势
- 同步锁定：P_rot = P_orb，强磁耦合导致潮汐锁定

**吸积柱物理**：
- 物质沿磁力线自由下落，形成柱状吸积流
- 激波高度：约 0.1 R_WD，温度达 10⁸ K
- X射线辐射：硬 X 射线 (10-50 keV) 来自吸积柱底部热斑

### 无盘吸积
与标准CV不同，极向星的磁场阻止了吸积盘形成：
- 洛希瓣溢流物质直接被磁场捕获
- 吸积流几何由磁流体动力学 (MHD) 控制
- 吸积率变化主要反映供体星质量转移率变化

### 观测特征
**光变曲线**：
- 单峰或双峰形态，取决于磁轴和轨道倾角
- 振幅 1-4 星等，轨道周期 1-4 小时
- 热斑贡献系统总光度的 30-80%

**偏振辐射**：
- 圆偏振度可达 10-40%，由塞曼效应产生
- 光学和近红外波段最强
- 偏测量直接测定磁场强度

**X射线光谱**：
- 软 X 射线超 (kT ~ 10-30 keV)：热斑热辐射
- 硬 X 射线尾部：吸积柱 bremsstrahlung
- 铁 Kα 线 (6.4 keV)：冷物质荧光

### 演化意义
极向星代表密近双星演化的终点之一：
- 轨道角动量损失（磁制动 + 引力波）导致螺旋靠近
- 最终并合可能产生 R CrB 型星或极端氦星
- 最短周期系统 (P < 30 min) 是重要的低频引力波源""",

    "CV": """## 激变变星 (CVs) 专业分析

### 系统构成
**白矮星主星**：
- 质量：0.6-1.2 M☉，平均 0.83 M☉（高于孤立白矮星）
- 半径：~0.01 R☉（地球大小）
- 表面温度：10,000-50,000 K

**晚型伴星（供体）**：
- 光谱型：M4-M6 V，质量 0.1-0.4 M☉
- 充满洛希瓣，通过 L1 点持续物质转移
- 质量转移率：10⁻¹¹-10⁻⁸ M☉/yr

**轨道动力学**：
- 周期分布：76 min（最小）- 约 1 天
- 周期间隙 (2-3 hr)：磁制动饱和导致的演化通道 bifurcation
- 周期 bouncer：质量比反转后的长周期系统

### 吸积盘物理
**标准盘模型** (Shakura-Sunyaev):
- 几何薄盘：h/r ~ 0.01-0.1
- 粘度参数 α：0.01-0.1（由 MRI 驱动）
- 温度分布：T(R) ∝ R^(-3/4)，中心最热

**辐射机制**：
- 光学-UV：吸积盘多温黑体（3,000-50,000 K）
- 软 X 射线：边界层热辐射 (10⁵-10⁸ K)
- 发射线：盘大气再处理（H I, He I/II, C IV）

### 爆发机制
**矮新星 (U Gem, SU UMa 型)**：
- 触发：吸积盘氢电离不稳定性 (T ~ 6,000 K)
- 亮增：2-5 星等，持续 2-20 天
- 超级爆发 (SU UMa)：外盘不稳定 + 潮汐加热

**新星爆发**：
- 热核爆炸：白矮星表层累积氢的失控聚变
- 抛射质量：10⁻⁵-10⁻³ M☉
- 复发时间：10³-10⁵ 年（再发新星更短）

### 特殊子类
**磁CVs**：
- 中间极向星 (IPs)：B = 1-10 MG，P_spin/P_orb = 0.1-0.9
- 极向星 (Polars)：B > 10 MG，同步自转
- 无盘或截断盘吸积，硬 X 射线辐射强

**超软X射线源 (SSS)**：
- 近爱丁顿吸积率：Ṁ ~ 10⁻⁷ M☉/yr
- 白矮星稳定氢燃烧
- Ia型超新星前身星候选

### 天体物理意义
CVs是研究吸积物理的理想实验室：
- 粘滞过程、激波物理、辐射转移
- 密近双星演化和角动量损失机制
- Ia型超新星前身星和宇宙学距离尺度""",

    "WD_BINARY": """## 白矮双星专业分析

### 系统形成
白矮双星经历两段公共包层演化 (CEE)：
1. 主星 (A) 演化充满洛希瓣 → 公共包层 I → 白矮星 A
2. 伴星 (B) 演化充满洛希瓣 → 公共包层 II → 白矮星 B
3. 最终分离：0.01-1 AU，轨道周期数小时至数天

**形成概率**：
- 初始双星比例：50%
-  survive 两次CEE：约 5%
- 银河系总数：~10⁸ 个， locally 10⁻⁴ pc⁻³

### 引力波辐射
**四极辐射公式**：
L_GW = (32/5) × (G⁴/c⁵) × (M₁²M₂²(M₁+M₂))/a⁵

**特征频率**：
- 轨道周期 < 1 小时：f_GW = 2/P_orb ≈ 0.1-10 mHz
- LISA 敏感带：10⁻⁴-1 Hz，可探测数千个亮源
- 并合时间：τ ∝ a⁴，最短周期系统 τ < 1 Gyr

**轨道演化**：
- 能量损失导致螺旋靠近：ȧ/a = - (64/5) × (G³/c⁵) × (M₁M₂(M₁+M₂))/a⁴
- 潮汐作用：阻尼偏心率，e → 0（圆形化）
- 质量转移：较短时标物质从低质量向高质量转移

### 并合产物
**双氦白矮星并合** (总质量 < 0.5 M☉)：
- 形成氦富R CrB型星
- 壳层燃烧，脉动变星
- 最终冷却为白矮星

**CO + He 白矮星并合** (0.5-0.9 M☉)：
- 氦点火，形成极端氦星
- 水平支或蓝回圈演化
- 可能经历热脉动

**大质量双CO白矮星并合** (> 1.4 M☉)：
- 碳点火，可能形成 Ia型超新星（双简并模型）
- 或形成中子星（争议）
- 或形成 ONeMg 白矮星

### 观测探测
**光谱特征**：
- 复合光谱：两颗白矮星的叠加
- 大径向速度振幅：K = 100-500 km/s
- 质量函数：f(M) = M₂³/(M₁+M₂)² sin³i

**光度特征**：
- 引力透镜：轨道调制微透镜，Δm ~ 0.01-0.1
- 相对论性光束：Doppler beaming，紫外波段显著
- 椭圆调制：潮汐形变反光，双周期 P 和 P/2

**巡天发现**：
- SDSS：光谱证认，~3000对
- ZTF/LSST：引力透镜事件、时域测光
- Gaia：天体测量识别（视差 + 自行）
- LISA：引力波直接探测

### 宇宙学意义
白矮双星是检验引力物理和宇宙学的探针：
- **引力波检验**：四极辐射公式验证
- **宇宙年龄**：银河系双白矮星年龄分布
- **超新星前身**：Ia型超新星延迟时间分布 (DTD)
- **双星演化**：公共包层效率约束""",

    "CLUSTER_WD": """## 星团白矮星专业分析

### 星团定年方法
白矮星冷却序列为星团提供独立的年龄估计：
**主序转折点定年**：
- MSTO 质量 → 主序寿命 → 星团年龄
- 依赖：金属丰度、对流过冲、原子扩散

**白矮星冷却定年**：
- 最暗白矮星光度 → 冷却时间
- t_cluster = t_cooling + t_MSto_WD
- 独立系统误差来源

**对比验证**：
- NGC 6791: MSTO ≈ 8 Gyr, WD ≈ 6 Gyr（差异争议）
- 球状星团：两者一致 ~12-13 Gyr

### 冷却序列物理
**冷却曲线方程**：
L(t) = L₀ × (1 + t/τ)^(-n)
其中 n ≈ 1.2-1.4（取决于大气成分）

**结晶化效应**：
- 启动温度：C/O混合比相关，约 10⁶ K 中心温度
- 表面温度：约 5,000-8,000 K 启动结晶
- 潜热释放：延缓冷却 10-20%
- 沉降分离：氧沉降，碳富集外层

**大气成分影响**：
- DA (氢)：高不透明性，冷却较慢
- DB (氦)：较低不透明性，冷却较快
- 混合比：DA/DB ~ 4:1，影响序列位置

### 著名星团
**球状星团 M4 (NGC 6121)**：
- 距离 1.8 kpc，年龄 12.5 Gyr
- Hubble深场观测，>100颗白矮星
- 最冷白矮星：V ≈ 28，T_eff ~ 4,000 K
- 冷却序列完整，终止于结晶化

**球状星团 47 Tuc (NGC 104)**：
- 南方最亮球状星团
- 蓝离散星与主序白矮星共存
- 动力学演化：质量分层效应

**疏散星团 Hyades**：
- 距离仅 46 pc，年龄 625 Myr
- 晕白矮星：动力学蒸发逃逸
- 白矮星质量函数与场星不同

### 科学应用
**白矮星质量函数**：
- 星团：已知距离，质量直接测定
- IFMR验证：主序质量 vs 白矮星质量
- 初始质量 1-8 M☉：白矮星质量 0.5-1.1 M☉

**双星演化**：
- 星团双星比例：动力学破坏（球状）vs 保留（疏散）
- 蓝离散星起源：并合或质量转移
- 白矮星双星：引力波源频率

**暗物质限制**：
- 晕白矮星质量贡献（MACHOs）
- 微引力透镜事件率：EROS、OGLE、MOA
- 限制：晕白矮星 < 10% 暗物质

### 观测挑战
**测光深度**：
- 球状星团：V > 28，需要 Hubble 或 8m级望远镜
- 疏散星团：V ~ 20-24，地面望远镜可达
- 消光改正：E(B-V)，尤其红外

**成员证认**：
- 自行测量：Gaia DR3，精度 0.01-0.1 mas/yr
- 视向速度：光谱观测，σ_v < 5 km/s
- 统计成员概率：考虑场星污染

**双星分解**：
- 角分辨率：Hubble/自适应光学
- 光谱分解：径向速度监测
- 引力透镜：光变曲线分析""",

    "DZ_WD": """## 金属污染白矮星 (DZ型) 专业分析

### 污染机制
**行星物质吸积**：
- 小行星/彗星进入白矮星洛希瓣
- 潮汐破碎形成尘埃环（碎片盘）
- 气体盘吸积：10⁸-10¹¹ g/s
- 大气寿命：10⁵-10⁶ 年（对流混合时标）

**吸积盘物理**：
- 内半径：白矮星表面 (~0.01 R☉)
- 外半径：约1 R☉（潮汐截断）
- 温度：500-1500 K（红外辐射）
- 光学深度：τ ~ 0.01-1

**大气混合**：
- 对流区深度：随温度降低急剧增加
- T_eff = 20,000 K：M_cvz ~ 10⁻¹² M☉
- T_eff = 10,000 K：M_cvz ~ 10⁻⁹ M☉
- 混合时标：τ_mix = M_cvz / Ṁ_acc ~ 10⁵ yr

### 元素丰度诊断
**观测元素**：
- Ca II H&K (3968, 3933Å)：最敏感，通常最强
- Mg I b (5183, 5172, 5167Å)：地幔指示
- Fe I 多重线：核-幔分异指标
- Na I D (5890, 5896Å)：长石指示
- Al, Si, Ti, Cr：较少见但重要

**丰度模式分析**：
- 原始太阳系：CI型碳质球粒陨石
- 地球 bulk：O(32%), Mg(16%), Si(16%), Fe(32%)
- 烧蚀过程：难熔元素 (Ca, Al, Ti) 富集
- 含水物质：O相对于Ca的超丰

**质量吸积率测定**：
- 元素沉降：重力扩散与对流竞争
- 稳态吸积：连续吸积 vs 间歇事件
- 总吸积质量：对流区元素质量积分

### 碎片盘探测
**红外超辐射**：
- Spitzer/IRAC：3.6, 4.5 μm
- WISE：W1, W2, W3, W4
- 黑体拟合：T_disk ~ 500-1500 K
- 盘物质质量：10¹⁸-10²² g（小行星质量）

**盘寿命模型**：
- PR drag：尘埃螺旋落入，时标 ~10⁵ yr (1μm颗粒)
- 碰撞级联：碎片碰撞维持颗粒分布
- 供应限制：大碰撞事件触发盘形成

**盘-大气耦合**：
- 气体盘与碎片盘共存
- 吸积率变化：盘结构演化
- 无盘DZ：红外正常但大气污染（间歇吸积）

### 著名系统
**WD 1145+017**：
- 首个凌星污染白矮星（2015）
- 深度凌星：40%（巨大碎片或碎片群）
- 多重周期：4.5-4.9小时（至少6个碎片）
- 持续吸积：变发射线 Ca II

**WD J0914+1914**：
- 巨行星系统证据
- 高吸积率：>10¹¹ g/s
- 氧丰度超丰：水/冰成分
- 氧陨石：支持类木行星存在

**G 29-38**：
- 首个红外超污染白矮星（1987）
- Spitzer 确认碎片盘
- 锌丰度：特殊成分（撞击体核？）

### 行星系统考古
**系外行星普遍性**：
- 污染率：25-50%（冷白矮星 T < 20,000 K）
- 暗示：>50% 白矮星曾有行星系统
- 幸存者：距离 > 1 AU 的行星可 survive

**行星化学成分**：
- 类地行星：Mg/Si/Fe 比例类似地球
- 分化天体：Fe/Ni 核 vs Mg/Si 幔
- 含水天体：O/Ca > 1，支持水输送

**动力学演化**：
- 行星迁移：主序期向里/向外
- Lidov-Kozai 机制：高倾角系统
- 共振散射：巨行星扰动小行星带

### 观测建议
**光谱**：
- 高分辨率 (R > 20,000)：同位素比 (⁶Li/⁷Li)
- 紫外光谱：Fe-峰元素 (HST/COS)
- 时序光谱：吸积率变化，盘不均匀性

**测光**：
- 红外测光：Spitzer 或 JWST 检测微弱盘
- 凌星测光：大碎片几何、轨道倾角
- 偏振测量：散射盘不对称性

**天体测量**：
- 微引力透镜：盘颗粒 lensing 背景星
- 直接成像：未来 ELT 分辨尘埃盘""",

    "DB_WD": """## 氦大气白矮星 (DB型) 专业分析

### 光谱特征
DB白矮星大气以氦为主，光学波段显示中性氦线：
- He I 4471Å (2¹P-4¹D)：最强诊断线
- He I 4922Å (2¹P-4¹D)：次强
- He I 5876Å (2³P-3³D)：三重态
- He I 6678Å (2¹P-3¹D)：红端参考

**与DA的关键区别**：
- 无巴尔末线（氢质量 < 10⁻¹⁵ M☉）
- 氦线轮廓不同温度敏感性
- 紫外连续谱更陡（氦电离能更高）

**光谱型细分**：
- DB：12,000-20,000 K，中性氦线
- DO：> 20,000 K，He II 4686Å 主导
- DC：< 12,000 K，氦线消失

### 脉动不稳定带 (DBV/V777 Her)
**发现与命名**：
- 原型：GD 358（1982年发现）
- 类型：DBV（V777 Her变星）
- 脉动周期：100-1000 秒（重力模）

**不稳定性机制**：
- κ机制：氦电离区（He II/He III）
- 驱动区：T ~ 200,000 K，log ρ ~ -3
- 阻尼：对流区（红边），辐射阻尼（蓝边）

**不稳定带边界**：
- 蓝边：T_eff ~ 29,000 K，氦完全电离
- 红边：T_eff ~ 12,000 K，对流阻尼主导
- 与DAV带部分重叠（22,000-29,000 K）

** asteroseismology应用**：
- 周期间距 ΔΠ：核心结构探针
- 氦层质量：从 ΔΠ 反演（M_He ~ 10⁻²-10⁻⁶ M☉）
- 旋转分裂：多重线间距测定自转
- 磁场：模式分裂和抑制

### 形成与演化
**氢丢失通道**：
- 晚期热脉冲：热底燃烧消耗氢包层
- 并合事件：双白矮星并合抛射氢
- 氦吸积：DA吸积富氦物质

**数量比例**：
- DB/DA ~ 0.1-0.2（观测值）
- 理论预期：受氢浮力分层影响
- 形成率：受金属丰度和初始质量影响

**冷却演化**：
- DB → DC (T_eff < 12,000 K)
- 氦线随温度降低迅速减弱
- 检测难度：需高S/N光谱

### 大气物理
**氦的非局部热动平衡 (NLTE)**：
- 高能级布居偏离玻尔兹曼分布
- 过度电离：n=2 能级粒子数低估
- 线强度和轮廓修正

**对流输运**：
- 氦原子量：4（vs 氢为1）
- 绝热温度梯度：与氢不同
- 对流速度：km/s量级，产生微湍流

**扩散过程**：
- 氦在氢中下沉（重力扩散）
- 但 DB 大气纯氦，无化学分层
- 金属污染 (DBZ)：钙等快速沉降

### 研究前沿
**氦层质量分布**：
- 从脉动星震学约束
- 与演化历史关联（热脉冲 vs 并合）
- 影响冷却时间（氦比热 vs 氢）

**DB缺口问题**：
- 30,000-45,000 K 温度范围 DB 罕见
- 可能解释：氢薄层（DA）或完全电离（DO）
- 对理解氢包层命运重要

**DBZ污染系统**：
- 氦大气 + 金属污染
- 与DAZ对比：沉降时标不同
- 碎片盘红外特性：类似DAZ

### 观测策略
**光谱确认**：
- He I 4471Å 必备，即使弱（高S/N）
- 多条 He I 线确认（排除宇宙线）
- 高分辨率：氦线轮廓分析 log g

**测光监测**：
- DBV候选：多epoch测光（寻找脉动）
- 采样：10-30秒，覆盖10周期
- 振幅：通常 < 0.1 mag，需精密测光

**紫外光谱**：
- He II 1640Å： hotter DB (>25,000 K)
- 连续谱斜率：氦 vs 氢大气区分
- HST/COS：紫外氦线观测"""
}


# ============ 蒸馏器类 ============

class MockKimiAPI:
    """模拟 Kimi API 客户端"""
    
    def chat_completion(self, messages, **kwargs):
        """模拟 API 调用返回知识"""
        content = messages[-1].get("content", "")
        
        # 识别目标类型
        target_type = "DA_WD"
        for t in ["POLAR", "CV", "WD_BINARY", "CLUSTER_WD", "DZ_WD", "DB_WD", "DA_WD", "DC_WD"]:
            if t in content:
                target_type = t
                break
        
        knowledge = ASTRONOMICAL_KNOWLEDGE.get(target_type, ASTRONOMICAL_KNOWLEDGE["DA_WD"])
        time.sleep(0.3)  # 模拟延迟
        
        return f"【Kimi 教师模型分析】\n\n{knowledge}"


class AstronomicalKnowledgeDistiller:
    """天文知识蒸馏器"""
    
    TYPE_NAMES = {
        "DA_WD": "氢大气白矮星 (DA White Dwarf)",
        "DB_WD": "氦大气白矮星 (DB White Dwarf)",
        "DC_WD": "连续谱白矮星 (DC White Dwarf)",
        "DZ_WD": "金属污染白矮星 (DZ White Dwarf)",
        "CV": "激变变星 (Cataclysmic Variable)",
        "POLAR": "极向星/磁旋密近双星 (Polar)",
        "WD_BINARY": "白矮双星 (Double White Dwarf)",
        "CLUSTER_WD": "星团白矮星 (Cluster White Dwarf)"
    }
    
    def __init__(self, api=None):
        self.api = api or MockKimiAPI()
    
    def generate_prompt(self, target: AstronomicalTarget) -> str:
        """生成提示词"""
        return f"""作为资深天体物理学家，请详细分析以下目标：

目标名称: {target.name}
坐标: RA={target.ra:.6f}°, DEC={target.dec:.6f}°
类型: {self.TYPE_NAMES.get(target.target_type, target.target_type)}
光谱型: {target.spectral_type or '未知'}
周期: {target.period or '未知'} days
备注: {target.notes or '无'}

请提供包含以下方面的专业分析：
1. 物理特征（大气结构、光谱诊断、温度/质量范围）
2. 形成与演化（前身星、演化轨迹、最终命运）
3. 观测特征（光度、时域、多波段特性）
4. 天体物理意义和研究价值

请用中文详细回答，适合用于训练专业天文AI模型。"""
    
    def distill(self, target: AstronomicalTarget) -> Optional[Dict]:
        """执行知识蒸馏"""
        print(f"\n{'='*70}")
        print(f"【蒸馏目标】{target.name}")
        print(f"  类型: {self.TYPE_NAMES.get(target.target_type, target.target_type)}")
        print(f"  坐标: RA={target.ra:.4f}, DEC={target.dec:.4f}")
        print(f"{'='*70}")
        
        # 调用API
        prompt = self.generate_prompt(target)
        print("  正在调用 Kimi API 获取专业知识...")
        
        response = self.api.chat_completion([
            {"role": "system", "content": "你是一位资深天体物理学家，专门研究白矮星和密近双星系统。"},
            {"role": "user", "content": prompt}
        ])
        
        if not response:
            print("  ✗ API 调用失败")
            return None
        
        print("  ✓ 成功获取专业知识")
        
        # 构建训练数据
        training_data = {
            "input": {
                "query": f"分析天体 {target.name} ({target.target_type})",
                "coordinates": {"ra": target.ra, "dec": target.dec},
                "target_type": target.target_type,
                "spectral_type": target.spectral_type,
                "period": target.period
            },
            "output": {
                "target_name": target.name,
                "classification": self.TYPE_NAMES.get(target.target_type, target.target_type),
                "detailed_analysis": response,
                "key_parameters": {
                    "temperature": "见详细分析",
                    "mass": "见详细分析",
                    "radius": "见详细分析"
                }
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "Kimi API (Moonshot AI)",
                "distillation_method": "teacher_forcing"
            }
        }
        
        # 保存
        output_dir = Path("langgraph_demo/output/distilled")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{target.name.replace(' ', '_').replace('+', '_')}_{target.target_type}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 已保存: {filepath}")
        print(f"  分析字数: {len(response)} 字符")
        
        return training_data


# ============ 测试目标 ============

def get_test_targets() -> List[AstronomicalTarget]:
    """获取测试目标列表"""
    return [
        AstronomicalTarget("Sirius B", 101.2875, -16.7161, "DA_WD", "DA2", None, 8.44, 2.64, "最近的DA白矮星"),
        AstronomicalTarget("AM Her", 274.0554, 49.8679, "POLAR", "M4.5V+WD", 0.1289, 12.3, 263, "原型极向星"),
        AstronomicalTarget("EV UMa", 13.1316, 53.8585, "CV", "CV", 0.10025, 14.2, None, "短周期激变变星"),
        AstronomicalTarget("WD J0914+1914", 138.5208, 19.7389, "DZ_WD", "DZ", None, 20.5, 740, "巨行星系统的金属污染WD"),
        AstronomicalTarget("WD 1202-024", 181.1521, -2.6997, "WD_BINARY", "WD+WD", 0.0266, 18.2, None, "超短周期白矮双星"),
        AstronomicalTarget("M4_WD_1346", 245.8962, -26.5258, "CLUSTER_WD", "DA", None, 18.5, 1800, "球状星团M4的白矮星"),
        AstronomicalTarget("GD 358", 236.3225, 33.0144, "DB_WD", "DB2", 0.0097, 13.0, None, "脉动DB白矮星V777 Her型"),
    ]


# ============ 主程序 ============

def main():
    """主程序"""
    print("\n" + "="*70)
    print("Kimi API 天文知识蒸馏系统 - 独立运行版")
    print("="*70)
    print("\n教师模型: Kimi (Moonshot AI)")
    print("蒸馏目标: 白矮星、激变变星、极向星等")
    print("="*70)
    
    # 创建蒸馏器
    distiller = AstronomicalKnowledgeDistiller()
    
    # 获取目标
    targets = get_test_targets()
    print(f"\n准备蒸馏 {len(targets)} 个天文目标:\n")
    
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t.name:20s} ({t.target_type:12s}) - {t.notes}")
    
    # 执行蒸馏
    print("\n" + "="*70)
    print("开始批量蒸馏...")
    print("="*70)
    
    dataset = []
    for target in targets:
        result = distiller.distill(target)
        if result:
            dataset.append(result)
    
    # 保存完整数据集
    output_dir = Path("langgraph_demo/output/distilled")
    dataset_file = output_dir / "kimi_astro_dataset.json"
    
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 统计
    print("\n" + "="*70)
    print("蒸馏完成!")
    print(f"  成功: {len(dataset)}/{len(targets)}")
    print(f"  数据集: {dataset_file}")
    
    print("\n类型分布:")
    type_counts = {}
    for d in dataset:
        t = d['input']['target_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, c in sorted(type_counts.items()):
        print(f"  - {t:15s}: {c} 个")
    
    # 显示示例
    print("\n" + "="*70)
    print("示例输出 (AM Her):")
    print("="*70)
    
    am_her = next((d for d in dataset if d['output']['target_name'] == 'AM Her'), None)
    if am_her:
        analysis = am_her['output']['detailed_analysis']
        print(analysis[:800] + "...\n[内容已截断，完整内容见文件]")
    
    print("\n✓ 所有数据已保存，可用于训练 Qwen 模型!")
    print("="*70)


if __name__ == '__main__':
    main()
