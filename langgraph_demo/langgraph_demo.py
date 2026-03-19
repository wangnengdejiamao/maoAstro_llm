#!/usr/bin/env python3
"""
LangGraph 天文数据分析 Agent Demo
=================================
演示如何使用 LangGraph 构建天文数据分析工作流

功能:
1. 多 Agent 协作处理天文数据
2. 使用 Qwen 模型进行推理
3. 生成训练数据进行模型蒸馏

作者: AI Assistant
日期: 2026-03-02
"""

import os
import sys
import json
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# LangGraph 导入
try:
    from langgraph.graph import Graph, StateGraph
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️  LangGraph 未安装，使用模拟模式")
    LANGGRAPH_AVAILABLE = False

# 状态定义
class AstroState(TypedDict):
    """天文数据分析状态"""
    query: str                          # 用户查询
    target_name: str                    # 目标名称
    ra: float                          # 赤经
    dec: float                         # 赤纬
    fits_data: Dict[str, Any]          # FITS 数据
    vsx_data: Dict[str, Any]           # VSX 变星数据
    lamost_data: Dict[str, Any]        # LAMOST 光谱数据
    analysis_result: str               # 分析结果
    classification: str                # 分类结果
    physical_params: Dict[str, float]  # 物理参数
    confidence: float                  # 置信度
    reasoning_chain: List[str]         # 推理链
    training_data: Dict[str, Any]      # 训练数据 (用于蒸馏)


# ==================== 工具函数 ====================

def query_vsx_tool(ra: float, dec: float, radius: float = 2.0) -> Dict[str, Any]:
    """
    VSX 变星数据库查询工具
    """
    # 模拟 VSX 查询
    # 实际使用时应该调用 VSP 模块
    mock_data = {
        "found": True,
        "name": "V* EV UMa",
        "type": "CV (Cataclysmic Variable)",
        "period": 0.10025,
        "max_mag": 14.2,
        "min_mag": 18.5,
        "coordinates": {"ra": ra, "dec": dec},
        "distance_arcsec": 0.8
    }
    return mock_data


def query_lamost_tool(ra: float, dec: float, radius: float = 2.0) -> Dict[str, Any]:
    """
    LAMOST 光谱查询工具
    """
    # 模拟 LAMOST 查询
    mock_data = {
        "found": True,
        "obsid_list": ["123456789", "987654321"],
        "num_observations": 2,
        "class": "STAR",
        "subclass": "CV",
        "snr": 45.6,
        "z": -0.0001,
        "coordinates": {"ra": ra, "dec": dec}
    }
    return mock_data


def spectral_analysis_tool(fits_path: str) -> Dict[str, Any]:
    """
    光谱分析工具
    """
    # 模拟光谱分析
    return {
        "spectral_type": "DA (Hydrogen White Dwarf)",
        "temperature": 18500,
        "temperature_err": 500,
        "features": ["H-alpha", "H-beta", "H-gamma"],
        "signal_to_noise": 45.6,
        "analysis_method": "Blackbody fitting"
    }


def generate_training_data(state: AstroState) -> Dict[str, Any]:
    """
    生成训练数据用于模型蒸馏
    """
    training_example = {
        "input": {
            "query": state["query"],
            "coordinates": {"ra": state["ra"], "dec": state["dec"]},
            "vsx_data": state.get("vsx_data", {}),
            "lamost_data": state.get("lamost_data", {})
        },
        "output": {
            "classification": state.get("classification", ""),
            "physical_params": state.get("physical_params", {}),
            "reasoning": state.get("reasoning_chain", [])
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "confidence": state.get("confidence", 0.0),
            "data_sources": ["VSX", "LAMOST", "SIMBAD"]
        }
    }
    return training_example


# ==================== Agent 节点 ====================

def router_agent(state: AstroState) -> AstroState:
    """
    路由 Agent - 分析用户查询并决定执行路径
    """
    print("\n🔀 [Router Agent] 分析查询...")
    
    query = state["query"].lower()
    
    # 简单路由逻辑
    if "variable" in query or "cv" in query or "nova" in query:
        state["classification_target"] = "variable_star"
    elif "spectrum" in query or "spectral" in query:
        state["classification_target"] = "spectral_classification"
    else:
        state["classification_target"] = "general_analysis"
    
    state["reasoning_chain"] = ["Query classified as: " + state["classification_target"]]
    print(f"   查询类型: {state['classification_target']}")
    
    return state


def data_retrieval_agent(state: AstroState) -> AstroState:
    """
    数据检索 Agent - 查询外部数据库
    """
    print("\n📡 [Data Retrieval Agent] 查询外部数据库...")
    
    ra, dec = state["ra"], state["dec"]
    
    # 查询 VSX
    print("   - 查询 VSX 变星数据库...")
    vsx_data = query_vsx_tool(ra, dec)
    state["vsx_data"] = vsx_data
    
    if vsx_data.get("found"):
        print(f"   ✓ 找到变星: {vsx_data['name']}, 类型: {vsx_data['type']}")
        state["reasoning_chain"].append(f"VSX match: {vsx_data['name']} ({vsx_data['type']})")
    
    # 查询 LAMOST
    print("   - 查询 LAMOST DR10...")
    lamost_data = query_lamost_tool(ra, dec)
    state["lamost_data"] = lamost_data
    
    if lamost_data.get("found"):
        print(f"   ✓ 找到 LAMOST 光谱: {lamost_data['num_observations']} 次观测")
        state["reasoning_chain"].append(f"LAMOST spectra: {lamost_data['num_observations']} observations")
    
    return state


def analysis_agent(state: AstroState) -> AstroState:
    """
    分析 Agent - 执行科学分析
    """
    print("\n🔬 [Analysis Agent] 执行科学分析...")
    
    # 模拟分析过程
    vsx = state.get("vsx_data", {})
    lamost = state.get("lamost_data", {})
    
    analysis_steps = []
    
    # 基于 VSX 的分析
    if vsx.get("found"):
        analysis_steps.append(f"Variable star classification: {vsx.get('type', 'Unknown')}")
        analysis_steps.append(f"Period analysis: P = {vsx.get('period', 'N/A')} days")
    
    # 基于 LAMOST 的分析
    if lamost.get("found"):
        analysis_steps.append(f"Spectral type from LAMOST: {lamost.get('subclass', 'Unknown')}")
        analysis_steps.append(f"Radial velocity: z = {lamost.get('z', 'N/A')}")
    
    # 综合分类
    if vsx.get("type", "").lower().find("cv") >= 0:
        classification = "Cataclysmic Variable (CV)"
        confidence = 0.92
    elif vsx.get("type", "").lower().find("pulsating") >= 0:
        classification = "Pulsating Variable"
        confidence = 0.85
    else:
        classification = "Unknown/Uncertain"
        confidence = 0.45
    
    state["classification"] = classification
    state["confidence"] = confidence
    state["analysis_result"] = "\n".join(analysis_steps)
    state["reasoning_chain"].extend(analysis_steps)
    
    print(f"   分类结果: {classification} (置信度: {confidence:.2%})")
    
    return state


def reasoning_agent(state: AstroState) -> AstroState:
    """
    推理 Agent - 使用 LLM 进行科学推理
    这里模拟 Qwen 模型的推理过程
    """
    print("\n🧠 [Reasoning Agent] LLM 科学推理...")
    
    # 模拟 Qwen 的推理过程
    reasoning_prompt = f"""
基于以下观测数据进行推理:
- 坐标: RA={state['ra']:.4f}, DEC={state['dec']:.4f}
- VSX 分类: {state.get('vsx_data', {}).get('type', 'N/A')}
- 周期: {state.get('vsx_data', {}).get('period', 'N/A')} days
- LAMOST 光谱类型: {state.get('lamost_data', {}).get('subclass', 'N/A')}

推理步骤:
1. 变星类型判断基于光变周期和光谱特征
2. 周期 ~0.1d 且光谱显示发射线 → 吸积盘系统
3. 结合多波段测光确认热斑存在
    """
    
    # 模拟 LLM 输出
    llm_output = {
        "scientific_reasoning": "The target exhibits characteristics of a cataclysmic variable (CV) system:",
        "key_evidence": [
            "Short orbital period (P ≈ 0.1d) indicates compact binary",
            "Spectrum shows Balmer emission lines from accretion disk",
            "Amplitude variation (Δm ≈ 4.3 mag) consistent with CV outbursts"
        ],
        "physical_interpretation": "Polars (AM Her type) - magnetic CV with synchronized rotation",
        "confidence": 0.89
    }
    
    state["physical_params"] = {
        "orbital_period": state.get("vsx_data", {}).get("period", 0),
        "spectral_type": "DA/Magnetic",
        "distance_estimate": 250,  # pc
        "accretion_rate": 1e-9,    # Msun/yr
        "magnetic_field": 10       # MG
    }
    
    state["reasoning_chain"].append("LLM reasoning: " + llm_output["scientific_reasoning"])
    
    print("   推理完成")
    print(f"   物理解释: {llm_output['physical_interpretation']}")
    
    return state


def verification_agent(state: AstroState) -> AstroState:
    """
    验证 Agent - 交叉验证结果
    """
    print("\n✓ [Verification Agent] 结果验证...")
    
    # 验证逻辑
    checks = []
    
    # 检查 1: 数据一致性
    if state.get("vsx_data", {}).get("found") and state.get("lamost_data", {}).get("found"):
        checks.append("✓ Multiple data sources consistent")
    
    # 检查 2: 置信度阈值
    if state.get("confidence", 0) > 0.8:
        checks.append("✓ Confidence threshold passed")
    else:
        checks.append("⚠ Low confidence - needs expert review")
    
    # 检查 3: 物理参数合理性
    params = state.get("physical_params", {})
    if 0 < params.get("orbital_period", 0) < 1:
        checks.append("✓ Physical parameters within expected range")
    
    state["verification_result"] = checks
    state["reasoning_chain"].extend(checks)
    
    for check in checks:
        print(f"   {check}")
    
    return state


def output_agent(state: AstroState) -> AstroState:
    """
    输出 Agent - 生成最终报告和训练数据
    """
    print("\n📄 [Output Agent] 生成最终报告...")
    
    # 生成科学报告
    report = f"""
========================================
天文目标分析报告
========================================
目标: {state['target_name']}
坐标: RA={state['ra']:.6f}°, DEC={state['dec']:.6f}°
查询: {state['query']}

【分类结果】
{state.get('classification', 'Unknown')}
置信度: {state.get('confidence', 0):.2%}

【分析详情】
{state.get('analysis_result', 'N/A')}

【物理参数】
- 轨道周期: {state.get('physical_params', {}).get('orbital_period', 'N/A')} d
- 光谱类型: {state.get('physical_params', {}).get('spectral_type', 'N/A')}
- 距离估计: {state.get('physical_params', {}).get('distance_estimate', 'N/A')} pc
- 吸积率: {state.get('physical_params', {}).get('accretion_rate', 'N/A')} M☉/yr

【推理链】
"""
    for i, step in enumerate(state.get("reasoning_chain", []), 1):
        report += f"{i}. {step}\n"
    
    report += """
【验证结果】
"""
    for check in state.get("verification_result", []):
        report += f"{check}\n"
    
    report += "\n========================================"
    
    state["final_report"] = report
    
    # 生成训练数据 (用于模型蒸馏)
    print("   生成训练数据用于模型蒸馏...")
    training_data = generate_training_data(state)
    state["training_data"] = training_data
    
    print("   ✓ 报告生成完成")
    
    return state


# ==================== 构建 LangGraph ====================

def create_astro_graph():
    """
    创建天文数据分析 LangGraph
    """
    # 创建工作流图
    workflow = StateGraph(AstroState)
    
    # 添加节点
    workflow.add_node("router", router_agent)
    workflow.add_node("data_retrieval", data_retrieval_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("verification", verification_agent)
    workflow.add_node("output", output_agent)
    
    # 设置入口点
    workflow.set_entry_point("router")
    
    # 添加边 (工作流连接)
    workflow.add_edge("router", "data_retrieval")
    workflow.add_edge("data_retrieval", "analysis")
    workflow.add_edge("analysis", "reasoning")
    workflow.add_edge("reasoning", "verification")
    workflow.add_edge("verification", "output")
    workflow.add_edge("output", "__end__")
    
    # 编译图
    app = workflow.compile()
    
    return app


# ==================== 主程序 ====================

def run_demo(target_name: str = "EV_UMa", ra: float = 13.1316, dec: float = 53.8585):
    """
    运行 LangGraph Demo
    """
    print("=" * 70)
    print("LangGraph 天文数据分析 Agent Demo")
    print("=" * 70)
    print(f"\n目标: {target_name}")
    print(f"坐标: RA={ra:.6f}°, DEC={dec:.6f}°")
    
    # 初始化状态
    initial_state = AstroState(
        query=f"Analyze variable star {target_name}",
        target_name=target_name,
        ra=ra,
        dec=dec,
        fits_data={},
        vsx_data={},
        lamost_data={},
        analysis_result="",
        classification="",
        physical_params={},
        confidence=0.0,
        reasoning_chain=[],
        training_data={}
    )
    
    if not LANGGRAPH_AVAILABLE:
        print("\n⚠️  LangGraph 不可用，按顺序执行 Agent...")
        # 模拟执行
        state = router_agent(initial_state)
        state = data_retrieval_agent(state)
        state = analysis_agent(state)
        state = reasoning_agent(state)
        state = verification_agent(state)
        state = output_agent(state)
    else:
        # 使用 LangGraph 执行
        print("\n🚀 启动 LangGraph 工作流...\n")
        graph = create_astro_graph()
        
        # 执行工作流
        state = graph.invoke(initial_state)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(state.get("final_report", "No report generated"))
    
    # 保存训练数据
    output_dir = "langgraph_demo/output"
    os.makedirs(output_dir, exist_ok=True)
    
    training_file = f"{output_dir}/{target_name}_training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(state.get("training_data", {}), f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 训练数据已保存: {training_file}")
    
    # 保存完整状态
    state_file = f"{output_dir}/{target_name}_state.json"
    # 移除不可序列化的数据
    state_clean = {k: v for k, v in state.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_clean, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"✓ 状态已保存: {state_file}")
    
    return state


def batch_generate_training_data(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量生成训练数据
    用于模型蒸馏的数据准备
    """
    print("\n" + "=" * 70)
    print("批量生成训练数据")
    print("=" * 70)
    
    training_dataset = []
    
    for target in targets:
        print(f"\n处理: {target['name']}")
        state = run_demo(
            target_name=target['name'],
            ra=target['ra'],
            dec=target['dec']
        )
        
        if state.get("training_data"):
            training_dataset.append(state["training_data"])
    
    # 保存完整数据集
    output_file = "langgraph_demo/output/training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 训练数据集已保存: {output_file}")
    print(f"  样本数: {len(training_dataset)}")
    
    return training_dataset


if __name__ == '__main__':
    # 单个目标分析
    result = run_demo("EV_UMa", 13.1316, 53.8585)
    
    # 批量生成训练数据 (可选)
    print("\n\n是否批量生成训练数据? (y/n): ", end="")
    # response = input().strip().lower()
    response = 'n'  # 默认不执行批量
    
    if response == 'y':
        targets = [
            {"name": "AM_Her", "ra": 274.0554, "dec": 49.8679},
            {"name": "V1500_Cyg", "ra": 304.2875, "dec": 42.0250},
            {"name": "SS_Cyg", "ra": 325.4125, "dec": 43.5833},
        ]
        batch_generate_training_data(targets)
    
    print("\n✓ Demo 执行完成!")
