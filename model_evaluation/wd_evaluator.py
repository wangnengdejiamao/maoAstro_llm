#!/usr/bin/env python3
"""
白矮星专用评估器

评估模型在白矮星领域的知识掌握程度：
1. 单白矮星基本性质
2. 双白矮星系统
3. 磁性白矮星
4. 吸积白矮星
5. 反幻觉能力
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from astronomy_evaluator import AstronomyEvaluator, EvalResult
from typing import List, Dict
import json


class WhiteDwarfEvaluator(AstronomyEvaluator):
    """
    白矮星专用评估器
    """
    
    # 白矮星领域幻觉检测
    WD_HALLUCINATION_CHECKS = {
        # AM CVn 相关
        "am_cvn_neutron_star": {
            "pattern": r"中子星|neutron star",
            "context": r"AM CVn|AM\s+CVn",
            "should_exist": False,
            "description": "AM CVn 不含中子星"
        },
        "am_cvn_period": {
            "pattern": r"(5-65\s*min|5.*65.*分钟|period.*5.*65)",
            "context": r"AM CVn|AM\s+CVn.*周期",
            "should_exist": True,
            "description": "AM CVn 周期应为 5-65 分钟"
        },
        
        # 白矮星组成
        "wd_composition": {
            "pattern": r"(碳氧|CO|氧氖镁|ONeMg|氦|helium).*组成|composition.*(carbon|oxygen|helium)",
            "context": r"白矮星|white dwarf|WD",
            "should_exist": True,
            "description": "白矮星应提及组成（C/O 或 O/Ne/Mg 或 He）"
        },
        
        # 钱德拉塞卡极限
        "chandrasekhar_mass": {
            "pattern": r"1\.[34]|1\.38|~1\.4",
            "context": r"钱德拉|Chandrasekhar|极限|limit",
            "should_exist": True,
            "description": "钱德拉塞卡极限应约为 1.4 M☉"
        },
        
        # 双白矮星并合
        "dwd_merger_sn": {
            "pattern": r"总是|always|一定|definitely.*超新星|supernova",
            "context": r"双白矮星|double white dwarf|并合|merger",
            "should_exist": False,
            "description": "双白矮星并合不总是产生超新星"
        },
        
        # 磁性白矮星
        "mwd_fraction": {
            "pattern": r"(10|20).*%.+|约.*(10|20)|10-20",
            "context": r"磁性白矮星|magnetic white dwarf|MWD|比例|fraction",
            "should_exist": True,
            "description": "磁性白矮星约占 10-20%"
        },
        
        # 吸积白矮星
        "accretion_nova": {
            "pattern": r"都会|all.*新星|always.*nova",
            "context": r"吸积|accretion|白矮星|white dwarf",
            "should_exist": False,
            "description": "吸积白矮星不都会发生新星爆发"
        },
        
        # 冷却
        "wd_cooling_constant": {
            "pattern": r"恒定|constant.*rate|不变",
            "context": r"白矮星.*冷却|white dwarf.*cooling",
            "should_exist": False,
            "description": "白矮星冷却速率不恒定"
        }
    }
    
    # 白矮星领域基准题库
    WD_BENCHMARK = [
        # 单白矮星
        {
            "id": "wd_001",
            "question": "白矮星的主要组成成分是什么？",
            "options": {
                "A": "氢和氦",
                "B": "碳和氧（或氧氖镁）",
                "C": "铁和镍",
                "D": "中子简并物质"
            },
            "correct": "B",
            "subfield": "single_wd",
            "difficulty": "easy",
            "explanation": "大多数白矮星由碳和氧组成，大质量白矮星可能由氧氖镁组成，小质量白矮星可能由氦组成。"
        },
        {
            "id": "wd_002",
            "question": "钱德拉塞卡极限的数值约为：",
            "options": {
                "A": "0.5 M☉",
                "B": "1.0 M☉",
                "C": "1.4 M☉",
                "D": "2.0 M☉"
            },
            "correct": "C",
            "subfield": "single_wd",
            "difficulty": "easy",
            "explanation": "钱德拉塞卡极限约为 1.38-1.4 倍太阳质量，是白矮星能通过电子简并压支撑的最大质量。"
        },
        {
            "id": "wd_003",
            "question": "白矮星冷却的主要机制是什么？",
            "options": {
                "A": "核聚变",
                "B": "光度辐射",
                "C": "中微子辐射",
                "D": "对流"
            },
            "correct": "B",
            "subfield": "single_wd",
            "difficulty": "medium",
            "explanation": "白矮星没有核能源，通过表面辐射损失热量而冷却。中微子辐射只在早期高温阶段重要。"
        },
        
        # 双白矮星
        {
            "id": "dwd_001",
            "question": "双白矮星系统的主要引力波频率范围是：",
            "options": {
                "A": "Hz 量级",
                "B": "mHz 量级",
                "C": "kHz 量级",
                "D": "MHz 量级"
            },
            "correct": "B",
            "subfield": "double_wd",
            "difficulty": "medium",
            "explanation": "双白矮星系统周期为数小时到数分钟，对应的引力波频率在 mHz 范围，是 LISA 等空间探测器的主要目标。"
        },
        {
            "id": "dwd_002",
            "question": "关于双白矮星并合，以下说法正确的是：",
            "options": {
                "A": "所有双白矮星并合都会产生 Ia 型超新星",
                "B": "并合产物总质量超过钱德拉塞卡极限时可能产生超新星",
                "C": "并合总是产生中子星",
                "D": "并合产物只能是 R CrB 型星"
            },
            "correct": "B",
            "subfield": "double_wd",
            "difficulty": "hard",
            "explanation": "只有总质量超过钱德拉塞卡极限的双白矮星并合才可能产生 Ia 型超新星；亚钱德拉塞卡并合可能产生其他类型天体。"
        },
        {
            "id": "dwd_003",
            "question": "双白矮星系统的轨道周期如何演化？",
            "options": {
                "A": "由于引力波辐射而逐渐增大",
                "B": "由于引力波辐射而逐渐减小",
                "C": "保持恒定",
                "D": "随机变化"
            },
            "correct": "B",
            "subfield": "double_wd",
            "difficulty": "medium",
            "explanation": "双白矮星辐射引力波损失轨道能量，导致轨道收缩，周期减小，最终可能并合。"
        },
        
        # 磁性白矮星
        {
            "id": "mwd_001",
            "question": "大约多少比例的白矮星表现出可探测的磁场？",
            "options": {
                "A": "1-5%",
                "B": "10-20%",
                "C": "50-60%",
                "D": "90% 以上"
            },
            "correct": "B",
            "subfield": "magnetic_wd",
            "difficulty": "medium",
            "explanation": "约 10-20% 的白矮星表现出可探测的磁场，场强范围从 10^3 G 到 10^9 G。"
        },
        {
            "id": "mwd_002",
            "question": "极高磁场（>10^8 G）的白矮星称为：",
            "options": {
                "A": "DA 型",
                "B": "DB 型",
                "C": "DQ 型",
                "D": "高场白矮星（High-field WD）"
            },
            "correct": "D",
            "subfield": "magnetic_wd",
            "difficulty": "medium",
            "explanation": "磁场超过 10^8 G 的白矮星被称为高场白矮星，约占磁性白矮星的一部分。"
        },
        {
            "id": "mwd_003",
            "question": "磁性白矮星的磁场可能来源包括：",
            "options": {
                "A": "原恒星磁场冻结",
                "B": "双星并合发电机效应",
                "C": "氦核闪期间的湍流发电机",
                "D": "以上皆是"
            },
            "correct": "D",
            "subfield": "magnetic_wd",
            "difficulty": "hard",
            "explanation": "磁性白矮星的强磁场可能来源于原恒星磁场的冻结和放大、双星并合过程中的发电机效应，或氦核闪期间的湍流发电机。"
        },
        
        # 吸积白矮星
        {
            "id": "acw_001",
            "question": "激变变星（CV）中，吸积白矮星的光变主要来自：",
            "options": {
                "A": "恒星脉动",
                "B": "吸积盘不稳定性",
                "C": "潮汐相互作用",
                "D": "恒星自转"
            },
            "correct": "B",
            "subfield": "accreting_wd",
            "difficulty": "medium",
            "explanation": "激变变星的光变主要来自吸积盘的不稳定性（如 DIM 模型），导致亮度周期性变化（矮新星爆发）。"
        },
        {
            "id": "acw_002",
            "question": "经典新星爆发发生在：",
            "options": {
                "A": "白矮星核心",
                "B": "白矮星表面吸积层",
                "C": "伴星表面",
                "D": "吸积盘"
            },
            "correct": "B",
            "subfield": "accreting_wd",
            "difficulty": "medium",
            "explanation": "新星爆发发生在白矮星表面积累的氢层，当温度和压力达到临界值时发生失控的热核聚变。"
        },
        {
            "id": "acw_003",
            "question": "AM CVn 型系统的组成是：",
            "options": {
                "A": "白矮星 + 中子星",
                "B": "白矮星 + 氦星（或白矮星）",
                "C": "白矮星 + 主序星",
                "D": "两颗主序星"
            },
            "correct": "B",
            "subfield": "accreting_wd",
            "difficulty": "hard",
            "explanation": "AM CVn 是双白矮星系统，其中一颗已失去氢包层，成为氦星或裸露的氦白矮星。绝不含中子星！"
        },
        {
            "id": "acw_004",
            "question": "AM CVn 型系统的典型轨道周期范围是：",
            "options": {
                "A": "数天到数周",
                "B": "数小时",
                "C": "5-65 分钟",
                "D": "数秒"
            },
            "correct": "C",
            "subfield": "accreting_wd",
            "difficulty": "hard",
            "explanation": "AM CVn 是超紧凑双星，轨道周期极短，典型范围是 5-65 分钟，是已知周期最短的双星系统之一。"
        },
        
        # 综合/反幻觉
        {
            "id": "hal_001",
            "question": "关于白矮星，以下说法错误的是：",
            "options": {
                "A": "白矮星主要由碳和氧组成",
                "B": "白矮星通过核聚变产生能量",
                "C": "钱德拉塞卡极限约为 1.4 M☉",
                "D": "白矮星是恒星演化的最终阶段之一"
            },
            "correct": "B",
            "subfield": "anti_hallucination",
            "difficulty": "easy",
            "explanation": "白矮星没有核聚变反应，通过残余热量辐射冷却。这是常见误解。"
        },
        {
            "id": "hal_002",
            "question": "关于双白矮星并合，以下说法错误的是：",
            "options": {
                "A": "并合可能产生 Ia 型超新星",
                "B": "所有双白矮星并合都会产生超新星",
                "C": "并合产物可能是 R CrB 型星",
                "D": "并合过程辐射引力波"
            },
            "correct": "B",
            "subfield": "anti_hallucination",
            "difficulty": "medium",
            "explanation": "只有总质量超过钱德拉塞卡极限的双白矮星并合才可能产生超新星，并非所有并合都如此。"
        },
    ]
    
    def __init__(self, model_interface=None, output_dir: str = "./eval_results"):
        super().__init__(model_interface, output_dir)
        
        # 扩展幻觉检测
        self.HALLUCINATION_CHECKS.update(self.WD_HALLUCINATION_CHECKS)
    
    def load_wd_benchmark(self) -> List[Dict]:
        """加载白矮星基准题库"""
        print(f"✓ 加载白矮星基准题库: {len(self.WD_BENCHMARK)} 题")
        return self.WD_BENCHMARK
    
    def evaluate_wd_knowledge(self) -> Dict:
        """
        评估白矮星领域知识
        
        Returns:
            评估结果汇总
        """
        dataset = self.load_wd_benchmark()
        summary = self.evaluate(dataset)
        
        # 白矮星特有分析
        print("\n" + "=" * 60)
        print("白矮星领域专项分析")
        print("=" * 60)
        
        # 子领域分析
        subfields = {
            "single_wd": "单白矮星",
            "double_wd": "双白矮星",
            "magnetic_wd": "磁性白矮星",
            "accreting_wd": "吸积白矮星",
            "anti_hallucination": "反幻觉"
        }
        
        for sf_key, sf_name in subfields.items():
            sf_results = [r for r in summary.results if r.subfield == sf_key]
            if sf_results:
                sf_correct = sum(1 for r in sf_results if r.is_correct)
                sf_acc = sf_correct / len(sf_results)
                print(f"  {sf_name:12s}: {sf_acc*100:5.1f}% ({sf_correct}/{len(sf_results)})")
        
        # 幻觉检测统计
        hall_results = [r for r in summary.results if r.subfield == "anti_hallucination"]
        if hall_results:
            hall_correct = sum(1 for r in hall_results if r.is_correct)
            print(f"\n  反幻觉测试: {hall_correct}/{len(hall_results)} 通过")
        
        return summary
    
    def generate_wd_report(self, summary) -> str:
        """生成白矮星专用报告"""
        report = super().generate_report(summary)
        
        # 添加白矮星领域专项评估
        wd_analysis = "\n" + "=" * 70 + "\n"
        wd_analysis += "【白矮星领域专项评估】\n"
        wd_analysis += "=" * 70 + "\n"
        
        # 关键概念掌握度
        key_concepts = [
            ("钱德拉塞卡极限", "wd_002"),
            ("双白矮星引力波", "dwd_001"),
            ("磁性白矮星比例", "mwd_001"),
            ("AM CVn 组成", "acw_003"),
            ("AM CVn 周期", "acw_004"),
        ]
        
        wd_analysis += "\n关键概念掌握:\n"
        for concept_name, qid in key_concepts:
            result = next((r for r in summary.results if r.question_id == qid), None)
            if result:
                status = "✓" if result.is_correct else "✗"
                wd_analysis += f"  {status} {concept_name}\n"
        
        # 常见错误分析
        errors = [r for r in summary.results if not r.is_correct]
        wd_errors = [e for e in errors if e.subfield in 
                     ["double_wd", "magnetic_wd", "accreting_wd", "anti_hallucination"]]
        
        if wd_errors:
            wd_analysis += "\n白矮星领域错误分析:\n"
            for e in wd_errors[:5]:
                wd_analysis += f"  - {e.question[:50]}...\n"
                wd_analysis += f"    正确答案: {e.correct_answer}, 模型: {e.model_answer}\n"
        
        wd_analysis += "\n" + "=" * 70 + "\n"
        
        return report + wd_analysis


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="白矮星专用评估")
    parser.add_argument("--model", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--interface", type=str, default="ollama", 
                       choices=["ollama", "hf"], help="模型接口")
    parser.add_argument("--output-dir", type=str, default="./eval_results", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌟 白矮星专用模型评估")
    print("=" * 60)
    
    # 创建模型接口
    if args.interface == "ollama":
        from astronomy_evaluator import OllamaModelInterface
        model = OllamaModelInterface(args.model)
    else:
        from astronomy_evaluator import HuggingFaceModelInterface
        model = HuggingFaceModelInterface(args.model)
    
    # 创建评估器
    evaluator = WhiteDwarfEvaluator(model, args.output_dir)
    
    # 运行评估
    summary = evaluator.evaluate_wd_knowledge()
    
    # 生成报告
    report = evaluator.generate_wd_report(summary)
    print("\n" + report)
    
    # 保存结果
    output_path = evaluator.save_results(summary)
    print(f"\n✓ 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
