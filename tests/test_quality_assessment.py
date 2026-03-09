#!/usr/bin/env python3
"""
模型质量评估测试
================
基于评估报告中的问题，测试修改后的改进效果

评估维度：
1. AM CVn 知识准确性
2. 周期-光度关系理解
3. 常见错误避免（幻觉检测）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_system import AstronomyRAG


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        self.rag = AstronomyRAG()
        self.results = []
        self.total_score = 0
        self.max_score = 0
    
    def test_am_cvn_composition(self):
        """测试1: AM CVn 系统组成（不应包含中子星）"""
        print("\n" + "="*70)
        print("测试1: AM CVn 系统组成")
        print("="*70)
        print("标准：AM CVn 是白矮星+氦星，绝对不含中子星")
        
        query = "AM CVn 系统组成 中子星"
        result = self.rag.search(query, top_k=2)
        
        checks = {
            "包含'白矮星'": "白矮星" in result,
            "不含'中子星'错误": "中子星" not in result or "不含" in result or "NO" in result or "错误" in result,
            "提到氦星/He star": "氦" in result or "He" in result,
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("AM CVn系统组成", score, checks, result[:500])
        return score
    
    def test_am_cvn_period_range(self):
        """测试2: AM CVn 周期范围"""
        print("\n" + "="*70)
        print("测试2: AM CVn 周期范围")
        print("="*70)
        print("标准：轨道周期 5-65 分钟")
        
        query = "AM CVn 周期范围"
        result = self.rag.search(query, top_k=2)
        
        checks = {
            "提到5-65分钟": "5-65" in result or "5~65" in result or ("5" in result and "65" in result and "分钟" in result),
            "提到极短周期": "极短" in result or "ultra-compact" in result.lower() or "extremely short" in result.lower(),
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("AM CVn周期范围", score, checks, result[:500])
        return score
    
    def test_am_cvn_variability_mechanism(self):
        """测试3: AM CVn 光变机制（非潮汐）"""
        print("\n" + "="*70)
        print("测试3: AM CVn 光变机制")
        print("="*70)
        print("标准：光变来自吸积盘不稳定性，不是潮汐")
        
        query = "AM CVn 光变 潮汐"
        result = self.rag.search(query, top_k=2)
        
        # 获取常见错误提示
        mistake_hint = self.rag.get_common_mistakes_hint("am_cvn")
        
        checks = {
            "提到吸积盘": "吸积" in result or "accretion" in result.lower(),
            "提到不稳定性/instability": "不稳定" in result or "instability" in result.lower(),
            "纠正潮汐误解": "不是潮汐" in result or "NOT" in result or "错误" in result,
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("AM CVn光变机制", score, checks, result[:500] + "\n" + mistake_hint[:300])
        return score
    
    def test_period_luminosity_relation(self):
        """测试4: 周期-光度关系理解"""
        print("\n" + "="*70)
        print("测试4: 周期-光度关系")
        print("="*70)
        print("标准：AM CVn 是统计相关，无严格公式")
        
        query = "AM CVn 周期-光度关系 公式"
        result = self.rag.search(query, top_k=2)
        
        checks = {
            "说明是统计相关": "统计" in result or "statistical" in result.lower(),
            "说明无严格公式": "无严格" in result or "no strict" in result.lower() or "not a strict" in result.lower(),
            "提到Mdot关系": "Mdot" in result or "质量转移" in result,
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("周期-光度关系", score, checks, result[:500])
        return score
    
    def test_gravitational_wave_sources(self):
        """测试5: 引力波源重要性"""
        print("\n" + "="*70)
        print("测试5: 引力波源")
        print("="*70)
        print("标准：提到LISA和引力波辐射驱动")
        
        query = "AM CVn 引力波 LISA"
        result = self.rag.search(query, top_k=2)
        
        checks = {
            "提到LISA": "LISA" in result,
            "提到引力波": "引力波" in result or "gravitational wave" in result.lower(),
            "提到驱动机制": "驱动" in result or "radiation" in result.lower() or "辐射" in result,
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("引力波源", score, checks, result[:500])
        return score
    
    def test_rag_knowledge_coverage(self):
        """测试6: RAG知识库覆盖度"""
        print("\n" + "="*70)
        print("测试6: RAG知识库覆盖度")
        print("="*70)
        
        topics = self.rag.list_topics()
        expected_topics = ['am_cvn', 'cataclysmic_variable', 'white_dwarf', 'polar', 'rr_lyrae']
        
        checks = {}
        for topic in expected_topics:
            checks[f"包含{topic}"] = topic in topics
        
        # 检查文档块数量
        doc_count = len(self.rag.documents)
        checks["文档块数量>5"] = doc_count > 5
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("RAG知识覆盖", score, checks, f"知识主题: {len(topics)}个, 文档块: {doc_count}个")
        return score
    
    def test_common_mistakes_prevention(self):
        """测试7: 常见错误防范提示"""
        print("\n" + "="*70)
        print("测试7: 常见错误防范")
        print("="*70)
        
        hint = self.rag.get_common_mistakes_hint("am_cvn")
        
        checks = {
            "提到中子星错误": "中子星" in hint and ("错误" in hint or "❌" in hint),
            "提到潮汐错误": "潮汐" in hint and ("错误" in hint or "❌" in hint),
            "提到公式错误": "公式" in hint and ("错误" in hint or "❌" in hint or "编造" in hint),
        }
        
        score = sum(checks.values()) / len(checks) * 100
        self._record_result("错误防范", score, checks, hint[:500])
        return score
    
    def _record_result(self, test_name: str, score: float, checks: dict, evidence: str):
        """记录测试结果"""
        self.results.append({
            'test': test_name,
            'score': score,
            'checks': checks,
            'evidence': evidence
        })
        self.total_score += score
        self.max_score += 100
        
        # 打印结果
        status = "✅ PASS" if score >= 70 else ("⚠️ WARNING" if score >= 50 else "❌ FAIL")
        print(f"\n得分: {score:.1f}/100 {status}")
        for check, passed in checks.items():
            print(f"  {'✓' if passed else '✗'} {check}")
        print(f"\n证据片段:\n{evidence[:400]}...")
    
    def generate_report(self):
        """生成评估报告"""
        overall_score = self.total_score / len(self.results) if self.results else 0
        
        report = []
        report.append("\n" + "="*70)
        report.append("模型质量评估报告")
        report.append("="*70)
        report.append(f"总体评分: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            report.append("评级: 🟢 优秀 - 知识库完善，错误防范到位")
        elif overall_score >= 60:
            report.append("评级: 🟡 良好 - 基本正确，仍有改进空间")
        elif overall_score >= 40:
            report.append("评级: 🟠 及格 - 存在明显问题")
        else:
            report.append("评级: 🔴 不及格 - 需要重大改进")
        
        report.append("\n详细评分:")
        for r in self.results:
            status = "🟢" if r['score'] >= 70 else ("🟡" if r['score'] >= 50 else "🔴")
            report.append(f"  {status} {r['test']}: {r['score']:.1f}/100")
        
        report.append("\n改进建议:")
        
        # 根据得分生成建议
        low_scores = [r for r in self.results if r['score'] < 60]
        if not low_scores:
            report.append("  ✅ 所有测试项表现良好，知识库已完善")
        else:
            for r in low_scores:
                report.append(f"  ⚠️ {r['test']}: 需要加强相关知识")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report), overall_score


def main():
    """主函数"""
    print("="*70)
    print("AstroSage 模型质量评估")
    print("基于 AM CVn 知识准确性测试")
    print("="*70)
    
    assessor = QualityAssessor()
    
    # 运行所有测试
    assessor.test_am_cvn_composition()
    assessor.test_am_cvn_period_range()
    assessor.test_am_cvn_variability_mechanism()
    assessor.test_period_luminosity_relation()
    assessor.test_gravitational_wave_sources()
    assessor.test_rag_knowledge_coverage()
    assessor.test_common_mistakes_prevention()
    
    # 生成报告
    report, score = assessor.generate_report()
    print(report)
    
    # 保存报告
    report_file = "quality_assessment_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存: {report_file}")
    
    return score


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 60 else 1)
