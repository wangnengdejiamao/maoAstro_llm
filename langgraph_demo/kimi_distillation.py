#!/usr/bin/env python3
"""
Kimi API 天文知识蒸馏系统
=========================
使用 Kimi API 作为教师模型，蒸馏白矮星、CVs、Polars 等天文知识

支持的天体类型:
- 白矮星 (White Dwarfs) - DA, DB, DC, DO, DQ, DZ 等光谱型
- 激变变星 (Cataclysmic Variables)
- 磁旋密近双星 (Polars/AM Her type)
- 白矮双星 (White Dwarf Binaries)
- 星团白矮星 (Cluster White Dwarfs)

API: Kimi (Moonshot AI)
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Kimi API 配置
KIMI_API_KEY = "sk-kimi-SoSpEtPpAUpQN94Mng37gYiJ5scgv3WyDR7AfKDhyv01Awca6yfUKod9lcbNa6Uj"
KIMI_API_BASE = "https://api.moonshot.cn/v1"


@dataclass
class AstronomicalTarget:
    """天文目标数据结构"""
    name: str
    ra: float
    dec: float
    target_type: str  # WD, CV, Polar, WD_binary, Cluster_WD
    spectral_type: Optional[str] = None
    period: Optional[float] = None
    magnitude: Optional[float] = None
    distance: Optional[float] = None  # pc
    notes: Optional[str] = None


class KimiAPIClient:
    """Kimi API 客户端"""
    
    def __init__(self, api_key: str = KIMI_API_KEY):
        self.api_key = api_key
        self.base_url = KIMI_API_BASE
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "moonshot-v1-128k",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Optional[str]:
        """
        调用 Kimi API 进行对话
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
        
        Returns:
            生成的文本内容
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ✗ API 调用失败: {e}")
            return None


class AstronomicalKnowledgeDistiller:
    """天文知识蒸馏器"""
    
    # 天体类型定义
    TARGET_TYPES = {
        "DA_WD": {
            "name": "氢大气白矮星 (DA White Dwarf)",
            "features": ["Balmer线", "氢大气", "光谱型DA"],
            "temperature_range": "8000-40000 K",
            "mass_range": "0.5-1.4 M☉"
        },
        "DB_WD": {
            "name": "氦大气白矮星 (DB White Dwarf)",
            "features": ["氦线", "无氢线", "光谱型DB"],
            "temperature_range": "12000-20000 K",
            "mass_range": "0.5-1.2 M☉"
        },
        "DC_WD": {
            "name": "连续谱白矮星 (DC White Dwarf)",
            "features": ["无明显谱线", "连续谱", "低温"],
            "temperature_range": "4000-8000 K",
            "mass_range": "0.5-1.0 M☉"
        },
        "CV": {
            "name": "激变变星 (Cataclysmic Variable)",
            "features": ["白矮星+红矮星", "吸积盘", "爆发"],
            "period_range": "0.05-0.5 days",
            "types": ["新星", "矮新星", "极向星"]
        },
        "POLAR": {
            "name": "磁旋密近双星 / 极向星 (Polar/AM Her)",
            "features": ["强磁场(10-100 MG)", "同步自转", "吸积柱"],
            "period_range": "0.05-0.3 days",
            "magnetic_field": "10-100 MG"
        },
        "WD_BINARY": {
            "name": "白矮双星 (White Dwarf Binary)",
            "features": ["双白矮星", "引力波源", "并合候选体"],
            "period_range": "0.01-10 days",
            "mass_total": "0.8-2.8 M☉"
        },
        "CLUSTER_WD": {
            "name": "星团白矮星 (Cluster White Dwarf)",
            "features": ["星团成员", "年龄示踪", "冷却序列"],
            "ages": "100 Myr - 10 Gyr",
            "examples": ["M4", "47 Tuc", "NGC 6397"]
        },
        "DZ_WD": {
            "name": "金属线白矮星 (DZ White Dwarf)",
            "features": ["金属线(Ca, Mg, Fe)", "污染大气", "行星残骸"],
            "temperature_range": "4000-12000 K",
            "pollution_source": "行星物质吸积"
        }
    }
    
    def __init__(self, api_client: Optional[KimiAPIClient] = None):
        self.api = api_client or KimiAPIClient()
        self.distilled_data: List[Dict] = []
    
    def generate_knowledge_prompt(self, target: AstronomicalTarget) -> str:
        """生成知识提取提示词"""
        
        target_info = self.TARGET_TYPES.get(target.target_type, {})
        
        prompt = f"""你是一位专业的白矮星和变星天体物理学家。请详细分析以下天体目标，提供专业级的物理分析和观测特征。

## 目标信息
- 名称: {target.name}
- 坐标: RA={target.ra:.6f}°, DEC={target.dec:.6f}°
- 类型: {target_info.get('name', target.target_type)}
- 光谱型: {target.spectral_type or '未知'}
- 周期: {target.period or '未知'} days
- 星等: {target.magnitude or '未知'}

## 请提供以下分析（用中文回答）:

### 1. 基本物理特征
- 光谱特征和大气成分
- 温度、质量估计
- 半径和光度

### 2. 形成与演化
- 形成机制
- 演化阶段
- 最终命运

### 3. 观测特征
- 光变特征（如果有）
- 光谱诊断特征
- 多波段测光特征

### 4. 物理机制（针对特定类型）
"""
        
        # 根据类型添加特定问题
        if target.target_type == "POLAR":
            prompt += """
- 磁场强度和结构
- 吸积柱物理
- 同步自转机制
- X射线辐射机制
"""
        elif target.target_type == "CV":
            prompt += """
- 吸积盘结构
- 热斑位置
- 爆发机制（新星/矮新星）
- 质量转移率
"""
        elif target.target_type == "WD_BINARY":
            prompt += """
- 轨道衰减与引力波辐射
- 并合时间尺度
- Ia型超新星爆发可能性
- 潮汐锁定效应
"""
        elif target.target_type in ["DA_WD", "DB_WD", "DC_WD", "DZ_WD"]:
            prompt += """
- 大气层结构
- 冷却过程
- 结晶化过程
- 年龄估计方法
"""
        
        prompt += """

### 5. 研究意义
- 在天体物理中的重要性
- 推荐的后续观测

请提供详细、专业的分析，适合用于训练天文领域的AI模型。
"""
        return prompt
    
    def distill_target(self, target: AstronomicalTarget, save_intermediate: bool = True) -> Optional[Dict]:
        """
        对单个目标进行知识蒸馏
        
        Args:
            target: 天文目标
            save_intermediate: 是否保存中间结果
        
        Returns:
            蒸馏后的知识数据
        """
        print(f"\n{'='*60}")
        print(f"蒸馏目标: {target.name} ({target.target_type})")
        print(f"坐标: RA={target.ra:.6f}, DEC={target.dec:.6f}")
        print(f"{'='*60}")
        
        # 生成提示词
        prompt = self.generate_knowledge_prompt(target)
        
        # 调用 Kimi API
        print("  调用 Kimi API 获取专业知识...")
        messages = [
            {"role": "system", "content": "你是一位资深的天体物理学家，专门研究白矮星和密近双星系统。请提供准确、详细、专业的分析。"},
            {"role": "user", "content": prompt}
        ]
        
        knowledge_text = self.api.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=4096
        )
        
        if knowledge_text is None:
            print("  ✗ 知识蒸馏失败")
            return None
        
        print("  ✓ 成功获取专业知识")
        
        # 构建训练数据格式
        training_example = {
            "input": {
                "query": f"分析天体 {target.name} ({target.target_type})",
                "coordinates": {
                    "ra": target.ra,
                    "dec": target.dec
                },
                "target_type": target.target_type,
                "spectral_type": target.spectral_type,
                "period": target.period
            },
            "output": {
                "target_name": target.name,
                "classification": self.TARGET_TYPES.get(target.target_type, {}).get("name", target.target_type),
                "detailed_analysis": knowledge_text,
                "key_features": self.TARGET_TYPES.get(target.target_type, {}).get("features", []),
                "physical_params": {
                    "temperature": "见详细分析",
                    "mass": "见详细分析",
                    "radius": "见详细分析"
                }
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "Kimi API (Moonshot)",
                "distillation_method": "teacher_forcing",
                "target_category": target.target_type
            }
        }
        
        # 保存中间结果
        if save_intermediate:
            output_dir = Path("langgraph_demo/output/distilled")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{target.name.replace(' ', '_')}_{target.target_type}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_example, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 已保存: {filepath}")
        
        return training_example
    
    def batch_distill(self, targets: List[AstronomicalTarget], delay: float = 1.0) -> List[Dict]:
        """
        批量蒸馏多个目标
        
        Args:
            targets: 目标列表
            delay: API 调用间隔（秒）
        
        Returns:
            蒸馏后的数据集
        """
        print("\n" + "="*70)
        print("开始批量知识蒸馏")
        print(f"目标数量: {len(targets)}")
        print("="*70)
        
        dataset = []
        success_count = 0
        
        for i, target in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] 处理 {target.name}...")
            
            result = self.distill_target(target)
            if result:
                dataset.append(result)
                success_count += 1
            
            # API 限流保护
            if i < len(targets):
                time.sleep(delay)
        
        # 保存完整数据集
        output_dir = Path("langgraph_demo/output/distilled")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = output_dir / "astro_knowledge_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print(f"批量蒸馏完成!")
        print(f"  成功: {success_count}/{len(targets)}")
        print(f"  数据集保存: {dataset_file}")
        print(f"  数据条数: {len(dataset)}")
        print("="*70)
        
        return dataset


# ==================== 预定义目标列表 ====================

def create_test_targets() -> List[AstronomicalTarget]:
    """创建测试目标列表"""
    
    targets = [
        # 白矮星样本
        AstronomicalTarget(
            name="Sirius B",
            ra=101.2875,
            dec=-16.7161,
            target_type="DA_WD",
            spectral_type="DA2",
            magnitude=8.44,
            notes="最近的DA白矮星，天狼星伴星"
        ),
        AstronomicalTarget(
            name="40 Eri B",
            ra=64.3333,
            dec=-7.5167,
            target_type="DA_WD",
            spectral_type="DA4",
            magnitude=9.52,
            notes="三合星系统中的白矮星"
        ),
        
        # 激变变星
        AstronomicalTarget(
            name="AM Her",
            ra=274.0554,
            dec=49.8679,
            target_type="POLAR",
            spectral_type="M4.5V+WD",
            period=0.1289,
            magnitude=12.3,
            notes="原型极向星，强磁场"
        ),
        AstronomicalTarget(
            name="EV UMa",
            ra=13.1316,
            dec=53.8585,
            target_type="CV",
            spectral_type="CV",
            period=0.10025,
            magnitude=14.2,
            notes="短周期激变变星"
        ),
        
        # 白矮双星
        AstronomicalTarget(
            name="WD 1202-024",
            ra=181.1521,
            dec=-2.6997,
            target_type="WD_BINARY",
            spectral_type="WD+WD",
            period=0.0266,
            notes="超短周期白矮双星"
        ),
        
        # 星团白矮星
        AstronomicalTarget(
            name="M4 WD 1346",
            ra=245.8962,
            dec=-26.5258,
            target_type="CLUSTER_WD",
            spectral_type="DA",
            magnitude=18.5,
            notes="球状星团M4中的白矮星"
        ),
        
        # 金属污染白矮星
        AstronomicalTarget(
            name="WD J0914+1914",
            ra=138.5208,
            dec=19.7389,
            target_type="DZ_WD",
            spectral_type="DZ",
            notes="带有巨行星的金属污染白矮星"
        ),
        
        # DB白矮星
        AstronomicalTarget(
            name="GD 358",
            ra=236.3225,
            dec=33.0144,
            target_type="DB_WD",
            spectral_type="DB2",
            period=0.0097,
            notes="脉动DB白矮星，ZZ Ceti型"
        ),
        
        # 更多CVs
        AstronomicalTarget(
            name="SS Cyg",
            ra=325.4125,
            dec=43.5833,
            target_type="CV",
            spectral_type="CV",
            period=0.275,
            magnitude=8.2,
            notes="最亮的矮新星之一"
        ),
    ]
    
    return targets


# ==================== 主程序 ====================

def test_api_connection():
    """测试 API 连接"""
    print("="*70)
    print("测试 Kimi API 连接...")
    print("="*70)
    
    api = KimiAPIClient()
    messages = [
        {"role": "user", "content": "你好，请简要介绍一下白矮星的天体物理特征（用中文回答，100字以内）"}
    ]
    
    response = api.chat_completion(messages, max_tokens=200)
    
    if response:
        print(f"✓ API 连接成功!\n")
        print("响应示例:")
        print(f"{response[:300]}...")
        return True
    else:
        print("✗ API 连接失败")
        return False


def main():
    """主程序"""
    print("\n" + "="*70)
    print("Kimi API 天文知识蒸馏系统")
    print("="*70)
    
    # 测试 API
    if not test_api_connection():
        print("\n✗ 无法连接到 Kimi API，请检查 API key")
        return
    
    # 创建蒸馏器
    distiller = AstronomicalKnowledgeDistiller()
    
    # 创建测试目标
    targets = create_test_targets()
    
    print(f"\n准备蒸馏 {len(targets)} 个天文目标:")
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t.name} ({t.target_type})")
    
    # 询问是否开始
    print(f"\n是否开始批量蒸馏? (y/n): ", end="")
    # response = input().strip().lower()
    response = 'y'  # 自动开始
    
    if response == 'y':
        # 执行批量蒸馏
        dataset = distiller.batch_distill(targets, delay=2.0)
        
        print("\n✓ 所有目标蒸馏完成!")
        print(f"  生成的训练数据集可用于微调 Qwen 模型")
        
        # 显示数据集统计
        print("\n数据集统计:")
        type_counts = {}
        for item in dataset:
            t_type = item['input']['target_type']
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
        
        for t_type, count in sorted(type_counts.items()):
            print(f"  {t_type}: {count} 条")
    else:
        # 单独测试一个目标
        print("\n测试单个目标...")
        target = targets[0]  # Sirius B
        result = distiller.distill_target(target)
        
        if result:
            print("\n蒸馏结果预览:")
            print(f"  目标: {result['output']['target_name']}")
            print(f"  分类: {result['output']['classification']}")
            print(f"  分析长度: {len(result['output']['detailed_analysis'])} 字符")


if __name__ == '__main__':
    main()
