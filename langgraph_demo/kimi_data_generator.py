#!/usr/bin/env python3
"""
Kimi API 训练数据生成器
=======================
使用 Kimi API 生成高质量天文领域训练数据，用于后续模型微调或 RAG

API Key: Set via KIMI_API_KEY environment variable

功能：
1. 生成天文领域专业问答对
2. 生成 RAG 文档片段
3. 生成 Tool 调用示例
"""

import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class KimiConfig:
    """Kimi API 配置"""
    api_key: str = os.getenv("KIMI_API_KEY", "YOUR_API_KEY_HERE")
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "moonshot-v1-128k"  # 或 moonshot-v1-8k, moonshot-v1-32k
    temperature: float = 0.3  # 低温度确保输出稳定
    max_tokens: int = 4096


class KimiDataGenerator:
    """Kimi API 数据生成器"""
    
    def __init__(self, config: KimiConfig = None):
        self.config = config or KimiConfig()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            print("✓ Kimi API 客户端初始化成功")
        except ImportError:
            print("✗ 请先安装 openai: pip install openai")
            raise
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """调用 Kimi API 生成文本"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"✗ API 调用失败: {e}")
            return ""
    
    def generate_training_data(self, targets: List[Dict]) -> List[Dict]:
        """
        生成训练数据（问答对）
        
        Args:
            targets: 天体目标列表，每个包含 name, ra, dec, type 等信息
        
        Returns:
            训练数据列表，每个包含 input 和 output
        """
        print("\n" + "=" * 70)
        print("使用 Kimi API 生成训练数据")
        print("=" * 70)
        
        training_data = []
        system_prompt = """你是一位资深天文学家，专门研究变星、双星和系外行星。
你的回答应该专业、准确、详细，包含：
1. 精确的天体分类
2. 具体的物理参数（带数值和单位）
3. 观测建议（望远镜参数、滤镜、最佳观测时间）
4. 相关的科学背景知识"""
        
        for target in tqdm(targets, desc="生成数据"):
            # 构建 prompt
            prompt = self._build_astronomy_prompt(target)
            
            # 调用 Kimi API
            response = self.generate(prompt, system_prompt)
            
            if response:
                data_item = {
                    "input": {
                        "query": f"分析天体 {target.get('name', 'Unknown')}",
                        "coordinates": {
                            "ra": target.get('ra'),
                            "dec": target.get('dec')
                        },
                        "context": target
                    },
                    "output": {
                        "classification": target.get('type', 'Unknown'),
                        "analysis": response,
                        "raw_params": target
                    }
                }
                training_data.append(data_item)
                
                # 避免 API 限流
                time.sleep(0.5)
        
        print(f"✓ 生成完成: {len(training_data)} 条训练数据")
        return training_data
    
    def generate_rag_documents(self, topics: List[str]) -> List[Dict]:
        """
        生成 RAG 文档片段
        
        Args:
            topics: 天文主题列表，如 ["Cataclysmic Variables", "Cepheid Stars"]
        
        Returns:
            文档列表，每个包含 content 和 metadata
        """
        print("\n" + "=" * 70)
        print("生成 RAG 知识库文档")
        print("=" * 70)
        
        documents = []
        
        for topic in tqdm(topics, desc="生成文档"):
            system_prompt = f"你是一位天文学专家。请详细解释 {topic}，包括定义、特征、分类、观测方法和重要实例。"
            
            prompt = f"请提供关于 {topic} 的详细知识，适合作为 RAG 系统的知识库文档。包含：\n1. 定义和基本概念\n2. 物理特征和分类\n3. 观测方法和判据\n4. 重要实例和数据\n5. 研究意义"
            
            content = self.generate(prompt, system_prompt)
            
            if content:
                doc = {
                    "content": content,
                    "metadata": {
                        "topic": topic,
                        "category": "astronomy",
                        "source": "kimi_generated"
                    }
                }
                documents.append(doc)
                time.sleep(0.5)
        
        print(f"✓ 生成完成: {len(documents)} 篇文档")
        return documents
    
    def generate_tool_examples(self, tools: List[Dict], num_examples: int = 50) -> List[Dict]:
        """
        生成 Tool 调用训练示例
        
        Args:
            tools: 可用工具列表
            num_examples: 生成示例数量
        
        Returns:
            Tool 调用示例列表
        """
        print("\n" + "=" * 70)
        print("生成 Tool 调用示例")
        print("=" * 70)
        
        examples = []
        
        tool_descriptions = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in tools
        ])
        
        system_prompt = f"""你是一个天文 AI 助手，可以使用以下工具：
{tool_descriptions}

当用户的问题需要使用工具时，你应该输出 JSON 格式的工具调用：
{{"tool": "工具名", "parameters": {{参数}}}}"""
        
        # 预定义的查询模板
        query_templates = [
            "查询坐标 RA={ra}, DEC={dec} 的光变曲线",
            "获取 {star_name} 的 SDSS 光谱",
            "分析 {ra}, {dec} 的消光数据",
            "搜索 {ra}, {dec} 附近的变星",
            "计算 {ra}, {dec} 的天区覆盖",
        ]
        
        import random
        
        for i in tqdm(range(num_examples), desc="生成示例"):
            # 随机生成坐标
            ra = round(random.uniform(0, 360), 4)
            dec = round(random.uniform(-90, 90), 4)
            star_name = random.choice(["EV UMa", "VY CMa", "Betelgeuse", "Algol", "Mira"])
            
            # 随机选择模板
            template = random.choice(query_templates)
            query = template.format(ra=ra, dec=dec, star_name=star_name)
            
            prompt = f"用户查询: {query}\n\n请判断是否使用工具，如需要则输出工具调用 JSON。"
            
            response = self.generate(prompt, system_prompt)
            
            if response:
                example = {
                    "query": query,
                    "context": {"ra": ra, "dec": dec, "star_name": star_name},
                    "response": response
                }
                examples.append(example)
                time.sleep(0.3)
        
        print(f"✓ 生成完成: {len(examples)} 个示例")
        return examples
    
    def _build_astronomy_prompt(self, target: Dict) -> str:
        """构建天文学分析 prompt"""
        name = target.get('name', 'Unknown')
        ra = target.get('ra', 'N/A')
        dec = target.get('dec', 'N/A')
        obj_type = target.get('type', 'Unknown')
        
        return f"""请详细分析以下天体目标：

天体名称: {name}
赤经 (RA): {ra}°
赤纬 (DEC): {dec}°
已知类型: {obj_type}

请提供以下信息：
1. 精确的天体分类（具体到子类型）
2. 关键物理参数（周期、光度、质量、距离等，带具体数值）
3. 观测建议（推荐望远镜口径、滤镜、曝光时间、最佳观测季节）
4. 该天体的科学价值和研究意义
5. 已知的类似天体对比

请用专业天文学术语回答。"""


def create_sample_targets() -> List[Dict]:
    """创建示例天体目标"""
    return [
        {
            "name": "EV UMa",
            "ra": 13.1316,
            "dec": 53.8585,
            "type": "Cataclysmic Variable (CV)",
            "period": 0.10025,
            "magnitude": 14.5,
            "references": ["Gaia DR3", "SIMBAD"]
        },
        {
            "name": "VY Canis Majoris",
            "ra": 110.743,
            "dec": -25.767,
            "type": "Red Hypergiant",
            "radius": 1420,  # 太阳半径
            "luminosity": 2.7e5,  # 太阳光度
            "references": ["Humphreys 2005"]
        },
        {
            "name": "Algol (Beta Persei)",
            "ra": 47.042,
            "dec": 40.956,
            "type": "Eclipsing Binary (Algol type)",
            "period": 2.867,
            "magnitude_range": [2.1, 3.4],
            "references": ["Batten 1989"]
        },
        {
            "name": "Mira (Omicron Ceti)",
            "ra": 34.836,
            "dec": -2.977,
            "type": "Mira Variable",
            "period": 332,
            "magnitude_range": [2.0, 10.1],
            "references": ["AAVSO"]
        },
        {
            "name": "HD 209458",
            "ra": 330.795,
            "dec": 18.884,
            "type": "Exoplanet Host Star",
            "planet": "HD 209458 b",
            "method": "Transit",
            "references": ["Charbonneau 2000"]
        }
    ]


def main():
    """主函数 - 生成完整数据集"""
    
    # 初始化生成器
    config = KimiConfig(
        api_key=os.getenv("KIMI_API_KEY", "YOUR_API_KEY_HERE"),
        model="moonshot-v1-32k"  # 使用 32k 上下文模型
    )
    
    generator = KimiDataGenerator(config)
    
    output_dir = "langgraph_demo/output/kimi_generated"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成训练数据
    print("\n【阶段 1/3】生成训练数据")
    targets = create_sample_targets()
    training_data = generator.generate_training_data(targets)
    
    with open(f"{output_dir}/training_data.json", 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 训练数据已保存: {output_dir}/training_data.json")
    
    # 2. 生成 RAG 文档
    print("\n【阶段 2/3】生成 RAG 知识库")
    topics = [
        "Cataclysmic Variables - Types and Characteristics",
        "Cepheid Variable Stars - Period-Luminosity Relation",
        "Eclipsing Binary Stars - Classification and Analysis",
        "Mira Variables - Long-Period Pulsating Stars",
        "Exoplanet Detection Methods - Transit Photometry",
        "Astronomical Photometry - Techniques and Calibration",
        "Time-Series Analysis for Variable Stars",
        "Gaia Mission - Variable Star Applications"
    ]
    documents = generator.generate_rag_documents(topics)
    
    with open(f"{output_dir}/rag_documents.json", 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"✓ RAG 文档已保存: {output_dir}/rag_documents.json")
    
    # 3. 生成 Tool 示例
    print("\n【阶段 3/3】生成 Tool 调用示例")
    tools = [
        {
            "name": "query_light_curve",
            "description": "查询指定坐标的光变曲线数据",
            "parameters": {"ra": "float", "dec": "float", "radius": "float"}
        },
        {
            "name": "get_spectrum",
            "description": "获取 SDSS 光谱数据",
            "parameters": {"ra": "float", "dec": "float"}
        },
        {
            "name": "query_extinction",
            "description": "查询消光数据",
            "parameters": {"ra": "float", "dec": "float"}
        },
        {
            "name": "search_variables",
            "description": "搜索变星",
            "parameters": {"ra": "float", "dec": "float", "radius": "float"}
        }
    ]
    tool_examples = generator.generate_tool_examples(tools, num_examples=30)
    
    with open(f"{output_dir}/tool_examples.json", 'w', encoding='utf-8') as f:
        json.dump(tool_examples, f, ensure_ascii=False, indent=2)
    print(f"✓ Tool 示例已保存: {output_dir}/tool_examples.json")
    
    # 生成汇总报告
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_model": config.model,
        "datasets": {
            "training_data": len(training_data),
            "rag_documents": len(documents),
            "tool_examples": len(tool_examples)
        },
        "output_directory": output_dir
    }
    
    with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("数据生成完成!")
    print("=" * 70)
    print(f"训练数据: {len(training_data)} 条")
    print(f"RAG 文档: {len(documents)} 篇")
    print(f"Tool 示例: {len(tool_examples)} 个")
    print(f"\n所有文件保存在: {output_dir}/")
    print("\n下一步:")
    print("  1. 使用这些数据训练/微调模型")
    print("  2. 或直接使用 RAG + Tool 架构")


if __name__ == "__main__":
    main()
