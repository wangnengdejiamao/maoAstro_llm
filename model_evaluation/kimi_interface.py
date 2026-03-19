#!/usr/bin/env python3
"""
Kimi API 接口模块
用于：
1. 生成白矮星领域问答对
2. 论文摘要生成
3. 概念解释生成
4. 反幻觉数据生成
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass


@dataclass
class KimiConfig:
    """Kimi API 配置"""
    api_key: str = "9cccf25-f532-861b-8000-000042a859dc"
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "moonshot-v1-32k"  # 可选: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
    temperature: float = 0.3
    max_tokens: int = 4096
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("KIMI_API_KEY", "")


class KimiInterface:
    """
    Kimi API 接口
    
    功能:
    - 聊天生成
    - 流式输出
    - 批量处理
    - 重试机制
    """
    
    def __init__(self, config: Optional[KimiConfig] = None):
        self.config = config or KimiConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
        # 验证连接
        self._check_connection()
    
    def _check_connection(self):
        """检查 API 连接"""
        try:
            response = self.session.get(f"{self.config.base_url}/models")
            if response.status_code == 200:
                print(f"✓ Kimi API 连接成功")
                models = response.json().get('data', [])
                print(f"  可用模型: {', '.join([m['id'] for m in models[:3]])}")
            else:
                print(f"⚠ Kimi API 检查失败: {response.status_code}")
        except Exception as e:
            print(f"✗ Kimi API 连接错误: {e}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: Optional[float] = None, max_retries: int = 3) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度 (覆盖默认配置)
            max_retries: 最大重试次数
            
        Returns:
            生成的文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                elif response.status_code == 429:
                    #  rate limit
                    wait_time = 2 ** attempt
                    print(f"    触发速率限制，等待 {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    错误 {response.status_code}: {response.text[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    
            except Exception as e:
                print(f"    请求异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return "[生成失败]"
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None) -> Generator[str, None, None]:
        """流式生成"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True
        }
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta'].get('content', '')
                            if delta:
                                yield delta
                        except:
                            pass
        except Exception as e:
            print(f"流式生成错误: {e}")
            yield "[生成失败]"
    
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None,
                      max_workers: int = 3) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词
            max_workers: 并行数
            
        Returns:
            生成结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompt, system_prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"批量生成失败 (idx={idx}): {e}")
                    results[idx] = "[生成失败]"
        
        return results


class WhiteDwarfDataGenerator:
    """
    白矮星领域训练数据生成器
    使用 Kimi API 生成高质量问答对
    """
    
    def __init__(self, kimi: Optional[KimiInterface] = None):
        self.kimi = kimi or KimiInterface()
        
        # 白矮星领域系统提示词
        self.system_prompt = """你是白矮星天体物理学专家，专门研究：
- 单白矮星演化与冷却
- 双白矮星系统与并合
- 磁性白矮星（分类、磁场产生机制）
- 吸积白矮星与激变变星（CVs, nova, dwarf nova）

要求：
1. 使用精确的天文术语
2. 包含定量数据（质量范围、温度、磁场强度等）
3. 引用关键物理过程和机制
4. 区分不同类型的白矮星系统
5. 避免常见误解（如 AM CVn 不含中子星）
"""
    
    def generate_qa_pairs(self, topic: str, n_pairs: int = 5) -> List[Dict]:
        """
        生成特定主题的问答对
        
        Args:
            topic: 主题（如 "双白矮星并合", "磁性白矮星分类"）
            n_pairs: 生成数量
            
        Returns:
            问答对列表
        """
        prompt = f"""请生成 {n_pairs} 个关于「{topic}」的高质量问答对。

要求：
1. 问题要有深度，不是简单的定义查询
2. 答案要包含定量数据和物理解释
3. 涵盖不同难度级别
4. 使用标准天文术语

请按以下格式输出（JSON格式）：
```json
[
  {{
    "question": "问题内容",
    "answer": "详细答案",
    "difficulty": "easy/medium/hard",
    "type": "concept/calculation/analysis",
    "key_points": ["关键点1", "关键点2"]
  }}
]
```"""
        
        response = self.kimi.generate(prompt, self.system_prompt)
        
        # 提取 JSON
        try:
            # 尝试从 Markdown 代码块中提取
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            qa_pairs = json.loads(json_str)
            return qa_pairs
        except Exception as e:
            print(f"解析失败: {e}")
            print(f"响应: {response[:200]}")
            return []
    
    def generate_mcq(self, concept: str, n_questions: int = 3) -> List[Dict]:
        """
        生成选择题
        
        Args:
            concept: 概念（如 "白矮星钱德拉塞卡极限"）
            n_questions: 题目数量
            
        Returns:
            MCQ 列表
        """
        prompt = f"""请生成 {n_questions} 道关于「{concept}」的多选题。

要求：
1. 包含 4 个选项（A、B、C、D）
2. 干扰项要合理，有一定迷惑性
3. 附带详细解释
4. 标注正确答案

格式（JSON）：
```json
[
  {{
    "question": "问题",
    "options": {{"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}},
    "correct": "A",
    "explanation": "解释",
    "difficulty": "medium"
  }}
]
```"""
        
        response = self.kimi.generate(prompt, self.system_prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            print(f"MCQ 解析失败: {e}")
            return []
    
    def generate_anti_hallucination_data(self) -> List[Dict]:
        """生成反幻觉训练数据（白矮星领域常见误解）"""
        
        misconceptions = [
            {
                "topic": "白矮星组成",
                "misconception": "白矮星由氢和氦组成",
                "truth": "白矮星主要由碳和氧组成（大多数），或氧氖镁（大质量），或氦（小质量）"
            },
            {
                "topic": "双白矮星并合",
                "misconception": "双白矮星并合总是产生 I 型超新星",
                "truth": "只有总质量超过钱德拉塞卡极限 (~1.4 M☉) 的并合才可能产生超新星；亚钱德拉塞卡并合产生 R CrB 星或其他奇异天体"
            },
            {
                "topic": "磁性白矮星",
                "misconception": "所有白矮星都有强磁场",
                "truth": "只有约 10-20% 的白矮星表现出可探测磁场，场强从 10^3 到 10^9 G"
            },
            {
                "topic": "白矮星冷却",
                "misconception": "白矮星冷却速率恒定",
                "truth": "冷却速率随温度变化，高温时快（辐射主导），低温时慢（结晶化、氘燃烧）"
            },
            {
                "topic": "吸积白矮星",
                "misconception": "吸积白矮星都会发生新星爆发",
                "truth": "取决于吸积率，稳定的吸积不会导致新星；只有特定吸积率范围内才会积累足够的物质引发热核爆发"
            },
            {
                "topic": "AM CVn 系统",
                "misconception": "AM CVn 是白矮星与中子星系统",
                "truth": "AM CVn 是双白矮星系统，其中一颗已失去氦包层，周期 5-65 分钟"
            }
        ]
        
        results = []
        
        for item in misconceptions:
            prompt = f"""基于以下误解生成训练数据：

误解: {item['misconception']}
真相: {item['truth']}

请生成：
1. 一个引导模型纠正误解的问题
2. 正确的答案（包含详细解释）
3. 说明为什么这是常见误解

格式（JSON）：
```json
{{
  "question": "...",
  "correct_answer": "...",
  "misconception_explanation": "...",
  "topic": "{item['topic']}"
}}
```"""
            
            response = self.kimi.generate(prompt, self.system_prompt)
            
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response
                
                data = json.loads(json_str)
                data['misconception'] = item['misconception']
                data['truth'] = item['truth']
                results.append(data)
            except Exception as e:
                print(f"生成失败: {e}")
        
        return results
    
    def generate_paper_summary(self, title: str, abstract: str) -> Dict:
        """
        生成论文摘要问答
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            包含问答对的字典
        """
        prompt = f"""基于以下论文信息生成训练数据：

标题: {title}
摘要: {abstract}

请生成：
1. 论文核心发现的总结（2-3句话）
2. 3个关于论文内容的问题及答案
3. 论文在天体物理中的意义

格式（JSON）：
```json
{{
  "summary": "...",
  "qa_pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ],
  "significance": "..."
}}
```"""
        
        response = self.kimi.generate(prompt, self.system_prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            print(f"论文摘要解析失败: {e}")
            return {}
    
    def generate_calculation_problems(self, topic: str, n_problems: int = 3) -> List[Dict]:
        """
        生成计算题
        
        Args:
            topic: 主题（如 "白矮星冷却时标", "吸积盘温度"）
            n_problems: 题目数量
            
        Returns:
            计算题列表
        """
        prompt = f"""请生成 {n_problems} 道关于「{topic}」的计算题。

要求：
1. 包含具体数值
2. 提供完整解题步骤
3. 使用正确的公式和单位
4. 难度适中，有实际物理意义

格式（JSON）：
```json
[
  {{
    "problem": "题目描述",
    "given": ["已知条件1", "已知条件2"],
    "find": "要求解的量",
    "solution_steps": ["步骤1", "步骤2", "步骤3"],
    "formula": "使用的公式",
    "answer": "最终答案（含单位）",
    "difficulty": "medium"
  }}
]
```"""
        
        response = self.kimi.generate(prompt, self.system_prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            print(f"计算题解析失败: {e}")
            return []


# ==================== 便捷函数 ====================

def quick_generate(prompt: str, api_key: Optional[str] = None) -> str:
    """快速生成（单行调用）"""
    config = KimiConfig(api_key=api_key) if api_key else None
    kimi = KimiInterface(config)
    return kimi.generate(prompt)


def test_kimi_api():
    """测试 Kimi API"""
    print("=" * 60)
    print("测试 Kimi API")
    print("=" * 60)
    
    kimi = KimiInterface()
    
    # 测试简单生成
    print("\n1. 测试简单生成...")
    response = kimi.generate("什么是白矮星的钱德拉塞卡极限？简要回答。")
    print(f"响应: {response[:200]}...")
    
    # 测试批量生成
    print("\n2. 测试批量生成...")
    prompts = [
        "什么是双白矮星系统？",
        "磁性白矮星的分类有哪些？",
        "吸积白矮星的光变机制是什么？"
    ]
    responses = kimi.batch_generate(prompts, max_workers=2)
    for prompt, response in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {response[:100]}...")
        print()
    
    # 测试数据生成器
    print("\n3. 测试数据生成器...")
    generator = WhiteDwarfDataGenerator(kimi)
    qa_pairs = generator.generate_qa_pairs("双白矮星引力波辐射", n_pairs=2)
    for qa in qa_pairs:
        print(f"Q: {qa.get('question', 'N/A')}")
        print(f"A: {qa.get('answer', 'N/A')[:150]}...")
        print()
    
    print("✓ 测试完成")


if __name__ == "__main__":
    test_kimi_api()
