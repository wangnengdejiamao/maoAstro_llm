#!/usr/bin/env python3
"""
Ollama Qwen 接口
================
整合 Ollama 本地部署的 Qwen 模型
"""

import os
import sys
import json
import requests
from typing import Optional, List, Dict
from PIL import Image
import base64


class OllamaQwenInterface:
    """
    Ollama Qwen 模型接口
    
    用于与本地 Ollama 服务通信，调用 Qwen 模型进行天文数据分析
    """
    
    def __init__(self, model_name: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        """
        初始化 Ollama 接口
        
        Args:
            model_name: Ollama 模型名称，默认 qwen3:8b
            base_url: Ollama 服务地址，默认 http://localhost:11434
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_generate = f"{self.base_url}/api/generate"
        self.api_chat = f"{self.base_url}/api/chat"
        
        # 检查服务是否可用
        self._check_service()
    
    def _check_service(self):
        """检查 Ollama 服务状态"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name') for m in models]
                if self.model_name in model_names:
                    print(f"✓ Ollama 服务正常，模型 {self.model_name} 可用")
                else:
                    print(f"⚠ Ollama 服务正常，但未找到模型 {self.model_name}")
                    print(f"  可用模型: {model_names}")
            else:
                print(f"✗ Ollama 服务返回错误: {response.status_code}")
        except Exception as e:
            print(f"✗ 无法连接到 Ollama 服务: {e}")
            print(f"  请确保 Ollama 正在运行: ollama serve")
    
    def analyze_text(self, prompt: str, system_prompt: str = None, 
                     stream: bool = False, max_retries: int = 2) -> str:
        """
        分析文本，带重试机制
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            stream: 是否流式输出
            max_retries: 最大重试次数
            
        Returns:
            模型回复文本
        """
        import time
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": stream
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                
                response = requests.post(
                    self.api_generate, 
                    json=payload, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    error_msg = f"HTTP {response.status_code}"
                    if attempt < max_retries:
                        print(f"    ⚠ Ollama请求失败 ({error_msg})，{attempt+1}/{max_retries+1}次重试...")
                        time.sleep(1)
                        continue
                    return f"[Ollama服务暂时不可用: {error_msg}]"
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries:
                    print(f"    ⚠ Ollama连接失败，{attempt+1}/{max_retries+1}次重试...")
                    time.sleep(1)
                    continue
                return "[Ollama服务未运行，请执行: ollama serve]"
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"    ⚠ Ollama请求异常，{attempt+1}/{max_retries+1}次重试...")
                    time.sleep(1)
                    continue
                return f"[Ollama请求失败: {str(e)}]"
        
        return "[Ollama服务暂时不可用]"
    
    def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=3
            )
            return response.status_code == 200
        except:
            return False
    
    def analyze_image(self, image_path: str, prompt: str, system_prompt: str = None) -> str:
        """
        分析图像
        
        Args:
            image_path: 图像文件路径
            prompt: 用户提示词
            system_prompt: 系统提示词
            
        Returns:
            模型回复文本
        """
        try:
            # 读取并编码图像
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(self.api_chat, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                return f"错误: HTTP {response.status_code}"
                
        except Exception as e:
            return f"请求失败: {str(e)}"
    
    def analyze_sed_plot(self, sed_plot_path: str) -> str:
        """
        专门分析 SED 图
        
        Args:
            sed_plot_path: SED 图像路径
            
        Returns:
            分析结果
        """
        prompt = """请分析这张光谱能量分布(SED)图：

1. 描述图中显示的波段覆盖范围
2. 分析流量随波长的变化趋势
3. 判断这可能是什么类型的天体（恒星、星系、激变变星等）
4. 指出任何异常或特殊特征
5. 给出后续观测建议

请用专业的天文术语回答。"""

        return self.analyze_image(sed_plot_path, prompt)
    
    def analyze_hr_diagram(self, hr_plot_path: str) -> str:
        """
        专门分析赫罗图
        
        Args:
            hr_plot_path: 赫罗图路径
            
        Returns:
            分析结果
        """
        prompt = """请分析这张赫罗图(Hertzsprung-Russell Diagram)：

1. 描述目标星在赫罗图上的位置
2. 与背景星场比较，目标星有何特殊之处
3. 根据位置判断目标星的演化阶段（主序星、巨星、白矮星等）
4. 估计目标星的有效温度和光度
5. 给出天体类型判断

请用专业的天文术语回答。"""

        return self.analyze_image(hr_plot_path, prompt)
    
    def analyze_light_curve(self, lc_plot_path: str) -> str:
        """
        专门分析光变曲线
        
        Args:
            lc_plot_path: 光变曲线图像路径
            
        Returns:
            分析结果
        """
        prompt = """请分析这张光变曲线图：

1. 描述光变曲线的整体形态
2. 判断变星类型（食双星、脉动变星、激变变星等）
3. 估计周期（如果有）
4. 分析振幅和变化特征
5. 给出可能的物理解释

请用专业的天文术语回答。"""

        return self.analyze_image(lc_plot_path, prompt)
    
    def analyze_target_summary(self, data_json: dict) -> str:
        """
        基于汇总数据生成分析
        
        Args:
            data_json: 包含天体数据的字典
            
        Returns:
            分析结果
        """
        # 构建提示词
        prompt = f"""请基于以下天体观测数据给出专业分析：

目标名称: {data_json.get('name', 'Unknown')}
坐标: RA={data_json.get('ra', 'N/A')}, DEC={data_json.get('dec', 'N/A')}

观测数据:
"""
        
        # 添加消光信息
        ext = data_json.get('extinction', {})
        if ext.get('success'):
            prompt += f"- 消光: A_V = {ext.get('A_V', 'N/A')}, E(B-V) = {ext.get('E_B_V', 'N/A')}\n"
        
        # 添加 SIMBAD 信息
        simbad = data_json.get('simbad', {})
        if simbad.get('matched'):
            prompt += f"- SIMBAD 匹配: {simbad.get('main_id', 'N/A')}\n"
            prompt += f"- 天体类型: {simbad.get('otype', 'N/A')}\n"
            prompt += f"- 光谱型: {simbad.get('sp_type', 'N/A')}\n"
        
        # 添加测光信息
        phot = data_json.get('photometry', {})
        if phot.get('success'):
            cats = phot.get('catalogs', {})
            prompt += f"- 测光目录匹配: {sum(1 for v in cats.values() if v > 0)} 个\n"
        
        # 添加周期信息
        if 'period' in data_json:
            prompt += f"- 光变周期: {data_json['period']:.6f} 天 ({data_json['period']*24:.4f} 小时)\n"
        
        prompt += """
请给出：
1. 天体类型判断及依据
2. 物理特性分析（距离、消光、光度等）
3. 科学价值和研究意义
4. 后续观测建议

请用专业的天文术语回答。"""

        system_prompt = "你是一位专业的天体物理学家，擅长分析天体观测数据并给出专业判断。"
        
        return self.analyze_text(prompt, system_prompt)


# ==================== 便捷函数 ====================

def get_ollama_interface(model_name: str = "qwen3:8b") -> OllamaQwenInterface:
    """
    获取 Ollama 接口实例
    
    Args:
        model_name: 模型名称
        
    Returns:
        OllamaQwenInterface 实例
    """
    return OllamaQwenInterface(model_name=model_name)


def quick_analyze(image_path: str, prompt: str = "描述这张天文图像", model_name: str = "qwen3:8b") -> str:
    """
    快速分析图像
    
    Args:
        image_path: 图像路径
        prompt: 提示词
        model_name: 模型名称
        
    Returns:
        分析结果
    """
    interface = OllamaQwenInterface(model_name=model_name)
    return interface.analyze_image(image_path, prompt)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Ollama Qwen 接口测试")
    print("=" * 60)
    
    # 初始化接口
    interface = OllamaQwenInterface(model_name="qwen3:8b")
    
    # 测试文本分析
    print("\n1. 测试文本分析...")
    result = interface.analyze_text(
        "请简述激变变星(Cataclysmic Variable)的主要特征",
        system_prompt="你是一位天体物理学家"
    )
    print(f"回复: {result[:200]}...")
    
    # 测试图像分析（如果有图像）
    test_image = "./output/figures/AM_Her_sed.png"
    if os.path.exists(test_image):
        print(f"\n2. 测试图像分析 ({test_image})...")
        result = interface.analyze_sed_plot(test_image)
        print(f"回复: {result[:200]}...")
    else:
        print(f"\n2. 跳过图像测试（文件不存在: {test_image}）")
    
    print("\n测试完成!")
