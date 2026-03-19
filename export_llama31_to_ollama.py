#!/usr/bin/env python3
"""
将训练好的 maoAstro-Llama31 导出到 Ollama
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def export_to_ollama(model_path: str, model_name: str = "maoAstro-llama31"):
    """导出到 Ollama"""
    print("="*70)
    print(f"📤 导出到 Ollama: {model_name}")
    print("="*70)
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 创建 Modelfile
    modelfile_content = f'''FROM {model_path.absolute()}

SYSTEM """你是 maoAstro-Llama31，一个经过天文领域专业训练的AI助手。

你基于 Meta Llama-3.1-8B-Instruct 模型，使用 3,413 条高质量天文问答数据进行了LoRA微调训练。

你的专长包括：
- 灾变变星(CV)和双星系统
- 赫罗图和恒星演化
- 光变曲线分析
- 能谱分布(SED)
- 天文观测技术

请用专业、准确的中文回答天文问题。如果问题超出你的知识范围，请诚实说明。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
'''
    
    modelfile_path = model_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\n✅ Modelfile 已创建: {modelfile_path}")
    print("\n📄 Modelfile 内容:")
    print("-" * 70)
    print(modelfile_content)
    print("-" * 70)
    
    # 创建 Ollama 模型
    print(f"\n🔧 创建 Ollama 模型...")
    print(f"   ollama create {model_name} -f {modelfile_path}")
    
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"\n✅ Ollama 模型创建成功!")
        print(f"\n🚀 使用方法:")
        print(f"   ollama run {model_name}")
        print(f"   或")
        print(f"   ollama run {model_name}:latest")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 创建失败: {e}")
        print(f"   错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Ollama 未安装或未在 PATH 中")
        print(f"   请确保 Ollama 已正确安装")
        return False


def test_model(model_name: str):
    """测试模型"""
    print("\n" + "="*70)
    print("🧪 测试模型")
    print("="*70)
    
    test_question = "什么是灾变变星?"
    
    print(f"\n问题: {test_question}")
    print("回答:\n")
    
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, test_question],
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(result.stdout)
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="导出到 Ollama")
    parser.add_argument(
        "--model",
        type=str,
        default="train_qwen/maoAstro-Llama31-8B/final_model",
        help="模型路径",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="maoAstro-llama31",
        help="Ollama 模型名称",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="创建后测试模型",
    )
    
    args = parser.parse_args()
    
    # 导出
    success = export_to_ollama(args.model, args.name)
    
    if success and args.test:
        test_model(args.name)
    
    print("\n" + "="*70)
    print("✅ 完成")
    print("="*70)


if __name__ == "__main__":
    main()
