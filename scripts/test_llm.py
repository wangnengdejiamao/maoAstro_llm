#!/usr/bin/env python3
"""
LLM 功能测试脚本
================
测试 Ollama 服务和模型可用性

使用方法:
    python scripts/test_llm.py [--model MODEL_NAME]

作者: Assistant
日期: 2026-03-09
"""

import sys
import json
import subprocess
import argparse
import urllib.request
import urllib.error


DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"


def check_ollama_service():
    """检查 Ollama 服务是否运行"""
    try:
        response = urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
        )
        data = json.loads(response.read().decode())
        return True, data.get('models', [])
    except urllib.error.URLError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_model_available(model_name):
    """检查指定模型是否已下载"""
    success, models = check_ollama_service()
    if not success:
        return False, "服务未运行"
    
    for model in models:
        if model.get('name') == model_name or model.get('model') == model_name:
            return True, model
    
    return False, f"模型 {model_name} 未找到"


def test_model_response(model_name, prompt="你好，请简要介绍太阳系的八大行星。"):
    """测试模型响应"""
    try:
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200,
            }
        }
        
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        response = urllib.request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode())
        
        return True, result.get('response', '无响应')
        
    except Exception as e:
        return False, str(e)


def list_available_models():
    """列出所有可用模型"""
    success, models = check_ollama_service()
    if not success:
        return []
    
    return [m.get('name', m.get('model', 'unknown')) for m in models]


def start_ollama_service():
    """尝试启动 Ollama 服务"""
    print("  尝试启动 Ollama 服务...")
    try:
        # 后台启动服务
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        import time
        time.sleep(2)  # 等待服务启动
        return True
    except Exception as e:
        return False


def install_model(model_name):
    """安装模型"""
    print(f"\n  正在安装模型 {model_name}...")
    print(f"  运行: ollama pull {model_name}")
    print(f"  请在新终端中手动运行上述命令")
    print(f"  安装完成后按回车继续...")
    input()


def main():
    parser = argparse.ArgumentParser(
        description="测试 Ollama LLM 功能"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"要测试的模型名称 (默认: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用模型"
    )
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="如果模型不存在则自动安装"
    )
    parser.add_argument(
        "--prompt", "-p",
        help="自定义测试提示词"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("  LLM 功能测试")
    print("="*60)
    
    # 1. 检查 Ollama 服务
    print("\n1️⃣  检查 Ollama 服务...")
    success, info = check_ollama_service()
    
    if success:
        print(f"   ✓ Ollama 服务运行正常")
        print(f"   已安装 {len(info)} 个模型")
    else:
        print(f"   ✗ Ollama 服务未运行: {info}")
        print("   尝试启动服务...")
        
        if start_ollama_service():
            # 重新检查
            success, info = check_ollama_service()
            if success:
                print(f"   ✓ 服务已启动")
            else:
                print(f"   ✗ 无法启动服务")
                print("\n请手动启动 Ollama:")
                print("  ollama serve")
                sys.exit(1)
        else:
            print("   ✗ 自动启动失败")
            print("\n请检查 Ollama 是否已安装:")
            print("  安装指南: https://ollama.com/download")
            sys.exit(1)
    
    # 列出模型
    if args.list:
        print("\n📋 已安装模型:")
        models = list_available_models()
        if models:
            for i, model in enumerate(models, 1):
                marker = "👉 " if model == args.model else "   "
                print(f"  {marker}{i}. {model}")
        else:
            print("  (无)")
        print()
        return
    
    # 2. 检查指定模型
    print(f"\n2️⃣  检查模型: {args.model}")
    available, info = check_model_available(args.model)
    
    if available:
        print(f"   ✓ 模型已安装")
        print(f"   详情: {info}")
    else:
        print(f"   ✗ {info}")
        
        if args.install:
            install_model(args.model)
            # 重新检查
            available, info = check_model_available(args.model)
            if not available:
                print("   ✗ 安装失败")
                sys.exit(1)
        else:
            print(f"\n请安装模型:")
            print(f"  ollama pull {args.model}")
            print(f"\n或使用其他模型:")
            print(f"  python scripts/test_llm.py --model llama3.1:8b")
            sys.exit(1)
    
    # 3. 测试模型响应
    print(f"\n3️⃣  测试模型响应...")
    prompt = args.prompt or "你好，请简要介绍太阳系的八大行星。"
    print(f"   提示词: {prompt[:50]}...")
    
    success, response = test_model_response(args.model, prompt)
    
    if success:
        print(f"   ✓ 模型响应正常")
        print(f"\n📝 模型回复:\n{'-'*60}")
        print(response[:500] + "..." if len(response) > 500 else response)
        print('-'*60)
        
        print("\n✅ LLM 功能测试通过！")
        print(f"\n现在可以在代码中使用:")
        print(f"  from src.ollama_qwen_interface import OllamaQwenInterface")
        print(f"  ollama = OllamaQwenInterface(model_name='{args.model}')")
        
    else:
        print(f"   ✗ 模型响应失败: {response}")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
