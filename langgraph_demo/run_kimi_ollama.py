#!/usr/bin/env python3
"""
Kimi + Ollama 集成系统启动脚本
==============================
一键启动完整的天文 AI 助手系统

用法:
    python run_kimi_ollama.py [命令]

命令:
    generate    - 使用 Kimi API 生成数据
    rag         - 启动 RAG + Tool 系统
    chat        - 启动交互式聊天（完整功能）
    all         - 生成数据并启动系统
    distill     - 运行模型蒸馏（可选）

示例:
    python run_kimi_ollama.py generate    # 仅生成数据
    python run_kimi_ollama.py chat        # 直接聊天
    python run_kimi_ollama.py all         # 完整流程
"""

import sys
import os
import argparse


def check_ollama():
    """检查 Ollama 服务"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if 'qwen3:8b' in model_names:
                print("✓ Ollama 服务正常，qwen3:8b 已安装")
                return True
            else:
                print("⚠ Ollama 服务正常，但未找到 qwen3:8b")
                print("  请运行: ollama pull qwen3:8b")
                return False
    except Exception as e:
        print(f"✗ 无法连接到 Ollama: {e}")
        print("  请确保 Ollama 正在运行: ollama serve")
        return False


def generate_data():
    """生成训练数据"""
    print("\n" + "=" * 70)
    print("步骤 1: 使用 Kimi API 生成数据")
    print("=" * 70)
    
    try:
        from kimi_data_generator import main as generate_main
        generate_main()
        print("\n✓ 数据生成完成")
        return True
    except Exception as e:
        print(f"\n✗ 数据生成失败: {e}")
        return False


def start_rag_system():
    """启动 RAG 系统"""
    print("\n" + "=" * 70)
    print("步骤 2: 启动 RAG + Tool 系统")
    print("=" * 70)
    
    if not check_ollama():
        return False
    
    try:
        from rag_tool_system import AstronomyAssistant
        
        assistant = AstronomyAssistant(use_kimi=True)
        
        # 加载知识库
        docs_path = "langgraph_demo/output/kimi_generated/rag_documents.json"
        if os.path.exists(docs_path):
            assistant.load_kimi_documents(docs_path)
        else:
            print(f"⚠ 知识库文件不存在: {docs_path}")
            print("  请先运行: python run_kimi_ollama.py generate")
        
        # 启动交互模式
        assistant.chat()
        return True
        
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("  请安装: pip install openai sentence-transformers faiss-cpu")
        return False
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_distillation():
    """运行模型蒸馏"""
    print("\n" + "=" * 70)
    print("运行模型蒸馏")
    print("=" * 70)
    
    try:
        from model_distillation_ollama import main as distill_main
        distill_main()
        return True
    except Exception as e:
        print(f"✗ 蒸馏失败: {e}")
        return False


def quick_demo():
    """快速演示模式"""
    print("\n" + "=" * 70)
    print("天文 AI 助手 - 快速演示")
    print("=" * 70)
    
    if not check_ollama():
        return
    
    # 简单查询示例
    import requests
    
    queries = [
        "什么是灾变变星？",
        "造父变星的周期-光度关系是什么？",
    ]
    
    print("\n使用 Ollama qwen3:8b 回答示例问题:\n")
    
    for query in queries:
        print(f"问题: {query}")
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen3:8b",
            "prompt": f"你是一位天文学家。{query}",
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            answer = response.json().get('response', '')
            print(f"回答: {answer[:300]}...")
        except Exception as e:
            print(f"错误: {e}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Kimi + Ollama 天文 AI 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_kimi_ollama.py generate    # 生成训练数据
  python run_kimi_ollama.py chat        # 启动交互聊天
  python run_kimi_ollama.py all         # 完整流程
  python run_kimi_ollama.py demo        # 快速演示
  python run_kimi_ollama.py             # 默认启动聊天
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='chat',
        choices=['generate', 'rag', 'chat', 'all', 'distill', 'demo', 'check'],
        help='要执行的命令 (默认: chat)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'check':
        print("检查系统状态...")
        check_ollama()
        
        # 检查依赖
        deps = ['openai', 'sentence_transformers', 'faiss', 'requests']
        print("\n检查依赖:")
        for dep in deps:
            try:
                __import__(dep.replace('_', '-'))
                print(f"  ✓ {dep}")
            except ImportError:
                print(f"  ✗ {dep} (未安装)")
    
    elif args.command == 'generate':
        generate_data()
    
    elif args.command == 'rag':
        start_rag_system()
    
    elif args.command == 'chat':
        start_rag_system()
    
    elif args.command == 'all':
        # 完整流程
        if generate_data():
            start_rag_system()
        else:
            print("\n数据生成失败，是否仍要启动系统？")
            choice = input("继续? (y/n): ").strip().lower()
            if choice == 'y':
                start_rag_system()
    
    elif args.command == 'distill':
        run_distillation()
    
    elif args.command == 'demo':
        quick_demo()


if __name__ == "__main__":
    main()
