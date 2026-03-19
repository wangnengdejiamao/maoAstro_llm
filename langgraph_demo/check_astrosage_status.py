#!/usr/bin/env python3
"""
AstroSage 部署状态检查
======================
检查模型下载、转换和部署状态
"""

import os
import sys
import json


def get_size(path):
    """获取目录或文件大小 (GB)"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024**3)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total / (1024**3)
    return 0


def check_ollama():
    """检查 Ollama 状态"""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get('models', [])
            return True, models
    except:
        pass
    return False, []


def main():
    print("=" * 70)
    print("AstroSage 部署状态检查")
    print("=" * 70)
    
    MODEL_DIR = "models/astrosage-llama-3.1-8b"
    
    # 检查各阶段状态
    status = {
        "hf_model": False,
        "gguf": False,
        "ollama": False
    }
    
    sizes = {}
    
    print("\n1. 原始模型 (HuggingFace)")
    hf_dir = f"{MODEL_DIR}/hf_model"
    if os.path.exists(hf_dir):
        size = get_size(hf_dir)
        sizes['hf'] = size
        file_count = sum(1 for _, _, files in os.walk(hf_dir) for _ in files)
        print(f"   状态: ✓ 已下载 ({size:.2f} GB, {file_count} 个文件)")
        status["hf_model"] = True
        
        # 列出关键文件
        key_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        for f in key_files:
            path = os.path.join(hf_dir, f)
            if os.path.exists(path):
                fsize = os.path.getsize(path) / (1024**3)
                print(f"      ✓ {f} ({fsize:.2f} GB)")
    else:
        print(f"   状态: ✗ 未下载")
        print(f"   路径: {hf_dir}")
    
    print("\n2. GGUF 转换")
    gguf_path = f"{MODEL_DIR}/model.gguf"
    if os.path.exists(gguf_path):
        size = get_size(gguf_path)
        sizes['gguf'] = size
        print(f"   状态: ✓ 已转换 ({size:.2f} GB)")
        print(f"   路径: {gguf_path}")
        status["gguf"] = True
    else:
        print(f"   状态: ✗ 未转换")
        print(f"   预计输出: {gguf_path}")
    
    print("\n3. Ollama 集成")
    ollama_ok, models = check_ollama()
    if ollama_ok:
        print(f"   状态: ✓ Ollama 运行中")
        model_names = [m['name'] for m in models]
        if 'astrosage-llama-3.1-8b' in model_names:
            print(f"   模型: ✓ astrosage-llama-3.1-8b 已导入")
            status["ollama"] = True
        else:
            print(f"   模型: ✗ astrosage-llama-3.1-8b 未导入")
            print(f"   可用模型: {', '.join(model_names[:5])}")
    else:
        print(f"   状态: ✗ Ollama 未运行")
        print(f"   请运行: ollama serve")
    
    # 总结
    print("\n" + "=" * 70)
    print("部署进度")
    print("=" * 70)
    
    if all(status.values()):
        print("✓ 完整部署完成！可以运行: python run_astrosage.py")
    elif status["hf_model"] and not status["gguf"]:
        print("⚠ 步骤: 下载完成 → 需要转换 GGUF")
        print(f"   运行: python langgraph_demo/deploy_astrosage.py")
    elif status["gguf"] and not status["ollama"]:
        print("⚠ 步骤: 转换完成 → 需要导入 Ollama")
        print(f"   运行: ollama create astrosage-llama-3.1-8b -f {MODEL_DIR}/Modelfile")
    elif not status["hf_model"]:
        print("⚠ 步骤: 需要下载模型")
        print(f"   运行: python langgraph_demo/deploy_astrosage.py")
    
    # 磁盘空间
    print("\n" + "=" * 70)
    print("磁盘使用")
    print("=" * 70)
    total = sum(sizes.values())
    for key, size in sizes.items():
        print(f"  {key}: {size:.2f} GB")
    print(f"  总计: {total:.2f} GB")
    
    # 下一步建议
    print("\n" + "=" * 70)
    print("下一步")
    print("=" * 70)
    
    if all(status.values()):
        print("模型已就绪！启动命令:")
        print("  python run_astrosage.py")
        print("  或: python langgraph_demo/astro_assistant_astrosage.py")
    else:
        print("继续部署:")
        print("  python langgraph_demo/deploy_astrosage.py")


if __name__ == "__main__":
    main()
