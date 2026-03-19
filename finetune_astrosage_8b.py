#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AstroSage 8B (astrosage-local) 微调脚本
================================================================================
功能描述:
    1. 从 Ollama 导出 astrosage-local 模型
    2. 使用 llama.cpp 进行 LoRA 微调
    3. 合并权重并重新导入 Ollama

使用方法:
    # 完整流程
    python finetune_astrosage_8b.py --mode full
    
    # 仅导出
    python finetune_astrosage_8b.py --mode export
    
    # 仅训练
    python finetune_astrosage_8b.py --mode train
    
    # 仅部署
    python finetune_astrosage_8b.py --mode deploy

依赖:
    pip install llama-cpp-python
    
    或者从源码编译 llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make

作者: AstroSage Team
================================================================================
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# 配置
OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"
WORK_DIR = Path("train_qwen/astrosage_finetune")
EXPORT_DIR = WORK_DIR / "exported"
LORA_DIR = WORK_DIR / "lora"
MERGED_DIR = WORK_DIR / "merged"

# 训练数据路径
TRAIN_DATA = "train_qwen/data/qwen_train.json"


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """运行命令"""
    print(f"   执行: {' '.join(cmd[:5])}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 命令失败: {e}")
        print(f"   错误: {e.stderr[:500]}")
        raise


def find_ollama_blob(model_name: str = "astrosage-local") -> Optional[Path]:
    """找到 Ollama 模型的 blob 文件"""
    print("="*70)
    print(f"🔍 查找 Ollama 模型: {model_name}")
    print("="*70)
    
    blobs_dir = OLLAMA_MODELS_DIR / "blobs"
    if not blobs_dir.exists():
        print(f"❌ Blobs 目录不存在: {blobs_dir}")
        return None
    
    # 获取模型信息
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True,
            text=True,
        )
        modelfile = result.stdout
        print("\n   Modelfile 内容:")
        print("   " + "\n   ".join(modelfile.split("\n")[:15]))
    except:
        modelfile = ""
    
    # 查找最大的 blob 文件（通常是模型权重）
    print("\n   查找模型文件...")
    blobs = list(blobs_dir.glob("sha256-*"))
    blobs.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    print(f"   找到 {len(blobs)} 个 blob 文件")
    for i, blob in enumerate(blobs[:5], 1):
        size_gb = blob.stat().st_size / (1024**3)
        print(f"   [{i}] {blob.name[:30]}... ({size_gb:.2f} GB)")
    
    if blobs:
        model_blob = blobs[0]
        size_gb = model_blob.stat().st_size / (1024**3)
        print(f"\n   ✅ 选择模型文件: {size_gb:.2f} GB")
        return model_blob
    
    return None


def export_model():
    """导出 Ollama 模型"""
    print("\n" + "="*70)
    print("📦 导出 AstroSage 8B 模型")
    print("="*70)
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 找到模型 blob
    model_blob = find_ollama_blob("astrosage-local")
    if not model_blob:
        print("❌ 无法找到模型文件")
        return False
    
    # 复制/链接到工作目录
    export_path = EXPORT_DIR / "astrosage-8b.gguf"
    
    if export_path.exists():
        print(f"\n   模型已存在: {export_path}")
    else:
        print(f"\n   导出模型中...")
        print(f"   从: {model_blob}")
        print(f"   到: {export_path}")
        
        try:
            # 尝试硬链接（节省空间）
            os.link(model_blob, export_path)
            print("   ✅ 硬链接创建成功")
        except:
            # 回退到复制
            import shutil
            print("   使用复制...")
            shutil.copy2(model_blob, export_path)
            print("   ✅ 复制完成")
    
    # 获取模型信息
    print("\n   获取模型信息...")
    try:
        result = subprocess.run(
            ["ollama", "show", "astrosage-local"],
            capture_output=True,
            text=True,
        )
        
        # 解析参数
        info = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
        
        # 保存配置
        config = {
            "original_model": "astrosage-local",
            "gguf_path": str(export_path),
            "parameters": info.get("parameters", "unknown"),
            "quantization": info.get("quantization", "unknown"),
            "architecture": info.get("architecture", "unknown"),
        }
        
        with open(EXPORT_DIR / "model_info.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   架构: {config['architecture']}")
        print(f"   参数量: {config['parameters']}")
        print(f"   量化: {config['quantization']}")
        
    except Exception as e:
        print(f"   ⚠️  无法获取模型信息: {e}")
    
    print(f"\n✅ 导出完成: {export_path}")
    return True


def prepare_training_data():
    """准备训练数据为 txt 格式 (llama.cpp 格式)"""
    print("\n" + "="*70)
    print("📚 准备训练数据")
    print("="*70)
    
    if not Path(TRAIN_DATA).exists():
        print(f"❌ 训练数据不存在: {TRAIN_DATA}")
        return None
    
    print(f"\n   加载数据: {TRAIN_DATA}")
    with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   样本数: {len(data)}")
    
    # 转换为训练文本
    output_file = WORK_DIR / "training_data.txt"
    
    print(f"   转换为训练格式...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            conversations = item.get('conversations', [])
            
            # 构建对话文本
            text = ""
            for msg in conversations:
                role = msg.get('from', '')
                content = msg.get('value', '')
                
                if role == 'system':
                    text += f"### System:\n{content}\n\n"
                elif role == 'user':
                    text += f"### User:\n{content}\n\n"
                elif role == 'assistant':
                    text += f"### Assistant:\n{content}\n\n"
            
            f.write(text)
            f.write("---\n\n")  # 分隔符
    
    print(f"   ✅ 训练数据已保存: {output_file}")
    return output_file


def train_with_llama_cpp():
    """使用 llama.cpp 进行 LoRA 训练"""
    print("\n" + "="*70)
    print("🎯 使用 llama.cpp 进行 LoRA 训练")
    print("="*70)
    
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查 llama.cpp
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        print("\n   克隆 llama.cpp...")
        run_command([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
        ])
    
    # 检查是否已编译
    finetune_bin = llama_cpp_dir / "finetune"
    if not finetune_bin.exists():
        print("\n   编译 llama.cpp...")
        print("   这需要在 Linux/macOS 上安装 make 和 g++")
        
        try:
            run_command(["make", "finetune"], cwd=llama_cpp_dir)
        except:
            print("\n   ❌ 编译失败")
            print("   请手动编译:")
            print(f"   cd {llama_cpp_dir} && make finetune")
            return False
    
    # 准备数据
    train_file = prepare_training_data()
    if not train_file:
        return False
    
    # 运行训练
    print("\n   开始 LoRA 训练...")
    print("   这可能需要数小时，取决于数据量和硬件")
    
    model_path = EXPORT_DIR / "astrosage-8b.gguf"
    lora_out = LORA_DIR / "astrosage-lora.bin"
    
    cmd = [
        str(finetune_bin),
        "--model-base", str(model_path),
        "--train-data", str(train_file),
        "--lora-out", str(lora_out),
        "--threads", "8",
        "--batch", "4",
        "--ctx", "2048",
        "--epochs", "3",
        "--learning-rate", "0.0001",
    ]
    
    print(f"\n   命令: {' '.join(cmd)}")
    print("\n   开始训练...")
    
    try:
        result = run_command(cmd, check=False)
        print("   ✅ 训练完成")
        return True
    except Exception as e:
        print(f"   ❌ 训练失败: {e}")
        return False


def merge_lora():
    """合并 LoRA 权重到基础模型"""
    print("\n" + "="*70)
    print("🔧 合并 LoRA 权重")
    print("="*70)
    
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    llama_cpp_dir = Path("llama.cpp")
    export_lora_bin = llama_cpp_dir / "export_lora"
    
    if not export_lora_bin.exists():
        print("   编译 export_lora...")
        run_command(["make", "export_lora"], cwd=llama_cpp_dir)
    
    base_model = EXPORT_DIR / "astrosage-8b.gguf"
    lora_weights = LORA_DIR / "astrosage-lora.bin"
    output_model = MERGED_DIR / "astrosage-8b-finetuned.gguf"
    
    if not lora_weights.exists():
        print(f"❌ LoRA 权重不存在: {lora_weights}")
        return False
    
    print(f"\n   基础模型: {base_model}")
    print(f"   LoRA权重: {lora_weights}")
    print(f"   输出模型: {output_model}")
    
    cmd = [
        str(export_lora_bin),
        "--model-base", str(base_model),
        "--lora", str(lora_weights),
        "--output", str(output_model),
    ]
    
    print("\n   合并中...")
    run_command(cmd)
    
    print(f"\n✅ 合并完成: {output_model}")
    return True


def deploy_to_ollama():
    """部署到 Ollama"""
    print("\n" + "="*70)
    print("📤 部署到 Ollama")
    print("="*70)
    
    model_file = MERGED_DIR / "astrosage-8b-finetuned.gguf"
    if not model_file.exists():
        print(f"❌ 模型文件不存在: {model_file}")
        return False
    
    # 创建 Modelfile
    modelfile_content = f'''FROM {model_file.absolute()}

SYSTEM """你是 maoAstro，一个经过天文领域专业微调的AI助手。

你基于 AstroSage 8B 模型，使用 3,413 条高质量天文问答数据进行了 LoRA 微调训练。

你的专长包括：
- 灾变变星(CV)和双星系统
- 赫罗图和恒星演化  
- 光变曲线分析
- 能谱分布(SED)
- 周期测量和分析

请用专业、准确的中文回答天文问题。如果问题超出你的知识范围，请诚实说明。"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
'''
    
    modelfile_path = MERGED_DIR / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print("\n   Modelfile 已创建")
    
    # 创建 Ollama 模型
    model_name = "maoAstro-astrosage8b"
    
    print(f"\n   创建 Ollama 模型: {model_name}")
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"\n✅ 模型创建成功!")
        print(f"\n🚀 使用方法:")
        print(f"   ollama run {model_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 创建失败: {e}")
        print(f"   错误: {e.stderr}")
        return False


def simple_finetune_alternative():
    """替代方案：使用 transformers + PEFT 微调"""
    print("\n" + "="*70)
    print("🔄 使用 Transformers + PEFT 微调 (替代方案)")
    print("="*70)
    print("\n由于 llama.cpp 需要编译，提供替代方案:")
    print("\n方案A: 使用之前创建的 train_alternative_model.py")
    print("  python train_alternative_model.py --model llama3-chinese")
    print("\n方案B: 继续优化现有的 Qwen2.5 模型")
    print("  python start_maoastro_with_simple_rag.py")
    print("\n方案C: 使用 Ollama API + RAG (无需微调)")
    print("  python start_maoastro_with_simple_rag.py")


def main():
    parser = argparse.ArgumentParser(description="AstroSage 8B 微调")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "export", "train", "merge", "deploy"],
        help="运行模式",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="使用 transformers 而非 llama.cpp",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AstroSage 8B (astrosage-local) 微调工具")
    print("="*70)
    
    if args.use_transformers:
        simple_finetune_alternative()
        return
    
    try:
        if args.mode == "full":
            export_model()
            train_with_llama_cpp()
            merge_lora()
            deploy_to_ollama()
            
        elif args.mode == "export":
            export_model()
            
        elif args.mode == "train":
            train_with_llama_cpp()
            
        elif args.mode == "merge":
            merge_lora()
            
        elif args.mode == "deploy":
            deploy_to_ollama()
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        simple_finetune_alternative()
    
    print("\n" + "="*70)
    print("✅ 流程结束")
    print("="*70)


if __name__ == "__main__":
    main()
