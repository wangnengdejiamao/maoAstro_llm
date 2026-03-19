#!/usr/bin/env python3
"""
AstroSage-Llama-3.1-8B 模型设置脚本
=====================================
下载天文专用模型并导入 Ollama

模型信息:
- 名称: AstroSage-Llama-3.1-8B
- 基础: Meta-Llama-3.1-8B
- 特色: 天文领域持续预训练 + 微调
- 性能: AstroMLab-1 基准 89.0%

来源: https://huggingface.co/Spectroscopic/AstroSage-Llama-3.1-8B
"""

import os
import sys
import subprocess
import json


def check_ollama():
    """检查 Ollama 状态"""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return True, models
    except:
        pass
    return False, []


def download_model():
    """
    下载 AstroSage-Llama-3.1-8B 模型
    方案 1: 直接下载 GGUF 格式（如果可用）
    方案 2: 从 HuggingFace 下载并转换
    """
    print("=" * 70)
    print("下载 AstroSage-Llama-3.1-8B 模型")
    print("=" * 70)
    
    model_dir = "models/astrosage-llama-3.1-8b"
    os.makedirs(model_dir, exist_ok=True)
    
    # 检查是否已有模型
    if os.path.exists(f"{model_dir}/model.gguf"):
        print(f"✓ 模型已存在: {model_dir}/model.gguf")
        return model_dir
    
    print("\n方案选择:")
    print("1. 从 HuggingFace 直接下载 (需要 llama.cpp 转换)")
    print("2. 使用量化版 GGUF (如果社区有提供)")
    print("3. 手动下载后导入")
    
    choice = input("\n选择方案 (1/2/3): ").strip() or "1"
    
    if choice == "1":
        return download_from_hf(model_dir)
    elif choice == "2":
        return download_gguf_direct(model_dir)
    else:
        print(f"\n请手动下载模型到: {model_dir}/")
        print("下载地址: https://huggingface.co/Spectroscopic/AstroSage-Llama-3.1-8B")
        return None


def download_from_hf(model_dir: str) -> str:
    """从 HuggingFace 下载并转换"""
    print("\n正在从 HuggingFace 下载...")
    
    # 检查 huggingface-cli
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
    except:
        print("✗ 请先安装 huggingface-hub: pip install huggingface-hub")
        return None
    
    repo_id = "Spectroscopic/AstroSage-Llama-3.1-8B"
    local_dir = f"{model_dir}/hf_model"
    
    print(f"\n下载模型: {repo_id}")
    print(f"保存到: {local_dir}")
    print("\n这将下载约 16GB 数据，可能需要较长时间...")
    
    confirm = input("继续? (y/n): ").strip().lower()
    if confirm != 'y':
        return None
    
    try:
        # 下载模型
        cmd = [
            "huggingface-cli", "download",
            repo_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False"
        ]
        subprocess.run(cmd, check=True)
        
        print("\n✓ 下载完成")
        print("\n现在需要转换为 GGUF 格式...")
        print("请安装 llama.cpp 并运行转换脚本:")
        print(f"  python convert_hf_to_gguf.py {local_dir} --outfile {model_dir}/model.gguf")
        
        return model_dir
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 下载失败: {e}")
        return None


def download_gguf_direct(model_dir: str) -> str:
    """直接下载 GGUF（社区版本）"""
    print("\n尝试下载社区 GGUF 版本...")
    
    # 一些可能的社区 GGUF 源
    urls = [
        # 如果有社区转换的版本
        # "https://huggingface.co/TheBloke/AstroSage-Llama-3.1-8B-GGUF/resolve/main/astrosage-llama-3.1-8b.Q4_K_M.gguf",
    ]
    
    if not urls:
        print("⚠ 暂无社区 GGUF 版本")
        print("建议:")
        print("1. 使用方案 1 从 HF 下载原始模型")
        print("2. 或使用替代模型: llama3.1:8b + RAG 增强")
        return None
    
    # 尝试下载
    for url in urls:
        filename = url.split('/')[-1]
        filepath = f"{model_dir}/{filename}"
        
        print(f"\n下载: {filename}")
        try:
            subprocess.run(["wget", "-c", "-O", filepath, url], check=True)
            print(f"✓ 下载完成: {filepath}")
            return model_dir
        except:
            continue
    
    return None


def create_modelfile(model_dir: str):
    """创建 Ollama Modelfile"""
    modelfile = f"""FROM {model_dir}/model.gguf

SYSTEM """你是一位顶尖天文学家，专门研究变星、双星、系外行星和宇宙学。
你接受过最新天文文献的训练，熟悉 GAIA、SDSS、ZTF、TESS 等巡天项目。

核心能力：
1. 精确的天体物理分析
2. 光变曲线解读
3. 光谱分析
4. 观测策略制定
5. 实时数据查询和解读

回答规范：
- 使用最新天文学术语
- 数值精确并带单位
- 引用数据来源
- 不确定时说明置信度
- 主动调用可用工具获取实时数据"""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER stop <|end_of_text|>
PARAMETER stop <|eot_id|>
"""
    
    modelfile_path = f"{model_dir}/Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f"\n✓ Modelfile 创建: {modelfile_path}")
    return modelfile_path


def import_to_ollama(model_dir: str, modelfile_path: str):
    """导入模型到 Ollama"""
    print("\n" + "=" * 70)
    print("导入模型到 Ollama")
    print("=" * 70)
    
    model_name = "astrosage-llama-3.1-8b"
    
    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        print(f"运行: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"\n✓ 模型导入成功: {model_name}")
        print(f"\n测试命令:")
        print(f"  ollama run {model_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 导入失败: {e}")
        return False


def main():
    """主函数"""
    print("AstroSage-Llama-3.1-8B Ollama 导入工具")
    print("=" * 70)
    
    # 检查 Ollama
    ok, models = check_ollama()
    if not ok:
        print("✗ Ollama 未运行")
        print("  请运行: ollama serve")
        return
    
    print(f"✓ Ollama 运行中")
    print(f"  当前模型: {', '.join(models) if models else '无'}")
    
    # 检查是否已有 AstroSage
    if "astrosage-llama-3.1-8b" in models:
        print("\n✓ AstroSage-Llama-3.1-8B 已安装")
        return
    
    # 下载模型
    model_dir = download_model()
    if not model_dir:
        print("\n✗ 模型下载失败")
        return
    
    # 检查是否有 GGUF 文件
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    if not gguf_files:
        print(f"\n⚠ 未找到 GGUF 文件于 {model_dir}")
        print("请确保模型已转换为 GGUF 格式")
        return
    
    # 创建 Modelfile
    modelfile_path = create_modelfile(model_dir)
    
    # 导入到 Ollama
    import_to_ollama(model_dir, modelfile_path)


if __name__ == "__main__":
    main()
