#!/usr/bin/env python3
"""
AstroSage-Llama-3.1-8B 一键部署脚本
=====================================
自动完成：下载 → 转换 → 导入 Ollama → 集成工具箱

使用方法:
    python langgraph_demo/deploy_astrosage.py

流程:
    1. 检查环境（空间、依赖）
    2. 从 HuggingFace 下载模型 (~16GB)
    3. 使用 llama.cpp 转换为 GGUF (~5GB)
    4. 导入 Ollama
    5. 测试运行
    6. 生成启动脚本
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

# 配置
MODEL_NAME = "AstroSage-Llama-3.1-8B"
HF_REPO = "Spectroscopic/AstroSage-Llama-3.1-8B"
MODEL_DIR = "models/astrosage-llama-3.1-8b"
OLLAMA_MODEL_NAME = "astrosage-llama-3.1-8b"


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_step(step_num, total_steps, message):
    """打印步骤信息"""
    print(f"\n{Colors.BLUE}[{step_num}/{total_steps}]{Colors.END} {message}")
    print("=" * 70)


def run_cmd(cmd, check=True, capture_output=False):
    """运行命令"""
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=isinstance(cmd, str), 
                                  check=check, capture_output=True, text=True)
            return result
        else:
            subprocess.run(cmd, shell=isinstance(cmd, str), check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ 命令失败: {e}{Colors.END}")
        raise


def check_disk_space():
    """检查磁盘空间"""
    print("检查磁盘空间...")
    
    # 获取当前目录所在磁盘
    stat = os.statvfs('.')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    print(f"  可用空间: {free_gb:.1f} GB")
    
    # 需要：16GB (下载) + 6GB (转换) + 5GB (GGUF) = ~27GB
    required_gb = 30
    
    if free_gb < required_gb:
        print(f"{Colors.RED}✗ 空间不足！需要约 {required_gb} GB，当前只有 {free_gb:.1f} GB{Colors.END}")
        print("  建议清理空间或使用外部存储")
        return False
    
    print(f"{Colors.GREEN}✓ 空间充足{Colors.END}")
    return True


def check_dependencies():
    """检查依赖"""
    print("\n检查依赖...")
    
    deps = {
        "requests": "requests",
        "tqdm": "tqdm",
        "torch": "torch",
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
        "huggingface_hub": "huggingface-hub"
    }
    
    missing = []
    for module, pkg in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (未安装)")
            missing.append(pkg)
    
    if missing:
        print(f"\n{Colors.YELLOW}安装缺失依赖...{Colors.END}")
        run_cmd([sys.executable, "-m", "pip", "install"] + missing)
    
    print(f"{Colors.GREEN}✓ 依赖检查完成{Colors.END}")
    return True


def download_model():
    """下载模型"""
    print_step(1, 5, "下载 AstroSage 模型")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_dir = f"{MODEL_DIR}/hf_model"
    
    # 检查是否已下载
    if os.path.exists(hf_dir) and len(os.listdir(hf_dir)) > 0:
        print(f"{Colors.YELLOW}模型文件已存在于 {hf_dir}{Colors.END}")
        choice = input("是否重新下载? (y/N): ").strip().lower()
        if choice != 'y':
            print("跳过下载")
            return hf_dir
    
    print(f"从 HuggingFace 下载 {HF_REPO}")
    print(f"保存到: {hf_dir}")
    print(f"大小: ~16 GB")
    print(f"预计时间: 30-120 分钟 (取决于网速)\n")
    
    confirm = input("开始下载? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("取消下载")
        return None
    
    # 使用 huggingface-cli 下载
    try:
        from huggingface_hub import snapshot_download
        
        print("开始下载...")
        start_time = time.time()
        
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=hf_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n{Colors.GREEN}✓ 下载完成！耗时: {elapsed/60:.1f} 分钟{Colors.END}")
        
        # 显示下载的文件
        files = os.listdir(hf_dir)
        print(f"  文件数: {len(files)}")
        
        return hf_dir
        
    except Exception as e:
        print(f"{Colors.RED}✗ 下载失败: {e}{Colors.END}")
        print("\n尝试使用 Git 下载...")
        
        # 备选：使用 git
        git_url = f"https://huggingface.co/{HF_REPO}"
        try:
            run_cmd(["git", "clone", "--depth", "1", git_url, hf_dir])
            return hf_dir
        except:
            print(f"{Colors.RED}✗ Git 下载也失败{Colors.END}")
            return None


def setup_llama_cpp():
    """设置 llama.cpp"""
    print_step(2, 5, "设置 llama.cpp")
    
    llama_cpp_dir = "models/llama.cpp"
    
    if os.path.exists(llama_cpp_dir):
        print(f"{Colors.GREEN}✓ llama.cpp 已存在{Colors.END}")
        return llama_cpp_dir
    
    print("克隆 llama.cpp 仓库...")
    run_cmd(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir])
    
    print("\n安装 Python 依赖...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", f"{llama_cpp_dir}/requirements.txt"])
    
    print(f"{Colors.GREEN}✓ llama.cpp 设置完成{Colors.END}")
    return llama_cpp_dir


def convert_to_gguf(hf_dir, llama_cpp_dir):
    """转换为 GGUF"""
    print_step(3, 5, "转换为 GGUF 格式")
    
    gguf_path = f"{MODEL_DIR}/model.gguf"
    
    # 检查是否已转换
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"{Colors.YELLOW}GGUF 文件已存在: {gguf_path} ({size_gb:.2f} GB){Colors.END}")
        choice = input("是否重新转换? (y/N): ").strip().lower()
        if choice != 'y':
            return gguf_path
    
    print("转换参数:")
    print("  格式: Q4_K_M (4-bit 量化，平衡质量和大小)")
    print("  输出: model.gguf (~5 GB)")
    print("  预计时间: 10-30 分钟\n")
    
    confirm = input("开始转换? (Y/n): ").strip().lower()
    if confirm == 'n':
        return None
    
    # 运行转换脚本
    convert_script = f"{llama_cpp_dir}/convert_hf_to_gguf.py"
    
    # 确保脚本存在
    if not os.path.exists(convert_script):
        # 旧版本可能使用不同名称
        convert_script = f"{llama_cpp_dir}/convert.py"
    
    cmd = [
        sys.executable, convert_script,
        hf_dir,
        "--outfile", gguf_path,
        "--outtype", "q4_K_M"
    ]
    
    print("开始转换...")
    start_time = time.time()
    
    try:
        run_cmd(cmd)
        
        elapsed = time.time() - start_time
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        
        print(f"\n{Colors.GREEN}✓ 转换完成！{Colors.END}")
        print(f"  文件: {gguf_path}")
        print(f"  大小: {size_gb:.2f} GB")
        print(f"  耗时: {elapsed/60:.1f} 分钟")
        
        return gguf_path
        
    except Exception as e:
        print(f"{Colors.RED}✗ 转换失败: {e}{Colors.END}")
        return None


def create_modelfile(gguf_path):
    """创建 Ollama Modelfile"""
    print_step(4, 5, "创建 Ollama 配置")
    
    modelfile_content = f'''FROM {gguf_path}

SYSTEM """You are AstroSage, a domain-specialized AI assistant for astronomy, astrophysics, and space science.

Your expertise includes:
- Variable stars: Cepheids, RR Lyrae, Mira variables, Cataclysmic Variables
- Binary systems: Eclipsing binaries, spectroscopic binaries, contact binaries
- Exoplanets: Detection methods, characterization, habitability
- Stellar astrophysics: Evolution, nucleosynthesis, atmospheres
- Galactic astronomy: Structure, dynamics, stellar populations
- Cosmology: Distance ladder, Hubble constant, dark energy
- Observational astronomy: Photometry, spectroscopy, time-domain surveys
- Data analysis: GAIA, SDSS, ZTF, TESS, Kepler, JWST

Response guidelines:
1. Use precise astronomical terminology
2. Provide quantitative estimates with uncertainties when possible
3. Cite specific surveys, catalogs, or recent literature
4. Consider observational constraints: extinction, sky brightness, seeing
5. When coordinates are provided, query available databases and incorporate real-time data
6. Suggest specific observing strategies: telescope aperture, filters, exposure times, cadence
7. For variable stars, estimate periods, amplitudes, and absolute magnitudes when possible

You have access to real-time astronomical databases through the tools provided.
Always use available tools to enrich your analysis with current data from GAIA, extinction maps, and other surveys."""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop <|end_of_text|>
PARAMETER stop <|eot_id|>
'''
    
    modelfile_path = f"{MODEL_DIR}/Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"{Colors.GREEN}✓ Modelfile 创建完成{Colors.END}")
    print(f"  路径: {modelfile_path}")
    
    return modelfile_path


def import_to_ollama(modelfile_path):
    """导入到 Ollama"""
    print_step(5, 5, "导入 Ollama")
    
    # 检查 Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print(f"{Colors.RED}✗ Ollama 未运行{Colors.END}")
            print("  请运行: ollama serve")
            return False
    except:
        print(f"{Colors.RED}✗ 无法连接到 Ollama{Colors.END}")
        return False
    
    print("创建 Ollama 模型...")
    
    # 删除旧模型（如果存在）
    try:
        run_cmd(["ollama", "rm", OLLAMA_MODEL_NAME], check=False)
    except:
        pass
    
    # 创建新模型
    run_cmd(["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path])
    
    print(f"\n{Colors.GREEN}✓ 模型导入成功！{Colors.END}")
    print(f"  模型名: {OLLAMA_MODEL_NAME}")
    
    # 验证
    r = requests.get("http://localhost:11434/api/tags")
    models = [m['name'] for m in r.json().get('models', [])]
    
    if OLLAMA_MODEL_NAME in models:
        print(f"  验证: ✓ 模型已在列表中")
    
    return True


def create_launcher():
    """创建启动脚本"""
    print("\n创建启动脚本...")
    
    # Windows batch
    bat_content = f'''@echo off
echo AstroSage 天文助手
echo ====================
cd /d "{os.path.abspath('.')}"
python langgraph_demo/astro_assistant_astrosage.py --model {OLLAMA_MODEL_NAME}
pause
'''
    with open("run_astrosage.bat", 'w') as f:
        f.write(bat_content)
    
    # Linux/Mac shell
    sh_content = f'''#!/bin/bash
echo "AstroSage 天文助手"
echo "===================="
cd "{os.path.abspath('.')}"
python langgraph_demo/astro_assistant_astrosage.py --model {OLLAMA_MODEL_NAME}
'''
    with open("run_astrosage.sh", 'w') as f:
        f.write(sh_content)
    os.chmod("run_astrosage.sh", 0o755)
    
    # Python launcher
    py_content = f'''#!/usr/bin/env python3
"""AstroSage 天文助手启动器"""
import subprocess
import sys

subprocess.run([
    sys.executable, 
    "langgraph_demo/astro_assistant_astrosage.py",
    "--model", "{OLLAMA_MODEL_NAME}"
])
'''
    with open("run_astrosage.py", 'w') as f:
        f.write(py_content)
    
    print(f"{Colors.GREEN}✓ 启动脚本已创建{Colors.END}")
    print("  run_astrosage.bat  (Windows)")
    print("  run_astrosage.sh   (Linux/Mac)")
    print("  run_astrosage.py   (Python)")


def test_model():
    """测试模型"""
    print("\n测试模型...")
    
    try:
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL_NAME,
                "prompt": "What is a Cepheid variable star? (One sentence)",
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            print(f"{Colors.GREEN}✓ 模型测试成功{Colors.END}")
            print(f"  示例回答: {result[:100]}...")
            return True
        else:
            print(f"{Colors.RED}✗ 测试失败: HTTP {response.status_code}{Colors.END}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}✗ 测试出错: {e}{Colors.END}")
        return False


def main():
    """主函数"""
    print(f"""
{Colors.BLUE}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     AstroSage-Llama-3.1-8B 一键部署工具                          ║
║                                                                  ║
║     天文领域专用大语言模型                                       ║
║     AstroMLab-1 基准: 89.0% (与 GPT-4 相当)                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{Colors.END}
""")
    
    print("本脚本将自动完成以下步骤：")
    print("  1. 从 HuggingFace 下载模型 (~16 GB)")
    print("  2. 使用 llama.cpp 转换为 GGUF (~5 GB)")
    print("  3. 创建 Ollama 配置")
    print("  4. 导入 Ollama")
    print("  5. 集成到你的工具箱")
    print()
    
    # 预检查
    if not check_disk_space():
        return 1
    
    if not check_dependencies():
        return 1
    
    # 确认开始
    confirm = input(f"{Colors.YELLOW}开始部署? (Y/n): {Colors.END}").strip().lower()
    if confirm == 'n':
        print("取消部署")
        return 0
    
    try:
        # 步骤 1: 下载
        hf_dir = download_model()
        if not hf_dir:
            return 1
        
        # 步骤 2: 设置 llama.cpp
        llama_cpp_dir = setup_llama_cpp()
        
        # 步骤 3: 转换
        gguf_path = convert_to_gguf(hf_dir, llama_cpp_dir)
        if not gguf_path:
            return 1
        
        # 步骤 4: 创建 Modelfile
        modelfile_path = create_modelfile(gguf_path)
        
        # 步骤 5: 导入 Ollama
        if not import_to_ollama(modelfile_path):
            return 1
        
        # 创建启动脚本
        create_launcher()
        
        # 测试
        test_model()
        
        # 完成
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}✓ AstroSage 部署完成！{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}")
        print()
        print("使用方法:")
        print(f"  1. 直接运行: python run_astrosage.py")
        print(f"  2. 或: ollama run {OLLAMA_MODEL_NAME}")
        print(f"  3. 集成工具箱: python langgraph_demo/astro_assistant_astrosage.py")
        print()
        print("文件位置:")
        print(f"  原始模型: {hf_dir}/")
        print(f"  GGUF: {gguf_path}")
        print(f"  Modelfile: {modelfile_path}")
        print()
        
        # 立即运行选项
        run_now = input("立即启动天文助手? (Y/n): ").strip().lower()
        if run_now != 'n':
            subprocess.run([sys.executable, "run_astrosage.py"])
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}用户取消{Colors.END}")
        return 130
    except Exception as e:
        print(f"\n{Colors.RED}✗ 部署失败: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
