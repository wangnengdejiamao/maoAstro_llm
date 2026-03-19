#!/usr/bin/env python3
"""
AstroSage 自动部署脚本（非交互式）
==================================
自动完成全部部署流程，无需人工干预

使用方法:
    python langgraph_demo/deploy_astrosage_auto.py

会在后台自动:
    1. 下载模型
    2. 转换 GGUF
    3. 导入 Ollama
    
进度保存在 deploy_progress.log
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# 配置
MODEL_NAME = "AstroSage-Llama-3.1-8B"
HF_REPO = "Spectroscopic/AstroSage-Llama-3.1-8B"
MODEL_DIR = "models/astrosage-llama-3.1-8b"
OLLAMA_MODEL_NAME = "astrosage-llama-3.1-8b"
LOG_FILE = "deploy_progress.log"


def log(msg):
    """记录日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def run_cmd(cmd, check=True):
    """运行命令"""
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), 
                          check=check, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"Error: {result.stderr}")
    return result


def step_1_download():
    """步骤 1: 下载模型"""
    log("=" * 70)
    log("步骤 1/4: 下载 AstroSage 模型")
    log("=" * 70)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_dir = f"{MODEL_DIR}/hf_model"
    
    # 检查是否已下载
    if os.path.exists(hf_dir) and len(os.listdir(hf_dir)) > 0:
        log("模型文件已存在，跳过下载")
        return True
    
    log(f"开始从 HuggingFace 下载 {HF_REPO}")
    log(f"保存到: {hf_dir}")
    log("预计大小: ~16 GB，预计时间: 30-120 分钟")
    
    try:
        from huggingface_hub import snapshot_download
        
        log("开始下载...")
        start = time.time()
        
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=hf_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        elapsed = time.time() - start
        log(f"✓ 下载完成！耗时: {elapsed/60:.1f} 分钟")
        return True
        
    except Exception as e:
        log(f"✗ 下载失败: {e}")
        return False


def step_2_setup_llamacpp():
    """步骤 2: 设置 llama.cpp"""
    log("=" * 70)
    log("步骤 2/4: 设置 llama.cpp")
    log("=" * 70)
    
    llama_cpp_dir = "models/llama.cpp"
    
    if os.path.exists(llama_cpp_dir):
        log("llama.cpp 已存在")
        return llama_cpp_dir
    
    log("克隆 llama.cpp...")
    run_cmd(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir])
    
    log("安装依赖...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", 
             f"{llama_cpp_dir}/requirements.txt"], check=False)
    
    return llama_cpp_dir


def step_3_convert(llama_cpp_dir):
    """步骤 3: 转换 GGUF"""
    log("=" * 70)
    log("步骤 3/4: 转换为 GGUF 格式")
    log("=" * 70)
    
    hf_dir = f"{MODEL_DIR}/hf_model"
    gguf_path = f"{MODEL_DIR}/model.gguf"
    
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        log(f"GGUF 已存在: {gguf_path} ({size_gb:.2f} GB)")
        return gguf_path
    
    log("开始转换 (Q4_K_M 量化)...")
    log("预计时间: 10-30 分钟")
    
    convert_script = f"{llama_cpp_dir}/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        convert_script = f"{llama_cpp_dir}/convert.py"
    
    cmd = [
        sys.executable, convert_script,
        hf_dir,
        "--outfile", gguf_path,
        "--outtype", "q4_K_M"
    ]
    
    try:
        start = time.time()
        run_cmd(cmd)
        
        elapsed = time.time() - start
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        log(f"✓ 转换完成！耗时: {elapsed/60:.1f} 分钟，大小: {size_gb:.2f} GB")
        return gguf_path
        
    except Exception as e:
        log(f"✗ 转换失败: {e}")
        return None


def step_4_import_ollama(gguf_path):
    """步骤 4: 导入 Ollama"""
    log("=" * 70)
    log("步骤 4/4: 导入 Ollama")
    log("=" * 70)
    
    # 创建 Modelfile
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
5. When coordinates are provided, query available databases
6. Suggest specific observing strategies"""

PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER stop <|end_of_text|>
PARAMETER stop <|eot_id|>
'''
    
    modelfile_path = f"{MODEL_DIR}/Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    log("Modelfile 创建完成")
    
    # 导入 Ollama
    log("创建 Ollama 模型...")
    
    # 删除旧模型
    run_cmd(["ollama", "rm", OLLAMA_MODEL_NAME], check=False)
    
    # 创建新模型
    run_cmd(["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path])
    
    log("✓ Ollama 模型创建成功！")
    return True


def create_launchers():
    """创建启动脚本"""
    log("创建启动脚本...")
    
    # Python 启动器
    with open("run_astrosage.py", 'w') as f:
        f.write(f'''#!/usr/bin/env python3
import subprocess
import sys
subprocess.run([
    sys.executable, 
    "langgraph_demo/astro_assistant_astrosage.py",
    "--model", "{OLLAMA_MODEL_NAME}"
])
''')
    
    # Shell 启动器
    with open("run_astrosage.sh", 'w') as f:
        f.write(f'''#!/bin/bash
cd "{os.path.abspath('.')}"
python langgraph_demo/astro_assistant_astrosage.py --model {OLLAMA_MODEL_NAME}
''')
    os.chmod("run_astrosage.sh", 0o755)
    
    log("✓ 启动脚本已创建")


def main():
    """主函数"""
    log("\n" + "=" * 70)
    log("AstroSage 自动部署开始")
    log("=" * 70)
    log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 步骤 1
        if not step_1_download():
            log("✗ 部署失败: 下载步骤")
            return 1
        
        # 步骤 2
        llama_cpp_dir = step_2_setup_llamacpp()
        
        # 步骤 3
        gguf_path = step_3_convert(llama_cpp_dir)
        if not gguf_path:
            log("✗ 部署失败: 转换步骤")
            return 1
        
        # 步骤 4
        if not step_4_import_ollama(gguf_path):
            log("✗ 部署失败: 导入步骤")
            return 1
        
        # 创建启动脚本
        create_launchers()
        
        log("\n" + "=" * 70)
        log("✓ 部署完成！")
        log("=" * 70)
        log(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log("\n使用方法:")
        log("  python run_astrosage.py")
        log("  或: ollama run astrosage-llama-3.1-8b")
        
        return 0
        
    except Exception as e:
        log(f"\n✗ 部署失败: {e}")
        import traceback
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # 清空旧日志
    open(LOG_FILE, 'w').close()
    sys.exit(main())
