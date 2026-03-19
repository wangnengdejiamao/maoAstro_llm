#!/usr/bin/env python3
"""
AstroSage 自动部署脚本 - 使用镜像源
=====================================
使用 hf-mirror.com 加速下载
"""

import os
import sys
import subprocess
import time

# 配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

MODEL_DIR = "models/astrosage-llama-3.1-8b"
HF_REPO = "Spectroscopic/AstroSage-Llama-3.1-8B"
LOG_FILE = "deploy_mirror.log"


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")


def main():
    log("=" * 70)
    log("AstroSage 部署 - 使用镜像源 hf-mirror.com")
    log("=" * 70)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_dir = f"{MODEL_DIR}/hf_model"
    
    # 安装必要的包
    log("检查依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "huggingface-hub", "hf-transfer"], check=False)
    
    # 下载
    log(f"从镜像下载 {HF_REPO}")
    log(f"保存到: {hf_dir}")
    
    try:
        from huggingface_hub import snapshot_download
        
        log("开始下载（使用 hf-mirror.com 加速）...")
        start = time.time()
        
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=hf_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # 多线程下载
        )
        
        elapsed = time.time() - start
        log(f"✓ 下载完成！耗时: {elapsed/60:.1f} 分钟")
        
        # 显示文件
        files = os.listdir(hf_dir)
        log(f"文件数: {len(files)}")
        
        # 计算总大小
        total_size = sum(os.path.getsize(os.path.join(hf_dir, f)) 
                        for f in files if os.path.isfile(os.path.join(hf_dir, f)))
        log(f"总大小: {total_size/(1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        log(f"✗ 下载失败: {e}")
        import traceback
        log(traceback.format_exc())
        return False


if __name__ == "__main__":
    open(LOG_FILE, 'w').close()
    success = main()
    sys.exit(0 if success else 1)
