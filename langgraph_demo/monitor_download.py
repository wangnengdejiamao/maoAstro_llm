#!/usr/bin/env python3
"""
监控 AstroSage 下载进度
"""

import os
import time
import subprocess

MODEL_DIR = "models/astrosage-llama-3.1-8b/hf_model"

def get_dir_size(path):
    """获取目录大小 (GB)"""
    if not os.path.exists(path):
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)

def count_files(path):
    """统计文件数"""
    if not os.path.exists(path):
        return 0
    return sum(1 for _, _, files in os.walk(path) for _ in files)

def main():
    print("=" * 70)
    print("AstroSage 下载监控")
    print("=" * 70)
    print(f"监控目录: {MODEL_DIR}")
    print(f"预计大小: ~16 GB")
    print("=" * 70)
    
    start_time = time.time()
    last_size = 0
    
    while True:
        try:
            size = get_dir_size(MODEL_DIR)
            files = count_files(MODEL_DIR)
            elapsed = time.time() - start_time
            speed = (size - last_size) * 1024 / 60 if elapsed > 0 else 0  # MB/min
            
            print(f"\r时间: {elapsed/60:.1f}min | 大小: {size:.2f} GB | 文件: {files} | 速度: {speed:.1f} MB/min", end="", flush=True)
            
            last_size = size
            
            # 检查是否完成（model.safetensors 文件大于 15GB）
            model_file = os.path.join(MODEL_DIR, "model.safetensors")
            if os.path.exists(model_file):
                model_size = os.path.getsize(model_file) / (1024**3)
                if model_size > 15:
                    print(f"\n\n✓ 下载完成！总大小: {size:.2f} GB")
                    break
            
            time.sleep(60)  # 每分钟更新
            
        except KeyboardInterrupt:
            print("\n\n监控停止")
            break

if __name__ == "__main__":
    main()
