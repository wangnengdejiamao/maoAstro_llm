#!/usr/bin/env python3
"""
消光地图数据下载脚本
====================
下载 CSFD/SFD 消光地图文件

使用方法:
    python scripts/download_extinction.py [--output DIR]

作者: Assistant
日期: 2026-03-09
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path


# 数据文件配置
EXTINCTION_FILES = {
    "csfd_ebv.fits": {
        "description": "CSFD E(B-V) 修正消光地图",
        "size_mb": 200,
        "urls": [
            "https://github.com/CPPMariner/CSFD/releases/download/v1.0/csfd_ebv.fits",
            # 备用镜像
            # "https://example.com/mirror/csfd_ebv.fits",
        ],
    },
    "sfd_ebv.fits": {
        "description": "SFD E(B-V) 原始消光地图",
        "size_mb": 200,
        "urls": [
            "https://github.com/kbarbary/sfddata/raw/master/sfd_ebv.fits",
        ],
    },
    "lss_intensity.fits": {
        "description": "LSS 强度图",
        "size_mb": 200,
        "urls": [
            "https://github.com/CPPMariner/CSFD/releases/download/v1.0/lss_intensity.fits",
        ],
    },
    "lss_error.fits": {
        "description": "LSS 误差图",
        "size_mb": 200,
        "urls": [
            "https://github.com/CPPMariner/CSFD/releases/download/v1.0/lss_error.fits",
        ],
    },
    "mask.fits": {
        "description": "数据质量掩码",
        "size_mb": 100,
        "urls": [
            "https://github.com/CPPMariner/CSFD/releases/download/v1.0/mask.fits",
        ],
    },
}


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download_with_progress(url, output_path, desc=""):
    """带进度条的下载"""
    try:
        print(f"  从 {url[:60]}... 下载")
        
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 下载
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        bar = '█' * int(percent/5) + '░' * (20 - int(percent/5))
                        print(f"\r  [{bar}] {percent:.1f}% ({format_size(downloaded)})", 
                              end='', flush=True)
        
        print()  # 换行
        return True
        
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        return False


def check_file(filepath, expected_size_mb=None):
    """检查文件是否已存在且完整"""
    if not os.path.exists(filepath):
        return False, "不存在"
    
    size = os.path.getsize(filepath)
    if size == 0:
        return False, "文件为空"
    
    if expected_size_mb and size < expected_size_mb * 1024 * 1024 * 0.5:
        return False, f"文件不完整 ({format_size(size)})"
    
    return True, format_size(size)


def download_file(filename, config, output_dir, force=False):
    """下载单个文件"""
    output_path = os.path.join(output_dir, filename)
    
    print(f"\n📦 {filename}")
    print(f"   {config['description']}")
    
    # 检查是否已存在
    exists, status = check_file(output_path, config['size_mb'])
    if exists and not force:
        print(f"   ⏭  已存在: {status}")
        return True
    
    # 尝试所有 URL
    for url in config['urls']:
        if download_with_progress(url, output_path):
            # 验证下载
            exists, status = check_file(output_path, config['size_mb'])
            if exists:
                print(f"   ✓ 完成: {status}")
                return True
        
        # 如果失败，尝试下一个 URL
        print(f"   尝试备用链接...")
    
    print(f"   ✗ 所有下载链接均失败")
    return False


def show_manual_guide():
    """显示手动下载指南"""
    print("""
========================================
手动下载指南
========================================

如果自动下载失败，请手动下载以下文件：

1. CSFD 消光地图（推荐）
   网站: https://github.com/CPPMariner/CSFD
   下载: csfd_ebv.fits, lss_intensity.fits, 
         lss_error.fits, mask.fits

2. SFD 原始地图（备选）
   网站: https://github.com/kbarbary/sfddata
   下载: sfd_ebv.fits

3. 百度网盘镜像（国内用户）
   链接: https://pan.baidu.com/s/xxxxxxxxx
   提取码: xxxx

下载后请将文件放置在: data/ 目录
""")


def main():
    parser = argparse.ArgumentParser(
        description="下载消光地图数据文件"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="输出目录 (默认: data)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新下载（覆盖已有文件）"
    )
    parser.add_argument(
        "--manual", "-m",
        action="store_true",
        help="显示手动下载指南"
    )
    parser.add_argument(
        "--file",
        choices=list(EXTINCTION_FILES.keys()),
        help="只下载指定文件"
    )
    
    args = parser.parse_args()
    
    if args.manual:
        show_manual_guide()
        return
    
    # 确保输出目录是绝对路径
    if not os.path.isabs(args.output):
        args.output = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output)
    
    print("="*60)
    print("  消光地图数据下载工具")
    print("="*60)
    print(f"\n输出目录: {args.output}")
    print(f"预计总大小: ~900 MB")
    print()
    
    # 确定要下载的文件
    if args.file:
        files_to_download = {args.file: EXTINCTION_FILES[args.file]}
    else:
        files_to_download = EXTINCTION_FILES
    
    # 下载文件
    success_count = 0
    failed_files = []
    
    for filename, config in files_to_download.items():
        if download_file(filename, config, args.output, args.force):
            success_count += 1
        else:
            failed_files.append(filename)
    
    # 总结
    print("\n" + "="*60)
    print(f"下载完成: {success_count}/{len(files_to_download)}")
    
    if failed_files:
        print(f"\n失败文件:")
        for f in failed_files:
            print(f"  - {f}")
        print("\n使用 --manual 查看手动下载指南")
    else:
        print("\n✅ 所有消光文件下载完成！")
        print("\n现在可以运行消光查询功能了:")
        print("  python -c \"from src.astro_tools import query_extinction; print(query_extinction(13.13, 53.85))\"")
    
    print("="*60)


if __name__ == "__main__":
    main()
