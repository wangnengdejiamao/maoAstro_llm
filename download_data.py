#!/usr/bin/env python3
"""
数据下载脚本
============
此脚本用于下载项目运行所需的数据文件。
由于这些文件较大，不适合放入 Git 仓库，需要单独下载。

使用方法:
    python download_data.py [选项]

选项:
    --extinction    下载消光地图数据 (~800MB)
    --catalogs      下载天文星表数据 (~30GB)
    --models        下载模型文件 (~900MB)
    --all           下载所有数据
    --output DIR    指定下载目录 (默认: ./data)

示例:
    python download_data.py --extinction
    python download_data.py --all --output /path/to/data

作者: Assistant
日期: 2026-03-09
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path


# ===== 数据文件配置 =====
DATA_SOURCES = {
    "extinction": {
        "description": "消光地图数据 (CSFD/SFD)",
        "size": "约 800MB",
        "files": [
            {
                "name": "csfd_ebv.fits",
                "url": "https://example.com/data/csfd_ebv.fits",  # 请替换为实际URL
                "description": "CSFD E(B-V) 修正消光地图",
            },
            {
                "name": "sfd_ebv.fits",
                "url": "https://example.com/data/sfd_ebv.fits",  # 请替换为实际URL
                "description": "SFD E(B-V) 原始消光地图",
            },
            {
                "name": "lss_intensity.fits",
                "url": "https://example.com/data/lss_intensity.fits",
                "description": "LSS 强度图",
            },
            {
                "name": "lss_error.fits",
                "url": "https://example.com/data/lss_error.fits",
                "description": "LSS 误差图",
            },
            {
                "name": "mask.fits",
                "url": "https://example.com/data/mask.fits",
                "description": "数据质量掩码",
            },
        ],
    },
    "models": {
        "description": "污染检测模型",
        "size": "约 900MB",
        "files": [
            {
                "name": "contamination_umap_reducer_20251211IIv3b.1.joblib",
                "url": "https://example.com/models/contamination_umap_reducer.joblib",
                "description": "UMAP 降维器",
            },
            {
                "name": "contamination_random_forest_20251211IIv3b.1.joblib",
                "url": "https://example.com/models/contamination_random_forest.joblib",
                "description": "随机森林分类器",
            },
            {
                "name": "contamination_threshold_20251211IIv3b.1.npy",
                "url": "https://example.com/models/contamination_threshold.npy",
                "description": "分类阈值",
            },
        ],
    },
    "catalogs": {
        "description": "天文星表数据 (LAMOST DR10等)",
        "size": "约 30GB",
        "files": [
            {
                "name": "LAMOST_DR10/README.md",
                "url": None,
                "description": "LAMOST DR10 数据获取指南（请访问官网下载）",
            },
        ],
    },
}


def format_size(size_bytes):
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def download_file(url, output_path, desc=""):
    """下载单个文件"""
    print(f"  正在下载: {desc}")
    print(f"  来源: {url}")
    print(f"  目标: {output_path}")
    
    try:
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 下载文件
        urllib.request.urlretrieve(url, output_path)
        
        size = os.path.getsize(output_path)
        print(f"  ✓ 完成 ({format_size(size)})\n")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}\n")
        return False


def download_with_progress(url, output_path, desc=""):
    """带进度条的下载"""
    print(f"  正在下载: {desc}")
    print(f"  来源: {url}")
    print(f"  目标: {output_path}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用 urlopen 获取文件大小
        with urllib.request.urlopen(url) as response:
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
                        print(f"\r  进度: {percent:.1f}% ({format_size(downloaded)} / {format_size(total_size)})", 
                              end='', flush=True)
        
        print(f"\n  ✓ 完成\n")
        return True
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}\n")
        return False


def check_file_exists(filepath):
    """检查文件是否已存在"""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def download_category(category, output_dir, force=False):
    """下载指定类别的数据"""
    if category not in DATA_SOURCES:
        print(f"未知的数据类别: {category}")
        return False
    
    info = DATA_SOURCES[category]
    print(f"\n{'='*60}")
    print(f"下载: {info['description']}")
    print(f"预计大小: {info['size']}")
    print(f"{'='*60}\n")
    
    success_count = 0
    
    for file_info in info['files']:
        name = file_info['name']
        url = file_info['url']
        desc = file_info['description']
        
        output_path = os.path.join(output_dir, name)
        
        # 检查文件是否已存在
        if check_file_exists(output_path) and not force:
            size = os.path.getsize(output_path)
            print(f"  ⏭  已存在: {name} ({format_size(size)})")
            success_count += 1
            continue
        
        if url is None:
            # 手动下载说明
            print(f"  📋 {name}")
            print(f"     {desc}")
            print(f"     请手动下载此文件\n")
        else:
            # 自动下载
            if download_with_progress(url, output_path, desc):
                success_count += 1
    
    print(f"完成: {success_count}/{len(info['files'])} 个文件")
    return success_count == len(info['files'])


def show_manual_download_info():
    """显示手动下载信息"""
    print("""
============================================
手动下载指南
============================================

由于部分数据文件较大或需要注册，请访问以下链接手动下载：

1. 消光地图数据 (CSFD/SFD)
   - 网站: https://github.com/CPPMariner/CSFD
   - 下载: csfd_ebv.fits, sfd_ebv.fits 等

2. LAMOST DR10 星表数据
   - 网站: http://www.lamost.org/dr10/v2.0/
   - 需要注册账号
   - 下载低分辨率光谱星表

3. Gaia 数据
   - 网站: https://www.cosmos.esa.int/web/gaia/dr3
   - 或使用 Gaia Archive: https://gea.esac.esa.int/archive/

4. 模型文件
   - 请联系项目维护者获取
   - 或训练自己的模型（见 docs/model_training.md）

下载后请将文件放置在以下目录：
  - 消光地图: data/
  - 星表数据: lib/
  - 模型文件: 项目根目录 或 models/
""")


def main():
    parser = argparse.ArgumentParser(
        description="下载项目所需的数据文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_data.py --extinction    # 下载消光地图
  python download_data.py --all          # 下载所有数据
  python download_data.py --models       # 下载模型文件
        """
    )
    
    parser.add_argument(
        "--extinction",
        action="store_true",
        help="下载消光地图数据 (~800MB)"
    )
    parser.add_argument(
        "--catalogs",
        action="store_true",
        help="下载天文星表数据 (~30GB)"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="下载模型文件 (~900MB)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有数据"
    )
    parser.add_argument(
        "--output",
        default="data",
        help="指定下载目录 (默认: ./data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（覆盖已有文件）"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="显示手动下载指南"
    )
    
    args = parser.parse_args()
    
    # 显示手动下载指南
    if args.manual:
        show_manual_download_info()
        return
    
    # 如果没有指定任何选项，显示帮助
    if not any([args.extinction, args.catalogs, args.models, args.all]):
        parser.print_help()
        print("\n" + "="*60)
        print("提示: 使用 --manual 查看手动下载指南")
        print("="*60)
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("  天文AI分析平台 - 数据下载工具")
    print("="*60)
    
    # 下载指定类别
    success = True
    
    if args.all or args.extinction:
        success = download_category("extinction", args.output, args.force) and success
    
    if args.all or args.catalogs:
        success = download_category("catalogs", args.output, args.force) and success
    
    if args.all or args.models:
        success = download_category("models", ".", args.force) and success
    
    print("\n" + "="*60)
    if success:
        print("✅ 所有数据下载完成！")
    else:
        print("⚠️  部分数据下载失败，请检查网络连接或手动下载")
        print("使用 --manual 查看手动下载指南")
    print("="*60)


if __name__ == "__main__":
    main()
