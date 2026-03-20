#!/usr/bin/env python3
"""
项目清理脚本 - 为 GitHub 上传做准备
=====================================
此脚本帮助清理项目中的大文件和多余内容，使项目适合上传到 GitHub。

使用方法:
    python clean_for_github.py [--dry-run]

选项:
    --dry-run   仅显示将要删除的内容，不实际删除

作者: Assistant
日期: 2026-03-09
"""

import os
import shutil
import argparse
from pathlib import Path


# ===== 配置区域 =====
# 这些文件/目录将被删除（相对于项目根目录）
ITEMS_TO_REMOVE = [
    # 大型数据文件（lib/ 目录包含 LAMOST DR10 等大型天文数据）
    "lib",
    
    # 模型文件（用户需要自行下载）
    "models",
    "contamination_umap_reducer_20251211IIv3b.1.joblib",
    "contamination_random_forest_20251211IIv3b.1.joblib",
    "contamination_threshold_20251211IIv3b.1.npy",
    
    # 数据文件（可通过脚本下载）
    "data/csfd_ebv.fits",
    "data/sfd_ebv.fits",
    "data/lss_intensity.fits",
    "data/lss_error.fits",
    "data/mask.fits",
    "data/large_files",
    
    # 下载的光谱数据
    "SDSS_Spectra_Downloads",
    
    # 输出文件
    "output/data",
    "output/figures",
    "output/*.json",
    "output/*.png",
    "output/*.txt",
    "output/WD_OPC",
    "output/opcwd",
    
    # 缓存文件
    "cache",
    
    # 废弃文件
    "Useless",
    
    # 归档文件
    "archive",
    
    # 下载目录
    "downloads",
    
    # IDE 配置（个人偏好）
    ".idea",
    
    # Python 缓存
    "src/__pycache__",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    
    # Jupyter 检查点
    ".ipynb_checkpoints",
    "**/.ipynb_checkpoints",
    
    # 大型 Notebook（VSP 整合 notebook 有 170MB）
    "VSP/VSP_Integrate.ipynb",
]

# 这些文件将保留，但需要确认存在
ESSENTIAL_FILES = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "LICENSE",
    "src",
    "docs",
    "tests",
    "examples",
]


def get_size(path):
    """获取文件或目录的大小"""
    if not os.path.exists(path):
        return 0
    
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total


def format_size(size_bytes):
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def remove_item(item_path, dry_run=False):
    """删除文件或目录"""
    if not os.path.exists(item_path):
        return False, "不存在"
    
    size = get_size(item_path)
    size_str = format_size(size)
    
    if dry_run:
        return True, f"[预览] 将删除 ({size_str})"
    
    try:
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)
        return True, f"已删除 ({size_str})"
    except Exception as e:
        return False, f"删除失败: {e}"


def expand_glob_pattern(pattern, base_dir):
    """展开 glob 模式"""
    import glob
    full_pattern = os.path.join(base_dir, pattern)
    return glob.glob(full_pattern, recursive=True)


def main():
    parser = argparse.ArgumentParser(
        description="清理项目文件，为 GitHub 上传做准备"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="仅显示将要删除的内容，不实际删除"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="无需确认直接删除"
    )
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("  项目清理脚本 - 为 GitHub 上传做准备")
    print("=" * 60)
    print()
    
    if args.dry_run:
        print("⚠️  当前为预览模式 (--dry-run)，不会实际删除文件\n")
    
    # 收集所有要删除的项目
    items_to_delete = []
    total_size = 0
    
    for pattern in ITEMS_TO_REMOVE:
        # 检查是否是 glob 模式
        if '*' in pattern:
            matches = expand_glob_pattern(pattern, base_dir)
            for match in matches:
                if match not in [item[0] for item in items_to_delete]:
                    size = get_size(match)
                    items_to_delete.append((match, size))
                    total_size += size
        else:
            full_path = os.path.join(base_dir, pattern)
            if os.path.exists(full_path):
                if full_path not in [item[0] for item in items_to_delete]:
                    size = get_size(full_path)
                    items_to_delete.append((full_path, size))
                    total_size += size
    
    # 显示将要删除的内容
    print(f"发现 {len(items_to_delete)} 个项目将被清理:\n")
    print(f"{'类型':<8} {'大小':>12} {'路径':<50}")
    print("-" * 80)
    
    for item_path, size in sorted(items_to_delete, key=lambda x: x[1], reverse=True):
        rel_path = os.path.relpath(item_path, base_dir)
        item_type = "目录" if os.path.isdir(item_path) else "文件"
        print(f"{item_type:<8} {format_size(size):>12} {rel_path:<50}")
    
    print("-" * 80)
    print(f"{'总计':<8} {format_size(total_size):>12}")
    print()
    
    # 确认删除
    if not args.dry_run and not args.force:
        confirm = input("确认删除以上文件? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("已取消操作")
            return
    
    # 执行删除
    if not args.dry_run:
        print("\n开始清理...\n")
    
    success_count = 0
    failed_items = []
    
    for item_path, size in items_to_delete:
        rel_path = os.path.relpath(item_path, base_dir)
        success, msg = remove_item(item_path, args.dry_run)
        
        if success:
            print(f"✓ {rel_path:<50} {msg}")
            success_count += 1
        else:
            print(f"✗ {rel_path:<50} {msg}")
            failed_items.append(rel_path)
    
    print()
    print("=" * 60)
    if args.dry_run:
        print(f"预览完成，共 {len(items_to_delete)} 个项目将被清理")
        print(f"预计释放空间: {format_size(total_size)}")
        print()
        print("要实际执行清理，请运行: python clean_for_github.py")
    else:
        print(f"清理完成！成功删除 {success_count}/{len(items_to_delete)} 个项目")
        print(f"释放空间: {format_size(total_size)}")
        if failed_items:
            print(f"\n以下项目删除失败:")
            for item in failed_items:
                print(f"  - {item}")
    print("=" * 60)
    
    # 检查保留的必要文件
    print("\n检查必要文件...")
    missing = []
    for essential in ESSENTIAL_FILES:
        full_path = os.path.join(base_dir, essential)
        if not os.path.exists(full_path):
            missing.append(essential)
        else:
            print(f"  ✓ {essential}")
    
    if missing:
        print("\n⚠️  以下必要文件缺失:")
        for item in missing:
            print(f"  ✗ {item}")
    
    # 显示清理后的项目大小
    print("\n" + "=" * 60)
    print("  清理后项目信息")
    print("=" * 60)
    
    def get_dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total
    
    current_size = get_dir_size(base_dir)
    print(f"当前项目总大小: {format_size(current_size)}")
    print()
    print("✅ 项目已准备好上传到 GitHub！")
    print()
    print("下一步:")
    print("  1. 查看 GITHUB_UPLOAD_GUIDE.md 了解如何上传到 GitHub")
    print("  2. 运行数据下载脚本获取必要的数据文件")
    print("  3. 根据需要下载模型文件")


if __name__ == "__main__":
    main()
