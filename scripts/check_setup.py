#!/usr/bin/env python3
"""
环境配置检查脚本
================
检查所有组件是否正确安装配置

使用方法:
    python scripts/check_setup.py

作者: Assistant
日期: 2026-03-09
"""

import os
import sys
import json
import subprocess
import urllib.request
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 9
    return ok, f"{version.major}.{version.minor}.{version.micro}", "需要 Python 3.9+"


def check_conda_env():
    """检查 Conda 环境"""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            envs = result.stdout
            active = "*" in envs
            has_astromlab = "astromlab" in envs
            return has_astromlab, "astromlab 环境" + (" (已激活)" if active else ""), "建议创建 astromlab 环境"
        return False, "Conda 未配置", ""
    except:
        return False, "Conda 未安装", "建议安装 Miniconda"


def check_package(package_name, min_version=None):
    """检查 Python 包"""
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # 简单版本比较
            ok = version >= min_version
        else:
            ok = True
            
        return ok, version, f"需要 {min_version}" if min_version else ""
    except ImportError:
        return False, "未安装", f"pip install {package_name}"


def check_ollama():
    """检查 Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip() or "已安装"
            return True, version, ""
        return False, "安装异常", ""
    except FileNotFoundError:
        return False, "未安装", "https://ollama.com/download"


def check_ollama_service():
    """检查 Ollama 服务"""
    try:
        response = urllib.request.urlopen(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        data = json.loads(response.read().decode())
        models = len(data.get('models', []))
        return True, f"运行中 ({models} 个模型)", ""
    except:
        return False, "未运行", "运行: ollama serve"


def check_extinction_files():
    """检查消光文件"""
    data_dir = Path("data")
    required_files = [
        "csfd_ebv.fits",
        "sfd_ebv.fits",
        "lss_intensity.fits",
        "lss_error.fits",
        "mask.fits"
    ]
    
    found = []
    missing = []
    
    for f in required_files:
        path = data_dir / f
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            found.append(f"{f} ({size_mb:.0f}MB)")
        else:
            missing.append(f)
    
    if len(found) == len(required_files):
        return True, f"全部 {len(found)} 个文件", ""
    elif found:
        return False, f"部分 ({len(found)}/{len(required_files)})", f"缺少: {', '.join(missing)}"
    else:
        return False, "未找到", "运行: python scripts/download_extinction.py"


def check_models():
    """检查模型文件"""
    root_dir = Path(".")
    model_files = list(root_dir.glob("contamination_*.joblib"))
    
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / 1024 / 1024
        return True, f"{len(model_files)} 个文件 ({total_size:.0f}MB)", ""
    else:
        return False, "未找到", "需要手动下载或训练"


def print_check(name, ok, value, hint=""):
    """打印检查结果"""
    status = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"  # 绿色/红色
    reset = "\033[0m"
    
    print(f"  {color}{status}{reset} {name:<25} {value}")
    if hint:
        print(f"    {'└─>':<23} {hint}")


def main():
    print("="*70)
    print("  maoAstro LLM - 环境配置检查")
    print("="*70)
    
    all_ok = True
    
    # 1. 系统环境
    print("\n🐍 Python 环境")
    ok, value, hint = check_python_version()
    print_check("Python 版本", ok, value, hint)
    all_ok &= ok
    
    ok, value, hint = check_conda_env()
    print_check("Conda 环境", ok, value, hint)
    # Conda 可选，不影响 all_ok
    
    # 2. 核心包
    print("\n📦 核心依赖包")
    packages = [
        ("astropy", "5.0"),
        ("numpy", "1.21"),
        ("scipy", "1.7"),
        ("matplotlib", "3.5"),
        ("pandas", "1.3"),
        ("healpy", None),
    ]
    
    for pkg, min_ver in packages:
        ok, value, hint = check_package(pkg, min_ver)
        print_check(pkg, ok, value, hint)
        all_ok &= ok
    
    # 3. 天文包
    print("\n🔭 天文分析包")
    astro_packages = [
        ("astroquery", None),
        ("lightkurve", None),
    ]
    
    for pkg, min_ver in astro_packages:
        ok, value, hint = check_package(pkg, min_ver)
        print_check(pkg, ok, value, hint)
        all_ok &= ok
    
    # 4. LLM 环境
    print("\n🤖 LLM 环境")
    ok, value, hint = check_ollama()
    print_check("Ollama 安装", ok, value, hint)
    all_ok &= ok
    
    if ok:
        ok, value, hint = check_ollama_service()
        print_check("Ollama 服务", ok, value, hint)
        all_ok &= ok
        
        if ok:
            print("    可用模型:")
            try:
                response = urllib.request.urlopen(
                    "http://localhost:11434/api/tags",
                    timeout=3
                )
                data = json.loads(response.read().decode())
                for model in data.get('models', []):
                    name = model.get('name', model.get('model', 'unknown'))
                    print(f"      • {name}")
            except:
                pass
    
    # 5. 数据文件
    print("\n📁 数据文件")
    ok, value, hint = check_extinction_files()
    print_check("消光地图", ok, value, hint)
    # 数据文件可选，不影响 all_ok
    
    ok, value, hint = check_models()
    print_check("污染检测模型", ok, value, hint)
    # 模型文件可选
    
    # 6. 项目结构
    print("\n📂 项目结构")
    required_dirs = ["src", "data", "output", "scripts"]
    for d in required_dirs:
        ok = Path(d).exists()
        print_check(f"{d}/ 目录", ok, "存在" if ok else "缺失", f"mkdir {d}" if not ok else "")
    
    # 总结
    print("\n" + "="*70)
    if all_ok:
        print("✅ 所有核心组件检查通过！")
        print("\n可以开始使用了:")
        print("  • 运行示例: python examples/basic_analysis.py")
        print("  • 启动 Jupyter: jupyter lab")
        print("  • 测试 LLM: python scripts/test_llm.py")
    else:
        print("⚠️  部分组件未就绪")
        print("\n请根据上方提示安装缺失的组件")
        print("详细指南: SETUP_GUIDE.md")
    print("="*70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
