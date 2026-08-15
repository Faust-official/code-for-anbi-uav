# -*- coding: utf-8 -*-
"""
============================================================
 02_install_deps.py —— 安装 YOLOv5 依赖（不会重装你的 torch）
============================================================
你已经装好 PyTorch GPU 版，所以本脚本：
    1) 读取 yolov5/requirements.txt
    2) 跳过 torch / torchvision / opencv-python（避免误装/冲突）
    3) 只安装其余依赖
    4) 根据你的 torch 版本，提示应安装的 torchvision 版本

用法（在你自己的 VSCode 终端里）：
    python scripts/02_install_deps.py                 # 正常安装
    python scripts/02_install_deps.py --mirror        # 用清华镜像加速
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================
import os
import re
import sys
import subprocess

YOLOV5_DIR = r"C:\codex-workspace\1\yolov5"       # YOLOv5 仓库位置
SKIP = ["torch", "torchvision", "opencv-python"]  # 这些已装/由你手动处理

# torch 版本 -> 推荐 torchvision 版本（官方配套表）
TORCHVISION_MAP = {
    "2.0": "0.15", "2.1": "0.16", "2.2": "0.17", "2.3": "0.18",
    "2.4": "0.19", "2.5": "0.20", "2.6": "0.21", "2.7": "0.22",
    "2.8": "0.23", "2.9": "0.24", "2.12": "0.27", "2.13": "0.28",
}


def pip_install(pkgs, mirror=False):
    """调用当前 Python 的 pip 安装包列表。"""
    cmd = [sys.executable, "-m", "pip", "install"] + pkgs
    if mirror:
        cmd += ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
    print("运行:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    mirror = "--mirror" in sys.argv
    req_file = os.path.join(YOLOV5_DIR, "requirements.txt")
    if not os.path.exists(req_file):
        print(f"未找到 {req_file}，请确认 YOLOv5 仓库位置")
        return

    # 1) 读取 requirements.txt，过滤掉需要跳过的包
    to_install = []
    with open(req_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()   # 去掉行内注释（如 "psutil  # system resources"）
            if not line:
                continue
            line = line.split(";", 1)[0].strip()   # 去掉环境标记（如 "; python_version > '3.8'"）
            pkg = re.split(r"[<>=! ]", line)[0].strip().lower()
            if any(pkg == s or pkg.startswith(s + "[") for s in SKIP):
                print(f"跳过（你已装/单独处理）: {line}")
                continue
            to_install.append(line)

    # 2) 安装其余依赖
    print(f"\n共需安装 {len(to_install)} 个包：")
    for p in to_install:
        print("   ", p)
    pip_install(to_install, mirror)

    # 3) 检查 torch，提示对应 torchvision
    print("\n" + "=" * 60)
    try:
        import torch
        tv = torch.__version__
        print(f"检测到 torch 版本: {tv}")
        major_minor = ".".join(tv.split(".")[:2])
        rec = TORCHVISION_MAP.get(major_minor)
        if rec:
            print(f"请确认 torchvision 版本为 {rec}.x（当前: "
                  f"{__import__('torchvision').__version__ if _torchvision_ok() else '未安装'}）")
            print(f"安装命令：pip install torchvision=={rec}.x  （x 取最新小版本）")
        else:
            print("未能匹配配套版本，请查 https://pytorch.org/get-started/previous-versions/")
    except ImportError:
        print("未检测到 torch。请先安装 GPU 版 PyTorch（见说明文档），再运行本脚本。")

    print("\n完成！建议接着运行 scripts/01_check_env.py 做最终检查。")


def _torchvision_ok():
    try:
        import torchvision
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()