# -*- coding: utf-8 -*-
"""
============================================================
 01_check_env.py —— 检查训练环境是否就绪
============================================================
在你自己的 VSCode 终端里运行：
    python scripts/01_check_env.py
它会检查：Python / PyTorch GPU / torchvision / 常用依赖 / YOLOv5 仓库，
并给出缺什么、怎么补的建议。
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================
import importlib
import platform
import sys
import os

YOLOV5_DIR = r"C:\codex-workspace\1\yolov5"   # YOLOv5 仓库位置（可改）

def check(name):
    """尝试导入某个包，返回 (是否成功, 版本号)。"""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return True, str(ver)
    except Exception:
        return False, ""

print("=" * 60)
print("1) Python 版本")
print("=" * 60)
print(f"   {platform.python_version()}  (建议 3.8 ~ 3.10；3.11/3.12 也可)")
if not (3, 8) <= sys.version_info[:2] <= (3, 12):
    print("   [警告] Python 版本过新/过旧，YOLOv5 可能不兼容")

print("=" * 60)
print("2) PyTorch GPU")
print("=" * 60)
try:
    import torch
    print(f"   torch 版本      : {torch.__version__}")
    print(f"   CUDA 可用       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   显卡            : {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本       : {torch.version.cuda}")
    else:
        print("   [警告] CUDA 不可用！可能原因：")
        print("          1) 装了 CPU 版 torch（用 nvidia-smi 确认驱动存在）")
        print("          2) 显卡驱动未装/过旧（需 NVIDIA 驱动，与 CUDA Toolkit 无关）")
except ImportError:
    print("   [错误] 没有安装 torch！请先安装 GPU 版：")
    print("          https://pytorch.org/get-started/locally/ 选对应命令安装")

print("=" * 60)
print("3) torchvision（必须与 torch 配套）")
print("=" * 60)
ok, ver = check("torchvision")
if ok:
    print(f"   torchvision 版本: {ver}")
    try:
        import torch
        print("   [提示] 配套要求：torch 2.x ↔ torchvision 0.x")
    except Exception:
        pass
else:
    print("   [警告] 未安装 torchvision，请安装与 torch 匹配的版本（见说明文档）")

print("=" * 60)
print("4) 常用依赖")
print("=" * 60)
for pkg in ["cv2", "numpy", "matplotlib", "pandas", "seaborn", "yaml",
            "tqdm", "scipy", "requests", "PIL", "thop", "ultralytics"]:
    ok, ver = check(pkg)
    print(f"   {pkg:<14}: {'OK ' + ver if ok else '缺少！'}")

print("=" * 60)
print("5) YOLOv5 仓库")
print("=" * 60)
if os.path.isdir(YOLOV5_DIR):
    files = os.listdir(YOLOV5_DIR)
    need = ["train.py", "detect.py", "models", "requirements.txt"]
    missing = [n for n in need if n not in files]
    print(f"   {YOLOV5_DIR}")
    print(f"   结构完整: {('是' if not missing else '缺少 ' + str(missing))}")
else:
    print(f"   [警告] 未找到 {YOLOV5_DIR}，请先准备好 YOLOv5 代码")

print("=" * 60)
print("检查完成。有 [警告]/[错误] 的项按说明文档处理；全 OK 即可进入下一步。")