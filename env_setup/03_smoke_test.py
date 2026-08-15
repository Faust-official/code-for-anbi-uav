# -*- coding: utf-8 -*-
"""
============================================================
 03_smoke_test.py —— 最小冒烟测试：验证 数据+标签+YOLOv5 能跑通
============================================================
功能
----
从训练集里挑少量图片和标签，复制成一个迷你数据集，
用 YOLOv5 训练 1 个 epoch、batch=4，确认：
    - 图片和标签能被正确读取
    - 模型能构建、loss 能算
    - 能保存权重
跑通 = 环境与数据都没问题，可以进入正式训练。

用法（在你自己的 VSCode 终端，先 cd 到 yolov5 目录）：
    python ../video_analysis/env_setup/03_smoke_test.py
可选参数：
    --n 16         用多少张图（默认 16）
    --device cuda  指定设备（默认自动：有 GPU 用 GPU，否则 CPU）
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================
import os
import sys
import glob
import shutil
import subprocess
import argparse

WORKSPACE = r"C:\codex-workspace\1"
DATASET = os.path.join(WORKSPACE, "dataset")
YOLOV5 = os.path.join(WORKSPACE, "yolov5")
SMOKE = os.path.join(WORKSPACE, "_smoke")
CLASS_NAMES = ["bolt", "normal_column", "normal_mortar_layer",
               "surface_damage", "rusted_column", "deteriorated_mortar_layer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="用多少张图")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = parser.parse_args()

    # 1) 复制少量 图片+标签 到 _smoke
    src_imgs = sorted(glob.glob(os.path.join(DATASET, "images", "train", "*.jpg")))
    if not src_imgs:
        print("数据集里没有图片，请先运行 05 脚本生成数据。")
        return
    picks = src_imgs[:args.n]
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    for d in ["images", "labels"]:
        os.makedirs(os.path.join(SMOKE, d), exist_ok=True)
    for img in picks:
        base = os.path.splitext(os.path.basename(img))[0]
        shutil.copy(img, os.path.join(SMOKE, "images"))
        lbl = os.path.join(DATASET, "labels", "train", base + ".txt")
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(SMOKE, "labels"))
    print(f"迷你数据集：{len(picks)} 张图 -> {SMOKE}")

    # 2) 写迷你 data.yaml
    yaml_path = os.path.join(SMOKE, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {SMOKE.replace(os.sep, '/')}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names: " + repr(CLASS_NAMES).replace("'", '"') + "\n")

    # 3) 调用 yolov5/train.py
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    cmd = [
        sys.executable, os.path.join(YOLOV5, "train.py"),
        "--data", yaml_path,
        "--epochs", "1",
        "--batch", "4",
        "--imgsz", "640",
        "--weights", "",           # 从零开始，不下载预训练权重
        "--device", device,
        "--project", os.path.join(SMOKE, "runs"),
        "--name", "smoke",
        "--exist-ok",
    ]
    # 从零训练必须指定模型结构（本冒烟测试不用预训练权重）
    cmd += ["--cfg", os.path.join("models", "yolov5s.yaml")]
    print("运行:", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=YOLOV5)
        print("\n[OK] 冒烟测试通过：数据读取、模型构建、训练、保存全部正常！")
        print("接下来可以直接跑正式训练（见说明文档）。")
    except subprocess.CalledProcessError as e:
        print(f"\n[失败] 冒烟测试未通过（退出码 {e.returncode}）。")
        print("请把上方报错信息发给 Codex 协助排查。")


if __name__ == "__main__":
    main()