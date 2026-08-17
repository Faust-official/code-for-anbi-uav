# -*- coding: utf-8 -*-
"""
============================================================
 06_run_ablations.py —— 一键跑 ANBINet 消融实验（对照论文 Table I）
============================================================
功能
----
顺序训练 4 个配置（相同数据、相同超参，保证公平对比）：
    1. YOLOv5s 基线
    2. +SC3（backbone C3 -> SC3）
    3. +MAM（head 加多注意力）
    4. ANBINet-Light（SC3 + MAM，完整模型）

用法（VSCode 终端，在 yolov5 目录执行）：
    python ..\video_analysis\env_setup\06_run_ablations.py --epochs 300 --device 0
    先小规模试跑：--epochs 2 --batch 4

参数说明：
    --epochs    训练轮数（正式实验用 300，先试跑用 2~5）
    --batch     批大小（论文 16；显存不够就 8 或 4）
    --img       输入尺寸（论文 640）
    --device    显卡编号，如 0；CPU 用 cpu
    --data      数据集配置（默认用我们生成好的 data.yaml）
    --weights   预训练权重，默认空=从零训练；想用迁移学习传 yolov5s.pt
    --project   结果保存根目录（默认 runs/ablation）
"""
import os
import sys
import argparse
import subprocess

YOLOV5 = r"C:\codex-workspace\1\yolov5"
DATA = r"C:\codex-workspace\1\dataset\data.yaml"

# 消融配置：名称 -> 模型 yaml
ABLATIONS = [
    ("baseline", "models/yolov5s.yaml"),       # 基线
    ("sc3", "models/anbinet_sc3.yaml"),        # +SC3
    ("mam", "models/anbinet_mam.yaml"),        # +MAM
    ("anbinet", "models/anbinet_light.yaml"),  # SC3 + MAM（完整）
]


def run_one(tag, cfg, args, python):
    """用 train.py 训练一个配置。"""
    cmd = [
        python, os.path.join(YOLOV5, "train.py"),
        "--data", args.data,
        "--cfg", cfg,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--imgsz", str(args.img),
        "--device", args.device,
        "--project", os.path.join(args.project, tag),
        "--name", "train",
        "--exist-ok",
    ]
    if args.weights:
        cmd += ["--weights", args.weights]
    print("\n" + "=" * 70)
    print(f"开始训练：{tag}  ({cfg})")
    print("命令:", " ".join(cmd))
    print("=" * 70)
    subprocess.check_call(cmd, cwd=YOLOV5)
    print(f"完成：{tag} -> {os.path.join(args.project, tag)}")


def main():
    parser = argparse.ArgumentParser(description="ANBINet 消融实验（论文 Table I）")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数（正式用 300）")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--img", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", default="0", help="GPU 编号，如 0")
    parser.add_argument("--data", default=DATA, help="data.yaml 路径")
    parser.add_argument("--weights", default="", help="预训练权重（可选）")
    parser.add_argument("--project", default=r"C:\codex-workspace\1\runs\ablation", help="结果保存根目录")
    parser.add_argument("--only", default="", help="只跑某一个，如 --only anbinet")
    args = parser.parse_args()

    # 找到当前 Python（在用户环境里就是装好 torch 的那个）
    python = sys.executable

    targets = ABLATIONS
    if args.only:
        targets = [t for t in ABLATIONS if t[0] == args.only]
        if not targets:
            print(f"未知配置：{args.only}，可选：{[t[0] for t in ABLATIONS]}")
            return

    print(f"共 {len(targets)} 个配置，epochs={args.epochs}, batch={args.batch}, device={args.device}")
    print(f"结果将保存到: {args.project}")

    for tag, cfg in targets:
        run_one(tag, cfg, args, python)

    print("\n" + "=" * 70)
    print("全部消融训练完成！结果目录：")
    for tag, _ in targets:
        print(f"  {os.path.join(args.project, tag)}")
    print("每个目录下 runs/ 里有结果，results.csv 是验证指标汇总。")


if __name__ == "__main__":
    main()