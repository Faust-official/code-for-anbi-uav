# -*- coding: utf-8 -*-
"""
============================================================
 07_plot_ablation.py —— 消融实验结果对比可视化
============================================================
功能
----
读取 06_run_ablations.py 跑出的 4 组结果（runs/ablation/{tag}/**/results.csv），
绘制：
    1) 柱状对比图：mAP@0.5 / mAP@0.5:0.95 / Precision / Recall / F1
    2) 训练曲线图：mAP@0.5 随 epoch 的变化（4 组合在一张图）
输出：
    C:/codex-workspace/1/runs/ablation/ablation_summary.png
    C:/codex-workspace/1/runs/ablation/ablation_curves.png

用法（VSCode 终端，任意目录）：
    python video_analysis/env_setup/07_plot_ablation.py
"""
import os
import glob
import argparse

import matplotlib
matplotlib.use("Agg")  # 无窗口模式，适合脚本
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体（Windows 微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ABLATION_ROOT = r"C:\codex-workspace\1\runs\ablation"
TAGS = ["baseline", "sc3", "mam", "anbinet"]


def find_results(tag, root):
    """在 tag 目录下递归找 results.csv，返回路径或 None。"""
    hits = glob.glob(os.path.join(root, tag, "**", "results.csv"), recursive=True)
    return hits[0] if hits else None


def load_metrics(root):
    """读取 4 组结果，返回 {tag: 最后一行的指标 dict}。"""
    data = {}
    for tag in TAGS:
        p = find_results(tag, root)
        if not p:
            print(f"[提示] 未找到 {tag} 的结果（先跑 06_run_ablations.py）")
            continue
        df = pd.read_csv(p)
        row = df.iloc[-1]  # 最后一轮
        p_, r_ = row.get("metrics/precision", np.nan), row.get("metrics/recall", np.nan)
        data[tag] = {
            "mAP50": row.get("metrics/mAP_0.5", np.nan),
            "mAP50-95": row.get("metrics/mAP_0.5:0.95", np.nan),
            "P": p_, "R": r_,
            "F1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else np.nan,
        }
    return data


def plot_bars(data, out):
    """柱状对比图。"""
    if not data:
        return
    metrics = ["mAP50", "mAP50-95", "P", "R", "F1"]
    names = {"mAP50": "mAP@0.5", "mAP50-95": "mAP@0.5:0.95",
             "P": "Precision", "R": "Recall", "F1": "F1"}
    tags = list(data.keys())
    x = np.arange(len(metrics))
    w = 0.8 / len(tags)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#94a3b8", "#fbbf24", "#34d399", "#3d8dff"]
    for i, tag in enumerate(tags):
        vals = [data[tag][m] * 100 for m in metrics]
        ax.bar(x + (i - len(tags) / 2 + 0.5) * w, vals, w, label=tag, color=colors[i % 4])
        for xi, v in zip(x + (i - len(tags) / 2 + 0.5) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in metrics])
    ax.set_ylabel("(%)")
    ax.set_ylim(0, 105)
    ax.set_title("ANBINet 消融实验结果对比（最后一轮验证集）")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("已保存:", out)


def plot_curves(root, out):
    """mAP@0.5 训练曲线。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#94a3b8", "#fbbf24", "#34d399", "#3d8dff"]
    any_plot = False
    for i, tag in enumerate(TAGS):
        p = find_results(tag, root)
        if not p:
            continue
        df = pd.read_csv(p)
        if "metrics/mAP_0.5" not in df.columns:
            continue
        ax.plot(df["epoch"], df["metrics/mAP_0.5"] * 100,
                label=tag, color=colors[i % 4], linewidth=1.8)
        any_plot = True
    if not any_plot:
        print("[提示] 没有可绘制的曲线数据。")
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_title("消融实验 mAP@0.5 训练曲线")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("已保存:", out)


def main():
    parser = argparse.ArgumentParser(description="消融实验结果可视化")
    parser.add_argument("--root", default=ABLATION_ROOT, help="消融结果根目录")
    args = parser.parse_args()

    data = load_metrics(args.root)
    plot_bars(data, os.path.join(args.root, "ablation_summary.png"))
    plot_curves(args.root, os.path.join(args.root, "ablation_curves.png"))

    if not data:
        print("\n还没有任何结果。请先运行：")
        print("  cd C:\\codex-workspace\\1\\yolov5")
        print("  python ..\\video_analysis\\env_setup\\06_run_ablations.py --epochs 300 --device 0")
    else:
        print("\n完成！对比图已保存到:", args.root)


if __name__ == "__main__":
    main()