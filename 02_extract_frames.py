# -*- coding: utf-8 -*-
"""
============================================================
 02_extract_frames.py —— 从视频中抽取关键帧（图片）
============================================================
功能
----
把视频按"等间隔"抽成若干张图片并保存为 JPG，附带两个可选优化：
    1) 模糊帧过滤：用"拉普拉斯方差"衡量清晰度，跳过太模糊的帧；
    2) 相似帧去重：相邻帧画面几乎相同则跳过，避免训练集冗余。

适用场景
--------
声屏障无人机视频帧与帧之间高度相似（无人机缓慢飞行）。
标注前先用本脚本抽帧，得到"覆盖整段、互不重复"的训练图片，
再交给 LabelImg / X-AnyLabeling 标注。

使用方式
--------
1) 安装依赖：  pip install opencv-python numpy
2) 直接运行：  python 02_extract_frames.py
   或指定参数：python 02_extract_frames.py "视频路径" "输出目录" 12
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================

import os
import cv2
import numpy as np

# ============ 需要修改的地方 ============
VIDEO_PATH = r"C:\Users\Faust\Desktop\科研\无人机数据\2_赣州工务段-九江工务段等\赣州工务段京港高速线下行K1989+368-484.mp4"
OUTPUT_DIR = r"C:\codex-workspace\1\video_frames"   # 抽帧结果保存目录
N_FRAMES = 12                 # 想抽取多少张（均匀覆盖整段视频）
ENABLE_BLUR_FILTER = True     # 是否开启模糊帧过滤
BLUR_THRESHOLD = 60.0         # 清晰度阈值：拉普拉斯方差低于此值视为模糊
ENABLE_DEDUP = True           # 是否开启相似帧去重
DUP_THRESHOLD = 5.0           # 去重阈值(MAD,0~255)：平均绝对差低于此值视为重复帧
# ========================================


def laplacian_variance(gray):
    """
    计算灰度图的"拉普拉斯方差"，用于衡量清晰度。

    原理：清晰图片边缘多、灰度跳变剧烈，拉普拉斯算子能放大这种跳变，
    其方差越大说明边缘越锐利、画面越清晰；模糊图片方差很小。
    这是最常用的"免参考清晰度评价"方法。
    """
    # cv2.Laplacian 对图像做二阶微分，返回与输入同尺寸的梯度图；
    # .var() 是 numpy 数组的方差计算。
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path, output_dir, n_frames,
                   enable_blur_filter=True, blur_threshold=60.0,
                   enable_dedup=True, dup_threshold=5.0):
    """
    核心函数：从视频中等间隔抽帧，保存为 JPG，并返回保存的文件列表。
    """
    # 创建输出目录（存在则跳过，不会报错）
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息：{total} 帧，{fps:.2f} fps")

    # 等间隔选出 n_frames 个帧号（首尾都包含）。
    # 例如总 813 帧、抽 12 张，帧号约为 0, 74, 148, ..., 812。
    indices = [int(round(i * (total - 1) / (n_frames - 1))) for i in range(n_frames)]

    saved = []          # 记录成功保存的文件名
    prev_frame = None   # 保存上一张被保留的帧，用于相似度去重

    for idx in indices:
        # 1) 把读取指针定位到第 idx 帧
        #    （注意：set 之后必须再 read() 一次才能真正读到该帧）
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue    # 读不到（极少发生）就跳过

        # 2) 可选：模糊帧过滤
        if enable_blur_filter:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # BGR 转灰度
            var = laplacian_variance(gray)
            if var < blur_threshold:
                print(f"  跳过第 {idx} 帧：太模糊（清晰度 {var:.1f} < {blur_threshold}）")
                continue

        # 3) 可选：与上一张已保留帧做相似度去重
        #    做法：两张图都缩到 64x64 并转灰度，逐像素求"平均绝对差(MAD)"。
        #    MAD 越小表示画面越接近（0 表示完全一样）。
        #    对无人机慢速飞行这种"整体相似但局部在变化"的画面，
        #    像素差比直方图更敏感，能更准确地判断是否真的重复。
        if enable_dedup and prev_frame is not None:
            small1 = cv2.resize(frame, (64, 64))
            small2 = cv2.resize(prev_frame, (64, 64))
            gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
            # absdiff 逐像素相减取绝对值，mean 求平均 → 平均绝对差
            mad = float(cv2.absdiff(gray1, gray2).mean())
            if mad < dup_threshold:   # 差异太小 = 几乎重复
                print(f"  跳过第 {idx} 帧：与上一帧重复（MAD {mad:.2f} < {dup_threshold}）")
                continue

        # 4) 保存当前帧：文件名里同时带"帧号"和"时间点"，便于追溯
        timestamp = idx / fps                 # 帧号 → 时间（秒）
        name = f"frame_{idx:05d}_t{timestamp:06.2f}s.jpg"
        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])  # 92 = JPEG 质量
        saved.append((name, idx))

        # 记录当前帧作为下一轮的"上一帧"（copy 避免引用同一内存）
        prev_frame = frame.copy()

    cap.release()
    print(f"完成：共保存 {len(saved)} 张图片 → {output_dir}")
    for name, _ in saved:
        print("  ", name)
    return saved


if __name__ == "__main__":
    # 支持命令行传参：python 02_extract_frames.py 视频 输出目录 张数
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else N_FRAMES

    try:
        extract_frames(video_path, output_dir, n_frames,
                       ENABLE_BLUR_FILTER, BLUR_THRESHOLD,
                       ENABLE_DEDUP, DUP_THRESHOLD)
    except Exception as e:
        print(f"抽帧失败：{e}")