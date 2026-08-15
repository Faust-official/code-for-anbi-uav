# -*- coding: utf-8 -*-
"""
============================================================
 05_make_training_dataset.py —— 从无人机视频生成可标注的训练图片集
============================================================
功能
----
一键把"无人机视频"变成"可以直接标注的 YOLO 训练数据集"：
    视频 → 按时间间隔抽帧 → 自适应模糊过滤 → 相似帧去重
        → 把大图切成 640x640 小图（默认无重叠，可调 --overlap 增加重叠）
        → 按 7:2:1 划分 train/val/test
        → 生成 data.yaml + manifest.csv

参考论文
--------
ANBI-UAV（IEEE JSTARS 2026）：无人机采集 8192x5460 大图后，
用滑动窗口切成 640x640 小图再训练。本脚本复刻了这一步。

重要设计说明
------------
1) 模糊过滤默认"自适应"（BLUR_MODE="adaptive"）：
   不同视频的清晰度基线差异很大（大面蓝色声屏障偏平滑、镀锌立柱偏锐利），
   用固定阈值会误杀整段视频。自适应模式按"本视频自身的清晰度分布"
   丢弃最模糊的约 1/4 帧，既能去掉运动模糊，又不会误删清晰视频。
2) 切片内容过滤默认关闭（TILE_MIN_STD=0）：
   "纯天空/空白"和"平滑的声屏障板面"灰度标准差都很小，固定阈值无法区分，
   误删会丢掉真正需要标注的板面。需要清理空白切片时再手动开启并调参。

输出结构
--------
dataset/
├── data.yaml          # YOLOv5 训练配置（类名、路径）
├── manifest.csv       # 每张图片的来源追踪（视频/帧号/时间/切片坐标）
├── images/
│   ├── train/xxx.jpg
│   ├── val/xxx.jpg
│   └── test/xxx.jpg
└── labels/            # 标注后 LabelImg 把 txt 存到这里（当前为空）
    ├── train/
    ├── val/
    └── test/

使用方式
--------
1) 安装依赖：  pip install opencv-python numpy
2) 直接运行：  python 05_make_training_dataset.py
   或指定参数：python 05_make_training_dataset.py --input "视频或文件夹" --output "数据集目录" --interval-sec 1.0
3) 标注：用 LabelImg / X-AnyLabeling 打开 images/train 标注，
   标签保存到 labels/train（LabelImg 的"保存目录"设为 labels/train 即可）。
4) 训练：数据准备好后，YOLOv5 训练时直接引用 data.yaml。
"""
import os
import sys
import csv
import glob
import random
import argparse
import cv2
import numpy as np

# ============ 默认配置（可用命令行参数覆盖） ============
DEFAULT_INPUT = r"C:\Users\Faust\Desktop\科研\无人机数据\2_赣州工务段-九江工务段等"  # 视频或视频文件夹
DEFAULT_OUTPUT = r"C:\codex-workspace\1\dataset"          # 数据集输出根目录
INTERVAL_SEC = 1.0        # 每隔多少秒抽一帧（无人机慢速飞行，1 秒足够）
WINDOW = 640              # 切片窗口大小（论文用 640x640）
OVERLAP = 0.0             # 切片重叠比例（论文建数据集用无重叠 640x640；0=无重叠、数量少，>0 更抗目标被切碎）
BLUR_MODE = "adaptive"    # 模糊过滤：adaptive=按视频自适应 | absolute=固定阈值 | off=关闭
BLUR_THRESHOLD = 60.0     # 仅 absolute 模式使用：拉普拉斯方差低于此值视为模糊
DEDUP_THRESHOLD = 5.0     # 抽帧去重阈值(MAD,0~255)：低于此值视为重复帧
TILE_MIN_STD = 0.0        # 切片内容过滤：灰度标准差低于此值视为空白丢弃；0=关闭(默认)
SPLIT_RATIOS = (0.7, 0.2, 0.1)   # train / val / test 划分比例（论文为 7:2:1）
SEED = 42                 # 随机种子：保证每次划分结果一致（可复现）
# ========================================================

# 论文的 6 个类别（顺序即类别编号 0~5，YOLO 标签里用编号）
CLASS_NAMES = [
    "bolt",                       # 0 螺栓
    "normal_column",              # 1 正常立柱
    "normal_mortar_layer",        # 2 正常砂浆层
    "surface_damage",             # 3 表面损伤
    "rusted_column",              # 4 锈蚀立柱
    "deteriorated_mortar_layer",  # 5 砂浆劣化
]


def laplacian_variance(gray):
    """清晰度评价：拉普拉斯方差（越大越清晰）。"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def save_jpeg(path, img):
    """
    保存 JPEG 图片（兼容中文/任意路径）。

    说明：cv2.imwrite 在 Windows 上遇到非 ASCII（中文）路径时经常读取失败，
    所以这里先用 cv2.imencode 把图像编码成字节，再用 Python 内置 open 写文件，
    这样任何路径都能正常读写。
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"JPEG 编码失败：{path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def collect_videos(input_path):
    """
    收集要处理的视频文件。
    输入可以是单个视频文件，也可以是包含视频的文件夹。
    返回视频路径列表（按名称排序，保证顺序稳定）。
    """
    if os.path.isfile(input_path):
        return [input_path]
    # 文件夹：匹配常见视频扩展名
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v", "*.wmv", "*.MP4")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_path, e)))
    # Windows 上 glob 大小写不敏感，*.mp4 与 *.MP4 会匹配到同一文件，
    # 用 set 去重后再排序，避免同一视频被处理两遍。
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"在 {input_path} 中没有找到视频文件")
    return files


def sample_frames(video_path, interval_sec, blur_mode, blur_threshold, dedup_threshold):
    """
    从单个视频中抽帧：每隔 interval_sec 秒取一帧，
    先做"模糊过滤"（adaptive / absolute / off），再做"相似帧去重"。
    返回 (保留帧列表, 视频总帧数)，保留帧每项为 (帧号, 时间秒, 图像BGR)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [警告] 无法打开视频，跳过：{video_path}")
        return [], 0

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        return [], total

    # 每隔 interval_sec 秒的帧号列表（0, interval*fps, 2*interval*fps, ...）
    frame_step = max(1, int(round(interval_sec * fps)))
    indices = list(range(0, total, frame_step))

    # ---- 第一遍：按间隔取样，并计算每帧的清晰度 ----
    candidates = []   # 每项: (帧号, 时间, 图像, 拉普拉斯方差)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var = laplacian_variance(gray)
        candidates.append((idx, idx / fps, frame, var))
    cap.release()

    # ---- 模糊过滤 ----
    if blur_mode == "off":
        passed = candidates
    elif blur_mode == "absolute":
        # 固定阈值模式：清晰度低于 blur_threshold 的帧全部丢弃
        passed = [c for c in candidates if c[3] >= blur_threshold]
    else:
        # 自适应模式（默认）：
        # 取本视频所有取样帧清晰度的"第 25 百分位"作为阈值，
        # 即丢弃本视频里最模糊的约 1/4 帧。
        # 优点：整段视频偏平滑（如大面蓝色声屏障）也不会被误杀，
        #       同时能去掉偶尔出现的运动模糊帧。
        variances = sorted(c[3] for c in candidates)
        if len(variances) < 4:
            passed = candidates   # 取样帧太少，不做过滤
        else:
            p25 = variances[max(0, int(len(variances) * 0.25) - 1)]
            passed = [c for c in candidates if c[3] >= p25]
            print(f"    清晰度范围 {min(variances):.1f}~{max(variances):.1f}，"
                  f"自适应阈值 {p25:.1f}（丢弃最模糊约 25%）")

    # ---- 相似帧去重（与上一张保留帧比较平均绝对差 MAD）----
    # 做法：两张图都缩到 64x64 并转灰度，逐像素求平均绝对差。
    # MAD 越小表示画面越接近（0 表示完全一样）。
    kept = []          # 最终保留的帧
    prev_gray = None   # 上一张保留帧的灰度小图
    for idx, t, frame, _var in passed:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        if prev_gray is not None:
            mad = float(cv2.absdiff(small, prev_gray).mean())
            if mad < dedup_threshold:
                continue    # 和上一帧几乎一样，丢弃
        prev_gray = small
        kept.append((idx, t, frame))

    return kept, total


def slice_image(img, window, overlap, tile_min_std):
    """
    把一张大图切成 window x window 的小图（滑动窗口，带重叠）。
    - 边缘处理：最后一列/行窗口对齐到图像右/下边缘，保证每块都是完整大小
    - 内容过滤（默认关闭）：灰度标准差低于 tile_min_std 视为空白并丢弃
    返回 [(左上角x, 左上角y, 切片BGR), ...]
    """
    h, w = img.shape[:2]
    if h < window or w < window:
        return []    # 图比窗口还小，跳过（不会发生在 4K 视频上）

    stride = max(1, int(window * (1 - overlap)))

    # 生成窗口左上角的 x 坐标序列（首列 0，末列对齐到右边界）
    xs = list(range(0, w - window + 1, stride))
    if xs[-1] < w - window:
        xs.append(w - window)
    # 同理生成 y 坐标序列
    ys = list(range(0, h - window + 1, stride))
    if ys[-1] < h - window:
        ys.append(h - window)

    tiles = []
    for y in ys:
        for x in xs:
            tile = img[y:y + window, x:x + window]
            if tile_min_std > 0:
                # 内容过滤：灰度标准差太小 = 大片纯色（天空/空白），对标注无价值。
                # 注意：平滑的声屏障板面标准差也小，所以默认关闭，确需清理再开启。
                tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                if float(tile_gray.std()) < tile_min_std:
                    continue
            tiles.append((x, y, tile))
    return tiles


def build_dataset(videos, output_dir, interval_sec, blur_mode, window, overlap,
                  blur_threshold, dedup_threshold, tile_min_std,
                  split_ratios, seed):
    """
    主流程：处理所有视频 → 收集所有切片 → 划分 → 保存 → 写配置/清单。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有切片（文件名、来源信息、图像）
    all_items = []   # 每项: (image_name, video_stem, video_id, frame_idx, time_sec, x, y, img)
    video_ids = {}   # 视频路径 -> ASCII 短编号（避免中文文件名问题）
    total_frames = 0

    for video_path in videos:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"处理视频：{video_path}")
        frames, total_frames_in_video = sample_frames(
            video_path, interval_sec, blur_mode, blur_threshold, dedup_threshold)
        print(f"  抽帧保留 {len(frames)} 张（原始 {total_frames_in_video} 帧）")
        total_frames += len(frames)

        # 给当前视频分配 ASCII 短编号（v001、v002...）。
        # 原因：YOLOv5 训练时的图片加载用 cv2.imread，中文文件名会导致读取失败，
        # 所以输出文件名统一用 ASCII，原始视频名保存在 manifest.csv 里方便溯源。
        video_id = f"v{len(video_ids) + 1:03d}"
        video_ids[video_path] = video_id

        for frame_idx, time_sec, frame in frames:
            tiles = slice_image(frame, window, overlap, tile_min_std)
            for x, y, tile in tiles:
                # 文件名 = 视频编号_帧号_坐标.jpg（全部 ASCII）
                name = f"{video_id}_f{frame_idx:06d}_x{x:04d}y{y:04d}.jpg"
                all_items.append((name, video_stem, video_id, frame_idx, time_sec, x, y, tile))
            print(f"    帧 {frame_idx} (t={time_sec:.1f}s) → 切片 {len(tiles)} 张")

    if not all_items:
        raise RuntimeError("没有生成任何切片，请检查输入视频和过滤参数。")

    # ---- 按比例划分 train/val/test（固定随机种子，保证可复现）----
    random.seed(seed)
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    # 其余全部归 test
    parts = {
        "train": all_items[:n_train],
        "val": all_items[n_train:n_train + n_val],
        "test": all_items[n_train + n_val:],
    }

    # ---- 建目录并保存图片 ----
    for split_name, items in parts.items():
        img_dir = os.path.join(output_dir, "images", split_name)
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)   # 标注前为空目录
        for name, *_rest, img in items:
            save_jpeg(os.path.join(img_dir, name), img)

    # ---- 写 manifest.csv（记录每张图的来源与原图坐标，供后续映射回原图）----
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "video", "video_id", "frame_index", "time_sec", "tile_x", "tile_y", "tile_size"])
        for split_name, items in parts.items():
            for name, video_stem, video_id, frame_idx, time_sec, x, y, _img in items:
                writer.writerow([os.path.join("images", split_name, name),
                                 video_stem, video_id, frame_idx, round(time_sec, 2), x, y, window])

    # ---- 写 data.yaml（YOLOv5 训练配置）----
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# 自动生成：ANBI-UAV 复现数据集配置（{len(all_items)} 张切片）\n")
        f.write(f"path: {os.path.abspath(output_dir).replace(os.sep, '/')}  # 数据集根目录\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(CLASS_NAMES)}  # 类别数\n")
        f.write("names: " + repr(CLASS_NAMES).replace("'", '"') + "\n")

    # ---- 打印汇总 ----
    print("=" * 56)
    print("数据集生成完成：")
    print(f"  视频数        : {len(videos)}")
    print(f"  保留帧数      : {total_frames}")
    print(f"  切片总数      : {len(all_items)}")
    for split_name, items in parts.items():
        print(f"  {split_name:<6}: {len(items)} 张")
    print(f"  输出目录      : {output_dir}")
    print(f"  data.yaml     : {yaml_path}")
    print(f"  manifest.csv  : {manifest_path}")
    print("=" * 56)
    print("下一步：用 LabelImg 打开 images/train 标注，")
    print("        标签保存到 labels/train；YOLO 类别编号按 data.yaml 的 names 顺序。")


if __name__ == "__main__":
    # ---- 命令行参数（都可以省略，省略则用顶部默认值）----
    parser = argparse.ArgumentParser(description="从无人机视频生成可标注的 YOLO 训练图片集")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="视频文件或视频文件夹路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="数据集输出目录")
    parser.add_argument("--interval-sec", type=float, default=INTERVAL_SEC, help="抽帧间隔（秒）")
    parser.add_argument("--window", type=int, default=WINDOW, help="切片窗口大小（像素）")
    parser.add_argument("--overlap", type=float, default=OVERLAP, help="切片重叠比例 0~1")
    parser.add_argument("--blur-mode", default=BLUR_MODE, choices=["adaptive", "absolute", "off"],
                        help="模糊过滤方式：adaptive=自适应 / absolute=固定阈值 / off=关闭")
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD, help="absolute 模式的模糊阈值")
    parser.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD, help="去重阈值(MAD)")
    parser.add_argument("--tile-min-std", type=float, default=TILE_MIN_STD, help="切片灰度标准差下限(0=关闭)")
    args = parser.parse_args()

    try:
        videos = collect_videos(args.input)
        print(f"共发现 {len(videos)} 个视频：")
        for v in videos:
            print("  -", v)
        build_dataset(videos, args.output, args.interval_sec, args.blur_mode,
                      args.window, args.overlap, args.blur_threshold,
                      args.dedup_threshold, args.tile_min_std, SPLIT_RATIOS, SEED)
    except Exception as e:
        print(f"运行失败：{e}")
        sys.exit(1)