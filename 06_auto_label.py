# -*- coding: utf-8 -*-
"""
============================================================
 06_auto_label.py —— 用视觉大模型自动标注数据集（伪标签）
============================================================
功能
----
把 05 脚本生成的训练切片，交给视觉大模型（Qwen-VL）自动检测
6 类声屏障部件/病害，输出 YOLO 格式标签：
    images/train/v001_f000000_x0000y0000.jpg
        → labels/train/v001_f000000_x0000y0000.txt
    （txt 每行： class_id cx cy w h ，坐标已归一化到 0~1）

说明
----
- 复用 04_vision_analyze.py 的 API 调用方式（阿里云百炼 OpenAI 兼容接口，
  只用标准库 urllib），但提示词改为"检测并输出边界框 JSON"。
- 自动标注得到的是"伪标签(pseudo-label)"：速度快、可批量，
  但精度不如人工，小目标（螺栓）和细微病害（轻微锈蚀）可能有漏检/误检，
  训练前建议人工抽检修正，或用它做预训练/半监督。
- 支持断点续跑：已生成标签的图片自动跳过；--force 可强制重标。

使用方式
--------
1) 配置 API Key（同 04）：环境变量 DASHSCOPE_API_KEY 或同目录 .env
2) 标注全部：    python 06_auto_label.py --split all
   只标 train：  python 06_auto_label.py --split train
   先试 20 张：  python 06_auto_label.py --split train --limit 20
   4 线程加速：  python 06_auto_label.py --split all --workers 4
"""
import os
import sys
import re
import json
import time
import base64
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ 默认配置 ============
DATASET_ROOT = r"C:\codex-workspace\1\dataset"     # 05 脚本生成的数据集根目录
MODEL = "qwen-vl-max"                              # 视觉模型
MAX_WORKERS = 3                                    # 并发线程数（改大更快，但注意 API 限流）
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# ==================================

# 6 个类别（顺序与 05 脚本的 data.yaml 一致）
CLASS_NAMES = ["bolt", "normal_column", "normal_mortar_layer",
               "surface_damage", "rusted_column", "deteriorated_mortar_layer"]

# 模型输出中文类别名 → 类别编号 的映射（含常见同义词）
CLASS_ALIASES = {
    "螺栓": 0, "bolt": 0,
    "正常立柱": 1, "立柱": 1, "normal column": 1,
    "正常砂浆层": 2, "砂浆层": 2, "正常砂浆": 2,
    "表面损伤": 3, "损伤": 3, "破损": 3, "surface damage": 3,
    "锈蚀立柱": 4, "锈蚀": 4, "锈柱": 4, "rusted column": 4,
    "砂浆劣化": 5, "劣化": 5, "劣化砂浆": 5, "deteriorated mortar": 5,
}

# 标注提示词：要求模型输出严格 JSON 数组
PROMPT = (
    "你是铁路声屏障巡检标注助手。请检测这张 640x640 图片中所有声屏障部件与病害，"
    "只输出一个严格 JSON 数组（不要输出任何其他文字、不要用代码块包裹），格式："
    '[{"class": "类别中文名", "bbox": [x1, y1, x2, y2]}]，'
    "其中 bbox 是该目标在图片中的像素坐标（左上角 x1,y1，右下角 x2,y2，取值 0~640）。"
    "类别只能是以下之一：螺栓、正常立柱、正常砂浆层、表面损伤、锈蚀立柱、砂浆劣化。"
    "如果图片中没有目标，输出 []。"
)


def load_api_key():
    """读取 API Key：环境变量优先，其次同目录 .env 文件。"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def image_to_base64(path):
    """图片 → base64 字符串。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path, prompt, model, timeout=180):
    """
    调用 Qwen-VL，返回模型生成的文字。
    （与 04_vision_analyze.py 相同，只是超时和 max_tokens 加大）
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY。请配置环境变量或同目录 .env 文件。")

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def parse_boxes(text, img_size=640):
    """
    从模型返回的文字中解析出边界框。
    返回 [(class_id, x1, y1, x2, y2), ...]（像素坐标，已校验过滤）。

    解析策略：
    1) 去掉可能的 markdown 代码块包裹 ```json ... ```
    2) 用正则找到第一个 JSON 数组
    3) 对每一项做类别映射、坐标范围校验、最小面积过滤
    """
    # 去掉代码块包裹
    text = re.sub(r"```(?:json)?", "", text).strip()
    # 提取第一个 [ ... ] 数组
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None   # 解析失败（不是"无目标"，是格式异常）
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    boxes = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cls_name = str(it.get("class", "")).strip()
        cls_id = CLASS_ALIASES.get(cls_name)
        if cls_id is None:
            continue    # 类别不认识，跳过

        bbox = it.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue

        # 坐标范围校验与钳制
        if x2 <= x1 or y2 <= y1:
            continue
        if x2 < 0 or y2 < 0 or x1 > img_size or y1 > img_size:
            continue
        x1, x2 = max(0.0, x1), min(float(img_size), x2)
        y1, y2 = max(0.0, y1), min(float(img_size), y2)

        # ---- 后处理过滤（伪标签噪声很常见，过滤明显不合理的框）----
        bw, bh = x2 - x1, y2 - y1
        area_ratio = bw * bh / (img_size * img_size)
        # 1) 全图框：模型不确定时常把整张图框起来（如"正常砂浆层"），无效，丢弃
        if area_ratio > 0.5:
            continue
        # 2) 过小的框：可能是噪声（默认边长 < 1.5% 图片即 < 9.6px）
        if bw < img_size * 0.015 or bh < img_size * 0.015:
            continue
        # 3) 面积过小：< 0.05% 图片面积
        if area_ratio < 0.0005:
            continue

        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def to_yolo_line(cls_id, x1, y1, x2, y2, img_size=640):
    """像素框 → YOLO 格式一行：class_id cx cy w h（归一化 0~1）"""
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    # 钳制到 (0,1]，避免越界
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    w, h = max(1e-6, min(1.0, w)), max(1e-6, min(1.0, h))
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_one(image_path, label_path, model, force, retries=3):
    """
    标注单张图片：调用模型 → 解析 → 写标签。返回 (status, 目标数)。
    网络/API 偶发失败会自动重试（最多 retries 次，间隔 3 秒）。
    """
    if os.path.exists(label_path) and not force:
        return "skipped", None

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            text = analyze_image(image_path, PROMPT, model)
            boxes = parse_boxes(text)
            if boxes is None:
                return "parse_fail", None    # 格式异常，不写文件，下次可重试
            with open(label_path, "w", encoding="utf-8") as f:
                for cls_id, x1, y1, x2, y2 in boxes:
                    f.write(to_yolo_line(cls_id, x1, y1, x2, y2) + "\n")
            return "ok", len(boxes)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(3)    # 等 3 秒再重试（常见于限流/网络抖动）
    return f"error:{type(last_err).__name__}", None


def main():
    parser = argparse.ArgumentParser(description="用视觉大模型自动标注数据集（YOLO 伪标签）")
    parser.add_argument("--root", default=DATASET_ROOT, help="数据集根目录")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"], help="标注哪个划分")
    parser.add_argument("--limit", type=int, default=0, help="最多标注多少张（0=全部）")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    parser.add_argument("--model", default=MODEL, help="视觉模型名")
    parser.add_argument("--force", action="store_true", help="强制重标已标注的图片")
    args = parser.parse_args()

    # 收集待标注图片
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    tasks = []   # (image_path, label_path)
    for sp in splits:
        img_dir = os.path.join(args.root, "images", sp)
        lbl_dir = os.path.join(args.root, "labels", sp)
        if not os.path.isdir(img_dir):
            print(f"[警告] 目录不存在：{img_dir}")
            continue
        os.makedirs(lbl_dir, exist_ok=True)
        for name in sorted(os.listdir(img_dir)):
            if not name.lower().endswith(".jpg"):
                continue
            tasks.append((os.path.join(img_dir, name),
                          os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")))
    if args.limit > 0:
        tasks = tasks[:args.limit]

    if not tasks:
        print("没有找到待标注图片。")
        return

    print(f"待标注图片：{len(tasks)} 张（模型 {args.model}，线程 {args.workers}）")
    print("提示：自动标注为伪标签，训练前建议人工抽检修正。")

    stats = {"ok": 0, "empty": 0, "skipped": 0, "parse_fail": 0, "errors": 0, "boxes": 0}
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, img, lbl, args.model, args.force): img
                   for img, lbl in tasks}
        for fut in as_completed(futures):
            status, n = fut.result()
            done += 1
            if status == "ok":
                stats["ok"] += 1
                if n == 0:
                    stats["empty"] += 1
                else:
                    stats["boxes"] += n
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "parse_fail":
                stats["parse_fail"] += 1
            else:
                stats["errors"] += 1
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = done / el
                remain = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  进度 {done}/{len(tasks)}  已标注目标数 {stats['boxes']}  "
                      f"耗时 {el:.0f}s  预计剩余 {remain/60:.1f}min")

    el = time.time() - t0
    print("=" * 56)
    print("自动标注完成：")
    print(f"  成功标注      : {stats['ok']} 张（含空标签 {stats['empty']} 张）")
    print(f"  已跳过(续跑)  : {stats['skipped']} 张")
    print(f"  解析失败      : {stats['parse_fail']} 张")
    print(f"  出错          : {stats['errors']} 张")
    print(f"  共生成目标框  : {stats['boxes']} 个")
    print(f"  总耗时        : {el/60:.1f} 分钟")
    print("=" * 56)
    print("下一步建议：")
    print("  1) 抽检 labels 目录里的 txt 与图片是否吻合；")
    print("  2) 用 LabelImg 打开修正明显误标/漏标；")
    print("  3) 确认后即可用 data.yaml 训练 YOLOv5。")


if __name__ == "__main__":
    main()