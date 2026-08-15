# -*- coding: utf-8 -*-
"""
============================================================
 03_make_contact_sheet.py —— 把多张图片拼成一张总览缩略图
============================================================
功能
----
把指定目录下的所有图片按网格（如 4 列 x 3 行）拼接成一张大图，
每张小图下方标注文件名，方便一眼总览整段视频内容。

适用场景
--------
汇报 / 自查：把 12 张关键帧拼成一张图，直接放进 PPT；
也可以在标注前快速检查抽帧质量。

使用方式
--------
1) 安装依赖：  pip install pillow
2) 直接运行：  python 03_make_contact_sheet.py
   或指定参数：python 03_make_contact_sheet.py "图片目录" "输出图片.jpg" 4
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================

import os
import glob
from PIL import Image, ImageDraw, ImageFont   # Pillow：图像处理库

# ============ 需要修改的地方 ============
IMAGE_DIR = r"C:\codex-workspace\1\video_frames"   # 关键帧所在目录（02 脚本的输出）
OUTPUT_FILE = r"C:\codex-workspace\1\video_frames\contact_sheet.jpg"
COLS = 4                  # 每行放几张
THUMB_WIDTH = 460         # 每张缩略图的宽度（像素）
GAP = 10                  # 图片之间的间距（像素）
LABEL_HEIGHT = 36         # 底部文件名标注区的高度（像素）
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"  # 中文字体（微软雅黑）；找不到则回退默认字体
# ========================================


def load_font(size):
    """
    加载中文字体。Pillow 默认字体不支持中文，会显示成方块，
    所以优先用系统的微软雅黑（msyh.ttc）。
    """
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(image_dir, output_file, cols,
                       thumb_width=460, gap=10, label_height=36):
    """
    核心函数：把目录下所有图片拼成网格总览图，返回输出路径。
    """
    # 找出目录下所有 jpg/png 图片，按文件名排序，保证顺序稳定
    files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                   glob.glob(os.path.join(image_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"目录中没有找到图片：{image_dir}")

    # 网格行数 = 图片数 ÷ 每行列数，向上取整
    rows = (len(files) + cols - 1) // cols
    # 每格高度 = 缩略图高度（按 16:9 估算）+ 标注区高度
    cell_height = int(thumb_width * 9 / 16) + label_height

    # 整张画布的宽高
    width = cols * thumb_width + (cols + 1) * gap
    height = rows * cell_height + (rows + 1) * gap

    # 创建浅灰色画布（RGB 三通道）
    sheet = Image.new("RGB", (width, height), (243, 243, 244))
    draw = ImageDraw.Draw(sheet)     # 画笔，用于写文件名
    font = load_font(20)

    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        # 等比缩放缩略图，保持原始宽高比（LANCZOS 是高质量缩放算法）
        im = im.resize((thumb_width, int(thumb_width * im.height / im.width)),
                       Image.LANCZOS)
        # divmod 得到第 i 张图所在的行 r 和列 c
        r, c = divmod(i, cols)
        x = gap + c * (thumb_width + gap)   # 左上角 x 坐标
        y = gap + r * (cell_height + gap)   # 左上角 y 坐标
        # 把图片贴到格子里（垂直方向居中，看起来更整齐）
        sheet.paste(im, (x, y + (cell_height - label_height - im.height) // 2))
        # 在图片下方写文件名
        draw.text((x + 2, y + cell_height - label_height + 6),
                  os.path.basename(f), font=font, fill=(30, 41, 59))

    sheet.save(output_file, quality=92)    # JPEG 质量 92
    print(f"完成：{len(files)} 张图片 → {output_file}（{width}x{height}）")
    return output_file


if __name__ == "__main__":
    # 支持命令行传参：python 03_make_contact_sheet.py 图片目录 输出图片 列数
    image_dir = sys.argv[1] if len(sys.argv) > 1 else IMAGE_DIR
    output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else COLS

    try:
        make_contact_sheet(image_dir, output_file, cols,
                           THUMB_WIDTH, GAP, LABEL_HEIGHT)
    except Exception as e:
        print(f"拼接失败：{e}")