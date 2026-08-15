# -*- coding: utf-8 -*-
"""
============================================================
 01_video_info.py —— 读取无人机视频的基本参数
============================================================
功能
----
用 OpenCV 读取一段视频文件，并输出：
    - 分辨率（宽 x 高）
    - 帧率 FPS
    - 总帧数
    - 时长（秒）
    - 编码格式（fourcc）

适用场景
--------
在做"声屏障病害检测"数据预处理之前，先了解每段视频的参数，
便于决定抽帧间隔、是否需要切片等。

使用方式
--------
1) 安装依赖：      pip install opencv-python
2) 直接运行：      python 01_video_info.py
   或在命令行指定视频路径：python 01_video_info.py "视频路径.mp4"
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================

import cv2          # OpenCV：视频/图像处理库

# ============ 需要修改的地方 ============
# 你的无人机视频路径。
# 注意：Windows 路径建议用 r"..." 原始字符串，避免反斜杠被当成转义符；
#       中文路径（如"科研""无人机数据"）也可以正常处理。
VIDEO_PATH = r"C:\Users\Faust\Desktop\科研\无人机数据\2_赣州工务段-九江工务段等\赣州工务段京港高速线下行K1989+368-484.mp4"
# ========================================


def decode_fourcc(fourcc_value):
    """
    把 OpenCV 返回的 fourcc 整数解码成可读字符串（如 'h264'）。

    原理：fourcc 本质上是 4 个 ASCII 字符拼成的整数，
    例如 'h264' 存储为 0x34363268，按字节从低到高还原即可。
    """
    chars = []
    for i in range(4):
        chars.append(chr((fourcc_value >> (8 * i)) & 0xFF))
    return "".join(chars)


def get_video_info(path):
    """
    读取视频参数，返回一个包含各项参数的字典。
    """
    # VideoCapture 是 OpenCV 读取视频的统一入口（支持 mp4/avi/mov 等）
    cap = cv2.VideoCapture(path)

    # 打不开时 isOpened() 返回 False。
    # 常见原因：路径写错 / 文件不存在 / 系统缺少视频解码器。
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件：{path}")

    # CAP_PROP_* 是 OpenCV 预定义的"属性编号"，
    # 用 cap.get(属性编号) 读取对应参数（返回 float，需自行转 int/round）。
    fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 帧率：每秒多少帧
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))         # 画面宽度（像素）
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))       # 画面高度（像素）
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 总帧数
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))       # 编码格式（整数）

    cap.release()   # 用完后释放资源（否则视频文件会被占用，无法删除/移动）

    # 时长 = 总帧数 / 帧率；帧率为 0 时做保护，避免除零
    duration = frame_count / fps if fps > 0 else 0.0

    # 把结果整理成字典，方便打印和后续使用
    info = {
        "路径": path,
        "分辨率": f"{width} x {height}",
        "帧率": round(fps, 2),
        "总帧数": frame_count,
        "时长(秒)": round(duration, 2),
        "编码格式": decode_fourcc(fourcc_value),
    }
    return info


if __name__ == "__main__":
    # 程序入口：只有"直接运行本文件"时才执行这里。
    # 支持命令行传参：python 01_video_info.py "视频路径"，不传则用上面的 VIDEO_PATH。
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH

    try:
        result = get_video_info(video_path)
        print("=" * 50)
        print("视频参数如下：")
        print("=" * 50)
        # 逐行打印，f-string 中 <12 表示左对齐占 12 个字符宽，让输出对齐
        for key, value in result.items():
            print(f"{key:<12}: {value}")
    except Exception as e:
        # 出错时给出友好提示（而不是让程序直接崩溃）
        print(f"读取失败：{e}")
        print("请检查：1) 视频路径是否正确；2) 是否已安装 opencv-python。")