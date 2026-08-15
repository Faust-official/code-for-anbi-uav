# -*- coding: utf-8 -*-
"""
============================================================
 04_vision_analyze.py —— 调用视觉大模型识别画面内容（可选）
============================================================
功能
----
把一张视频帧发送给视觉大模型（阿里云百炼 Qwen-VL），
让模型用文字描述画面内容，并判断是否存在病害迹象。
这一步解决"人工逐帧看图太慢"的问题，是预分析阶段的关键提速手段。

说明
----
- 本脚本使用阿里云百炼（DashScope）的 OpenAI 兼容接口，
  只依赖 Python 标准库（urllib），无需额外安装包。
- 需要先申请一个 DashScope API Key：
    阿里云百炼控制台 → API-KEY 管理 → 创建
- Key 的配置方式（二选一）：
    方式A：设置环境变量  DASHSCOPE_API_KEY=你的key
    方式B：在本文件同目录创建 .env 文件，内容一行：
           DASHSCOPE_API_KEY=你的key

使用方式
--------
1) 配置好 API Key（见上面"说明"）
2) 直接运行：  python 04_vision_analyze.py
   或指定图片：python 04_vision_analyze.py "图片路径"
3) 输出：模型生成的画面描述文字
"""
# ========== 强制UTF-8输出编码（Windows中文显示） ==========
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ===========================================================

import os
import json
import base64
import urllib.request

# ============ 需要修改的地方 ============
IMAGE_PATH = r"C:\codex-workspace\1\video_frames\frame_00369_t006.16s.jpg"
PROMPT = ("这是一张无人机拍摄的铁路声屏障画面。请用中文描述："
          "1)画面里有什么结构（声屏障、立柱、螺栓、金属板、混凝土等）；"
          "2)是否存在锈蚀、表面损伤、开裂、砂浆劣化等病害迹象；"
          "3)拍摄距离与角度、光线天气；4)画面是否清晰、适合画框标注。")
MODEL = "qwen-vl-max"     # 模型名：qwen-vl-max（更强）/ qwen-vl-plus（更快更便宜）
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# ========================================


def load_api_key():
    """
    读取 API Key，优先级：环境变量 > 同目录 .env 文件。
    返回字符串；找不到返回 None。
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key.strip()

    # 尝试从本文件同目录的 .env 读取（格式：KEY=VALUE，一行一个）
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHSCOPE_API_KEY="):
                    # 去掉前缀，并兼容值两端的引号
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def image_to_base64(path):
    """把图片文件读成 base64 字符串（视觉模型通过 base64 接收图片）。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path, prompt, model):
    """
    调用 Qwen-VL 视觉模型，返回模型生成的文字描述。

    接口格式：OpenAI 兼容的 chat/completions，
    图片通过 content 数组里的 image_url 传入（data URI 形式）。
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY。请先申请阿里云百炼 API Key，"
            "并配置到环境变量或同目录 .env 文件。"
        )

    # 1) 构造请求体（JSON）
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    # 图片：base64 编码的 JPEG
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"}},
                    # 文字指令
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 800,     # 最多生成 800 个 token
    }

    # 2) 发送 HTTP POST 请求
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",   # 身份认证
            "Content-Type": "application/json",
        },
        method="POST",
    )

    # 3) 读取并解析响应（超时 120 秒，防止网络卡死）
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # 4) 返回模型生成的内容：choices[0].message.content
    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # 支持命令行传参：python 04_vision_analyze.py "图片路径"
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    try:
        print(f"正在分析：{image_path}")
        answer = analyze_image(image_path, PROMPT, MODEL)
        print("=" * 50)
        print("模型识别结果：")
        print("=" * 50)
        print(answer)
    except Exception as e:
        print(f"调用失败：{e}")