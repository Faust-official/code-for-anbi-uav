# 无人机视频预分析脚本（声屏障病害检测项目）

本目录的脚本对应"论文复现——数据预分析"这一步：
先用轻量脚本把无人机视频变成"看得懂、能标注"的图片，并借助视觉大模型快速判断画面内容。

## 目录结构

| 文件 | 作用 | 依赖 |
|---|---|---|
| `01_video_info.py` | 读取视频参数（分辨率/帧率/帧数/时长/编码） | opencv-python |
| `02_extract_frames.py` | 等间隔抽关键帧 + 模糊过滤 + 相似帧去重 | opencv-python, numpy |
| `03_make_contact_sheet.py` | 把关键帧拼成一张总览缩略图 | pillow |
| `04_vision_analyze.py` | 调用视觉大模型（Qwen-VL）描述画面内容 | 标准库（需 API Key） |
| `05_make_training_dataset.py` | 一键生成可标注的 YOLO 训练图片集（抽帧+去重+640 切片+7:2:1 划分） | opencv-python, numpy |
| `06_auto_label.py` | 用视觉大模型（Qwen-VL）自动标注，生成 YOLO 伪标签 | 标准库（需 API Key） |

## 在 VSCode 中运行

1. **打开文件夹**：VSCode → 文件 → 打开文件夹 → 选择本目录（或整个项目目录）。
2. **选择解释器**：`Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选择你装了
   PyTorch/OpenCV 的那个环境（建议先建一个虚拟环境，见下）。
3. **安装依赖**（终端里执行，任选其一）：
   ```bash
   pip install opencv-python pillow numpy
   ```
   或者一次性装齐（含后续训练需要的包）：
   ```bash
   pip install opencv-python pillow numpy torch torchvision
   ```
4. **修改路径**：每个脚本顶部都有 `VIDEO_PATH` / `OUTPUT_DIR` 等"需要修改的地方"，
   改成你自己的视频路径和输出目录。
5. **运行**：点脚本右上角的"运行"按钮（▶），或在终端执行：
   ```bash
   python 01_video_info.py
   ```

## 推荐执行顺序

```bash
# 1) 先看视频参数，决定抽帧策略
python 01_video_info.py

# 2) 抽 12 张关键帧（也可指定：python 02_extract_frames.py 视频 输出目录 张数）
python 02_extract_frames.py

# 3) 拼一张总览图，快速检查抽帧质量（也可直接放进 PPT）
python 03_make_contact_sheet.py

# 4) （可选，需要 API Key）让视觉大模型描述画面
python 04_vision_analyze.py
```

## 关于 04 视觉模型（可选步骤）

- 需要阿里云百炼（DashScope）API Key：控制台 → API-KEY 管理 → 创建。
- 配置方式二选一：
  - 环境变量：`set DASHSCOPE_API_KEY=你的key`（Windows CMD）或
    `$env:DASHSCOPE_API_KEY="你的key"`（PowerShell）
  - 在 `04_vision_analyze.py` 同目录创建 `.env` 文件，内容一行：
    ```
    DASHSCOPE_API_KEY=你的key
    ```
- 模型可选：`qwen-vl-max`（更强）/ `qwen-vl-plus`（更快更便宜），在脚本里改 `MODEL`。
- 注意：**不要把 API Key 提交到 Git / 发给别人**；`.env` 文件建议加入 `.gitignore`。


## 关于 05 训练数据集脚本（重点）

`05_make_training_dataset.py` 把视频直接变成 **YOLO 可直接训练的图片集**：

```
dataset/
├── data.yaml          # YOLOv5 训练配置（6 类，路径已配好）
├── manifest.csv       # 每张切片的来源（原始视频/帧号/时间/在原图的坐标）
├── images/ train|val|test
└── labels/ train|val|test   # 空目录，标注后存放 LabelImg 导出的 txt
```

常用参数（全部可省略）：
```bash
python 05_make_training_dataset.py \
  --input "视频或文件夹" \
  --output "数据集目录" \
  --interval-sec 1.0      # 抽帧间隔（秒）
  --window 640            # 切片大小
  --overlap 0.0           # 切片重叠比例（0=无重叠；设 0.25 更抗目标被切碎但数量多）
  --blur-mode adaptive    # 模糊过滤：adaptive 自适应(默认) / absolute 固定阈值 / off 关闭
  --tile-min-std 0        # 空白切片过滤（默认关闭，开启需先观察数据再调参）
```

三个设计要点（对应踩过的坑）：
1. **模糊过滤默认自适应**：不同视频清晰度基线差异大（蓝色板面偏平滑、镀锌立柱偏锐利），固定阈值会误杀整段视频（曾把 5 段里 2 段误删），自适应模式只丢弃本视频最模糊的约 1/4。
2. **输出文件名为 ASCII 编号（v001...）**：YOLOv5 的图片加载用 cv2.imread，中文文件名会导致读取失败；原始视频名记录在 manifest.csv。
3. **切片内容过滤默认关闭**：平滑声屏障板面的灰度标准差和天空一样小，固定阈值会误删有用的板面切片。

标注：用 LabelImg/X-AnyLabeling 打开 `images/train`，按 data.yaml 里 6 类标注，
标签保存到 `labels/train`；val/test 同理。训练时 YOLOv5 直接引用 `data.yaml`。


## 关于 06 自动标注脚本（伪标签）

`06_auto_label.py` 调用 Qwen-VL 对每张切片输出 6 类目标的边界框，自动写成 YOLO 标签：

```bash
# 先试 20 张
python 06_auto_label.py --split train --limit 20
# 标注全部（train/val/test）
python 06_auto_label.py --split all --workers 3
```

要点：
- **伪标签定位**：速度快、可批量，但精度不如人工。实测 80 张：约 87% 成功解析、
  框基本能对准立柱/损伤区，但存在砂浆层框偏大、漏框、误检等问题。
  **训练前务必人工抽检/修正**，或先用它做预训练再人工精标。
- **断点续跑**：已生成标签的图片自动跳过；`--force` 强制重标；解析失败的图片
  不会写标签，重跑一遍即可补上（脚本自带 3 次重试，已把网络失败率降到 0）。
- 后处理过滤已内置：全图框（>50% 面积）和过小噪声框会被丢弃。
- 成本与时间：按当前速度约 1.5 秒/张（3 线程并发），全部 2430 张约需 1~1.5 小时，
  调用按模型计费，建议先用 `--limit` 小批量试跑评估质量。

## 常见问题

- **打不开视频**：检查路径是否正确；确认已装 `opencv-python`（自带 H.264 解码）。
- **中文显示成方块**：脚本已优先使用微软雅黑字体；若还是不行，检查系统字体。
- **抽出的帧太少**：`02` 脚本开了模糊过滤和去重，可调低 `BLUR_THRESHOLD` /
  调低 `DUP_THRESHOLD`，或把两个开关改成 `False`。