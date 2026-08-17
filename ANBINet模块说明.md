# ANBINet 模块实现说明（SC3 + MAM）

> 论文：ANBI-UAV（IEEE JSTARS 2026），官方代码未开源，本实现按论文文字描述复现，
> 已在沙箱 CPU 环境验证"构建 + 前向 + 反向 + 端到端训练"全部通过。

## 一、改了什么（都在 yolov5 目录里）

| 文件 | 改动 |
|---|---|
| `yolov5\models\common.py` | 新增 **SC3**（简化 C3）和 **MAM**（多注意力）两个模块类 |
| `yolov5\models\yolo.py` | import 增加 SC3/MAM；`parse_model` 注册 SC3（按 C3 方式处理）、MAM（自动取输入通道） |
| `yolov5\models\anbinet.yaml` | **标准版**：backbone 的 C3→SC3，head 的 P3/P4/P5 前各插一个 MAM（融合 7×7，按论文公式） |
| `yolov5\models\anbinet_light.yaml` | **轻量版**：同上，但 MAM 融合用 1×1（参数贴近论文 6.7M 量级，训练快） |\n| `yolov5\models\anbinet_sc3.yaml` | 消融用：仅 +SC3（backbone 换 SC3，head 不加 MAM） |\n| `yolov5\models\anbinet_mam.yaml` | 消融用：仅 +MAM（backbone 保持 C3，head 加 MAM） |

## 二、两个新模块（论文结构）

### SC3 —— 简化 C3（替换 backbone 中的 C3）
1. 输入 → 1×1 卷积 + SiLU → 特征 F
2. F → 级联交叉结构（1×1 与 3×3 卷积交替、各接 SiLU）→ F′
3. F + F′（跳跃相加）后再与 F 拼接 → F″
4. F″ → 1×1 卷积 + SiLU → 输出

### MAM —— 多注意力（并联通道+空间注意力）
- **通道分支**（关注什么）：全局平均池化 → 1×1(降维) → ReLU → 1×1(还原) → sigmoid → 逐通道乘
- **空间分支**（关注哪里）：通道维 avg/max 池化 → 拼接 → 7×7 卷积 → sigmoid → 逐空间乘
- **融合**：F_MAM = F + Conv(Concat(F_c, F_s))

## 三、两种配置怎么选

| 配置 | 参数量 | GFLOPs | 说明 |
|---|---|---|---|
| YOLOv5s 基线 | 7.04 M | 16.0 | 对照用 |
| anbinet.yaml（7×7 融合） | 40.25 M | 75.6 | 严格按论文公式，参数偏大 |
| **anbinet_light.yaml（1×1 融合）** | **7.22 M** | **15.2** | **推荐**：贴近论文效率，训练/推理都快 |

> 为什么有差异：论文公式明确写融合是 7×7 卷积（F_MAM = F + Conv7x7(Concat(Fc,Fs))），
> 按字面实现在三处大通道特征图上参数会到 40M。论文最终模型参数未在文中给出，
> 官方代码也未开源，所以给出两种选择：忠实版(7×7)和轻量版(1×1)。
> 想切换只需把 yaml 里 MAM 的 `[16]` 改成 `[16, 1]`（或反过来）。

## 四、一键消融实验（论文 Table I）\n\n```bash\ncd C:\\codex-workspace\\1\\yolov5\n# 先小规模试跑（2 epoch）\npython ..\\video_analysis\\env_setup\\06_run_ablations.py --epochs 2 --batch 4\n# 正式实验（300 epoch）\npython ..\\video_analysis\\env_setup\\06_run_ablations.py --epochs 300 --batch 16 --device 0\n```\n\n顺序训练 4 个配置：`baseline` → `sc3` → `mam` → `anbinet`（ANBINet-Light），\n结果分别保存在 `C:\\codex-workspace\\1\\runs\\ablation\\{名称}\\`，每个目录里的 `results.csv` 是指标汇总。\n只跑某一个：`--only anbinet`。\n\n## 五、验证

```bash
# 在 VSCode 终端运行（任意目录）
python video_analysis/env_setup/04_verify_anbinet.py
```
会打印 5 种配置（基线/+SC3/+MAM/ANBINet7x7/ANBINet-Light）的参数量/GFLOPs，并验证 SC3/MAM 梯度都能正常回传。

端到端冒烟（1 epoch，16 张图）：
```bash
cd C:\codex-workspace\1\yolov5
python ..\video_analysis\env_setup\03_smoke_test.py --cfg models/anbinet_light.yaml
```

## 六、正式训练（手动跑单个配置）

在 `yolov5` 目录分别跑，对照论文 Table I：

```bash
# ① 基线 YOLOv5s
python train.py --data C:\codex-workspace\1\dataset\data.yaml --weights yolov5s.pt ^
  --img 640 --batch 16 --epochs 300 --device 0 --project runs\exp

# ② ANBINet-Light（推荐）
python train.py --data C:\codex-workspace\1\dataset\data.yaml --cfg models/anbinet_light.yaml ^
  --img 640 --batch 16 --epochs 300 --device 0 --project runs\anbinet_light
```

> 训练从零开始用 `--cfg`；用预训练权重则 `--weights yolov5s.pt`。
> 建议先用 `--epochs 5` 试跑确认正常，再跑 300 epochs。

## 七、实现假设（论文未写清楚、需注意的点）

1. **MAM 插入位置**：论文 Fig.7 有架构图，但正文未明说。本实现放在 head 的
   P3/P4/P5 三个检测尺度输出前（"自适应特征精炼"的常见做法）。如需调整，
   改 `anbinet.yaml` 里 MAM 所在行即可。
2. **SC3 级联深度**：论文"级联交叉结构"未给具体层数，本实现为 1×1→3×3 一次。
3. **MAM 融合核**：论文写 7×7，轻量版用 1×1（见第三节）。
4. 这几个点若之后能拿到官方代码，再按官方结构对齐。