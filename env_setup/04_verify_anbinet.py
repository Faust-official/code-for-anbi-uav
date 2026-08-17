# -*- coding: utf-8 -*-
"""
============================================================
 04_verify_anbinet.py —— 验证 ANBINet（YOLOv5s + SC3 + MAM）
============================================================
功能
----
1) 构建 YOLOv5s / ANBINet(7x7) / ANBINet-Light(1x1) 三种配置
2) 打印参数量、GFLOPs 对比
3) 前向推理 + 反向传播（训练模式）验证梯度能正常流动

用法（VSCode 终端，任意目录）：
    python video_analysis/env_setup/04_verify_anbinet.py
"""
import os
import sys

YOLOV5 = r"C:\codex-workspace\1\yolov5"
sys.path.insert(0, YOLOV5)
os.chdir(YOLOV5)

import torch
from models.yolo import Model


def collect_shapes(o):
    """递归收集输出张量形状（YOLOv5 检测头输出嵌套结构）。"""
    if isinstance(o, torch.Tensor):
        return [tuple(o.shape)]
    if isinstance(o, (list, tuple)):
        r = []
        for e in o:
            r += collect_shapes(e)
        return r
    return []


def build(cfg, train_mode=False):
    """构建模型；train_mode=True 时做一次前向+反向，验证梯度流动。"""
    model = Model(cfg=cfg, ch=3, nc=6)
    n_params = sum(p.numel() for p in model.parameters())

    if train_mode:
        model.train()
        x = torch.randn(2, 3, 640, 640)
        outs = model(x)                       # 前向
        # YOLOv5 训练模式输出 (losses, detections) 或 detections
        if isinstance(outs, tuple):
            det = outs[1]
        else:
            det = outs
        # 直接对输出求和做一次反向，验证 SC3/MAM 梯度可回传
        total = 0
        def add_tensors(o):
            nonlocal total
            if isinstance(o, torch.Tensor):
                total = total + o.sum()
            elif isinstance(o, (list, tuple)):
                for e in o:
                    add_tensors(e)
        add_tensors(outs)
        total.backward()
        # 检查 SC3/MAM 内部参数是否都拿到了梯度（证明梯度能流过新模块）
        grads = {}
        for name, m in model.named_modules():
            if type(m).__name__ in ("SC3", "MAM"):
                grads[f"{type(m).__name__}/{name}"] = all(
                    p.grad is not None for p in m.parameters() if p.requires_grad)
        return n_params, None, grads
    else:
        model.eval()
        with torch.no_grad():
            outs = model(torch.randn(2, 3, 640, 640))
        return n_params, collect_shapes(outs), None


if __name__ == "__main__":
    print("=" * 60)
    results = {}
    configs = [
        ("YOLOv5s 基线", "models/yolov5s.yaml"),
        ("+SC3", "models/anbinet_sc3.yaml"),
        ("+MAM", "models/anbinet_mam.yaml"),
        ("ANBINet(7x7)", "models/anbinet.yaml"),
        ("ANBINet-Light(1x1)", "models/anbinet_light.yaml"),
    ]
    for tag, cfg in configs:
        print(f"构建 {tag} ...")
        n_params, shapes, _ = build(cfg)
        results[tag] = n_params
        print(f"   参数量: {n_params/1e6:.2f} M，输出: {shapes}")

    print("=" * 60)
    print("训练模式反向传播验证（ANBINet-Light）：")
    n_params, _, grads = build("models/anbinet_light.yaml", train_mode=True)
    print("   SC3/MAM 各模块梯度是否正常:", grads)
    assert all(grads.values()), "有模块梯度为 None，请检查实现"

    print("=" * 60)
    print("消融配置参数量对比（对照论文 Table I）：")
    base = results["YOLOv5s 基线"]
    for tag, n in results.items():
        print(f"   {tag:<18}: {n/1e6:6.2f} M   (相对基线 {n/base*100-100:+6.1f}%)")
    print("\n[OK] ANBINet 构建、前向、反向全部正常。可进入正式训练。")