from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn


class EarlyFusionYOLO:
    def __init__(self, weights: str, in_channels: int, num_classes: int, device: str = "cpu", allow_download: bool = False):
        from ultralytics import YOLO
        path = Path(weights)
        if not path.exists() and not allow_download: raise FileNotFoundError(f"本地权重不存在，已禁止下载: {path}")
        self.wrapper = YOLO(str(path))
        self.model = self.wrapper.model
        # ultralytics >= 8.4 中 model.args 是 dict 且 loss 超参数（box/cls/dfl）缺失，需合并默认值
        if isinstance(self.model.args, dict):
            from ultralytics.utils import DEFAULT_CFG_DICT
            merged = {**DEFAULT_CFG_DICT, **self.model.args}
            self.model.args = Namespace(**merged)
        self._replace_first_conv(in_channels)
        head = self.model.model[-1]
        head.nc = num_classes
        self.model.nc = num_classes
        self.model.names = {i: str(i) for i in range(num_classes)}
        if hasattr(head, "cv3"):
            for branch in head.cv3:
                last = branch[-1]
                if isinstance(last, nn.Conv2d) and last.out_channels != num_classes:
                    branch[-1] = nn.Conv2d(last.in_channels, num_classes, last.kernel_size, last.stride, last.padding)
        self.model.to(device)

    def _replace_first_conv(self, in_channels):
        for module in self.model.modules():
            conv = getattr(module, "conv", None)
            if isinstance(conv, nn.Conv2d) and conv.in_channels == 3:
                new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
                with torch.no_grad():
                    new.weight[:, :3].copy_(conv.weight)
                    if in_channels > 3: new.weight[:, 3:].copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_channels - 3, 1, 1))
                    if conv.bias is not None: new.bias.copy_(conv.bias)
                module.conv = new
                return
        raise RuntimeError("未找到首个三通道卷积")
