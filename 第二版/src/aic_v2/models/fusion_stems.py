import torch
import torch.nn as nn


class QualityGatedStem(nn.Module):
    """轻量候选模块；基线不默认启用。"""
    def __init__(self, rgb_channels=3, auxiliary_channels=3, out_channels=32):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(rgb_channels, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.SiLU())
        self.aux = nn.Sequential(nn.Conv2d(auxiliary_channels, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.SiLU())
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels * 2, out_channels, 1), nn.Sigmoid())
    def forward(self, rgb, auxiliary):
        a, b = self.rgb(rgb), self.aux(auxiliary)
        gate = self.gate(torch.cat((a, b), 1))
        return a + gate * b
