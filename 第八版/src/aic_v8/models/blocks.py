from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=True):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2, activation=True):
        super().__init__()
        primary_channels = math.ceil(out_channels / ratio)
        cheap_channels = primary_channels * (ratio - 1)
        self.primary = ConvGNAct(in_channels, primary_channels, kernel_size, stride, activation=activation)
        self.cheap = ConvGNAct(primary_channels, cheap_channels, 3, 1, groups=primary_channels, activation=activation)
        self.out_channels = out_channels

    def forward(self, inputs):
        primary = self.primary(inputs)
        return torch.cat((primary, self.cheap(primary)), 1)[:, :self.out_channels]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2.0):
        super().__init__()
        hidden = max(8, int(out_channels * expansion))
        self.expand = GhostConv(in_channels, hidden, 1, 1)
        self.spatial = ConvGNAct(hidden, hidden, 3, stride, groups=hidden) if stride > 1 else nn.Identity()
        self.project = GhostConv(hidden, out_channels, 1, 1, activation=False)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                ConvGNAct(in_channels, in_channels, 3, stride, groups=in_channels, activation=False),
                ConvGNAct(in_channels, out_channels, 1, 1, activation=False),
            )

    def forward(self, inputs):
        return self.project(self.spatial(self.expand(inputs))) + self.shortcut(inputs)


class C3Ghost(nn.Module):
    def __init__(self, in_channels, out_channels, repeats=1, expansion=0.5):
        super().__init__()
        hidden = max(8, int(out_channels * expansion))
        self.left = ConvGNAct(in_channels, hidden, 1)
        self.right = ConvGNAct(in_channels, hidden, 1)
        self.blocks = nn.Sequential(*(GhostBottleneck(hidden, hidden) for _ in range(repeats)))
        self.output = ConvGNAct(hidden * 2, out_channels, 1)

    def forward(self, inputs):
        return self.output(torch.cat((self.blocks(self.left(inputs)), self.right(inputs)), 1))


class EMAAttention(nn.Module):
    """Efficient multi-scale attention with grouped cross-spatial interaction."""

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        groups = min(factor, channels)
        while groups > 1 and channels % groups:
            groups -= 1
        self.groups = groups
        grouped_channels = channels // groups
        self.conv1 = nn.Conv2d(grouped_channels, grouped_channels, 1)
        self.conv3 = nn.Conv2d(grouped_channels, grouped_channels, 3, padding=1)
        self.norm = nn.GroupNorm(grouped_channels, grouped_channels)

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        grouped = inputs.reshape(batch * self.groups, channels // self.groups, height, width)
        pooled_h = grouped.mean(dim=3, keepdim=True)
        pooled_w = grouped.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.conv1(torch.cat((pooled_h, pooled_w), dim=2))
        weight_h, weight_w = torch.split(encoded, (height, width), dim=2)
        branch1 = self.norm(grouped * weight_h.sigmoid() * weight_w.transpose(2, 3).sigmoid())
        branch2 = self.conv3(grouped)
        query1 = F.softmax(branch1.mean((2, 3), keepdim=True).flatten(2), dim=1).transpose(1, 2)
        query2 = F.softmax(branch2.mean((2, 3), keepdim=True).flatten(2), dim=1).transpose(1, 2)
        values1 = branch1.flatten(2)
        values2 = branch2.flatten(2)
        spatial = (query1 @ values2 + query2 @ values1).reshape(batch * self.groups, 1, height, width)
        return (grouped * spatial.sigmoid()).reshape(batch, channels, height, width)


class StarBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.depthwise1 = ConvGNAct(channels, channels, 3, groups=channels)
        self.branch1 = nn.Conv2d(channels, hidden, 1)
        self.branch2 = nn.Conv2d(channels, hidden, 1)
        self.project = ConvGNAct(hidden, channels, 1, activation=False)
        self.depthwise2 = ConvGNAct(channels, channels, 3, groups=channels, activation=False)

    def forward(self, inputs):
        features = self.depthwise1(inputs)
        features = self.branch1(features) * F.relu6(self.branch2(features))
        return inputs + self.depthwise2(self.project(features))


class GhostHGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, branches=3):
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.input = ConvGNAct(in_channels, hidden, 1)
        self.layers = nn.ModuleList(GhostConv(hidden, hidden, 3) for _ in range(branches))
        self.aggregate = ConvGNAct(hidden * (branches + 1), out_channels, 1)
        self.shortcut = nn.Identity() if in_channels == out_channels else ConvGNAct(in_channels, out_channels, 1, activation=False)

    def forward(self, inputs):
        features = [self.input(inputs)]
        for layer in self.layers:
            features.append(layer(features[-1]))
        return self.aggregate(torch.cat(features, 1)) + self.shortcut(inputs)
