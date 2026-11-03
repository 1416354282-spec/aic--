from __future__ import annotations

import torch
import torch.nn as nn


class ECAChannelScore(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, inputs):
        pooled = inputs.mean((2, 3), keepdim=False).unsqueeze(1)
        return self.conv(pooled).squeeze(1).unsqueeze(-1).unsqueeze(-1)


class TriCSSA(nn.Module):
    """Differentiable three-modal adaptation of channel switching and spatial attention.

    与 CMTF 相同的数值稳定性处理: AMP 下 softmax 分数可能溢出, 整个融合
    分支强制 float32 计算, 输出转回输入 dtype (v5 验证过的方案)。
    """

    def __init__(self, rgb_channels: int, ir_channels: int, depth_channels: int):
        super().__init__()
        self.ir_projection = nn.Conv2d(ir_channels, rgb_channels, 1, bias=False)
        self.depth_projection = nn.Conv2d(depth_channels, rgb_channels, 1, bias=False)
        self.rgb_score = ECAChannelScore(rgb_channels)
        self.ir_score = ECAChannelScore(rgb_channels)
        self.depth_score = ECAChannelScore(rgb_channels)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.delta_projection = nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, rgb, infrared, depth, quality):
        orig_dtype = rgb.dtype
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rgb = rgb.float()
            infrared = infrared.float()
            depth = depth.float()
            quality = quality.float() if torch.is_tensor(quality) else quality
            return self._forward_fp32(rgb, infrared, depth, quality).to(orig_dtype)

    def _forward_fp32(self, rgb, infrared, depth, quality):
        infrared = self.ir_projection(infrared)
        depth = self.depth_projection(depth)
        scores = torch.stack((self.rgb_score(rgb), self.ir_score(infrared), self.depth_score(depth)), dim=1)
        quality_bias = quality.clamp_min(1e-4).log().view(quality.shape[0], 3, 1, 1, 1)
        weights = torch.softmax(scores + quality_bias, dim=1)
        stacked = torch.stack((rgb, infrared, depth), dim=1)
        mixed = (weights * stacked).sum(dim=1)
        spatial_input = torch.cat((mixed.mean(1, keepdim=True), mixed.amax(1, keepdim=True)), dim=1)
        spatial = self.spatial(spatial_input).sigmoid()
        delta = self.delta_projection(mixed - rgb)
        return rgb + torch.tanh(self.residual_scale) * spatial * delta
