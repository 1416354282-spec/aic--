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

    def forward(self, rgb, infrared, depth, quality, depth_quality=None):
        orig_dtype = rgb.dtype
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rgb = rgb.float()
            infrared = infrared.float()
            depth = depth.float()
            quality = quality.float() if torch.is_tensor(quality) else quality
            if depth_quality is not None:
                depth_quality = depth_quality.float()
            return self._forward_fp32(rgb, infrared, depth, quality, depth_quality).to(orig_dtype)

    def _forward_fp32(self, rgb, infrared, depth, quality, depth_quality=None):
        infrared = self.ir_projection(infrared)
        depth = self.depth_projection(depth)
        scores = torch.stack((self.rgb_score(rgb), self.ir_score(infrared), self.depth_score(depth)), dim=1)
        quality_bias = quality.clamp_min(1e-4).log().view(quality.shape[0], 3, 1, 1)
        if depth_quality is None:
            # 整图标量先验 (旧行为): 3 个模态各一个 bias
            weights = torch.softmax(scores + quality_bias.unsqueeze(-1), dim=1)
        else:
            # 空间先验: depth 的标量 bias 换成逐 token 的有效率图,
            # 深度缺失区域 (含 mosaic 拼接/擦除/模态 dropout) 自动降权。
            height, width = depth_quality.shape[-2:]
            # scores 形状 (B, 3, C, 1, 1): 每个模态、每个通道一个分数
            spatial_scores = scores.expand(quality.shape[0], 3, scores.shape[2], height, width).clone()
            spatial_scores = spatial_scores + quality_bias.unsqueeze(-1)
            depth_bias = depth_quality.clamp_min(1e-4).log().unsqueeze(1)  # (B,1,1,H,W)
            spatial_scores[:, 2:3] = spatial_scores[:, 2:3] + depth_bias
            weights = torch.softmax(spatial_scores, dim=1)
        stacked = torch.stack((rgb, infrared, depth), dim=1)
        mixed = (weights * stacked).sum(dim=1)
        spatial_input = torch.cat((mixed.mean(1, keepdim=True), mixed.amax(1, keepdim=True)), dim=1)
        spatial = self.spatial(spatial_input).sigmoid()
        delta = self.delta_projection(mixed - rgb)
        return rgb + torch.tanh(self.residual_scale) * spatial * delta
