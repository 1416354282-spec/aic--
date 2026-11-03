from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tri_cssa import TriCSSA


class ReliabilityFusion(TriCSSA):
    """Tri-CSSA with spatial modality scores and a real depth-validity mask."""

    def __init__(self, rgb_channels: int, ir_channels: int, depth_channels: int):
        super().__init__(rgb_channels, ir_channels, depth_channels)
        self.local_score = nn.Conv2d(6, 3, 3, padding=1)
        nn.init.zeros_(self.local_score.weight)
        nn.init.zeros_(self.local_score.bias)

    def forward(self, rgb, infrared, depth, quality, depth_valid):
        original_dtype = rgb.dtype
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            result = self._forward_fp32(
                rgb.float(), infrared.float(), depth.float(), quality.float(), depth_valid.float()
            )
        return result.to(original_dtype)

    def _forward_fp32(self, rgb, infrared, depth, quality, depth_valid):
        valid = F.interpolate(depth_valid, size=rgb.shape[-2:], mode="area").clamp(0, 1)
        infrared = self.ir_projection(infrared)
        depth = self.depth_projection(depth) * valid

        scores = torch.stack(
            (self.rgb_score(rgb), self.ir_score(infrared), self.depth_score(depth)), dim=1
        )
        summaries = torch.cat(
            (
                rgb.mean(1, keepdim=True),
                infrared.mean(1, keepdim=True),
                depth.mean(1, keepdim=True),
                (infrared - rgb).abs().mean(1, keepdim=True),
                (depth - rgb).abs().mean(1, keepdim=True),
                valid,
            ),
            dim=1,
        )
        local = 2.0 * torch.tanh(self.local_score(summaries)).unsqueeze(2)
        quality_bias = quality.clamp_min(1e-4).log().view(quality.shape[0], 3, 1, 1, 1)
        depth_bias = valid.clamp_min(1e-4).log().unsqueeze(1)
        depth_bias = depth_bias.masked_fill(valid.unsqueeze(1) == 0, -1e4)
        quality_bias = quality_bias + torch.cat(
            (torch.zeros_like(depth_bias), torch.zeros_like(depth_bias), depth_bias), dim=1
        )
        weights = torch.softmax(scores + quality_bias + local, dim=1)
        mixed = (weights * torch.stack((rgb, infrared, depth), dim=1)).sum(dim=1)
        spatial_input = torch.cat((mixed.mean(1, keepdim=True), mixed.amax(1, keepdim=True)), dim=1)
        spatial = self.spatial(spatial_input).sigmoid()
        delta = self.delta_projection(mixed - rgb)
        return rgb + torch.tanh(self.residual_scale) * spatial * delta
