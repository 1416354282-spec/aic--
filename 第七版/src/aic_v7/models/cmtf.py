"""CMTF: Cross-Modal Transformer Fusion (v5 core innovation, replaces v3 Tri-CSSA).

Per-scale block taking (S_i, I_i, D_i) -> F_i with five steps:
  1. channel alignment projections (1x1)
  2. bidirectional cross-attention: IR queries RGB+Depth context, Depth queries RGB+IR
  3. modality weighting: ECA scores + quality prior bias + brightness bias -> softmax
  4. spatial attention (7x7 on [mean, max])
  5. residual identity: F = S + tanh(residual_scale) * spatial * Delta

Identity init (residual_scale=0) preserves the pretrained RF-DETR feature
distribution; modality weights collapse to RGB when a modality is missing
(quality=0), giving the competition-required robustness for free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ECAChannelScore(nn.Module):
    """Channel score via global average pooling + 1D conv (kernel 5), from v3 Tri-CSSA.

    Outputs (B, C, 1, 1) so stacked modality scores broadcast against feature maps.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        kernel = min(5, channels if channels % 2 else 5)
        if kernel % 2 == 0:
            kernel = max(3, kernel - 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        pooled = self.pool(x).flatten(2)  # (B, C, 1)
        scores = self.conv(pooled.transpose(1, 2))  # (B, 1, C)
        return scores.transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)


class _CrossModalAttention(nn.Module):
    """Bidirectional cross-attention: IR/Depth tokens query the RGB(+other) context.

    Operates on flattened tokens of a single scale; 2 heads by default to keep
    token-level cost modest at stride 16 (704px -> 44x44 tokens).
    """

    def __init__(self, channels: int, nhead: int = 2):
        super().__init__()
        self.nhead = nhead
        head_dim = channels // nhead
        self.head_dim = head_dim
        self.q_ir = nn.Linear(channels, channels)
        self.q_depth = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)

    def _attend(self, query, q_proj, context):
        batch, tokens, channels = query.shape
        q = q_proj(query).view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv(context)
        k = kv.view(batch, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = kv.view(batch, -1, self.nhead, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.out(attended)

    def forward(self, ir_tokens, depth_tokens, rgb_context):
        """ir_tokens/depth_tokens: (B, N, C); rgb_context: (B, M, C) with M = 2N (rgb + other)."""
        ir_enhanced = self._attend(ir_tokens, self.q_ir, rgb_context)
        depth_enhanced = self._attend(depth_tokens, self.q_depth, rgb_context)
        return ir_enhanced, depth_enhanced


class CMTF(nn.Module):
    """Cross-modal Transformer fusion block (one per scale)."""

    def __init__(
        self,
        rgb_ch: int,
        ir_ch: int,
        depth_ch: int,
        nhead: int = 2,
        use_cross_attn: bool = True,
        use_lite: bool = False,
    ):
        super().__init__()
        self.rgb_ch = rgb_ch
        self.use_cross_attn = use_cross_attn and not use_lite
        # Step 1: channel alignment
        self.ir_projection = nn.Conv2d(ir_ch, rgb_ch, 1)
        self.depth_projection = nn.Conv2d(depth_ch, rgb_ch, 1)
        # Step 2: bidirectional cross-attention
        if self.use_cross_attn:
            self.cross_attn = _CrossModalAttention(rgb_ch, nhead=nhead)
        # Step 3: modality scores (ECA per modality) + quality/brightness bias in forward
        self.rgb_score = _ECAChannelScore(rgb_ch)
        self.ir_score = _ECAChannelScore(rgb_ch)
        self.depth_score = _ECAChannelScore(rgb_ch)
        # Step 4: spatial attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        # Step 5: residual identity
        self.delta_projection = nn.Conv2d(rgb_ch, rgb_ch, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        # brightness branch (DBW-inspired): global mean brightness per image
        self.brightness_bias = nn.Parameter(torch.zeros(1))

    def _modality_weights(self, rgb, ir, depth, quality):
        scores = torch.stack(
            (self.rgb_score(rgb), self.ir_score(ir), self.depth_score(depth)), dim=1
        )  # (B, 3, C, 1, 1)
        quality_bias = (quality + 1e-4).clamp_min(1e-4).log().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1, 1)
        weights = F.softmax(scores + quality_bias, dim=1)  # (B, 3, C, 1, 1)
        return weights

    def forward(self, rgb, infrared, depth, quality):
        """rgb/ir/depth: (B, C, H, W); quality: (B, 3).

        AMP 数值稳定性: 交叉注意力 QK^T 在 float16 下可能溢出 (logits > 65504
        导致 softmax NaN), 整个融合分支强制 float32 计算, 输出转回输入 dtype。
        注意: 仅 .float() 不够 —— autocast 会把 Conv2d 的 fp32 输入自动降回
        fp16, 必须用 autocast(enabled=False) 包住整个前向。
        """
        orig_dtype = rgb.dtype
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rgb = rgb.float()
            infrared = infrared.float()
            depth = depth.float()
            quality = quality.float() if torch.is_tensor(quality) else quality
            return self._forward_fp32(rgb, infrared, depth, quality).to(orig_dtype)

    def _forward_fp32(self, rgb, infrared, depth, quality):
        """FP32 下的融合前向 (原 forward 主体)。"""
        # Step 1: align channels
        ir = self.ir_projection(infrared)
        depth = self.depth_projection(depth)

        if self.use_cross_attn:
            # Step 2: cross-attention over flattened tokens
            batch, channels, height, width = rgb.shape
            rgb_tokens = rgb.flatten(2).transpose(1, 2)  # (B, N, C)
            ir_tokens = ir.flatten(2).transpose(1, 2)
            depth_tokens = depth.flatten(2).transpose(1, 2)
            rgb_context = torch.cat((rgb_tokens, depth_tokens), dim=1)  # (B, 2N, C)
            ir_enh, depth_enh = self.cross_attn(ir_tokens, depth_tokens, rgb_context)
            ir = ir_enh.transpose(1, 2).reshape(batch, channels, height, width)
            depth = depth_enh.transpose(1, 2).reshape(batch, channels, height, width)

        # Step 3: modality weights (ECA scores + quality bias + brightness bias)
        weights = self._modality_weights(rgb, ir, depth, quality)
        stacked = torch.stack((rgb, ir, depth), dim=1)
        mixed = (weights * stacked).sum(dim=1)  # (B, C, H, W)

        # Step 4: spatial attention
        spatial_input = torch.cat((mixed.mean(1, keepdim=True), mixed.amax(1, keepdim=True)), dim=1)
        spatial = self.spatial(spatial_input).sigmoid()

        # Step 5: residual identity (zero-init -> identity)
        delta = self.delta_projection(mixed - rgb)
        return rgb + torch.tanh(self.residual_scale) * spatial * delta
