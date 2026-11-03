"""IR / Depth modality encoders (ported from v3, adapted to RT-DETR multi-scale).

RT-DETR backbone (HGNetv2) emits three scales S3/S4/S5 (strides 8/16/32), so
the encoders produce three aligned outputs by default; ``out_strides`` keeps
the scale set configurable (v5 single-scale mode for ablation).
"""

from __future__ import annotations

import torch.nn as nn

from .blocks import C3Ghost, ConvGNAct, EMAAttention, GhostHGBlock, StarBlock
from .depth_adapters import DepthTypeAdapters


def _scaled_channels(rgb_channels, width):
    return [max(16, int(round(channel * width / 8) * 8)) for channel in rgb_channels]


class _EncoderBase(nn.Module):
    """Shared 5-stage downsampling skeleton (stride 2/4/8/16/32) with selectable outputs."""

    STRIDE_OF_STAGE = {2: 8, 3: 16, 4: 32}  # stage index (1-based) -> feature stride

    def __init__(self, rgb_channels, width=0.375):
        super().__init__()
        # rgb_channels may carry one value (v5 single-scale) or three (v3 P3/P4/P5);
        # derive per-stage widths in either case.
        scaled = _scaled_channels(rgb_channels, width)
        if len(scaled) >= 3:
            c3, c4, c5 = scaled[:3]
        else:
            c4 = scaled[0]
            c3 = max(16, c4 // 2)
            c5 = c4
        self.channels = (c3, c4, c5)
        self.stem_channels = max(16, c3 // 4)
        self.stage2_channels = max(16, c3 // 2)
        self.c3, self.c4, self.c5 = c3, c4, c5

    def _stage_outputs(self, x, stages, out_strides):
        outputs = []
        for stage_index in range(2, 6):
            x = stages[stage_index](x)
            if self.STRIDE_OF_STAGE[stage_index] in out_strides:
                outputs.append(x)
        return tuple(outputs)


class IREncoder(_EncoderBase):
    """EGD-inspired single-channel encoder; adaptation, not an EGD reproduction."""

    def __init__(self, rgb_channels, width=0.375, attention_groups=8, out_strides=(8, 16, 32)):
        super().__init__(rgb_channels, width)
        self.out_strides = tuple(out_strides)
        stem_channels = self.stem_channels
        stage2_channels = self.stage2_channels
        c3, c4, c5 = self.c3, self.c4, self.c5
        self.stem = ConvGNAct(1, stem_channels, 3, 2)
        self.stage2 = nn.Sequential(
            ConvGNAct(stem_channels, stage2_channels, 3, 2), C3Ghost(stage2_channels, stage2_channels)
        )
        self.stage3 = nn.Sequential(
            ConvGNAct(stage2_channels, c3, 3, 2), C3Ghost(c3, c3, 2), EMAAttention(c3, attention_groups)
        )
        self.stage4 = nn.Sequential(
            ConvGNAct(c3, c4, 3, 2), C3Ghost(c4, c4, 2), EMAAttention(c4, attention_groups)
        )
        self.stage5 = nn.Sequential(
            ConvGNAct(c4, c5, 3, 2), C3Ghost(c5, c5, 1), EMAAttention(c5, attention_groups)
        )
        # out_channels in the same order as out_strides
        stage_to_ch = {8: c3, 16: c4, 32: c5}
        self.out_channels = tuple(stage_to_ch[s] for s in self.out_strides)

    def forward(self, inputs):
        features = self.stage2(self.stem(inputs))
        p3 = self.stage3(features)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        outputs = []
        for s in self.out_strides:
            outputs.append({8: p3, 16: p4, 32: p5}[s])
        return tuple(outputs)


class DepthEncoder(_EncoderBase):
    """GLS-inspired depth encoder for HHA / value+validity+format channels."""

    def __init__(self, rgb_channels, width=0.375, out_strides=(8, 16, 32), in_channels=3, type_adapters=False):
        super().__init__(rgb_channels, width)
        self.input_adapters = DepthTypeAdapters(in_channels) if type_adapters else None
        self.out_strides = tuple(out_strides)
        stem_channels = self.stem_channels
        stage2_channels = self.stage2_channels
        c3, c4, c5 = self.c3, self.c4, self.c5
        self.stem = ConvGNAct(in_channels, stem_channels, 3, 2)
        self.stage2 = nn.Sequential(
            ConvGNAct(stem_channels, stage2_channels, 3, 2), GhostHGBlock(stage2_channels, stage2_channels, 2)
        )
        self.stage3 = nn.Sequential(
            ConvGNAct(stage2_channels, c3, 3, 2), GhostHGBlock(c3, c3), StarBlock(c3)
        )
        self.stage4 = nn.Sequential(
            ConvGNAct(c3, c4, 3, 2), GhostHGBlock(c4, c4), StarBlock(c4)
        )
        self.stage5 = nn.Sequential(
            ConvGNAct(c4, c5, 3, 2), GhostHGBlock(c5, c5, 2), StarBlock(c5)
        )
        stage_to_ch = {8: c3, 16: c4, 32: c5}
        self.out_channels = tuple(stage_to_ch[s] for s in self.out_strides)

    def forward(self, inputs, depth_type=None, depth_valid=None):
        if self.input_adapters is not None:
            inputs = self.input_adapters(inputs, depth_type, depth_valid)
        features = self.stage2(self.stem(inputs))
        p3 = self.stage3(features)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        outputs = []
        for s in self.out_strides:
            outputs.append({8: p3, 16: p4, 32: p5}[s])
        return tuple(outputs)
