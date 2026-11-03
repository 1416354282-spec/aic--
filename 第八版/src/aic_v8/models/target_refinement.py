from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct


class TargetAwareRefiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels // 4)
        self.features = nn.Sequential(ConvGNAct(channels, hidden, 3), ConvGNAct(hidden, hidden, 3))
        self.mask = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)
        self.refinement_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        logits = self.mask(self.features(inputs))
        refined = inputs * (1 + torch.tanh(self.refinement_scale) * torch.tanh(logits))
        return refined, logits


def build_soft_box_targets(batch_indices, boxes, batch_size, feature_shapes, device, dtype, sigma=0.35):
    targets = [torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype) for height, width in feature_shapes]
    if boxes.numel() == 0:
        return targets
    for level, (height, width) in enumerate(feature_shapes):
        grid_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        grid_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        for image_index, box in zip(batch_indices.tolist(), boxes.to(dtype)):
            center_x, center_y, box_width, box_height = box
            if box_width <= 0 or box_height <= 0:
                continue
            # At least one grid center per axis must support a sub-cell target.
            inside = (xx - center_x).abs() <= box_width.clamp_min(1 / width) / 2
            inside &= (yy - center_y).abs() <= box_height.clamp_min(1 / height) / 2
            norm_x = (xx - center_x) / (box_width.clamp_min(1 / width) * sigma)
            norm_y = (yy - center_y) / (box_height.clamp_min(1 / height) * sigma)
            soft = torch.exp(-0.5 * (norm_x.square() + norm_y.square())) * inside
            targets[level][image_index, 0] = torch.maximum(targets[level][image_index, 0], soft)
    return targets


def target_mask_loss(logits, targets, gamma=2.0, dice_weight=1.0):
    if len(logits) != len(targets):
        raise ValueError("mask logits 与 target 层数不一致")
    losses = []
    for prediction, target in zip(logits, targets):
        bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        probability = prediction.sigmoid()
        probability_target = probability * target + (1 - probability) * (1 - target)
        focal = ((1 - probability_target).pow(gamma) * bce).mean()
        intersection = (probability * target).sum((1, 2, 3))
        denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
        dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
        losses.append(focal + dice_weight * dice)
    return torch.stack(losses).mean()
