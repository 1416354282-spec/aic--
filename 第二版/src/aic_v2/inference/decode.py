from __future__ import annotations

import torch
import torch.nn.functional as F


def make_anchors(feature_shapes, strides, device):
    anchors, stride_values = [], []
    for (height, width), stride in zip(feature_shapes, strides):
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        anchors.append(torch.stack(((x.flatten() + 0.5) * stride, (y.flatten() + 0.5) * stride), 1))
        stride_values.append(torch.full((height * width, 1), float(stride), device=device))
    return torch.cat(anchors), torch.cat(stride_values)


def unpack_raw_predictions(predictions, reg_max, num_classes):
    if isinstance(predictions, dict) and "boxes" in predictions: return predictions["boxes"], predictions["scores"]
    if isinstance(predictions, (tuple, list)):
        for item in predictions:
            if isinstance(item, dict) and "boxes" in item: return item["boxes"], item["scores"]
        tensors = [item for item in predictions if torch.is_tensor(item) and item.ndim == 3]
        if not tensors: raise TypeError("无法识别模型输出")
        predictions = torch.cat(tensors, -1)
    return predictions[:, :4 * reg_max], predictions[:, 4 * reg_max:4 * reg_max + num_classes]


def decode_dfl(box_logits, class_logits, anchors, stride_tensor, reg_max=16):
    batch, _, count = box_logits.shape
    if anchors.shape[0] != count: raise ValueError(f"anchor 数 {anchors.shape[0]} 与预测数 {count} 不一致")
    distribution = box_logits.reshape(batch, 4, reg_max, count).softmax(2)
    bins = torch.arange(reg_max, device=box_logits.device, dtype=box_logits.dtype).reshape(1, 1, -1, 1)
    distances = (distribution * bins).sum(2) * stride_tensor.T.unsqueeze(0)
    left_top, right_bottom = distances.chunk(2, 1)
    anchor = anchors.T.unsqueeze(0)
    boxes = torch.cat((anchor - left_top, anchor + right_bottom), 1).permute(0, 2, 1)
    return boxes, class_logits.sigmoid().permute(0, 2, 1)
