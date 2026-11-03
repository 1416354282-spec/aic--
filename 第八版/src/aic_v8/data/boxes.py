from __future__ import annotations

import numpy as np


def _boxes(value) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    return boxes


def xywh_to_xyxy(boxes):
    boxes = _boxes(boxes)
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def xyxy_to_xywh(boxes):
    boxes = _boxes(boxes)
    output = boxes.copy()
    output[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    output[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    output[:, 2] = boxes[:, 2] - boxes[:, 0]
    output[:, 3] = boxes[:, 3] - boxes[:, 1]
    return output


def box_area(boxes):
    boxes = _boxes(boxes)
    return np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2, eps=1e-7):
    first, second = _boxes(boxes1), _boxes(boxes2)
    intersection_size = np.maximum(
        0,
        np.minimum(first[:, None, 2:], second[None, :, 2:]) - np.maximum(first[:, None, :2], second[None, :, :2]),
    )
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return intersection / (union + eps)


def clip_labels(labels, min_size=1e-4, min_visible=0.1):
    labels = np.asarray(labels, dtype=np.float32)
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xyxy = xywh_to_xyxy(labels[:, 1:5])
    original_area = box_area(xyxy)
    clipped = xyxy.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, 1)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, 1)
    visible = box_area(clipped) / np.maximum(original_area, 1e-12)
    wh = clipped[:, 2:4] - clipped[:, 0:2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size) & (visible >= min_visible)
    return np.column_stack((labels[keep, 0], xyxy_to_xywh(clipped[keep]))).astype(np.float32)


cxcywh_to_xyxy = xywh_to_xyxy
xyxy_to_cxcywh = xyxy_to_xywh
