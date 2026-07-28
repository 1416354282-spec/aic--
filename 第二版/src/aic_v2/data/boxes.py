"""Bounding-box helpers using NumPy arrays."""
from __future__ import annotations

import numpy as np


def _boxes(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return a.reshape(0, 4)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    return a


def xywh_to_xyxy(boxes):
    a = _boxes(boxes).copy()
    out = a.copy()
    out[:, 0] = a[:, 0] - a[:, 2] / 2
    out[:, 1] = a[:, 1] - a[:, 3] / 2
    out[:, 2] = a[:, 0] + a[:, 2] / 2
    out[:, 3] = a[:, 1] + a[:, 3] / 2
    return out


def xyxy_to_xywh(boxes):
    a = _boxes(boxes).copy()
    out = a.copy()
    out[:, 0] = (a[:, 0] + a[:, 2]) / 2
    out[:, 1] = (a[:, 1] + a[:, 3]) / 2
    out[:, 2] = a[:, 2] - a[:, 0]
    out[:, 3] = a[:, 3] - a[:, 1]
    return out


def clip_boxes(boxes, width, height):
    out = _boxes(boxes).copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, float(width))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, float(height))
    return out


def box_area(boxes):
    a = _boxes(boxes)
    return np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])


def valid_boxes(boxes, min_size=1.0, min_area=1.0):
    a = _boxes(boxes)
    w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    return (w >= min_size) & (h >= min_size) & (w * h >= min_area) & np.isfinite(a).all(axis=1)


def box_iou(boxes1, boxes2, eps=1e-7):
    a, b = _boxes(boxes1), _boxes(boxes2)
    inter_wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter + eps)


def cxcywh_to_xyxy(boxes):
    return xywh_to_xyxy(boxes)


def xyxy_to_cxcywh(boxes):
    return xyxy_to_xywh(boxes)


def clip_labels(labels, min_size=1e-4, min_visible=0.1):
    labels = np.asarray(labels, dtype=np.float32)
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xyxy = xywh_to_xyxy(labels[:, 1:5])
    original = box_area(xyxy)
    clipped = clip_boxes(xyxy, 1.0, 1.0)
    visible = box_area(clipped) / np.maximum(original, 1e-12)
    wh = clipped[:, 2:4] - clipped[:, 0:2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size) & (visible >= min_visible)
    return np.column_stack((labels[keep, 0], xyxy_to_xywh(clipped[keep]))).astype(np.float32)


def scale_boxes(boxes, scale_x=1.0, scale_y=None, offset_x=0.0, offset_y=0.0):
    scale_y = scale_x if scale_y is None else scale_y
    out = _boxes(boxes).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * scale_x + offset_x
    out[:, [1, 3]] = out[:, [1, 3]] * scale_y + offset_y
    return out
