from __future__ import annotations

import random
from typing import Dict, Sequence

import cv2
import numpy as np

from .boxes import clip_labels, cxcywh_to_xyxy, xyxy_to_cxcywh


def mosaic4(samples: Sequence[tuple[Dict[str, np.ndarray], np.ndarray]], size: int) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    if len(samples) != 4:
        raise ValueError("mosaic4 必须接收四个样本")
    center_x = random.randint(size // 3, size * 2 // 3)
    center_y = random.randint(size // 3, size * 2 // 3)
    regions = [(0, 0, center_x, center_y), (center_x, 0, size, center_y), (0, center_y, center_x, size), (center_x, center_y, size, size)]
    modalities = samples[0][0].keys()
    output = {}
    for modality in modalities:
        ref = samples[0][0][modality]
        shape = (size, size) if ref.ndim == 2 else (size, size, ref.shape[2])
        output[modality] = np.zeros(shape, dtype=ref.dtype)
    all_labels = []
    for (images, labels), (x1, y1, x2, y2) in zip(samples, regions):
        tw, th = x2 - x1, y2 - y1
        h, w = next(iter(images.values())).shape[:2]
        scale = max(tw / w, th / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        crop_x, crop_y = max(0, (nw - tw) // 2), max(0, (nh - th) // 2)
        for modality, image in images.items():
            interp = cv2.INTER_NEAREST if modality in {"depth", "depth_valid"} else cv2.INTER_LINEAR
            resized = cv2.resize(image, (nw, nh), interpolation=interp)
            output[modality][y1:y2, x1:x2] = resized[crop_y:crop_y + th, crop_x:crop_x + tw]
        if len(labels):
            xyxy = cxcywh_to_xyxy(labels[:, 1:5]) * np.array([w, h, w, h], np.float32)
            xyxy = xyxy * scale + np.array([x1 - crop_x, y1 - crop_y, x1 - crop_x, y1 - crop_y], np.float32)
            converted = labels.copy()
            converted[:, 1:5] = xyxy_to_cxcywh(xyxy / size)
            all_labels.append(converted)
    merged = np.concatenate(all_labels) if all_labels else np.zeros((0, 5), np.float32)
    return output, clip_labels(merged, min_visible=0.2)
