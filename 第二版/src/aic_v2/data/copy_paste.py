from __future__ import annotations

import random
from typing import Dict, Iterable

import numpy as np

from .boxes import clip_labels, cxcywh_to_xyxy


def rare_class_copy_paste(target_images: Dict[str, np.ndarray], target_labels: np.ndarray, source_images: Dict[str, np.ndarray], source_labels: np.ndarray, rare_classes: Iterable[int], max_instances: int = 2):
    rare = set(int(x) for x in rare_classes)
    candidates = np.flatnonzero(np.isin(source_labels[:, 0].astype(int), list(rare)))
    if not len(candidates):
        return target_images, target_labels
    h, w = next(iter(source_images.values())).shape[:2]
    picks = random.sample(candidates.tolist(), min(max_instances, len(candidates)))
    output = {k: v.copy() for k, v in target_images.items()}
    labels = [target_labels]
    xyxy = cxcywh_to_xyxy(source_labels[:, 1:5])
    for index in picks:
        x1, y1, x2, y2 = (xyxy[index] * [w, h, w, h]).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        for key in output:
            output[key][y1:y2, x1:x2] = source_images[key][y1:y2, x1:x2]
        labels.append(source_labels[index:index + 1])
    return output, clip_labels(np.concatenate(labels))
