from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .boxes import clip_labels, cxcywh_to_xyxy, xyxy_to_cxcywh


@dataclass
class LetterboxMeta:
    original_hw: tuple[int, int]
    output_hw: tuple[int, int]
    ratio: float
    pad_xy: tuple[float, float]


def letterbox(images: Dict[str, np.ndarray], labels: np.ndarray, size: int) -> tuple[Dict[str, np.ndarray], np.ndarray, LetterboxMeta]:
    ref = next(iter(images.values()))
    h, w = ref.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    result = {}
    for key, image in images.items():
        interpolation = cv2.INTER_NEAREST if key in {"depth", "depth_valid"} else cv2.INTER_LINEAR
        resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
        shape = (size, size) if resized.ndim == 2 else (size, size, resized.shape[2])
        canvas = np.zeros(shape, dtype=resized.dtype)
        canvas[top:top + nh, left:left + nw] = resized
        result[key] = canvas
    output = labels.copy()
    if len(output):
        xyxy = cxcywh_to_xyxy(output[:, 1:5]) * np.array([w, h, w, h], np.float32)
        xyxy = xyxy * ratio + np.array([left, top, left, top], np.float32)
        output[:, 1:5] = xyxy_to_cxcywh(xyxy / size)
        output = clip_labels(output)
    return result, output, LetterboxMeta((h, w), (size, size), ratio, (left, top))


def synchronized_affine(images: Dict[str, np.ndarray], labels: np.ndarray, scale: float, translate: float) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    ref = next(iter(images.values()))
    h, w = ref.shape[:2]
    factor = random.uniform(1 - scale, 1 + scale)
    tx, ty = random.uniform(-translate, translate) * w, random.uniform(-translate, translate) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, factor)
    matrix[:, 2] += (tx, ty)
    result = {}
    for key, image in images.items():
        interp = cv2.INTER_NEAREST if key in {"depth", "depth_valid"} else cv2.INTER_LINEAR
        result[key] = cv2.warpAffine(image, matrix, (w, h), flags=interp, borderValue=0)
    output = labels.copy()
    if len(output):
        xyxy = cxcywh_to_xyxy(output[:, 1:5]) * np.array([w, h, w, h], np.float32)
        corners = np.stack((xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]), axis=1)
        transformed = corners @ matrix[:, :2].T + matrix[:, 2]
        new_xyxy = np.column_stack((transformed[:, :, 0].min(1), transformed[:, :, 1].min(1), transformed[:, :, 0].max(1), transformed[:, :, 1].max(1)))
        output[:, 1:5] = xyxy_to_cxcywh(new_xyxy / np.array([w, h, w, h], np.float32))
        output = clip_labels(output, min_visible=0.2)
    return result, output


class MultiModalTransform:
    def __init__(self, size: int, train: bool, config: dict | None = None):
        self.size, self.train, self.cfg = size, train, config or {}

    def __call__(self, images: Dict[str, np.ndarray], labels: np.ndarray):
        images = {k: v.copy() for k, v in images.items()}
        labels = clip_labels(labels)
        if self.train:
            if random.random() < self.cfg.get("horizontal_flip", 0.5):
                images = {k: cv2.flip(v, 1) for k, v in images.items()}
                labels[:, 1] = 1 - labels[:, 1]
            if random.random() < self.cfg.get("affine_probability", 0.33):
                images, labels = synchronized_affine(images, labels, self.cfg.get("scale", 0.2), self.cfg.get("translate", 0.1))
            if "infrared" in images and random.random() < self.cfg.get("infrared_dropout", 0.0):
                images["infrared"].fill(0)
            if "depth" in images and random.random() < self.cfg.get("depth_dropout", 0.0):
                images["depth"].fill(0)
                if "depth_valid" in images: images["depth_valid"].fill(0)
        return letterbox(images, labels, self.size)
