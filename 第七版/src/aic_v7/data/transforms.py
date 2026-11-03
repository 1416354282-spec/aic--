"""Multi-modal transforms for v5.

Changes vs v3:
  - letterbox targets an arbitrary (w, h) divisible by 32 (RF-DETR constraint,
    16:9 aspect preferred) instead of a square.
  - modality dropout is conditional (CoLA-style): in dark frames the auxiliary
    modality drop probability is halved; quality-0 pixels propagate to the
    CMTF quality prior through the returned ``meta``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .boxes import clip_labels, cxcywh_to_xyxy, xyxy_to_cxcywh


DEFAULT_NEAREST_KEYS = {"depth", "depth_valid", "depth_format"}


@dataclass
class LetterboxMeta:
    original_hw: tuple[int, int]
    output_hw: tuple[int, int]
    ratio: float
    pad_xy: tuple[float, float]
    infrared_dropped: bool = False
    depth_dropped: bool = False


def letterbox(images: Dict[str, np.ndarray], labels: np.ndarray, target_size: tuple[int, int], nearest_keys: set = None):
    if nearest_keys is None:
        nearest_keys = DEFAULT_NEAREST_KEYS
    size_w, size_h = target_size
    reference = next(iter(images.values()))
    height, width = reference.shape[:2]
    ratio = min(size_w / width, size_h / height)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    left, top = (size_w - new_width) // 2, (size_h - new_height) // 2
    result = {}
    for key, image in images.items():
        interpolation = cv2.INTER_NEAREST if key in nearest_keys else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        shape = (size_h, size_w) if resized.ndim == 2 else (size_h, size_w, resized.shape[2])
        canvas = np.zeros(shape, dtype=resized.dtype)
        canvas[top:top + new_height, left:left + new_width] = resized
        result[key] = canvas
    output = labels.copy()
    if len(output):
        xyxy = cxcywh_to_xyxy(output[:, 1:5]) * np.array([width, height, width, height], np.float32)
        xyxy = xyxy * ratio + np.array([left, top, left, top], np.float32)
        output[:, 1:5] = xyxy_to_cxcywh(xyxy / np.array([size_w, size_h, size_w, size_h], np.float32))
        output = clip_labels(output)
    return result, output, LetterboxMeta((height, width), (size_h, size_w), ratio, (left, top))


def synchronized_affine(images: Dict[str, np.ndarray], labels: np.ndarray, scale: float, translate: float, nearest_keys: set = None):
    if nearest_keys is None:
        nearest_keys = DEFAULT_NEAREST_KEYS
    reference = next(iter(images.values()))
    height, width = reference.shape[:2]
    factor = random.uniform(1 - scale, 1 + scale)
    tx = random.uniform(-translate, translate) * width
    ty = random.uniform(-translate, translate) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, factor)
    matrix[:, 2] += (tx, ty)
    result = {}
    for key, image in images.items():
        interpolation = cv2.INTER_NEAREST if key in nearest_keys else cv2.INTER_LINEAR
        result[key] = cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderValue=0)
    output = labels.copy()
    if len(output):
        xyxy = cxcywh_to_xyxy(output[:, 1:5]) * np.array([width, height, width, height], np.float32)
        corners = np.stack((xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]), 1)
        transformed = corners @ matrix[:, :2].T + matrix[:, 2]
        transformed_xyxy = np.column_stack((
            transformed[:, :, 0].min(1), transformed[:, :, 1].min(1),
            transformed[:, :, 0].max(1), transformed[:, :, 1].max(1),
        ))
        output[:, 1:5] = xyxy_to_cxcywh(transformed_xyxy / np.array([width, height, width, height], np.float32))
        output = clip_labels(output, min_visible=0.2)
    return result, output


class MultiModalTransform:
    """v5 transform: flip / affine / conditional modality dropout / 16:9 letterbox."""

    def __init__(self, size: tuple[int, int], train: bool, config: dict | None = None, nearest_keys: set = None):
        self.size = size
        self.train = train
        self.config = config or {}
        self.nearest_keys = nearest_keys or DEFAULT_NEAREST_KEYS

    def __call__(self, images: Dict[str, np.ndarray], labels: np.ndarray):
        images = {key: value.copy() for key, value in images.items()}
        labels = clip_labels(labels)
        meta = LetterboxMeta((0, 0), self.size, 1.0, (0, 0))
        if self.train:
            if random.random() < self.config.get("horizontal_flip", 0.5):
                images = {key: cv2.flip(value, 1) for key, value in images.items()}
                labels[:, 1] = 1 - labels[:, 1]
            if random.random() < self.config.get("affine_probability", 0.33):
                images, labels = synchronized_affine(
                    images, labels, self.config.get("scale", 0.2), self.config.get("translate", 0.1), self.nearest_keys
                )
            # Conditional modality dropout (CoLA-style): dark frames keep the
            # auxiliary modalities more often so IR/depth stay informative.
            brightness = float(images["visible"].mean()) / 255.0
            dark = brightness < self.config.get("dark_brightness", 0.35)
            ir_factor = 0.5 if dark else 1.0
            depth_factor = 0.5 if dark else 1.0
            if random.random() < self.config.get("infrared_dropout", 0.1) * ir_factor:
                images["infrared"].fill(0)
                meta.infrared_dropped = True
            if random.random() < self.config.get("depth_dropout", 0.1) * depth_factor:
                for key in ("depth", "depth_valid", "depth_format", "depth_h", "depth_ha", "depth_a"):
                    if key in images:
                        images[key].fill(0)
                meta.depth_dropped = True
        images, labels, letterbox_meta = letterbox(images, labels, self.size, self.nearest_keys)
        meta.original_hw, meta.ratio, meta.pad_xy = letterbox_meta.original_hw, letterbox_meta.ratio, letterbox_meta.pad_xy
        return images, labels, meta
