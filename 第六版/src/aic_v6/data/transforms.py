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
    # 真实模态有效率, 在 letterbox **之前** 统计: 填充的黑边不应计入分母,
    # 否则 640×640 方形阶段 43.8% 的零填充会把 q_depth 人为压低一半。
    ir_quality: float = 1.0
    depth_quality: float = 1.0


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


def synchronized_photometric(images: Dict[str, np.ndarray], brightness: float = 0.0,
                             contrast: float = 0.0, gamma: float = 0.0,
                             keys: tuple[str, ...] = ("visible", "infrared")):
    """Illumination jitter applied with the *same* parameters to RGB and IR.

    Cross-modal consistency matters here: if RGB and IR are brightened
    independently the fusion branch learns a spurious alignment that does not
    exist at test time.  Depth channels are geometric encodings (HHA values),
    so they are deliberately left untouched.

    ``brightness`` is an additive offset in [0, 1] units, ``contrast`` a
    multiplicative factor deviation from 1, ``gamma`` a power deviation from 1.
    All three are drawn once per sample and shared by every key.
    """
    if brightness <= 0 and contrast <= 0 and gamma <= 0:
        return images
    delta = random.uniform(-brightness, brightness) if brightness > 0 else 0.0
    factor = random.uniform(1.0 - contrast, 1.0 + contrast) if contrast > 0 else 1.0
    power = random.uniform(1.0 - gamma, 1.0 + gamma) if gamma > 0 else 1.0
    for key in keys:
        image = images.get(key)
        if image is None or image.dtype != np.uint8:
            continue
        scaled = image.astype(np.float32) / 255.0
        scaled = np.clip((scaled - 0.5) * factor + 0.5 + delta, 0.0, 1.0)
        if abs(power - 1.0) > 1e-6:
            scaled = np.power(scaled, power)
        images[key] = np.clip(scaled * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return images


def synchronized_erasing(images: Dict[str, np.ndarray], labels: np.ndarray,
                         scale_range: tuple[float, float] = (0.02, 0.15),
                         keys: tuple[str, ...] = ("visible", "infrared"),
                         depth_keys: tuple[str, ...] = ("depth", "depth_valid", "depth_h", "depth_ha", "depth_a")):
    """Random-erasing (cutout) that keeps all modalities consistent.

    A rectangle is erased in RGB and IR and the matching depth pixels are
    zeroed (HHA zero == invalid, raw ``depth_valid`` zero == missing) so the
    depth branch sees "no depth here" rather than a hole in the RGB scene.
    Rectangles containing a GT centre are rejected, which matters for this
    dataset: 40% of the boxes are <=64px and an unlucky cutout can delete a
    whole small object.
    """
    reference = next(iter(images.values()))
    height, width = reference.shape[:2]
    area = height * width
    low, high = float(scale_range[0]), float(scale_range[1])
    boxing = labels if labels is not None and len(labels) else None
    for _ in range(10):
        target_area = random.uniform(low, high) * area
        aspect = random.uniform(0.5, 2.0)
        erase_h = int(round((target_area / aspect) ** 0.5))
        erase_w = int(round((target_area * aspect) ** 0.5))
        if erase_h < 4 or erase_w < 4 or erase_h >= height or erase_w >= width:
            continue
        x0 = random.randint(0, width - erase_w)
        y0 = random.randint(0, height - erase_h)
        if boxing is not None:
            cx = boxing[:, 1] * width
            cy = boxing[:, 2] * height
            inside = (cx >= x0) & (cx <= x0 + erase_w) & (cy >= y0) & (cy <= y0 + erase_h)
            if bool(inside.any()):
                continue
        break
    else:
        return images
    for key in keys:
        if key in images:
            images[key][y0:y0 + erase_h, x0:x0 + erase_w] = 0
    for key in depth_keys:
        if key in images:
            images[key][y0:y0 + erase_h, x0:x0 + erase_w] = 0
    return images


def _mosaic_crop_size(height: int, width: int, target_aspect: float,
                      crop_scale: tuple[float, float]):
    """Pick a source crop with the quadrant aspect ratio; cap by source size."""
    crop_w = random.uniform(crop_scale[0], crop_scale[1]) * width
    crop_h = crop_w / target_aspect
    if crop_h > height:
        crop_h = float(height)
        crop_w = crop_h * target_aspect
    return max(crop_w, 2.0), max(crop_h, 2.0)


def mosaic4(samples, target_size: tuple[int, int], nearest_keys: set | None = None,
            crop_scale: tuple[float, float] = (0.35, 1.0), min_visible: float = 0.3):
    """Compose four samples into one 2x2 canvas at ``target_size``.

    This is the classic YOLO mosaic adapted to the tri-modal pipeline: every
    source gets an independent random crop, is resized into its quadrant and
    its boxes are re-projected into canvas coordinates.  Each source therefore
    contributes both a random scale and a random translation, which is exactly
    the scale diversity the regression head is missing (see
    第六版小目标专项改进方案.md §3.2).

    Unlike a plain crop, mosaic keeps the total number of objects per image
    high instead of dropping boxes, so the 12-class-equal-weight objective
    still sees the rare/small classes.

    ``samples`` is a list of ``(images_dict, labels)`` pairs (labels in
    normalized cxcywh, 5 columns).  Depth channels are composed the same way;
    seams between sources are not blended (standard mosaic behaviour).
    """
    if nearest_keys is None:
        nearest_keys = DEFAULT_NEAREST_KEYS
    size_w, size_h = int(target_size[0]), int(target_size[1])
    half_w, half_h = size_w // 2, size_h // 2
    quadrants = (
        (0, 0, half_w, half_h),
        (half_w, 0, size_w - half_w, half_h),
        (0, half_h, half_w, size_h - half_h),
        (half_w, half_h, size_w - half_w, size_h - half_h),
    )
    canvas: Dict[str, np.ndarray] = {}
    projected: list[np.ndarray] = []
    for (images, labels), (ox, oy, qw, qh) in zip(samples, quadrants):
        reference = next(iter(images.values()))
        height, width = reference.shape[:2]
        crop_w, crop_h = _mosaic_crop_size(height, width, qw / max(qh, 1), crop_scale)
        x0 = random.uniform(0.0, max(width - crop_w, 0.0))
        y0 = random.uniform(0.0, max(height - crop_h, 0.0))
        x1, y1 = x0 + crop_w, y0 + crop_h
        for key, image in images.items():
            interpolation = cv2.INTER_NEAREST if key in nearest_keys else cv2.INTER_LINEAR
            ix0, iy0 = int(round(x0)), int(round(y0))
            ix1, iy1 = min(int(round(x1)), width), min(int(round(y1)), height)
            crop = image[iy0:iy1, ix0:ix1]
            if crop.size == 0:
                crop = image
                ix0, iy0, ix1, iy1 = 0, 0, width, height
            resized = cv2.resize(crop, (qw, qh), interpolation=interpolation)
            if key not in canvas:
                shape = (size_h, size_w) if resized.ndim == 2 else (size_h, size_w, resized.shape[2])
                canvas[key] = np.zeros(shape, dtype=resized.dtype)
            canvas[key][oy:oy + qh, ox:ox + qw] = resized
        if labels is None or not len(labels):
            continue
        # source pixel xyxy -> quadrant crop pixels -> canvas pixels
        xyxy = cxcywh_to_xyxy(labels[:, 1:5]) * np.array([width, height, width, height], np.float32)
        scale_x, scale_y = qw / max(ix1 - ix0, 1), qh / max(iy1 - iy0, 1)
        mapped = np.column_stack((
            (xyxy[:, 0] - ix0) * scale_x + ox,
            (xyxy[:, 1] - iy0) * scale_y + oy,
            (xyxy[:, 2] - ix0) * scale_x + ox,
            (xyxy[:, 3] - iy0) * scale_y + oy,
        ))
        clipped = np.column_stack((
            np.clip(mapped[:, 0], ox, ox + qw), np.clip(mapped[:, 1], oy, oy + qh),
            np.clip(mapped[:, 2], ox, ox + qw), np.clip(mapped[:, 3], oy, oy + qh),
        ))
        mapped_area = (mapped[:, 2] - mapped[:, 0]) * (mapped[:, 3] - mapped[:, 1])
        visible = ((clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])) / np.maximum(mapped_area, 1e-6)
        keep = (visible >= min_visible) & (clipped[:, 2] - clipped[:, 0] >= 2) & (clipped[:, 3] - clipped[:, 1] >= 2)
        if not keep.any():
            continue
        normalized = clipped[keep] / np.array([size_w, size_h, size_w, size_h], np.float32)
        projected.append(np.column_stack((labels[keep, 0], xyxy_to_cxcywh(normalized))))
    if projected:
        labels = clip_labels(np.concatenate(projected, axis=0), min_visible=min_visible)
    else:
        labels = np.zeros((0, 5), dtype=np.float32)
    return canvas, labels


class MultiModalTransform:
    """flip / affine / photometric / erasing / conditional dropout / letterbox."""

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
            # Photometric jitter uses the same draw for RGB and IR (see the
            # function docstring); depth is a geometric encoding and is skipped.
            if random.random() < self.config.get("photometric_probability", 0.0):
                images = synchronized_photometric(
                    images,
                    brightness=float(self.config.get("brightness", 0.0)),
                    contrast=float(self.config.get("contrast", 0.0)),
                    gamma=float(self.config.get("gamma", 0.0)),
                )
            if random.random() < self.config.get("erasing_probability", 0.0):
                images = synchronized_erasing(
                    images, labels, self.config.get("erasing_scale", (0.02, 0.15)),
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
        # Quality is measured on the unpadded image: after letterbox, black
        # padding would otherwise be counted as "missing IR / missing depth".
        ir_quality = float((images["infrared"] > 0).mean()) if "infrared" in images else 1.0
        depth_mask = images.get("depth_valid")
        depth_quality = float(np.asarray(depth_mask).mean()) if depth_mask is not None else 1.0
        images, labels, letterbox_meta = letterbox(images, labels, self.size, self.nearest_keys)
        meta.original_hw, meta.ratio, meta.pad_xy = letterbox_meta.original_hw, letterbox_meta.ratio, letterbox_meta.pad_xy
        meta.ir_quality = ir_quality
        meta.depth_quality = depth_quality
        return images, labels, meta
