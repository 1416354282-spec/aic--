"""v5 dataset: dict input {img, ir, depth, depth_valid} + DETR-style targets.

Modality keys follow v3's HHA/raw channel conventions; the collate function
builds the per-image target lists expected by the RF-DETR SetCriterion
(boxes in normalized cxcywh, labels as float class ids).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import clip_labels
from .depth_encoding import depth_to_3channel_raw, hha_encode, jpg_depth_encode
from .manifest import find_image, read_labels
from .transforms import MultiModalTransform, mosaic4


# depth_valid 在 HHA 模式下也是真实有效性掩码（不是 HHA 的 HA 通道）:
# 它只用于质量先验与空间 bias，不进入 depth 的 3 通道堆叠。
CHANNEL_ORDER_HHA = ("visible", "infrared", "depth_h", "depth_ha", "depth_a", "depth_valid")
CHANNEL_ORDER_RAW = ("visible", "infrared", "depth", "depth_valid", "depth_format")


def _imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"无法解码图片: {path}")
    return image


class MultiModalDataset(Dataset):
    collate_fn = staticmethod(lambda batch: collate_fn(batch))

    def __init__(
        self,
        root: str,
        ids: list[str] | None,
        size: tuple[int, int],
        train: bool,
        data_config: dict,
        augmentation: dict | None = None,
    ):
        self.root = Path(root)
        self.size = size
        self.train = train
        self.config = data_config
        # 标注目录名可配置: 官方 zip 解压出来是 labels/, 本地修订版是 new_labels_2000/。
        # 默认 "labels" 与既有行为一致; 指向不存在的目录会让 GT 静默变空 (只影响训练/验证,
        # 推理不需要标注), 所以这里显式告警而不是等 mAP 莫名归零。
        self.label_dir = self.root / str(data_config.get("labels_subfolder", "labels"))
        label_dir = self.label_dir
        if not label_dir.exists():
            print(f"[WARNING] 标注目录不存在: {label_dir} → GT 全空 (训练/验证会学不到东西; 推理无影响)")
        if ids is not None:
            self.ids = ids
        elif label_dir.exists():
            self.ids = sorted(path.stem for path in label_dir.glob("*.txt"))
        else:
            self.ids = sorted(path.stem for path in (self.root / "visible").glob("*") if path.is_file())
        depth_encoding = data_config.get("depth_encoding", "hha")
        if depth_encoding == "hha":
            self._channel_order = CHANNEL_ORDER_HHA
            # HHA 三个通道是连续几何量 → 线性插值; 有效性掩码是二值的 → 最近邻
            self._nearest_keys = {"depth_valid"}
        else:
            self._channel_order = CHANNEL_ORDER_RAW
            self._nearest_keys = {"depth", "depth_valid", "depth_format"}
        # include_originals: 训练集 = 原图(仅 letterbox, 无随机增强) + 增强图, 物理翻倍
        # augment_ratio: 增强图数量 = 原图数量 × ratio (1.0=全量翻倍, 0.5=增强图减半)
        # augment_repeats: include_originals=false 时每条图每 epoch 被随机增强几次
        #   (只影响 epoch 长度, 用来在"全增强"下保住与 include_originals=true 相同的更新数)
        self.include_originals = bool(data_config.get("include_originals", False))
        self.augment_ratio = float(data_config.get("augment_ratio", 1.0))
        self.augment_repeats = max(1, int(data_config.get("augment_repeats", 1)))
        self.augment_count = int(len(self.ids) * self.augment_ratio) if self.include_originals and train else 0
        self.augmentation = dict(augmentation or {})
        self.transform = MultiModalTransform(size, train, augmentation, self._nearest_keys)
        if self.include_originals and train:
            self.transform_plain = MultiModalTransform(size, False, {}, self._nearest_keys)
        # mosaic 的关闭调度: trainer 每 epoch 调 set_epoch(), 在最后 N 个 epoch 关掉 mosaic。
        self.epoch = 0
        self.total_epochs = None
        self.phase = None

    def set_size(self, size: tuple[int, int]) -> None:
        """Switch the letterbox target (multi-resolution training per epoch)."""
        self.size = size
        self.transform.size = size
        if getattr(self, "transform_plain", None):
            self.transform_plain.size = size

    def __len__(self):
        if self.include_originals and self.train:
            return len(self.ids) + self.augment_count  # 原图 + 部分增强图
        if self.train:
            return len(self.ids) * self.augment_repeats
        return len(self.ids)

    def set_epoch(self, epoch: int, total_epochs: int | None = None, phase: int | None = None) -> None:
        """Update the counters used by the close-mosaic / min-phase schedule."""
        self.epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)
        if phase is not None:
            self.phase = int(phase)

    def _mosaic_probability(self) -> float:
        probability = float(self.augmentation.get("mosaic_probability", 0.0))
        if probability <= 0:
            return 0.0
        # Phase 1 runs at 640×384; a mosaic quadrant there would shrink a 24px
        # uav to ~12px, below the stride-8 grid.  Let the config gate mosaic to
        # the higher-resolution phases instead of guessing per dataset.
        min_phase = int(self.augmentation.get("mosaic_min_phase", 1))
        if self.phase is not None and self.phase < min_phase:
            return 0.0
        close_epochs = int(self.augmentation.get("mosaic_close_epochs", 0))
        if self.total_epochs is not None and close_epochs > 0:
            if self.epoch >= self.total_epochs - close_epochs:
                return 0.0
        return probability

    def _path(self, image_id: str, modality: str) -> Path:
        path = find_image(self.root / modality, image_id)
        if path is None:
            raise FileNotFoundError(f"{modality}/{image_id}")
        return path

    def _load_depth(self, image_id: str):
        """Read depth and return ``(depth_channels, valid, format)``.

        ``depth_channels`` is what the DepthEncoder consumes: 3 HHA channels,
        or 1 normalized channel in raw mode.  ``valid`` is the **real**
        validity mask (uint16 metric depth outside [300, 20000] mm is invalid;
        8-bit JPG relative depth is dense).  ``format`` is 1 for metric PNG
        and 0 for relative JPG.  Keeping ``valid`` separate fixes the old
        bug where the HHA "height above ground" channel was fed to the
        modality quality scalar as if it were a validity mask.
        """
        path = self._path(image_id, "depth")
        raw = _imread(path, cv2.IMREAD_UNCHANGED)
        encoding = self.config.get("depth_encoding", "hha")

        # ── PNG: uint16 度量深度 ──
        if raw.dtype == np.uint16:
            depth_f = raw.astype(np.float32)
            valid = ((depth_f >= 300.0) & (depth_f < 20000.0)).astype(np.float32)
            if encoding == "hha":
                hha = hha_encode(raw)
                channels = (hha[:, :, 0].copy(), hha[:, :, 1].copy(), hha[:, :, 2].copy())
            else:
                normalized, _, _ = depth_to_3channel_raw(raw, mode=self.config.get("depth_mode", "log"))
                channels = (normalized,)
            return channels, valid, np.ones_like(valid, np.float32)

        # ── JPG: 8-bit 相对深度 (无度量信息) ──
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        # 相对深度图是全图有效的 (0 只是"远", 不代表无效)
        valid = np.ones_like(gray, np.float32)
        if encoding == "hha":
            jpg = jpg_depth_encode(gray)
            channels = (jpg[:, :, 0].copy(), jpg[:, :, 1].copy(), jpg[:, :, 2].copy())
        else:
            channels = (gray.astype(np.float32) / 255,)
        return channels, valid, np.zeros_like(valid, np.float32)

    def _load_raw(self, image_id: str):
        """Read one sample as uint8/float images + normalized cxcywh labels."""
        visible = cv2.cvtColor(_imread(self._path(image_id, "visible")), cv2.COLOR_BGR2RGB)
        infrared_raw = _imread(self._path(image_id, "infrared"))
        infrared = infrared_raw.astype(np.float32).mean(2).astype(np.uint8)
        depth_channels, depth_valid, depth_format = self._load_depth(image_id)
        encoding = self.config.get("depth_encoding", "hha")
        images = {"visible": visible, "infrared": infrared, "depth_valid": depth_valid}
        if encoding == "hha":
            images["depth_h"], images["depth_ha"], images["depth_a"] = depth_channels
        else:
            images["depth"] = depth_channels[0]
            images["depth_format"] = depth_format
        labels, _ = read_labels(self.label_dir / f"{image_id}.txt")
        return images, clip_labels(labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        is_original = False
        if self.include_originals and self.train:
            # 前段 = 原图 (仅 letterbox), 后段 = 增强图 (均匀散布到全部 id)
            if index < len(self.ids):
                image_id = self.ids[index]
                transform = self.transform_plain
                is_original = True
            else:
                step = len(self.ids) / max(self.augment_count, 1)
                image_id = self.ids[min(int((index - len(self.ids)) * step), len(self.ids) - 1)]
                transform = self.transform
        else:
            image_id = self.ids[index % len(self.ids)]
            transform = self.transform
        images, labels = self._load_raw(image_id)
        # Mosaic only touches augmented samples: the "originals" segment keeps
        # its meaning (clean letterbox) for the identity/anchor of the epoch.
        if self.train and not is_original and random.random() < self._mosaic_probability():
            others = [
                self._load_raw(self.ids[random.randrange(len(self.ids))])
                for _ in range(3)
            ]
            images, labels = mosaic4(
                [(images, labels)] + others, self.size, self._nearest_keys,
                crop_scale=tuple(self.augmentation.get("mosaic_crop_scale", (0.35, 1.0))),
            )
        images, labels, meta = transform(images, labels)

        encoding = self.config.get("depth_encoding", "hha")
        channels = {}
        for key in self._channel_order:
            image = images[key]
            if image.ndim == 2:
                image = image[..., None]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
            channels[key] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

        depth_valid = channels["depth_valid"]
        if encoding == "hha":
            depth = torch.stack(
                (channels["depth_h"], channels["depth_ha"], channels["depth_a"]), dim=1
            ).squeeze(0)
        else:
            depth = channels["depth"]
        item = {
            "img": channels["visible"],
            "ir": channels["infrared"],
            "depth": depth,
            "depth_valid": depth_valid,
            "ir_quality": float(meta.ir_quality),
            "depth_quality": float(meta.depth_quality),
            "boxes": torch.tensor(np.asarray(labels[:, 1:5], np.float32)).reshape(-1, 4),
            "labels": torch.tensor(np.asarray(labels[:, 0], np.int64)).reshape(-1),
            "num_boxes": len(labels),
            "image_id": image_id,
            "meta": meta,
        }
        if encoding != "hha":
            item["depth_format"] = channels["depth_format"]
        return item


def collate_fn(batch):
    images = torch.stack([item["img"] for item in batch])
    irs = torch.stack([item["ir"] for item in batch])
    depths = torch.stack([item["depth"] for item in batch])
    depth_valids = torch.stack([item["depth_valid"] for item in batch])
    num_boxes = torch.tensor([item["num_boxes"] for item in batch], dtype=torch.long)
    max_boxes = int(num_boxes.max()) if len(batch) else 0
    boxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    for index, item in enumerate(batch):
        if item["num_boxes"]:
            boxes[index, : item["num_boxes"]] = item["boxes"]
            labels[index, : item["num_boxes"]] = item["labels"]
    collated = {
        "img": images,
        "ir": irs,
        "depth": depths,
        "depth_valid": depth_valids,
        "ir_quality": torch.tensor([item["ir_quality"] for item in batch], dtype=torch.float32),
        "depth_quality": torch.tensor([item["depth_quality"] for item in batch], dtype=torch.float32),
        "boxes": boxes,
        "labels": labels,
        "num_boxes": num_boxes,
        "image_ids": [item["image_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
    if "depth_format" in batch[0]:
        collated["depth_format"] = torch.stack([item["depth_format"] for item in batch])
    return collated
