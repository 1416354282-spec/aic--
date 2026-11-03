"""v5 dataset: dict input {img, ir, depth, depth_valid} + DETR-style targets.

Modality keys follow v3's HHA/raw channel conventions; the collate function
builds the per-image target lists expected by the RF-DETR SetCriterion
(boxes in normalized cxcywh, labels as float class ids).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import clip_labels
from .depth_encoding import depth_to_3channel_raw, hha_encode, jpg_depth_encode
from .manifest import find_image, read_labels
from .transforms import MultiModalTransform


CHANNEL_ORDER_HHA = ("visible", "infrared", "depth_h", "depth_ha", "depth_a")
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
        # v7: 标注子目录名可配置 (本机为 new_labels_2000, 远程沿用 labels)
        self.label_dir = self.root / data_config.get("labels_subfolder", "labels")
        if ids is not None:
            self.ids = ids
        elif self.label_dir.exists():
            self.ids = sorted(path.stem for path in self.label_dir.glob("*.txt"))
        else:
            self.ids = sorted(path.stem for path in (self.root / "visible").glob("*") if path.is_file())
        limit = int(data_config.get("limit", 0))
        if limit > 0:
            self.ids = self.ids[:limit]
        depth_encoding = data_config.get("depth_encoding", "hha")
        if depth_encoding == "hha":
            self._channel_order = CHANNEL_ORDER_HHA
            self._nearest_keys = set()  # HHA 三个通道都是图像, 线性插值
        else:
            self._channel_order = CHANNEL_ORDER_RAW
            self._nearest_keys = {"depth", "depth_valid", "depth_format"}
        # include_originals: 训练集 = 原图(仅 letterbox, 无随机增强) + 增强图, 物理翻倍
        # augment_ratio: 增强图数量 = 原图数量 × ratio (1.0=全量翻倍, 0.5=增强图减半)
        self.include_originals = bool(data_config.get("include_originals", False))
        self.augment_ratio = float(data_config.get("augment_ratio", 1.0))
        self.augment_count = int(len(self.ids) * self.augment_ratio) if self.include_originals and train else 0
        self.transform = MultiModalTransform(size, train, augmentation, self._nearest_keys)
        if self.include_originals and train:
            self.transform_plain = MultiModalTransform(size, False, {}, self._nearest_keys)

    def set_size(self, size: tuple[int, int]) -> None:
        """Switch the letterbox target (multi-resolution training per epoch)."""
        self.size = size
        self.transform.size = size
        if getattr(self, "transform_plain", None):
            self.transform_plain.size = size

    def __len__(self):
        if self.include_originals and self.train:
            return len(self.ids) + self.augment_count  # 原图 + 部分增强图
        return len(self.ids)

    def _path(self, image_id: str, modality: str) -> Path:
        path = find_image(self.root / modality, image_id)
        if path is None:
            raise FileNotFoundError(f"{modality}/{image_id}")
        return path

    def _load_depth(self, image_id: str):
        path = self._path(image_id, "depth")
        raw = _imread(path, cv2.IMREAD_UNCHANGED)
        encoding = self.config.get("depth_encoding", "hha")

        # ── PNG: uint16 度量深度 ──
        if raw.dtype == np.uint16:
            if encoding == "hha":
                hha = hha_encode(raw)
                return hha[:, :, 0].copy(), hha[:, :, 1].copy(), hha[:, :, 2].copy()
            return depth_to_3channel_raw(raw, mode=self.config.get("depth_mode", "log"))

        # ── JPG: 8-bit 相对深度 (无度量信息) ──
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        if encoding == "hha":
            jpg = jpg_depth_encode(gray)
            return jpg[:, :, 0].copy(), jpg[:, :, 1].copy(), jpg[:, :, 2].copy()
        rel = gray.astype(np.float32) / 255
        ones = np.ones_like(rel, np.float32)
        zeros = np.zeros_like(rel, np.float32)
        return rel, ones, zeros

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.include_originals and self.train:
            # 前段 = 原图 (仅 letterbox), 后段 = 增强图 (均匀散布到全部 id)
            if index < len(self.ids):
                image_id = self.ids[index]
                transform = self.transform_plain
            else:
                step = len(self.ids) / max(self.augment_count, 1)
                image_id = self.ids[min(int((index - len(self.ids)) * step), len(self.ids) - 1)]
                transform = self.transform
        else:
            image_id = self.ids[index]
            transform = self.transform
        visible = cv2.cvtColor(_imread(self._path(image_id, "visible")), cv2.COLOR_BGR2RGB)
        infrared_raw = _imread(self._path(image_id, "infrared"))
        infrared = infrared_raw.astype(np.float32).mean(2).astype(np.uint8)
        depth, depth_valid, depth_format = self._load_depth(image_id)
        encoding = self.config.get("depth_encoding", "hha")
        images = {"visible": visible, "infrared": infrared}
        if encoding == "hha":
            images["depth_h"] = depth
            images["depth_ha"] = depth_valid
            images["depth_a"] = depth_format
        else:
            images["depth"] = depth
            images["depth_valid"] = depth_valid
            images["depth_format"] = depth_format
        labels, _ = read_labels(self.label_dir / f"{image_id}.txt")
        images, labels, meta = transform(images, clip_labels(labels))

        channels = {}
        for key in self._channel_order:
            image = images[key]
            if image.ndim == 2:
                image = image[..., None]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
            channels[key] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

        depth_valid = channels["depth_ha" if encoding == "hha" else "depth_valid"]
        depth_key = "depth_h" if encoding == "hha" else "depth"
        if encoding == "hha":
            depth = torch.stack(
                (channels["depth_h"], channels["depth_ha"], channels["depth_a"]), dim=1
            ).squeeze(0)
        else:
            depth = channels["depth"]
        return {
            "img": channels["visible"],
            "ir": channels["infrared"],
            "depth": depth,
            "depth_valid": depth_valid,
            "boxes": torch.tensor(np.asarray(labels[:, 1:5], np.float32)).reshape(-1, 4),
            "labels": torch.tensor(np.asarray(labels[:, 0], np.int64)).reshape(-1),
            "num_boxes": len(labels),
            "image_id": image_id,
            "meta": meta,
        }


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
    return {
        "img": images,
        "ir": irs,
        "depth": depths,
        "depth_valid": depth_valids,
        "boxes": boxes,
        "labels": labels,
        "num_boxes": num_boxes,
        "image_ids": [item["image_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
