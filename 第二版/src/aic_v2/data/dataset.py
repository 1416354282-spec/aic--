from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .boxes import clip_labels
from .manifest import find_image, read_labels
from .transforms import MultiModalTransform


def _imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """跨平台图片读取，兼容中文路径（cv2.imread 不支持非 ASCII 路径）。"""
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"无法解码图片: {path}")
    return image


class MultiModalDataset(Dataset):
    collate_fn = staticmethod(lambda batch: collate_fn(batch))

    def __init__(self, root: str, ids: list[str] | None, size: int, train: bool, data_config: dict, augmentation: dict | None = None):
        self.root, self.size, self.train = Path(root), size, train
        label_dir = self.root / "labels"
        self.ids = ids or (sorted(p.stem for p in label_dir.glob("*.txt")) if label_dir.exists() else sorted(p.stem for p in (self.root / "visible").glob("*")))
        self.cfg = data_config
        self.modalities = data_config.get("modalities", ["visible", "infrared", "depth"])
        self.transform = MultiModalTransform(size, train, augmentation)

    def __len__(self): return len(self.ids)

    def _load(self, image_id: str, modality: str):
        path = find_image(self.root / modality, image_id)
        if path is None: raise FileNotFoundError(f"{modality}/{image_id}")
        if modality == "visible":
            return cv2.cvtColor(_imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if modality == "infrared":
            image = _imread(path, cv2.IMREAD_COLOR)
            mode = self.cfg.get("infrared_mode", "rgb")
            if mode == "first_channel": return image[:, :, 0]
            if mode == "mean": return image.astype(np.float32).mean(2).astype(np.uint8)
            return image  # "rgb": 保留三通道
        raw = _imread(path, cv2.IMREAD_UNCHANGED)
        if raw.dtype == np.uint16:
            valid = ((raw >= 300) & (raw < 20000)).astype(np.float32)
            depth = raw.astype(np.float32)
            mode = self.cfg.get("depth_mode", "linear")
            if mode == "log": depth = np.log1p(depth) / np.log1p(20000)
            elif mode == "inverse": depth = np.where(valid > 0, 300 / np.maximum(depth, 300), 0)
            else: depth = depth / 20000
            return (depth * valid).astype(np.float32), valid
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        if self.cfg.get("jpg_depth_mode") == "relative_8bit": return gray.astype(np.float32) / 255, np.ones_like(gray, np.float32)
        return np.zeros_like(gray, np.float32), np.zeros_like(gray, np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.ids[index]; images = {}
        for modality in self.modalities:
            loaded = self._load(image_id, modality)
            if modality == "depth":
                images["depth"], valid = loaded
                if self.cfg.get("add_depth_validity_mask", False): images["depth_valid"] = valid
            else: images[modality] = loaded
        labels, _ = read_labels(self.root / "labels" / f"{image_id}.txt")
        labels = clip_labels(labels)
        images, labels, meta = self.transform(images, labels)
        channels = []
        for key in ["visible", "infrared", "depth", "depth_valid"]:
            if key not in images: continue
            image = images[key]
            if image.ndim == 2: image = image[..., None]
            if image.dtype == np.uint8: image = image.astype(np.float32) / 255
            channels.append(image.transpose(2, 0, 1))
        return {"img": torch.from_numpy(np.ascontiguousarray(np.concatenate(channels))).float(), "labels": labels, "image_id": image_id, "meta": meta}


def collate_fn(batch):
    images = torch.stack([x["img"] for x in batch]); cls, boxes, batch_idx = [], [], []
    for i, item in enumerate(batch):
        for label in item["labels"]:
            cls.append([label[0]]); boxes.append(label[1:]); batch_idx.append(i)
    return {"img": images, "cls": torch.tensor(cls, dtype=torch.float32).reshape(-1, 1), "bboxes": torch.tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4), "batch_idx": torch.tensor(batch_idx, dtype=torch.long), "image_ids": [x["image_id"] for x in batch], "meta": [x["meta"] for x in batch]}
