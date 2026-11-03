"""Generalization-augmentation tests (CPU, no checkpoint needed).

Covers the three new knobs and the invariant that matters most: with the new
probabilities left at their defaults every path is a no-op, so existing runs
stay bit-identical.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.data.dataset import MultiModalDataset
from aic_v6.data.transforms import MultiModalTransform, mosaic4


def _images(height=96, width=128, seed=0):
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return {
        "visible": np.repeat(gray[..., None], 3, axis=2),
        "infrared": gray.copy(),
        "depth_h": rng.random((height, width), dtype=np.float32),
        "depth_ha": rng.random((height, width), dtype=np.float32),
        "depth_a": rng.random((height, width), dtype=np.float32),
    }


def _labels():
    return np.array([[3.0, 0.50, 0.50, 0.10, 0.20]], dtype=np.float32)


def test_defaults_are_a_noop():
    """默认配置（新旋钮全 0）必须与 train=False 的纯 letterbox 逐位相同。"""
    images, labels = _images(), _labels()
    defaults = {
        "horizontal_flip": 0.0, "affine_probability": 0.0,
        "photometric_probability": 0.0, "erasing_probability": 0.0,
        "infrared_dropout": 0.0, "depth_dropout": 0.0,
    }
    augmented, aug_labels, _ = MultiModalTransform((128, 64), True, defaults, set())(images, labels)
    plain, plain_labels, _ = MultiModalTransform((128, 64), False, {}, set())(images, labels)
    for key in images:
        assert np.array_equal(augmented[key], plain[key]), f"{key} 在默认配置下被改动"
    assert np.allclose(aug_labels, plain_labels)
    print("[aug] 默认配置逐位等价于纯 letterbox")


def test_photometric_is_synchronized():
    """RGB 与 IR 必须拿到同一组光度参数；深度通道不得被改动。"""
    images = _images()
    depth_before = {key: value.copy() for key, value in images.items() if key.startswith("depth")}
    config = {
        "horizontal_flip": 0.0, "affine_probability": 0.0,
        "photometric_probability": 1.0, "brightness": 0.25, "contrast": 0.25, "gamma": 0.25,
        "erasing_probability": 0.0, "infrared_dropout": 0.0, "depth_dropout": 0.0,
    }
    random.seed(0)
    out, _, _ = MultiModalTransform((128, 96), True, config, set())(images, _labels())
    assert not np.array_equal(out["visible"], images["visible"]), "光度增强未生效"
    # 输入可见光的三通道与 IR 完全相同 → 同步变换后仍应完全相同
    assert out["visible"][..., 0].shape == out["infrared"].shape
    assert np.array_equal(out["visible"][..., 0], out["infrared"]), "RGB/IR 光度参数不同步"
    for key, value in depth_before.items():
        assert np.array_equal(out[key], value), f"{key} 不应参与光度增强"
    print("[aug] 光度抖动 RGB/IR 同步, 深度不受影响")


def test_erasing_hits_all_modalities():
    """擦除区域必须在 RGB/IR/深度上同时置 0。"""
    images = _images(128, 160, seed=3)
    config = {
        "horizontal_flip": 0.0, "affine_probability": 0.0,
        "photometric_probability": 0.0, "erasing_probability": 1.0,
        "erasing_scale": [0.10, 0.10], "infrared_dropout": 0.0, "depth_dropout": 0.0,
    }
    random.seed(1)
    out, _, _ = MultiModalTransform((160, 128), True, config, set())(images, np.zeros((0, 5), np.float32))
    erased = (out["visible"][..., 0] == 0) & (images["visible"][..., 0] != 0)
    assert erased.any(), "随机擦除未生效"
    assert np.array_equal(out["infrared"] == 0, erased | (images["infrared"] == 0))
    assert (out["depth_h"][erased] == 0).all(), "擦除区域深度未同步置 0"
    assert (out["depth_ha"][erased] == 0).all()
    print(f"[aug] 擦除区域 {int(erased.sum())} px 在 RGB/IR/深度上同步置 0")


def test_mosaic_geometry():
    """4 图 mosaic: 输出为目标尺寸, 框仍落在各自象限且坐标合法。"""
    samples = []
    for index in range(4):
        images = _images(180, 320, seed=index + 10)
        samples.append((images, _labels()))
    random.seed(2)
    images, labels = mosaic4(samples, (640, 384), set(), crop_scale=(1.0, 1.0))
    for key in samples[0][0]:
        expected = (384, 640) if samples[0][0][key].ndim == 2 else (384, 640, 3)
        assert images[key].shape == expected, f"{key} 尺寸异常: {images[key].shape}"
    assert len(labels) == 4, f"4 个目标应全部保留, got {len(labels)}"
    assert labels[:, 1:].min() >= 0 and labels[:, 1:].max() <= 1
    centres = labels[:, 1:3]
    quadrants = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    for row in centres:
        assert (row[0] < 0.5) or (row[0] > 0.5)
        assert (row[1] < 0.5) or (row[1] > 0.5)
    assert len({tuple(v) for v in zip(centres[:, 0] > 0.5, centres[:, 1] > 0.5)}) == 4, \
        "四个框应各占一个象限"
    assert quadrants  # keep the mapping explicit for readers
    print(f"[aug] mosaic 输出 {images['visible'].shape}, 框数 {len(labels)}")


def test_mosaic_close_schedule():
    dataset = object.__new__(MultiModalDataset)
    dataset.augmentation = {
        "mosaic_probability": 0.5, "mosaic_close_epochs": 10, "mosaic_min_phase": 2,
    }
    dataset.total_epochs = 100
    dataset.epoch = 0
    dataset.phase = None
    assert dataset._mosaic_probability() == 0.5
    dataset.set_epoch(89, 100, phase=2)
    assert dataset._mosaic_probability() == 0.5
    dataset.set_epoch(90, 100, phase=2)
    assert dataset._mosaic_probability() == 0.0, "close-mosaic 窗口未生效"
    dataset.set_epoch(90, 100, phase=1)
    assert dataset._mosaic_probability() == 0.0
    dataset.set_epoch(10, 100, phase=1)
    assert dataset._mosaic_probability() == 0.0, "mosaic_min_phase 未生效"
    dataset.set_epoch(10, 100, phase=2)
    assert dataset._mosaic_probability() == 0.5
    print("[aug] close-mosaic: 最后 10 epoch 关闭; mosaic_min_phase=2 跳过 Phase 1")


def test_augment_repeats_length():
    dataset = object.__new__(MultiModalDataset)
    dataset.ids = [f"{i:04d}" for i in range(10)]
    dataset.include_originals = False
    dataset.train = True
    dataset.augment_repeats = 3
    dataset.augment_count = 0
    assert len(dataset) == 30
    dataset.train = False
    assert len(dataset) == 10, "验证集不应被重复"
    print("[aug] augment_repeats: 训练 30 / 验证 10")


if __name__ == "__main__":
    test_defaults_are_a_noop()
    test_photometric_is_synchronized()
    test_erasing_hits_all_modalities()
    test_mosaic_geometry()
    test_mosaic_close_schedule()
    test_augment_repeats_length()
    print("ALL AUGMENTATION TESTS PASSED")
