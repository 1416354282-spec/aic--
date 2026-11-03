"""Depth/IR quality-prior tests.

Covers:
  1. scalar quality: real validity fraction, not the HHA height channel;
  2. spatial quality: per-token bias in Tri-CSSA, with the all-valid map
     reducing to the old scalar behaviour;
  3. dataset: PNG validity mask differs from the HHA HA channel, JPG dense;
  4. wiring: depth_quality.yaml builds a 4-channel DepthEncoder and trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.data.dataset import MultiModalDataset
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR, modality_quality
from aic_v6.models.tri_cssa import TriCSSA

BASE = ROOT / "configs" / "base.yaml"
DATA_ROOT = ROOT.parent / "初赛数据集-面向城市场景的多模态目标检测" / "训练集" / "AIC2026_Train_2000"


def test_modality_quality_scalars():
    infrared = torch.zeros(2, 1, 4, 4)
    infrared[0, :, :2, :] = 1.0  # 一半非零 → q_ir = 0.5
    depth_valid = torch.zeros(2, 1, 4, 4)
    depth_valid[1, :, :, :2] = 1.0  # 一半有效 → q_depth = 0.5
    quality = modality_quality(infrared, depth_valid)
    assert quality.shape == (2, 3)
    assert abs(float(quality[0, 1]) - 0.5) < 1e-6, "q_ir 应为非零像素比例"
    assert abs(float(quality[1, 2]) - 0.5) < 1e-6, "q_depth 应为有效性比例"

    # 数据管线在 letterbox 之前算好的标量优先 (不受黑边影响)
    quality = modality_quality(
        infrared, depth_valid,
        ir_quality=torch.tensor([0.25, 0.75]),
        depth_quality=torch.tensor([0.10, 0.90]),
    )
    assert torch.allclose(quality[:, 1], torch.tensor([0.25, 0.75]))
    assert torch.allclose(quality[:, 2], torch.tensor([0.10, 0.90]))
    print("[depth_q] q_ir/q_depth 标量与显式覆盖均正确")


def test_spatial_quality_bias():
    torch.manual_seed(0)
    block = TriCSSA(32, 16, 16)
    with torch.no_grad():
        block.residual_scale.fill_(1.0)  # 初始为 0 会恒等输出, 看不出权重差异
    rgb = torch.randn(2, 32, 8, 8)
    infrared = torch.randn(2, 16, 8, 8)
    depth = torch.randn(2, 16, 8, 8)
    quality = torch.ones(2, 3)

    old = block(rgb, infrared, depth, quality)
    full = block(rgb, infrared, depth, quality, depth_quality=torch.ones(2, 1, 8, 8))
    empty = block(rgb, infrared, depth, quality, depth_quality=torch.zeros(2, 1, 8, 8))
    assert torch.allclose(old, full, atol=1e-6), "全有效空间图应等价于旧标量先验"
    assert not torch.allclose(old, empty, atol=1e-6), "全无效深度图应改变融合结果"
    print("[depth_q] 空间质量: 全 1 等价旧行为, 全 0 抑制 depth")


def _first_ids(suffix: str, count: int = 1):
    folder = DATA_ROOT / "visible"
    found = sorted(path.stem for path in folder.iterdir() if path.suffix.lower() == suffix)
    return found[:count]


def test_dataset_real_valid_mask():
    png_ids = _first_ids(".png")
    jpg_ids = _first_ids(".jpg")
    if not png_ids or not jpg_ids:
        print("[depth_q] 跳过: 本地数据不完整")
        return
    data_config = {
        "labels_subfolder": "new_labels_2000",
        "depth_encoding": "hha",
        "include_originals": False,
        "augment_repeats": 1,
    }
    dataset = MultiModalDataset(str(DATA_ROOT), png_ids + jpg_ids, (256, 192), False, data_config, {})
    png_item, jpg_item = dataset[0], dataset[1]
    assert png_item["depth"].shape[0] == 3
    assert png_item["depth_valid"].shape[0] == 1
    assert 0.0 <= float(png_item["depth_valid"].min()) <= float(png_item["depth_valid"].max()) <= 1.0
    assert 0.0 <= png_item["depth_quality"] <= 1.0
    assert jpg_item["depth_quality"] == 1.0, "JPG 相对深度是全图有效的"

    # 旧实现把 HHA 的 HA 通道当 valid; 两者必须不同, 否则测试没测到东西
    channels, valid, _ = dataset._load_depth(png_ids[0])
    ha_channel_mean = float(np.asarray(channels[1]).mean())
    assert not np.isclose(float(valid.mean()), ha_channel_mean, atol=1e-3), \
        "真实有效期与 HHA HA 通道均值不应相同"
    print(f"[depth_q] PNG valid={valid.mean():.3f} vs HHA-HA={ha_channel_mean:.3f}; JPG valid=1.0")


def _fake_batch(batch_size=1, height=128, width=128, num_boxes=2):
    return {
        "img": torch.rand(batch_size, 3, height, width),
        "ir": torch.rand(batch_size, 1, height, width),
        "depth": torch.rand(batch_size, 3, height, width),
        "depth_valid": torch.ones(batch_size, 1, height, width),
        "ir_quality": torch.tensor([0.8] * batch_size),
        "depth_quality": torch.tensor([0.7] * batch_size),
        "boxes": torch.tensor([[[0.3, 0.4, 0.1, 0.2]] * num_boxes] * batch_size),
        "labels": torch.zeros(batch_size, num_boxes, dtype=torch.long),
        "num_boxes": torch.tensor([num_boxes] * batch_size),
    }


def test_depth_quality_wiring():
    overlay = ROOT / "configs" / "experiments" / "depth_quality.yaml"
    config = load_config(BASE, (overlay,))
    model = TriModalRTDETR(config)
    assert model.depth_in_channels == 4 and model.spatial_quality
    assert model.depth_encoder.stem[0].in_channels == 4, "DepthEncoder 第一层未接收 4 通道"
    model.train()
    loss, metrics = model.loss(_fake_batch())
    assert torch.isfinite(loss) and metrics["total"] > 0
    # eval 路径也要能过: 空间质量图 + 4 通道 + decoder eval 输出
    model.eval()
    with torch.no_grad():
        outputs = model.predict_raw(_fake_batch())
    y = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    assert y.shape[0] == 1
    print(f"[depth_q] depth_quality.yaml 接线通过 loss={float(loss.detach()):.3f} eval={tuple(y.shape)}")


if __name__ == "__main__":
    test_modality_quality_scalars()
    test_spatial_quality_bias()
    test_dataset_real_valid_mask()
    test_depth_quality_wiring()
    print("ALL DEPTH-QUALITY TESTS PASSED")
