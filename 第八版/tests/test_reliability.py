from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.data.dataset import MultiModalDataset
from aic_v8.models.reliability_fusion import ReliabilityFusion
from aic_v8.models.tri_cssa import TriCSSA


def test_all_valid_matches_tri_cssa_at_initialization():
    torch.manual_seed(7)
    baseline = TriCSSA(8, 4, 4)
    proposed = ReliabilityFusion(8, 4, 4)
    missing, unexpected = proposed.load_state_dict(baseline.state_dict(), strict=False)
    assert set(missing) == {"local_score.weight", "local_score.bias"}
    assert not unexpected
    with torch.no_grad():
        baseline.residual_scale.fill_(0.4)
        proposed.residual_scale.fill_(0.4)
    rgb, ir, depth = (torch.randn(2, channels, 8, 8) for channels in (8, 4, 4))
    quality = torch.tensor([[1.0, 0.9, 1.0], [1.0, 0.5, 1.0]])
    valid = torch.ones(2, 1, 32, 32)
    assert torch.allclose(
        baseline(rgb, ir, depth, quality), proposed(rgb, ir, depth, quality, valid), atol=1e-6
    )


def test_invalid_depth_cannot_change_fused_output():
    torch.manual_seed(11)
    fusion = ReliabilityFusion(8, 4, 4)
    with torch.no_grad():
        fusion.residual_scale.fill_(0.5)
        fusion.local_score.weight.normal_(0, 0.05)
    rgb = torch.randn(1, 8, 8, 8)
    ir = torch.randn(1, 4, 8, 8)
    depth = torch.randn(1, 4, 8, 8)
    valid = torch.ones(1, 1, 32, 32)
    valid[:, :, :, 16:] = 0
    quality = torch.ones(1, 3)
    first = fusion(rgb, ir, depth, quality, valid)
    changed = depth.clone()
    changed[:, :, :, 4:] *= 100
    second = fusion(rgb, ir, changed, quality, valid)
    assert torch.allclose(first, second, atol=1e-6)


def test_hha_dataset_returns_sensor_validity_not_height(tmp_path):
    for folder in ("visible", "infrared", "depth", "labels"):
        (tmp_path / folder).mkdir()
    rgb = np.full((2, 2, 3), 128, np.uint8)
    depth = np.array([[0, 1000], [20000, 5000]], np.uint16)
    for folder in ("visible", "infrared"):
        assert cv2.imwrite(str(tmp_path / folder / "frame.png"), rgb)
    assert cv2.imwrite(str(tmp_path / "depth" / "frame.png"), depth)
    (tmp_path / "labels" / "frame.txt").write_text("", encoding="utf-8")
    dataset = MultiModalDataset(
        str(tmp_path), ["frame"], (32, 32), False, {"depth_encoding": "hha"}, {}
    )
    valid = dataset[0]["depth_valid"]
    assert valid.shape == (1, 32, 32)
    assert [valid[0, y, x].item() for y, x in ((8, 8), (8, 24), (24, 8), (24, 24))] == [0, 1, 0, 1]
