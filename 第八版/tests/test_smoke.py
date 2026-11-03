"""Smoke tests: config load, model build, forward pass, loss, tri-mode fusion.

Run with CPU (small sizes); the RT-DETR weights file must exist locally
(rtdetr-l.pt in the project root, downloaded via a github proxy mirror).
模型构建类测试在预训练权重缺失时自动 skip, 不阻塞其余用例。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v8.config import load_config
from aic_v8.models.multimodal_rtdetr import TriModalRTDETR, _limit_feature_gradient

BASE = ROOT / "configs" / "base.yaml"


def _pretrained_weights_available() -> bool:
    """按 base.yaml 的实际解析结果检查 model.weights 是否已离线放置。"""
    try:
        return Path(load_config(BASE)["model"]["weights"]).exists()
    except Exception:
        return False


_requires_pretrained = pytest.mark.skipif(
    not _pretrained_weights_available(),
    reason="需要离线 RT-DETR 预训练权重 (见 configs/base.yaml 的 model.weights)",
)


def _fake_batch(batch_size=2, height=256, width=256, num_boxes=4, device="cpu"):
    return {
        "img": torch.rand(batch_size, 3, height, width, device=device),
        "ir": torch.rand(batch_size, 1, height, width, device=device),
        "depth": torch.rand(batch_size, 3, height, width, device=device),
        "depth_valid": torch.rand(batch_size, 1, height, width, device=device),
        "boxes": torch.rand(batch_size, num_boxes, 4, device=device),
        "labels": torch.randint(0, 12, (batch_size, num_boxes), device=device),
        "num_boxes": torch.tensor([num_boxes, num_boxes], device=device),
    }


def test_config_loads():
    config = load_config(BASE)
    assert config["model"]["mode"] == "tri"
    assert config["model"]["fusion"] == "reliability"
    assert config["model"]["num_classes"] == 12
    assert len(config["train"]["phases"]) == 3


def test_feature_gradient_limit_preserves_direction():
    gradient = torch.tensor([-1.0e22, 5.0e21, 2.0])
    limited = _limit_feature_gradient(gradient, 1.0e-3)
    assert torch.isfinite(limited).all()
    assert torch.isclose(limited.abs().amax(), torch.tensor(1.0e-3))
    assert torch.equal(torch.sign(limited), torch.sign(gradient))


@_requires_pretrained
def test_rgb_only_forward_and_loss():
    config = load_config(BASE, (ROOT / "configs" / "experiments" / "rgb_only.yaml",))
    model = TriModalRTDETR(config)
    model.eval()
    batch = _fake_batch()
    outputs = model.predict_raw(batch)
    # RTDETRDecoder eval 返回 (y, x); y = (B, 300, 6) [cx,cy,w,h,score,cls]
    y = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    assert y.shape == (2, 300, 6), f"eval 输出形状异常: {tuple(y.shape)}"
    assert y[..., 4].min() >= 0 and y[..., 4].max() <= 1, "score 应在 [0,1]"
    # training loss path
    model.train()
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert metrics["total"] > 0
    print(f"[rgb_only] loss={float(loss.detach()):.3f} y={tuple(y.shape)}")


@_requires_pretrained
def test_tri_forward_and_loss():
    config = load_config(BASE)
    model = TriModalRTDETR(config)
    assert model.ir_encoder is not None
    assert len(model.fusions) == 3, "三尺度融合应有 3 个块"
    model.eval()
    batch = _fake_batch()
    outputs = model.predict_raw(batch)
    y = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    assert y.shape == (2, 300, 6)
    model.train()
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert "target_mask" in metrics
    # residual identity init: fused feature must equal RGB at start
    fused = model.fusions[0]
    with torch.no_grad():
        projection = fused.ir_projection
        rgb_ch, aux_ch = projection.weight.shape[0], projection.weight.shape[1]
        feature = torch.rand(2, rgb_ch, 16, 16)
        aux = torch.rand(2, aux_ch, 16, 16)
        out = fused(feature, aux, aux, torch.ones(2, 3), torch.ones(2, 1, 16, 16))
    assert torch.allclose(out, feature, atol=1e-5), "残差恒等初始化失败"
    print(f"[tri] loss={float(loss.detach()):.3f} target_mask={float(metrics['target_mask']):.4f}")


@_requires_pretrained
def test_cmtf_ablation():
    config = load_config(BASE, (ROOT / "configs" / "experiments" / "cmtf.yaml",))
    model = TriModalRTDETR(config)
    assert model.fusion_name == "cmtf"
    model.train()
    loss, metrics = model.loss(_fake_batch())
    assert torch.isfinite(loss)
    print(f"[cmtf] loss={float(loss.detach()):.3f}")


@_requires_pretrained
def test_phase_control():
    config = load_config(BASE)
    model = TriModalRTDETR(config)
    model.set_phase(1, config)
    frozen_rgb = [not p.requires_grad for n, p in model.named_parameters() if model.parameter_role(n) == "rgb"]
    assert all(frozen_rgb), "Phase 1 应冻结 backbone"
    frozen_detector = [not p.requires_grad for n, p in model.named_parameters() if model.parameter_role(n) == "detector"]
    assert all(frozen_detector), "Phase 1 应冻结 neck+decoder"
    new_grad = [p.requires_grad for n, p in model.named_parameters() if model.parameter_role(n) == "new"]
    assert all(new_grad), "Phase 1 应训练新分支"
    model.train()
    assert model.ir_encoder.training, "Phase 1 新分支应保持 train 模式"
    assert all(not layer.training for layer in model.detector.model[:-1]), (
        "model.train() 不应重新启用 Phase 1 已冻结 backbone/neck 的训练态"
    )
    assert model.detector.model[-1].training, "RT-DETR head 必须保持 train 模式以返回训练损失输入"
    model.set_phase(3, config)
    model.train()
    all_grad = [p.requires_grad for _, p in model.named_parameters()]
    assert all(all_grad), "Phase 3 应全解冻"
    assert all(layer.training for layer in model.detector.model), "Phase 3 detector 应恢复 train 模式"


if __name__ == "__main__":
    test_config_loads()
    test_rgb_only_forward_and_loss()
    test_tri_forward_and_loss()
    test_cmtf_ablation()
    test_phase_control()
    print("ALL SMOKE TESTS PASSED")
