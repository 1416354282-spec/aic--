"""Smoke tests: config load, model build, forward pass, loss, tri-mode fusion.

Run with CPU (small sizes); the RT-DETR weights file must exist locally
(rtdetr-l.pt in the project root, downloaded via a github proxy mirror).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_v6.config import load_config
from aic_v6.models.multimodal_rtdetr import TriModalRTDETR

BASE = ROOT / "configs" / "base.yaml"


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
    assert config["model"]["fusion"] == "tri_cssa"
    assert config["model"]["num_classes"] == 12
    assert len(config["train"]["phases"]) == 3


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
        out = fused(feature, aux, aux, torch.ones(2, 3))
    assert torch.allclose(out, feature, atol=1e-5), "残差恒等初始化失败"
    print(f"[tri] loss={float(loss.detach()):.3f} target_mask={float(metrics['target_mask']):.4f}")


def test_cmtf_ablation():
    config = load_config(BASE, (ROOT / "configs" / "experiments" / "cmtf.yaml",))
    model = TriModalRTDETR(config)
    assert model.fusion_name == "cmtf"
    model.train()
    loss, metrics = model.loss(_fake_batch())
    assert torch.isfinite(loss)
    print(f"[cmtf] loss={float(loss.detach()):.3f}")


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
    # trainer 每 epoch 调 model.train(); 冻结层必须保持 eval(), 否则 BN 统计会漂移
    model.train()
    backbone = model.detector.model[:model.backbone_end + 1]
    assert all(not layer.training for layer in backbone), "model.train() 不应撤销 backbone 冻结"
    neck = model.detector.model[model.backbone_end + 1:-1]
    assert all(not layer.training for layer in neck), "model.train() 不应撤销 neck 冻结"
    # RTDETRDecoder 必须保持 train(): 它的 training 标志决定输出 5 元组(训练态)还是 2 元组(eval)
    assert model.detector.model[-1].training, "decoder head 必须保持 train() 以提供训练态输出"
    model.set_phase(3, config)
    model.train()
    assert all(layer.training for layer in model.detector.model), "Phase 3 应全部处于 train()"
    all_grad = [p.requires_grad for _, p in model.named_parameters()]
    assert all(all_grad), "Phase 3 应全解冻"


def test_scale_balanced_loss():
    """小目标回归损失: 默认逐位等同官方实现; 尺度加权抬高小目标, NWD 在极小框上不饱和。"""
    from aic_v6.models.losses import ScaleBalancedRTDETRLoss, size_balanced_weights

    torch.manual_seed(0)
    # 5 个匹配对 (奇数 → torch.median 取中位无歧义): 2 个小框 (0.02 ≈ 38px@1920 = ball 中位),
    # 1 个中框 (0.04), 2 个大框 (0.112 ≈ 215px = seat 中位)
    gt = torch.tensor([[0.3, 0.3, 0.02, 0.02], [0.7, 0.3, 0.02, 0.02], [0.5, 0.5, 0.04, 0.04],
                       [0.3, 0.7, 0.112, 0.112], [0.7, 0.7, 0.112, 0.112]])
    pred = gt + torch.tensor([[0.004, 0.0, 0.0, 0.0], [0.0, 0.004, 0.0, 0.0], [0.004, 0.004, 0.0, 0.0],
                              [0.004, 0.0, 0.02, 0.0], [0.0, 0.004, 0.0, 0.02]])

    stock = ScaleBalancedRTDETRLoss(nc=12)
    tuned = ScaleBalancedRTDETRLoss(nc=12, l1_size_gamma=0.5, nwd_blend=0.5)
    stock.device = tuned.device = torch.device("cpu")
    base_items = stock._get_loss_bbox(pred, gt)
    tuned_items = tuned._get_loss_bbox(pred, gt)
    assert set(base_items) == set(tuned_items) == {"loss_bbox", "loss_giou"}
    assert all(torch.isfinite(v) for v in tuned_items.values())

    # 尺度权重: 平均值 1 (量级不变); gamma=1 时小框权重 / 大框权重 = 大框尺度 / 小框尺度
    sizes = (gt[:, 2] * gt[:, 3]).sqrt()
    weights = size_balanced_weights(sizes, gamma=1.0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6), "权重必须归一到均值 1"
    ratio = float(weights[0] / weights[3])
    assert abs(ratio - 0.112 / 0.02) < 1e-3, f"gamma=1 时权重比应等于尺度反比, got {ratio:.3f}"
    assert torch.allclose(size_balanced_weights(sizes, gamma=0.0), torch.ones_like(sizes)), "gamma=0 必须是原实现"
    print(f"[scale_balanced] 尺度权重 小框 {float(weights[0]):.3f} / 大框 {float(weights[3]):.3f} = {ratio:.2f}× "
          f"| L1 项 stock {float(base_items['loss_bbox']):.4f} → 加权 {float(tuned_items['loss_bbox']):.4f}")

    # NWD 在极小框上的动态范围: 同样的绝对误差, GIoU 已饱和而 NWD 仍给出梯度
    tiny_gt = torch.tensor([[0.5, 0.5, 0.012, 0.012]])              # ≈ 23px @1920 (uav 中位尺度)
    tiny_pred = tiny_gt + torch.tensor([[0.0026, 0.0, 0.0, 0.0]])   # 偏 5px ≈ IoU 0.65
    nwd_only = ScaleBalancedRTDETRLoss(nc=12, nwd_blend=1.0)
    giou_only = ScaleBalancedRTDETRLoss(nc=12)
    nwd_only.device = giou_only.device = torch.device("cpu")
    nwd_loss = float(nwd_only._get_loss_bbox(tiny_pred, tiny_gt)["loss_giou"])
    giou_loss = float(giou_only._get_loss_bbox(tiny_pred, tiny_gt)["loss_giou"])
    assert nwd_loss < giou_loss, "NWD 在极小框上应比 GIoU 温和, 避免高 IoU 区梯度饱和后无信号"
    print(f"[scale_balanced] 23px 框偏 5px: GIoU 项 {giou_loss:.4f} → NWD 项 {nwd_loss:.4f}")


def test_small_object_loss_wiring():
    """configs/experiments/small_object_loss.yaml 必须真的装上 ScaleBalancedRTDETRLoss 并跑通。"""
    from aic_v6.models.losses import ScaleBalancedRTDETRLoss

    overlay = ROOT / "configs" / "experiments" / "small_object_loss.yaml"
    config = load_config(BASE, (overlay,))
    model = TriModalRTDETR(config)
    criterion = model.detector.criterion
    assert isinstance(criterion, ScaleBalancedRTDETRLoss), "小目标损失配置未接入 criterion"
    assert criterion.enabled and criterion.l1_size_gamma == 0.5 and criterion.nwd_blend == 0.5
    model.train()
    loss, metrics = model.loss(_fake_batch())
    assert torch.isfinite(loss) and metrics["total"] > 0
    # 未配置时 criterion 由 loss() 懒加载为官方实现 (默认路径不受影响)
    plain = TriModalRTDETR(load_config(BASE))
    assert not isinstance(getattr(plain.detector, "criterion", None), ScaleBalancedRTDETRLoss), \
        "默认配置不应改变损失"
    print(f"[small_object_loss] loss={float(loss.detach()):.3f} criterion={type(criterion).__name__}")


if __name__ == "__main__":
    test_config_loads()
    test_rgb_only_forward_and_loss()
    test_tri_forward_and_loss()
    test_cmtf_ablation()
    test_phase_control()
    test_scale_balanced_loss()
    test_small_object_loss_wiring()
    print("ALL SMOKE TESTS PASSED")
