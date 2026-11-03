"""小目标回归损失改造: 尺度归一化 L1 + NWD (专治高 IoU 定位)。

实测依据 (见 `第六版小目标专项改进方案.md`):
  RT-DETR 原版 `DETRLoss._get_loss_bbox` / `HungarianMatcher` 都在**归一化坐标**上
  算 L1, 代价与梯度 ∝ 目标尺寸 —— 一个 seat (中位 215px) 的 L1 权重是 ball (38px)
  的 5.7 倍、uav (原生 24px) 的 9 倍; 而赛题是 12 类直接平均, 三者同价 (各 8.33 分)。
  同时 GIoU 在小框上极不平滑: 38px 目标偏 2px 就掉到 IoU 0.9, 梯度接近消失,
  这正是 mAP@50 (83.8) 与 mAP@50-95 (57.4) 之间那 36 分缺口的来源。

本模块提供两个**独立开关**的旋钮, 都只作用在已匹配对上 (不动 Hungarian 匹配代价,
匹配行为不变 → 消融变量最干净):

    model:
      regression_loss:
        l1_size_gamma: 0.5        # 匹配对按 (中位尺度 / 该目标尺度)^gamma 加权 L1
        l1_weight_bounds: [0.25, 4.0]
        nwd_blend: 0.5            # giou 槽 = (1-λ)·GIoU + λ·NWD, λ∈[0,1]
        nwd_c: 0.0                # NWD 常数 C; 0 = 用 batch 内 GT 尺度中位数自适应

**默认 (未配置 / 两个旋钮都为 0) 走 `super()._get_loss_bbox`, 与官方实现逐位相同。**
注意 `nwd_blend > 0` 时 criterion 返回的 `loss_giou` 语义变成 "GIoU 与 NWD 的混合",
指标名不变但含义已改 (日志里看到的小幅下降是预期)。
"""

from __future__ import annotations

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou


def normalized_wasserstein(pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, constant: float) -> torch.Tensor:
    """NWD = exp(-sqrt(W2) / C), 把框 (cx, cy, w, h) 建模为 2D 高斯 N(mu, diag((w/2)^2, (h/2)^2))。

    W2 = ||mu_p - mu_g||^2 + ||Sigma_p^{1/2} - Sigma_g^{1/2}||_F^2 —— 对平移与尺度都平滑,
    小框上的梯度不会像 IoU/GIoU 那样在目标附近消失, 也不像归一化 L1 那样随尺寸缩放
    (Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection")。
    """
    delta = torch.stack(
        (
            pred_bboxes[:, 0] - gt_bboxes[:, 0],
            pred_bboxes[:, 1] - gt_bboxes[:, 1],
            (pred_bboxes[:, 2] - gt_bboxes[:, 2]) * 0.5,
            (pred_bboxes[:, 3] - gt_bboxes[:, 3]) * 0.5,
        ),
        dim=-1,
    )
    wasserstein = delta.pow(2).sum(-1).clamp_min(1e-12).sqrt()
    return torch.exp(-wasserstein / max(constant, 1e-6))


def size_balanced_weights(sizes: torch.Tensor, gamma: float = 0.5, bounds=(0.25, 4.0)) -> torch.Tensor:
    """(批内中位尺度 / 各目标尺度)^gamma, 截断到 bounds 后归一到均值 1。

    均值归一保证回归项的总量级不变 —— 换 gamma 不需要重调 lr / loss_gain。
    gamma=0 返回全 1 (等价于原实现)。
    """
    if gamma <= 0:
        return torch.ones_like(sizes)
    weight = (sizes.median() / sizes.clamp_min(1e-12)).pow(gamma).clamp(bounds[0], bounds[1])
    return weight / weight.mean()


class ScaleBalancedRTDETRLoss(RTDETRDetectionLoss):
    """RT-DETR criterion + 尺度归一化回归 (默认 = 官方实现, 见模块文档)。"""

    def __init__(
        self,
        nc: int = 12,
        l1_size_gamma: float = 0.0,
        l1_weight_bounds: tuple[float, float] | list[float] = (0.25, 4.0),
        nwd_blend: float = 0.0,
        nwd_c: float = 0.0,
        use_vfl: bool = True,
    ):
        super().__init__(nc=nc, use_vfl=use_vfl)
        self.l1_size_gamma = float(l1_size_gamma)
        self.l1_weight_bounds = (float(l1_weight_bounds[0]), float(l1_weight_bounds[1]))
        self.nwd_blend = float(nwd_blend)
        self.nwd_c = float(nwd_c)
        if self.l1_weight_bounds[0] > self.l1_weight_bounds[1]:
            raise ValueError(f"l1_weight_bounds 必须升序: {self.l1_weight_bounds}")
        if not 0.0 <= self.nwd_blend <= 1.0:
            raise ValueError(f"nwd_blend 必须在 [0, 1]: {self.nwd_blend}")

    @property
    def enabled(self) -> bool:
        return self.l1_size_gamma > 0 or self.nwd_blend > 0

    def _get_loss_bbox(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""):
        if not self.enabled:
            return super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"
        if len(gt_bboxes) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {name_bbox: zero, name_giou: zero}

        size = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp_min(1e-12).sqrt()
        l1 = (pred_bboxes - gt_bboxes).abs().sum(-1)
        if self.l1_size_gamma > 0:
            # 归一到均值 1: 回归损失总量级不变, 不需要重调 lr / loss_gain
            l1 = l1 * size_balanced_weights(size, self.l1_size_gamma, self.l1_weight_bounds)
        loss_bbox = self.loss_gain["bbox"] * l1.sum() / len(gt_bboxes)

        giou = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True).reshape(-1)
        if self.nwd_blend > 0:
            constant = self.nwd_c if self.nwd_c > 0 else float(size.median().clamp_min(1e-3))
            nwd = 1.0 - normalized_wasserstein(pred_bboxes, gt_bboxes, constant)
            giou = (1.0 - self.nwd_blend) * giou + self.nwd_blend * nwd
        loss_giou = self.loss_gain["giou"] * giou.sum() / len(gt_bboxes)
        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}
