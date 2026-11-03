"""Relation-Aware Refiner (设计文档 §3.7, 本版核心新增).

挂载位置: 解码器输出之后、最终打分之前。输入解码器末层候选框 B (Q×4, 归一化
cxcywh)、类别 logits L (Q×12)、内容查询特征 Z (Q×d, 可选)。输出精炼 logits
L'，坐标不变 (保证与 v6 提交管线兼容):

    L' = L + α₁·π                       图级共现先验 (冻结, 共现矩阵 P_coocᵀ·d)
           + α₂·S                       逐 query 空间先验 (冻结, 逐类 2D 高斯)
           + α₃·(W_geo ⊙ W_app)·Ŷ       query 间几何/外观关系聚合 (几何 MLP 可训练)

- 共现矩阵与空间高斯由 scripts/build_priors.py 从官方训练集统计, 加载后注册为
  buffer 冻结 (非参数, 只提供归纳偏置); 熵加权已在统计阶段折叠进协方差。
- 几何 MLP 末层零初始化: W_geo=0 → 初始 L'=L, 不扰动预训练分布; 末层输出为
  线性 (不可再套 ReLU: ReLU 在 0 点导数为 0, 会把末层零初始化变成梯度死锁,
  隐层已有 ReLU 提供非线性); 外观注意力默认关闭 (小数据下易过拟合), 开启时
  以 tanh 门控零初始化保持恒等。
- α₁/α₂/α₃ 为配置常数, 前 warmup_epochs 轮线性 warmup (§3.7.5)。
- 训练时对精炼 logits 在已匹配 query-GT 对上施加 varifocal 辅助监督 (匹配
  结果复用 criterion.last_indices, 不重复匹配)。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2 * math.pi)


def _pair_features(boxes: torch.Tensor) -> torch.Tensor:
    """不对称几何特征 e_ij = [Δcx/w_i, Δcy/h_i, log(w_j/w_i), log(h_j/h_i), IoU_ij, 中心距].

    boxes: (B, Q, 4) 归一化 cxcywh。
    """
    centers = boxes[..., :2]                          # (B, Q, 2)
    wh = boxes[..., 2:].clamp_min(1e-6)               # (B, Q, 2)
    delta = centers.unsqueeze(2) - centers.unsqueeze(1)          # (B, Q, Q, 2)
    rel = delta / wh.unsqueeze(1).clamp_min(1e-6)                # (B, Q, Q, 2) 除以自身尺度
    size_ratio = torch.log(
        (wh.unsqueeze(2) / wh.unsqueeze(1)).clamp_min(1e-6)      # (B, Q, Q, 2)
    )
    x1 = centers[..., 0] - wh[..., 0] / 2
    y1 = centers[..., 1] - wh[..., 1] / 2
    x2 = centers[..., 0] + wh[..., 0] / 2
    y2 = centers[..., 1] + wh[..., 1] / 2
    xx1 = torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))
    yy1 = torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))
    xx2 = torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
    yy2 = torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))
    inter = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
    area = (wh[..., 0] * wh[..., 1]).clamp_min(1e-6)
    union = area.unsqueeze(2) + area.unsqueeze(1) - inter
    iou = (inter / union.clamp_min(1e-6)).unsqueeze(-1)          # (B, Q, Q, 1)
    dist = torch.norm(delta, dim=-1, keepdim=True)               # (B, Q, Q, 1)
    return torch.cat((rel, size_ratio, iou, dist), dim=-1)       # (B, Q, Q, 6)


class RelationAwareRefiner(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        hidden: int = 64,
        alpha_cooc: float = 0.15,
        alpha_spatial: float = 0.10,
        alpha_geo: float = 0.20,
        use_appearance: bool = False,
        warmup_epochs: int = 5,
        topk_for_graph: int = 50,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_cooc = float(alpha_cooc)
        self.alpha_spatial = float(alpha_spatial)
        self.alpha_geo = float(alpha_geo)
        self.use_appearance = use_appearance
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.topk_for_graph = int(topk_for_graph)
        self._epoch = 0

        # 几何关系 MLP (末层零初始化 → 恒等起点)
        self.geo_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.geo_mlp[-1].weight)
        nn.init.zeros_(self.geo_mlp[-1].bias)

        # 外观关系注意力 (默认关闭; tanh 门控零初始化保持恒等)
        self.app_gate = nn.Parameter(torch.zeros(()))
        if use_appearance:
            self.app_proj = nn.Linear(feature_dim, feature_dim)
            nn.init.eye_(self.app_proj.weight)
            nn.init.zeros_(self.app_proj.bias)

        # 先验 buffer: 由 build_priors.py 统计 (冻结, 非参数)
        self.register_buffer("cooc_prior", torch.zeros(num_classes, num_classes))
        self.register_buffer("has_cooc", torch.tensor(False))
        self.register_buffer("spatial_mu", torch.zeros(num_classes, 2))
        self.register_buffer("spatial_inv", torch.eye(2).repeat(num_classes, 1, 1))
        self.register_buffer("spatial_logdet", torch.zeros(num_classes))
        self.register_buffer("has_spatial", torch.tensor(False))

    # -- prior loading ---------------------------------------------------------
    def set_priors(
        self,
        cooc: Optional[torch.Tensor] = None,
        spatial_mu: Optional[torch.Tensor] = None,
        spatial_cov: Optional[torch.Tensor] = None,
    ) -> None:
        """注册共现矩阵 (C×C) 与逐类空间高斯 (mu: C×2, cov: C×2×2, 已含熵加权)。"""
        if cooc is not None:
            cooc = cooc.to(self.cooc_prior.device, self.cooc_prior.dtype)
            if cooc.shape != (self.num_classes, self.num_classes):
                raise ValueError(f"共现矩阵形状 {tuple(cooc.shape)} != {(self.num_classes, self.num_classes)}")
            self.cooc_prior.copy_(cooc)
            self.has_cooc.fill_(True)
        if spatial_mu is not None and spatial_cov is not None:
            mu = spatial_mu.to(self.spatial_mu.device, self.spatial_mu.dtype)
            cov = spatial_cov.to(self.spatial_mu.device, self.spatial_mu.dtype)
            eye = torch.eye(2, device=cov.device, dtype=cov.dtype)
            cov = cov + eye * 1e-4  # ridge 防奇异
            inv = torch.linalg.inv(cov)
            sign, logdet = torch.linalg.slogdet(cov)
            logdet = torch.where(sign > 0, logdet, torch.full_like(logdet, 1e-6))
            self.spatial_mu.copy_(mu)
            self.spatial_inv.copy_(inv)
            self.spatial_logdet.copy_(logdet)
            self.has_spatial.fill_(True)

    def load_priors_npz(self, path: str) -> None:
        import numpy as np

        data = np.load(path)
        self.set_priors(
            cooc=torch.from_numpy(data["cooc"]).float() if "cooc" in data else None,
            spatial_mu=torch.from_numpy(data["spatial_mu"]).float() if "spatial_mu" in data else None,
            spatial_cov=torch.from_numpy(data["spatial_cov"]).float() if "spatial_cov" in data else None,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def warmup(self) -> float:
        return min(1.0, (self._epoch + 1) / self.warmup_epochs)

    # -- refinement --------------------------------------------------------------
    def _class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # D-FINE 分类头为 focal/sigmoid 风格, 类别证据用 sigmoid 归一化
        # (设计文档公式写 softmax, 此处按头的先验语义实现, 消融结论不受影响)。
        return logits.sigmoid()

    def _cooc_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """图级共现先验 π = P_coocᵀ·d, d = 逐类 top-k 置信度均值 (向量, 广播到全部 query)。"""
        if not bool(self.has_cooc):
            return 0.0
        probs = self._class_probs(logits)                              # (B, Q, C)
        k = min(self.topk_for_graph, probs.shape[1])
        topk = probs.transpose(1, 2).topk(k, dim=-1).values.mean(-1)   # (B, C)
        pi = self.cooc_prior.t() @ topk.unsqueeze(-1)                  # (B, C, 1)
        return torch.log(pi.squeeze(-1).clamp_min(1e-4) + 1e-6).unsqueeze(1)  # (B, 1, C)

    def _spatial_bias(self, logits: torch.Tensor, boxes: torch.Tensor, flip_x: bool = False) -> torch.Tensor:
        """逐 query 空间先验 S_i = Σ_c σ(L_i)[c]·log N(center_i; μ_c, Σ_c) (软指派)。

        flip_x=True 时 μ_c 的 x 分量镜像 (TTA 水平翻转分支: 翻转图上的位置先验
        应与翻转后的坐标系一致); 几何特征与共现先验均为相对/图级量, 不受翻转影响。
        """
        if not bool(self.has_spatial):
            return 0.0
        mu = self.spatial_mu
        if flip_x:
            mu = mu.clone()
            mu[..., 0] = 1.0 - mu[..., 0]
        centers = boxes[..., :2]                                       # (B, Q, 2)
        diff = centers.unsqueeze(2) - mu                               # (B, Q, C, 2)
        quad = torch.einsum("bqci,cij,bqcj->bqc", diff, self.spatial_inv, diff).clamp_max(1e5)
        logdens = -0.5 * (quad + 2 * _LOG_2PI + self.spatial_logdet)   # (B, Q, C)
        probs = self._class_probs(logits)
        return (probs * logdens).sum(-1, keepdim=True)                 # (B, Q, 1)

    def _geo_message(self, logits: torch.Tensor, boxes: torch.Tensor, query_features=None) -> torch.Tensor:
        """几何/外观关系聚合 Σ_j (W_geo ⊙ W_app)[i,j]·Ŷ_j, 经 tanh 界定到 [-1, 1]。

        必须有界: 300 个 query 的求和无界, 而 varifocal 辅助监督的负样本压制
        激励 (目标 0, 压得越低 loss 越小) 会把无界消息演变成全域分数压制
        (实测 33 epoch 后 message < -1600, refined 分数全灭, val mAP 塌为 0)。
        tanh 界定后 geo 增量 ≤ α₃ (0.2 logit); tanh(0)=0 保持恒等起点,
        tanh'(0)=1 不堵梯度, 消息漂大时梯度自适应衰减。
        """
        pair = _pair_features(boxes)                                   # (B, Q, Q, 6)
        w_geo = self.geo_mlp(pair).squeeze(-1)                         # (B, Q, Q) 线性, 恒等起点由末层零初始化保证
        w_app = torch.ones_like(w_geo)
        if self.use_appearance and query_features is not None:
            z = F.normalize(self.app_proj(query_features), dim=-1)     # (B, Q, d)
            attn = torch.bmm(z, z.transpose(1, 2)) / math.sqrt(z.shape[-1])
            w_app = 1.0 + torch.tanh(self.app_gate) * attn.softmax(-1)
        weights = w_geo * w_app                                        # (B, Q, Q)
        evidence = self._class_probs(logits)                           # (B, Q, C)
        return torch.tanh(torch.bmm(weights, evidence))                # (B, Q, C) 有界

    def refine_logits(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        flip_x: bool = False,
    ) -> torch.Tensor:
        """精炼类别 logits (坐标不变)。输入 (B, Q, C) logits 与 (B, Q, 4) 归一化 cxcywh。"""
        scale = self.warmup
        refined = logits + scale * self.alpha_cooc * self._cooc_bias(logits)
        refined = refined + scale * self.alpha_spatial * self._spatial_bias(logits, boxes, flip_x)
        refined = refined + scale * self.alpha_geo * self._geo_message(logits, boxes, query_features)
        return refined

    # -- supervision ---------------------------------------------------------------
    def varifocal_loss(
        self,
        refined_logits: torch.Tensor,
        boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """精炼 logits 的 varifocal 辅助监督 (复用解码器匹配结果, §3.7.5)。

        正样本 = 匹配的 query-GT 对, 目标分数 = 该对 IoU (质量感知); 其余 query
        为负样本 (目标 0)。boxes 应传入 detach 后的解码器框。
        """
        targets_scores = torch.zeros_like(refined_logits)
        for image_index, (src, tgt) in enumerate(indices):
            if len(src) == 0:
                continue
            pred = boxes[image_index, src]                              # (M, 4)
            gt = targets[image_index]["boxes"][tgt]                     # (M, 4)
            labels = targets[image_index]["labels"][tgt].long()
            x1 = torch.maximum(pred[:, 0] - pred[:, 2] / 2, gt[:, 0] - gt[:, 2] / 2)
            y1 = torch.maximum(pred[:, 1] - pred[:, 3] / 2, gt[:, 1] - gt[:, 3] / 2)
            x2 = torch.minimum(pred[:, 0] + pred[:, 2] / 2, gt[:, 0] + gt[:, 2] / 2)
            y2 = torch.minimum(pred[:, 1] + pred[:, 3] / 2, gt[:, 1] + gt[:, 3] / 2)
            inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
            area_p = (pred[:, 2] * pred[:, 3]).clamp_min(1e-6)
            area_g = (gt[:, 2] * gt[:, 3]).clamp_min(1e-6)
            iou = inter / (area_p + area_g - inter).clamp_min(1e-6)
            targets_scores[image_index, src, labels] = iou.detach()

        num_queries = max(refined_logits.shape[0] * refined_logits.shape[1], 1)
        pred_score = refined_logits.sigmoid()
        weight = torch.where(
            targets_scores > 0, pred_score.pow(2.0), (1 - pred_score).pow(2.0)
        )
        loss = weight * F.binary_cross_entropy_with_logits(
            refined_logits, targets_scores, reduction="none"
        )
        return loss.sum() / num_queries

    @property
    def enabled(self) -> bool:
        return any(a > 0 for a in (self.alpha_cooc, self.alpha_spatial, self.alpha_geo))
