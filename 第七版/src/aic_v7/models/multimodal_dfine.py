"""TriModalDFINE: 第七版主包装器 (v6 TriModalRTDETR 的同位替换).

    TriModalDFINE
      ├── core: DFINECore (官方 D-FINE: HGNetv2 + AIFI/CCFF + OQD 解码器 + FDR 头,
      │         注入 Tri-CSSA/CMTF 三模态融合 + Target-Aware Refiner 辅助监督)
      └── relation_refiner: Relation-Aware Refiner (本版核心新增, 目标间关系建模)

接口契约与 v6 完全一致:
  - loss(batch) → (total, metrics); 训练循环零改动
  - predict_raw(batch) → (B, 300, 6) 归一化 [cx,cy,w,h,score,cls] (letterbox 空间,
    score 已 sigmoid, 无 NMS) → evaluator / 提交管线零改动
  - set_phase / parameter_role / train/eval 对齐 v6; set_epoch 供关系 warmup

模式:
  - model.base=dfine (默认): 官方 D-FINE 路径; 若 third_party/D-FINE 或权重缺失
    会给出明确报错。
  - model.base=rtdetr: 设计文档允许的回退 (第六版 RT-DETR 实现, 需第六版代码可用)。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .dfine_adapter import DFINECore
from .relation_refiner import RelationAwareRefiner


def _build_rtdetr_fallback(config: Dict[str, Any], device):
    """回退到第六版 RT-DETR 实现 (README 声明的降级路径)。

    第六版与第七版的相对位置在本地 (并列目录) 与远程 (aic/v6) 不同, 依次尝试;
    亦可用 model.rtdetr_fallback_root 显式指定。
    """
    from pathlib import Path
    import sys

    here = Path(__file__).resolve()
    candidates = []
    if config.get("model", {}).get("rtdetr_fallback_root"):
        candidates.append(Path(config["model"]["rtdetr_fallback_root"]))
    candidates += [
        here.parents[4] / "第六版",   # 本地: <workspace>/第六版 (与第七版并列)
        Path("/root/aic/v6/第六版"),  # 远程 westd 布局
    ]
    for v6_root in candidates:
        if not (v6_root / "src" / "aic_v6").exists():
            continue
        if str(v6_root / "src") not in sys.path:
            sys.path.insert(0, str(v6_root / "src"))
        from aic_v6.models.multimodal_rtdetr import TriModalRTDETR

        return TriModalRTDETR(config, device)
    raise FileNotFoundError(f"第六版代码不存在 (尝试过: {[str(c) for c in candidates]})")


class TriModalDFINE(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        model_cfg = config.get("model", {})
        self.mode = model_cfg.get("mode", "tri")
        self._uses_dfine_core = str(model_cfg.get("base", "dfine")).lower() == "dfine"

        if self._uses_dfine_core:
            self.core = DFINECore(config, device)
        else:
            self.core = _build_rtdetr_fallback(config, device)

        relation_cfg = model_cfg.get("relation", {})
        self.relation_refiner = RelationAwareRefiner(
            num_classes=int(model_cfg.get("num_classes", 12)),
            hidden=int(relation_cfg.get("hidden", 64)),
            alpha_cooc=float(relation_cfg.get("alpha_cooc", 0.15)),
            alpha_spatial=float(relation_cfg.get("alpha_spatial", 0.10)),
            alpha_geo=float(relation_cfg.get("alpha_geo", 0.20)),
            use_appearance=bool(relation_cfg.get("use_appearance", False)),
            warmup_epochs=int(relation_cfg.get("warmup_epochs", 5)),
            topk_for_graph=int(relation_cfg.get("topk_for_graph", 50)),
            feature_dim=int(relation_cfg.get("feature_dim", 256)),
        )
        priors_path = relation_cfg.get("priors")
        if priors_path:
            self.relation_refiner.load_priors_npz(str(priors_path))
        self.relation_weight = float(config.get("loss", {}).get("relation", {}).get("weight", 0.05))
        self.refiner_enabled = self.relation_refiner.enabled and self._uses_dfine_core
        self.to(device)
        self.phase = 1

    # -- forward paths ---------------------------------------------------------
    def predict_raw(self, batch: Dict[str, torch.Tensor], return_aux=False, flip_x: bool = False):
        """返回 (B, 300, 6) 归一化 [cx,cy,w,h,score,cls] (letterbox 空间, 无 NMS)。

        全前向强制 fp32 (v5/v6 验证的 AMP 数值稳定方案); Relation-Aware Refiner
        只改分数来源 logits、不改坐标。flip_x 供 TTA 翻转分支使用: 空间先验的
        μ_c.x 随坐标系镜像, 几何/共现先验不受影响。
        """
        if not self._uses_dfine_core:
            return self.core.predict_raw(batch, return_aux=return_aux) if hasattr(self.core, "predict_raw") else self.core.predict_raw(batch)
        with torch.autocast(device_type=batch["img"].device.type, enabled=False):
            outputs = self.core.eval_outputs(batch)
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            auxiliary: Dict[str, torch.Tensor] = {}
            if self.refiner_enabled:
                features = outputs.get("pred_features")
                logits = self.relation_refiner.refine_logits(logits, boxes, features, flip_x=flip_x)
                auxiliary["relation_warmup"] = logits.new_tensor(self.relation_refiner.warmup)
            predictions = self.core.to_predictions(logits, boxes, outputs["orig_sizes"])
        if return_aux:
            return predictions, auxiliary
        return predictions

    def loss(self, batch: Dict[str, torch.Tensor]):
        """训练损失 = 官方 DFINE 损失 + target mask 辅助 + λ_rel × 关系精炼辅助。"""
        if not self._uses_dfine_core:
            return self.core.loss(batch)
        with torch.autocast(device_type=batch["img"].device.type, enabled=False):
            total, metrics, decoder_out, targets = self.core.loss(batch)
            if self.refiner_enabled and self.relation_weight > 0:
                refined = self.relation_refiner.refine_logits(
                    decoder_out["pred_logits"].detach(),
                    decoder_out["pred_boxes"].detach(),
                    decoder_out.get("pred_features").detach() if decoder_out.get("pred_features") is not None else None,
                )
                indices = getattr(self.core.criterion, "last_indices", None)
                relation_loss = self.relation_refiner.varifocal_loss(
                    refined, decoder_out["pred_boxes"].detach(), targets, indices or []
                )
                total = total + self.relation_weight * relation_loss
                metrics["relation"] = relation_loss.detach()
                metrics["total"] = total.detach()
        return total, metrics

    def forward(self, batch: Dict[str, torch.Tensor]):
        if isinstance(batch, dict) and "boxes" in batch:
            return self.loss(batch)
        return self.predict_raw(batch)

    # -- training controls -------------------------------------------------------
    def set_phase(self, phase: int, config: dict):
        self.phase = phase
        if hasattr(self.core, "set_phase"):
            self.core.set_phase(phase, config)

    def set_epoch(self, epoch: int):
        self.relation_refiner.set_epoch(epoch)

    def parameter_role(self, name: str) -> str:
        if name.startswith("relation_refiner."):
            return "new"
        if name.startswith("core.") and hasattr(self.core, "parameter_role"):
            return self.core.parameter_role(name[len("core."):])
        if hasattr(self.core, "parameter_role"):
            return self.core.parameter_role(name)
        return "new"

    def train(self, mode: bool = True):
        self.training = mode
        nn.Module.train(self.core, mode)
        nn.Module.train(self.relation_refiner, mode)
        if mode:
            # 冻结层的 BN 统计在 model.train() 后仍须保持 eval (v6 经验)
            if hasattr(self.core, "_apply_frozen_eval"):
                self.core._apply_frozen_eval()
        return self

    def eval(self):
        return self.train(False)


def build_dfine(config: Union[str, Dict[str, Any]], device: str = "cpu"):
    if isinstance(config, (str,)):
        from ..config import load_config

        config = load_config(config).to_dict()
    return TriModalDFINE(config, device=device)
