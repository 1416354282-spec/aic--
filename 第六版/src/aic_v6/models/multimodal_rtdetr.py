"""TriModalRTDETR: multimodal wrapper around RT-DETR (ultralytics, HGNetv2 + hybrid encoder + transformer decoder).

Two operating modes:
  - rgb_only:  plain RT-DETR forward (E0 baseline) — the detector is used as-is.
  - tri-modal: manual forward that replaces the three backbone feature maps
               (S3/S4/S5, model layers 3/7/9) with the fused features before
               the pretrained hybrid encoder (AIFI + CCFF) and RTDETRDecoder (E1+).

Fusion follows v3's Tri-CSSA (default) or v5's CMTF (ablation), both with
residual identity init (F ≈ S) so the pretrained encoder/decoder input
distribution stays intact; the continuous quality prior makes missing
modalities collapse to RGB.  The RT-DETR detection loss (focal cls + L1 +
GIoU + denoising) is reused from ultralytics via ``detector.loss(preds=...)``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmtf import CMTF
from .losses import ScaleBalancedRTDETRLoss
from .modality_encoders import DepthEncoder, IREncoder
from .target_refinement import TargetAwareRefiner, build_soft_box_targets, target_mask_loss
from .tri_cssa import TriCSSA

# RT-DETR (ultralytics rtdetr-l.yaml) 层布局:
#   0-9    HGNetv2 backbone; S3/S4/S5 = 层 3/7/9 的输出 (stride 8/16/32)
#   10-27  hybrid encoder: input_proj + AIFI + CCFF (FPN/PAN)
#   28     RTDETRDecoder (IoU-aware query selection + 6 层 deformable decoder)
FEATURE_LAYERS = (3, 7, 9)
BACKBONE_END = 9


def _resolve_layer_input(layer, current, outputs):
    source = layer.f
    if source == -1:
        return current
    if isinstance(source, int):
        return outputs[source]
    return [current if index == -1 else outputs[index] for index in source]


def modality_quality(infrared, depth_valid, ir_quality=None, depth_quality=None):
    """Continuous per-sample modality quality in [0, 1].

    q_rgb = 1 (always available fallback); q_ir = fraction of non-zero IR
    pixels; q_depth = mean of the real depth validity mask (0 when depth is
    missing).  ``ir_quality`` / ``depth_quality`` are optional per-sample
    scalars computed by the data pipeline *before* letterbox, so black padding
    is not counted as missing modality.
    """
    batch = infrared.shape[0]
    rgb_quality = torch.ones(batch, device=infrared.device, dtype=infrared.dtype)
    if ir_quality is None:
        ir_quality = (infrared.abs() > 1e-6).to(infrared.dtype).mean((1, 2, 3))
    else:
        ir_quality = ir_quality.to(device=infrared.device, dtype=infrared.dtype).flatten()
    if depth_quality is None:
        depth_quality = depth_valid.flatten(1).mean(1)
    else:
        depth_quality = depth_quality.to(device=infrared.device, dtype=infrared.dtype).flatten()
    return torch.stack((rgb_quality, ir_quality, depth_quality), dim=1)


def build_rtdetr(
    weights: str,
    num_classes: int = 12,
    device: str = "cpu",
    yaml_cfg: str = "rtdetr-l.yaml",
):
    """Build an ultralytics RTDETRDetectionModel with nc=num_classes and load pretrained weights.

    Pretrained class heads / denoising embeddings are shape-incompatible
    (80 vs num_classes rows) and get skipped by intersect_dicts → randomly
    initialized; backbone / hybrid encoder / decoder body / bbox heads keep
    the full COCO pretraining.
    """
    from ultralytics.nn.tasks import RTDETRDetectionModel

    detector = RTDETRDetectionModel(yaml_cfg, nc=num_classes, verbose=False)
    detector.nc = num_classes  # ultralytics 8.4 模型实例不暴露 nc, init_criterion 需要
    if not weights:
        raise FileNotFoundError("model.weights 未配置")
    if not Path(weights).exists():
        raise FileNotFoundError(f"RT-DETR 权重不存在: {weights}（先用镜像下载并离线保存）")
    # checkpoint["model"] 是完整模型实例; BaseModel.load 内部 intersect_dicts
    # 会跳过形状不匹配的类别头/denoising 嵌入 (80 vs nc)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    detector.load(checkpoint)
    detector.to(device)
    return detector


class TriModalRTDETR(nn.Module):
    def __init__(self, config: dict, device: str | torch.device = "cpu"):
        super().__init__()
        model_config = config["model"]
        self.mode = model_config.get("mode", "rgb_only")  # "rgb_only" | "tri"
        self.device = device
        self.fusion_name = model_config.get("fusion", "tri_cssa")  # "tri_cssa" | "cmtf"
        # DepthEncoder 输入通道: 3 = HHA 三通道; 4 = 追加真实 validity 掩码 (需重训)
        self.depth_in_channels = int(model_config.get("depth_in_channels", 3))
        if self.depth_in_channels not in (3, 4):
            raise ValueError(f"depth_in_channels 只能是 3 或 4, 收到 {self.depth_in_channels}")
        # 空间质量先验: 把 depth 有效率图作为逐 token bias 注入 Tri-CSSA
        self.spatial_quality = bool(model_config.get("spatial_quality", False))
        if self.mode == "tri" and self.spatial_quality and self.fusion_name != "tri_cssa":
            raise ValueError("spatial_quality 目前只支持 fusion=tri_cssa")
        self.detector = build_rtdetr(
            str(model_config.get("weights", "")),
            num_classes=int(model_config["num_classes"]),
            device=device,
            yaml_cfg=str(model_config.get("yaml_cfg", "rtdetr-l.yaml")),
        )
        self.loss_config = config.get("loss", {})
        self.regression_loss = dict(model_config.get("regression_loss") or {})
        if self.regression_loss:
            # 尺度归一化回归 (小目标高 IoU 定位): 未配置时保持官方 criterion,
            # 配置里两个旋钮都是 0 时内部也回落到官方实现 → 逐位相同 (见 losses.py)
            self.detector.criterion = ScaleBalancedRTDETRLoss(
                nc=int(model_config["num_classes"]), **self.regression_loss
            )

        self.feature_layers = tuple(int(i) for i in model_config.get("feature_layers", FEATURE_LAYERS))
        self.backbone_end = int(model_config.get("backbone_end", BACKBONE_END))
        self.ir_encoder = None
        self.depth_encoder = None
        self.fusions = nn.ModuleList()
        self.refiners = nn.ModuleDict()
        if self.mode == "tri":
            self._build_multimodal(model_config)
        self.to(device)
        self.phase = 1

    # -- multimodal branches -------------------------------------------------
    def _build_multimodal(self, model_config):
        out_strides = tuple(int(s) for s in model_config.get("out_strides", [8, 16, 32]))
        self.out_strides = out_strides
        rgb_channels = self._probe_rgb_channels(int(model_config.get("probe_size", 256)))
        self.ir_encoder = IREncoder(
            rgb_channels,
            width=float(model_config.get("ir_width", 0.375)),
            attention_groups=int(model_config.get("attention_groups", 8)),
            out_strides=out_strides,
        )
        self.depth_encoder = DepthEncoder(
            rgb_channels,
            width=float(model_config.get("depth_width", 0.375)),
            out_strides=out_strides,
            in_channels=self.depth_in_channels,
        )
        fusion_cls = CMTF if self.fusion_name == "cmtf" else TriCSSA
        if fusion_cls is CMTF:
            self.fusions = nn.ModuleList(
                CMTF(
                    rgb_channels[i],
                    self.ir_encoder.out_channels[i],
                    self.depth_encoder.out_channels[i],
                    nhead=int(model_config.get("cmtf_nhead", 2)),
                    use_cross_attn=bool(model_config.get("cmtf_cross_attn", True)),
                    use_lite=bool(model_config.get("cmtf_lite", False)),
                )
                for i in range(len(out_strides))
            )
        else:
            self.fusions = nn.ModuleList(
                TriCSSA(rgb_channels[i], self.ir_encoder.out_channels[i], self.depth_encoder.out_channels[i])
                for i in range(len(out_strides))
            )
        refinement_levels = {int(i) for i in model_config.get("target_refinement_levels", [0, 1])}
        self.refiners = nn.ModuleDict({str(i): TargetAwareRefiner(rgb_channels[i]) for i in refinement_levels})

    def _probe_rgb_channels(self, size: int):
        """Run the pretrained backbone once to read the fusion feature channel counts."""
        training = self.detector.training
        self.detector.eval()
        parameter = next(self.detector.parameters())
        with torch.no_grad():
            features, _ = self._run_rgb_backbone(
                torch.zeros(1, 3, size, size, device=parameter.device, dtype=parameter.dtype)
            )
        self.detector.train(training)
        channels = tuple(int(feature.shape[1]) for feature in features)
        if len(channels) != len(self.feature_layers):
            raise RuntimeError(f"backbone 输出 {len(channels)} 个尺度, 配置期望 {len(self.feature_layers)}: {channels}")
        # 校验各尺度 stride 与 out_strides 一致 —— RT-DETR-L (层3/7/9) 与
        # RT-DETR-X (层4/10/13) 的 backbone 层索引不同, 防止配置错配导致融合尺寸崩坏
        strides = tuple(size // int(feature.shape[2]) for feature in features)
        if strides != self.out_strides:
            raise RuntimeError(
                f"backbone 实际 strides={strides} != 配置 out_strides={self.out_strides} "
                f"(feature_layers={self.feature_layers}, 通道={channels}); "
                f"L 用 [3,7,9]/9, X 用 [4,10,13]/13"
            )
        return channels

    # -- forward paths -------------------------------------------------------
    def _run_rgb_backbone(self, rgb):
        outputs, features = [], {}
        current = rgb
        for layer in self.detector.model[:self.backbone_end + 1]:
            current = layer(_resolve_layer_input(layer, current, outputs))
            if layer.i in self.feature_layers:
                features[layer.i] = current
            outputs.append(current if layer.i in self.detector.save else None)
        if set(features) != set(self.feature_layers):
            raise RuntimeError(f"未获取全部 RGB 融合层: expected={self.feature_layers}, actual={sorted(features)}")
        return tuple(features[i] for i in self.feature_layers), outputs

    def _run_neck_and_head(self, outputs, current, head_batch):
        """Layers backbone_end+1 .. -2 (AIFI + CCFF), then the RTDETRDecoder head."""
        for layer in self.detector.model[self.backbone_end + 1:-1]:
            current = layer(_resolve_layer_input(layer, current, outputs))
            outputs.append(current if layer.i in self.detector.save else None)
        head = self.detector.model[-1]
        return head([outputs[j] for j in head.f], head_batch)

    def _head_batch(self, batch):
        """RT-DETR denoising needs ultralytics-style targets; only in training with GT."""
        if not self.training or batch.get("num_boxes") is None:
            return None
        img = batch["img"]
        cls, bboxes, batch_idx = [], [], []
        for i, count in enumerate(batch["num_boxes"].tolist()):
            for j in range(int(count)):
                cls.append(batch["labels"][i, j])
                bboxes.append(batch["boxes"][i, j])
                batch_idx.append(i)
        device, dtype = img.device, img.dtype
        if bboxes:
            bboxes = torch.stack(bboxes).to(device)
            cls = torch.tensor(cls, device=device, dtype=torch.long)
            batch_idx = torch.tensor(batch_idx, device=device, dtype=torch.long)
        else:
            bboxes = torch.zeros(0, 4, device=device, dtype=dtype)
            cls = torch.zeros(0, device=device, dtype=torch.long)
            batch_idx = torch.zeros(0, device=device, dtype=torch.long)
        return {
            "img": img,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "gt_groups": [int(c) for c in batch["num_boxes"].tolist()],
        }

    def predict_raw(self, batch, return_aux=False):
        """batch: dict with img(B,3,H,W), ir(B,1,H,W), depth(B,C,H,W), depth_valid(B,1,H,W).

        训练态返回 RTDETRDecoder 的 5 元组 (dec_bboxes, dec_scores, enc_bboxes,
        enc_scores, dn_meta); eval 态返回 (y, x), y 为 (B, Q, 6) 归一化
        [cx, cy, w, h, score, cls]（score 已 sigmoid, 无 NMS）。
        """
        rgb = batch["img"]
        # 全前向强制 fp32: fp16 AMP 下 backbone/编码器在高分辨率 (1920x1088)
        # 训练模式偶发溢出 NaN (criterion fp32 后仍出现 5 次), 已无保留 fp16 的意义
        orig_dtype = rgb.dtype
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            rgb = rgb.float()
            rgb_features, outputs = self._run_rgb_backbone(rgb)

            auxiliary = {}
            current = rgb_features[-1]
            if self.mode == "tri":
                infrared = batch["ir"].float()
                depth = batch["depth"].float()
                depth_valid = batch["depth_valid"].float()
                if depth.shape[1] == 1:
                    # raw 模式: 归一化深度 + valid + format 组成 3 通道编码器输入
                    depth_format = batch.get("depth_format")
                    if depth_format is None:
                        depth_format = torch.zeros_like(depth_valid)
                    else:
                        depth_format = depth_format.to(device=depth.device, dtype=depth.dtype)
                    depth = torch.cat((depth, depth_valid, depth_format), dim=1)
                    if self.depth_in_channels != 3:
                        raise ValueError("raw 深度编码固定为 3 通道, depth_in_channels 必须为 3")
                elif self.depth_in_channels == 4:
                    # HHA 3 通道 + 真实有效性掩码 → 显式告诉编码器哪里没有深度
                    depth = torch.cat((depth, depth_valid), dim=1)
                if depth.shape[1] != self.depth_in_channels:
                    raise ValueError(
                        f"depth 输入 {depth.shape[1]} 通道 != depth_in_channels={self.depth_in_channels}"
                    )
                quality = modality_quality(
                    infrared, depth_valid, batch.get("ir_quality"), batch.get("depth_quality")
                )
                ir_feats = self.ir_encoder(infrared)
                depth_feats = self.depth_encoder(depth)
                fused, mask_logits = [], []
                for i, (rgb_feature, ir_feature, depth_feature) in enumerate(
                    zip(rgb_features, ir_feats, depth_feats)
                ):
                    fusion_kwargs = {}
                    if self.spatial_quality:
                        # 逐 token 的深度有效率, 让融合权重在深度缺失区域自动降权
                        fusion_kwargs["depth_quality"] = F.interpolate(
                            depth_valid, size=rgb_feature.shape[-2:], mode="area"
                        )
                    feature = self.fusions[i](
                        rgb_feature, ir_feature, depth_feature, quality, **fusion_kwargs
                    )
                    refiner = self.refiners[str(i)] if str(i) in self.refiners else None
                    if refiner is not None:
                        feature, logits = refiner(feature)
                        mask_logits.append(logits)
                    fused.append(feature)
                for layer_index, feature in zip(self.feature_layers, fused):
                    outputs[layer_index] = feature
                current = fused[-1]
                auxiliary["mask_logits"] = mask_logits
                auxiliary["quality"] = quality

            # neck (AIFI + CCFF) + RTDETRDecoder 也全部 fp32 (同一 autocast 块)
            result = self._run_neck_and_head(outputs, current, self._head_batch(batch))
        if return_aux:
            return result, auxiliary
        return result

    def loss(self, batch):
        """batch: dict with img/ir/depth/depth_valid + boxes(B,N,4) + labels(B,N) + num_boxes(B)."""
        preds, auxiliary = self.predict_raw(batch, return_aux=True)
        head_batch = self._head_batch(batch)
        if not hasattr(self.detector, "criterion"):
            self.detector.criterion = self.detector.init_criterion()
        # criterion (GIoU/L1/focal) 在 fp16 下可能数值溢出产生 NaN (epoch 67 实测 12 次),
        # 强制 fp32 计算损失 (前向已在 predict_raw 内部分 fp32)
        with torch.autocast(device_type=batch["img"].device.type, enabled=False):
            total, detection_items = self.detector.loss(head_batch, preds=preds)
        metrics = {"detection": total.detach()}
        for key, value in detection_items.items():
            if torch.is_tensor(value):
                metrics[f"loss_{key}"] = value.detach()

        mask_config = self.loss_config.get("target_mask", {})
        if self.mode == "tri" and mask_config.get("enabled", True) and auxiliary["mask_logits"]:
            # mask 辅助损失同样强制 fp32 (与 criterion 同一 autocast 块内)
            with torch.autocast(device_type=batch["img"].device.type, enabled=False):
                shapes = [tuple(logits.shape[-2:]) for logits in auxiliary["mask_logits"]]
                batch_indices, flat_boxes = self._flatten_boxes(batch)
                soft_targets = build_soft_box_targets(
                    batch_indices, flat_boxes, batch["img"].shape[0],
                    shapes, batch["img"].device, batch["img"].dtype,
                    float(mask_config.get("center_sigma", 0.35)),
                )
                mask_loss = target_mask_loss(
                    auxiliary["mask_logits"], soft_targets,
                    gamma=float(mask_config.get("focal_gamma", 2.0)),
                    dice_weight=float(mask_config.get("dice_weight", 1.0)),
                )
                total = total + batch["img"].shape[0] * float(mask_config.get("weight", 0.1)) * mask_loss
                metrics["target_mask"] = mask_loss.detach()
        metrics["total"] = total.detach()
        return total, metrics

    @staticmethod
    def _flatten_boxes(batch):
        """Convert padded (B, N, 4) boxes to v3-style flat (T, 4) + batch indices."""
        boxes, indices = [], []
        for i, count in enumerate(batch["num_boxes"].tolist()):
            for j in range(int(count)):
                boxes.append(batch["boxes"][i, j])
                indices.append(i)
        if not boxes:
            return torch.zeros(0, device=batch["boxes"].device, dtype=torch.long), batch["boxes"].reshape(-1, 4)[:0]
        return (
            torch.tensor(indices, device=batch["boxes"].device, dtype=torch.long),
            torch.stack(boxes),
        )

    def forward(self, batch):
        if isinstance(batch, dict) and "boxes" in batch:
            return self.loss(batch)
        return self.predict_raw(batch)

    # -- training controls ---------------------------------------------------
    def set_phase(self, phase: int, config: dict):
        """Freeze/unfreeze by training phase (1: new branches only; 2: + neck/decoder;
        3: everything). Frozen modules switch to eval() so BN statistics stay fixed.

        Classifier heads (dec_score_head / enc_score_head / denoising_class_embed)
        are newly-added parameters (nc 80->12) and train from Phase 1 by default
        (freeze_classifier: false), so the 12-class predictor is shaped early.
        """
        self.phase = phase
        phases = config.get("train", {}).get("phases", [])
        spec = phases[phase - 1] if 0 < phase <= len(phases) else {}
        freeze_backbone = bool(spec.get("freeze_backbone", phase == 1))
        freeze_decoder = bool(spec.get("freeze_decoder", phase == 1))
        freeze_classifier = bool(spec.get("freeze_classifier", False))
        for name, param in self.named_parameters():
            role = self.parameter_role(name)
            if role == "rgb":
                param.requires_grad_(not freeze_backbone)
            elif role == "detector":
                param.requires_grad_(not freeze_decoder)
            elif role == "classifier":
                param.requires_grad_(not freeze_classifier)
            else:
                param.requires_grad_(True)
        self._frozen_eval = (freeze_backbone, freeze_decoder)
        self._set_frozen_eval(freeze_backbone, freeze_decoder)

    def train(self, mode: bool = True):
        """Keep frozen phases in eval() even after ``model.train()``.

        ``trainer.py`` calls ``model.train()`` every epoch, which would undo
        the eval() set by ``set_phase()`` and let the frozen backbone's
        BatchNorm running statistics drift on the augmented training set —
        exactly the contamination the phase freezing is meant to prevent.
        Re-apply the boundary state after the recursive train() call.
        """
        super().train(mode)
        if mode and getattr(self, "_frozen_eval", None):
            self._set_frozen_eval(*self._frozen_eval)
        return self

    def _set_frozen_eval(self, freeze_backbone: bool, freeze_decoder: bool):
        """Frozen submodules must run in eval() so BatchNorm statistics are not updated.

        The last layer (``RTDETRDecoder``) is deliberately excluded: its
        ``training`` flag selects between the 5-tuple training output (with
        denoising) and the 2-tuple eval output, and ``detector.loss`` unpacks
        the former.  It has no BatchNorm, so there is nothing to freeze in it
        beyond ``requires_grad`` (already handled by ``set_phase``).
        """
        layers = self.detector.model
        if freeze_backbone:
            for layer in layers[:self.backbone_end + 1]:
                layer.eval()
        if freeze_decoder:
            for layer in layers[self.backbone_end + 1:-1]:
                layer.eval()

    def parameter_role(self, name: str) -> str:
        if not name.startswith("detector.model."):
            return "new"
        parts = name.split(".")
        try:
            layer_index = int(parts[2])
        except (IndexError, ValueError):
            return "detector"
        if layer_index <= self.backbone_end:
            return "rgb"
        # RTDETRDecoder 的分类头 (新增参数, 80->12 随机初始化): 单独角色, Phase 1 起训练
        if layer_index == len(self.detector.model) - 1 and any(
            key in name for key in ("dec_score_head", "enc_score_head", "denoising_class_embed")
        ):
            return "classifier"
        return "detector"
